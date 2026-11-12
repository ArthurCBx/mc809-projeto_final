import grpc
import os
import argparse
import proto_config.message_pb2 as pb2
import proto_config.message_pb2_grpc as pb2_grpc
import logging
import ray
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from dataset_setup import CLASS_NAMES, list_dir, list_files, split_clients, split_dataset, split_external, make_client_dataset

BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
NUM_CLIENTS = 4
INIT_EPOCHS = 50              # máximo de épocas do treino inicial nos dados públicos (early stopping decide antes)
PRIVATE_INIT_EPOCHS = 5       # máximo do treino inicial privado: com uma classe só, a val_loss não para de cair
                              # e o early stopping deixaria o cliente colapsar por dezenas de épocas
PATIENCE = 4                  # épocas sem melhora na validação antes de parar
VAL_FRACTION = 0.2            # fração separada para validação, do público e do privado de cada cliente
PUBLIC_ROUND_FRACTION = 0.8   # fração do conjunto público sorteada a cada rodada federada
MIXED_PRECISION = False       # mixed_float16: com 4 clientes permite batch 64 em vez de 32
OUTPUT_DIR = 'results'

# Uma arquitetura diferente por cliente, como no FedMD: (filtros, kernel) de cada camada convolucional.
CLIENT_ARCHITECTURES = [
    [(16, 3), (32, 3), (64, 3)],
    [(32, 3), (64, 3), (128, 3)],
    [(16, 5), (32, 5), (64, 3), (128, 3)],
    [(32, 3), (32, 3), (64, 3), (64, 3), (128, 3)],
]

TRAIN_COLUMNS = ['seed', 'public_fraction', 'alpha', 'client_id', 'stage', 'round', 'epoch',
                 'loss', 'accuracy', 'val_loss', 'val_accuracy', 'restored', 'public_accuracy']
ENSEMBLE_COLUMNS = ['seed', 'public_fraction', 'alpha', 'stage', 'round', 'ensemble_accuracy', 'mean_client_accuracy']
EXTRA_TEST_COLUMNS = ['seed', 'public_fraction', 'alpha', 'stage', 'round', 'model', 'accuracy', 'balanced_accuracy']
TEST_COLUMNS  = ['seed', 'public_fraction', 'alpha', 'client_id', 'stage', 'round',
                 'n_private', 'private_malignant_fraction', 'test_loss', 'test_accuracy']


def small_cnn(filters):
    """Definição de função que constrói uma CNN pequena com a quantidade de filtros especificada em cada camada convolucional.
    Args:
        filters list(tuple(int)): Lista com uma tupla contendo quantidade de filtros para cada camada e o tamanho do kernel.

    Returns:
        function: Função que aplica as camadas convolucionais.
    """

    def f(x):
        for n in filters:
            x = keras.layers.Conv2D(n[0], n[1], padding="same", activation="gelu")(x)
            x = keras.layers.BatchNormalization(momentum=0.9)(x)
            x = keras.layers.MaxPooling2D()(x)
        return x
    return f


def build_model(backbone_fn, num_classes=2, image_size=(224, 224), jit_compile='auto'):
    inputs = keras.Input((*image_size, 3))
    x = keras.layers.Rescaling(1 / 255)(inputs)   # make_client_dataset entrega pixels em 0–255
    x = backbone_fn(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(128, activation="gelu")(x) # logits: é o que o FedMD troca no consenso
    outputs = keras.layers.Dense(num_classes, dtype="float32")(outputs)  # logits em float32 mesmo com mixed precision
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.AdamW(1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        jit_compile=jit_compile,
    )
    return model


def augmentation_pipeline(image_size=(224, 224), train: bool = True):
    """Cria um pipeline de aumento de dados para o conjunto de treinamento em cima dos dados privados."""
    if not train:
        return keras.Sequential([
            keras.layers.Resizing(*image_size),
        ])
    return keras.Sequential([
        keras.layers.Resizing(*image_size),
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1),
    ])


def distillation_kl(temperature: float):
    """Loss de destilação: T² · KL(softmax(consenso/T) ‖ softmax(cliente/T)), com consenso e cliente em logits.

    O fator T² mantém a escala do gradiente quando T cresce (Hinton et al., 2015).
    """
    def loss(consensus_logits, client_logits):
        target = keras.ops.softmax(consensus_logits / temperature, axis=-1)
        log_pred = keras.ops.log_softmax(client_logits / temperature, axis=-1)
        log_target = keras.ops.log(keras.ops.clip(target, 1e-7, 1.0))
        return temperature ** 2 * keras.ops.sum(target * (log_target - log_pred), axis=-1)
    return loss


def with_label_anchor(distill_loss, ce_weight: float):
    """Mistura a loss do digest com a entropia cruzada no rótulo verdadeiro: (1 − w) · destilação + w · CE.

    O alvo vira [consenso (num_classes logits), rótulo]: as primeiras colunas vão para distill_loss e a última é o
    rótulo inteiro. Com w = 1, o digest é só treino supervisionado nas imagens públicas, sem destilação.
    """
    def loss(target, client_logits):
        consensus, label = target[:, :-1], keras.ops.cast(target[:, -1], 'int32')
        ce = keras.losses.sparse_categorical_crossentropy(label, client_logits, from_logits=True)
        return (1 - ce_weight) * distill_loss(consensus, client_logits) + ce_weight * ce
    return loss


def holdout_split(indices, val_fraction: float, seed: int):
    """Separa aleatoriamente uma fração dos índices para validação.

    Não é estratificado: na divisão não IID um cliente pode ter só uma ou duas amostras de uma classe.
    """
    indices = np.random.default_rng(seed).permutation(indices)
    n_val = max(1, int(round(len(indices) * val_fraction)))
    return indices[n_val:], indices[:n_val]


@ray.remote(num_cpus=1, num_gpus=1 / NUM_CLIENTS)
class Client():
    def __init__(self, client_id: int, filters, file_paths: np.ndarray, labels: np.ndarray,
                 private_train_idx: np.ndarray, private_val_idx: np.ndarray,
                 test_paths: np.ndarray, test_labels: np.ndarray,
                 batch_size: int=BATCH_SIZE, gpu_memory_mb: int=None, mixed_precision: bool=MIXED_PRECISION, seed: int=42,
                 digest_lr: float=1e-3, digest_temperature: float=None, digest_ce_weight: float=0.0,
                 private_batch_size: int=None, jit_compile='auto'):
        """Cliente do FedMD.

        Recebe índices em vez de tf.data.Dataset porque datasets não podem ser serializados para o actor;
        os datasets são montados aqui dentro a partir de file_paths e labels.

        Args:
            filters (list(tuple(int))): Arquitetura do modelo do cliente, no formato de small_cnn.
            file_paths (np.ndarray): Caminhos de todo o conjunto de treino (público e privado).
            labels (np.ndarray): Rótulos de todo o conjunto de treino.
            private_train_idx (np.ndarray): Índices privados do cliente usados no treino.
            private_val_idx (np.ndarray): Índices privados do cliente usados na validação.
            test_paths (np.ndarray): Caminhos do conjunto de teste.
            test_labels (np.ndarray): Rótulos do conjunto de teste.
            gpu_memory_mb (int, optional): Limite de memória de GPU deste cliente. Os clientes dividem a mesma GPU
                                           e, sem limite, o primeiro TensorFlow a subir reserva quase toda a memória.
            mixed_precision (bool, optional): Treina em mixed_float16, o que reduz a memória das ativações.
            digest_lr (float, optional): Learning rate do digest (aproximação do consenso). Defaults to 1e-3.
            digest_temperature (float, optional): Se definido, o digest usa KL com essa temperatura em vez de MAE
                                                  nos logits. Defaults to None.
            digest_ce_weight (float, optional): Peso w da entropia cruzada no rótulo verdadeiro dentro do digest:
                                                (1 − w) · destilação + w · CE. Defaults to 0 (só destilação).
            private_batch_size (int, optional): Batch do treino privado (private_init e revisit), para clientes com
                                                poucos dados privados. Defaults to None (o mesmo batch_size).
        """
        gpus = tf.config.list_physical_devices('GPU')
        if gpus and gpu_memory_mb:
            tf.config.set_logical_device_configuration(
                gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=gpu_memory_mb)])
        if mixed_precision:
            # Ativações em float16 ocupam metade da memória; pesos e logits continuam em float32.
            keras.mixed_precision.set_global_policy('mixed_float16')
        keras.utils.set_random_seed(seed + client_id)

        self.client_id         = client_id
        # jit_compile=False evita os picos de memória da compilação XLA (um formato de batch novo recompila e autotuna).
        self.model             = build_model(small_cnn(filters), num_classes=len(CLASS_NAMES), image_size=IMAGE_SIZE,
                                             jit_compile=jit_compile)
        # Mesmas camadas (e pesos) de self.model, compilado uma vez só com a loss do digest. Recompilar self.model
        # a cada rodada recria e recompila a função de treino, e a RAM do worker cresce ~85 MB por rodada.
        self.digest_model      = keras.Model(self.model.inputs, self.model.outputs)
        digest_loss = (keras.losses.mean_absolute_error if digest_temperature is None
                       else distillation_kl(digest_temperature))
        self.digest_ce_weight = digest_ce_weight
        if digest_ce_weight > 0:
            digest_loss = with_label_anchor(digest_loss, digest_ce_weight)
        self.digest_model.compile(optimizer=keras.optimizers.AdamW(digest_lr), loss=digest_loss, jit_compile=jit_compile)
        self.file_paths        = file_paths    # dataset inteiro; fed_train recorta o subconjunto público
        self.labels            = labels
        self.private_train_idx = private_train_idx
        self.private_val_idx   = private_val_idx
        self.batch_size        = batch_size
        self.private_batch_size = private_batch_size or batch_size
        self.augment           = augmentation_pipeline(IMAGE_SIZE, train=True)
        self.test_dataset      = make_client_dataset(test_paths, test_labels, np.arange(len(test_labels)),
                                                     IMAGE_SIZE, batch_size, shuffle=False)

        self.host        = 'localhost'
        self.server_port = 50051
        self.channel     = grpc.insecure_channel(f'{self.host}:{self.server_port}')
        self.stub        = pb2_grpc.FedMDLearningStub(self.channel)

    def train(self, train_idx, val_idx, epochs: int=1, patience: int=PATIENCE, augment: bool=True,
              learning_rate: float=None, steps_per_epoch: int=None, rehearsal_idx: np.ndarray=None,
              rehearsal_fraction: float=0.5, batch_size: int=None):
        """Treina o modelo do cliente com os rótulos verdadeiros, com early stopping na validação.

        Args:
            train_idx (np.ndarray): Índices de treino, relativos a self.file_paths.
            val_idx (np.ndarray): Índices de validação, relativos a self.file_paths.
            epochs (int, optional): Número máximo de épocas. Defaults to 1.
            patience (int, optional): Número de épocas sem melhoria para parar o treinamento. Defaults to 4.
            augment (bool, optional): Aplica augmentation_pipeline nas imagens de treino. Defaults to True.
            learning_rate (float, optional): Novo learning rate do otimizador, mantido nas chamadas seguintes.
                                             Se None, usa o atual. Defaults to None.
            steps_per_epoch (int, optional): Passos de gradiente por época, iguais para todo cliente. O dataset é
                                             repetido, então clientes com poucas imagens também dão todos os passos.
                                             Se None, uma época é uma passada pelos dados. Defaults to None.
            rehearsal_idx (np.ndarray, optional): Rehearsal: índices públicos (com rótulo verdadeiro) misturados aos
                                                  dados de treino, para o cliente não esquecer as classes que não
                                                  tem nos dados privados. Defaults to None.
            rehearsal_fraction (float, optional): Fração esperada de amostras públicas em cada batch. Defaults to 0.5.
            batch_size (int, optional): Batch deste treino. Defaults to None (self.batch_size).

        Sem índices de validação, treina todas as épocas sem early stopping e devolve NaN nas métricas de validação.

        Returns:
            list[dict]: Uma linha por época com loss e acurácia em treino e validação. 'restored' marca a
                        época cujos pesos ficaram no modelo.
        """
        batch_size = batch_size or self.batch_size
        train_dataset = make_client_dataset(self.file_paths, self.labels, train_idx, IMAGE_SIZE, batch_size)
        if rehearsal_idx is not None:
            # Sorteia cada amostra do privado ou do público e remonta os batches. Os dois fluxos são infinitos, então
            # a época precisa de steps_per_epoch; sem ele, mantém o número de passos de uma passada pelo privado.
            public_dataset = make_client_dataset(self.file_paths, self.labels, rehearsal_idx, IMAGE_SIZE, batch_size)
            train_dataset = tf.data.Dataset.sample_from_datasets(
                [train_dataset.unbatch().repeat(), public_dataset.unbatch().repeat()],
                weights=[1 - rehearsal_fraction, rehearsal_fraction], seed=self.client_id,
            ).batch(batch_size)
            steps_per_epoch = steps_per_epoch or int(np.ceil(len(train_idx) / batch_size))
        elif steps_per_epoch is not None:
            train_dataset = train_dataset.repeat()
        if augment:
            train_dataset = train_dataset.map(lambda x, y: (self.augment(x, training=True), y),
                                              num_parallel_calls=tf.data.AUTOTUNE)
        has_val = len(val_idx) > 0
        validation_dataset = (make_client_dataset(self.file_paths, self.labels, val_idx, IMAGE_SIZE,
                                                  batch_size, shuffle=False) if has_val else None)

        if learning_rate is not None:
            # assign em vez de recompilar: não recria a função de treino (ver digest_model no __init__).
            self.model.optimizer.learning_rate.assign(learning_rate)

        callbacks = ([keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, min_delta=1e-4, restore_best_weights=True)]
                     if has_val else [])
        history = self.model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks,
            verbose=0
        )
        h = history.history
        if not has_val:
            h['val_loss'] = h['val_accuracy'] = [float('nan')] * len(h['loss'])
        best = int(np.nanargmin(h['val_loss'])) if has_val else len(h['loss']) - 1
        return [
            {'epoch': epoch, 'loss': h['loss'][epoch], 'accuracy': h['accuracy'][epoch],
             'val_loss': h['val_loss'][epoch], 'val_accuracy': h['val_accuracy'][epoch], 'restored': epoch == best}
            for epoch in range(len(h['loss']))
        ]

    def train_private(self, epochs: int=1, patience: int=PATIENCE, augment: bool=True, learning_rate: float=None,
                      steps_per_epoch: int=None, rehearsal_idx: np.ndarray=None, rehearsal_fraction: float=0.5):
        """Treina com os dados privados do cliente. Mesmo retorno de train."""
        return self.train(self.private_train_idx, self.private_val_idx, epochs, patience, augment, learning_rate,
                          steps_per_epoch, rehearsal_idx, rehearsal_fraction, self.private_batch_size)

    def fed_train(self, public_idx: np.ndarray, epochs: int=1, steps_per_epoch: int=None):
        """Treina o modelo do cliente com os dados públicos, usando o consenso do servidor como alvo.

        Args:
            public_idx (np.ndarray): Índices do subconjunto público desta rodada, relativos a self.file_paths.
            epochs (int, optional): Número de épocas. Defaults to 1.
            steps_per_epoch (int, optional): Passos do digest por época, iguais em qualquer fração pública. Os logits
                                             e o consenso continuam calculados sobre todo public_idx; só o treino
                                             usa os N batches sorteados. Se None, uma passada por public_idx.

        Returns:
            list[dict]: Uma linha por época com a perda do modelo em relação ao consenso e a acurácia do
                        cliente no subconjunto público antes do digest (o peso enviado ao servidor).
        """
        # Sem shuffle: a linha i dos logits precisa corresponder a public_idx[i].
        public_dataset = make_client_dataset(self.file_paths, self.labels, public_idx, IMAGE_SIZE, self.batch_size, shuffle=False)
        logits = self.model.predict(public_dataset, verbose=0)
        # O conjunto público é rotulado: o servidor usa essa acurácia para pesar o cliente no consenso.
        public_accuracy = float(np.mean(np.argmax(logits, axis=1) == self.labels[public_idx]))

        # O proto só aceita floats em uma lista plana, então achata e remonta (N, num_classes).
        logits_message = pb2.ClientLogits(client_id=self.client_id, logits=logits.ravel().tolist(),
                                          public_accuracy=public_accuracy)
        response = self.stub.SendClientLogits(logits_message)
        consensus = np.array(response.consensus_logits, dtype=np.float32).reshape(logits.shape)

        # Mesmas imagens, agora embaralhadas junto com o consenso de cada uma (e o rótulo, se o digest usa CE).
        targets = consensus
        if self.digest_ce_weight > 0:
            targets = np.concatenate([consensus, self.labels[public_idx, None].astype(np.float32)], axis=1)
        digest_dataset = make_client_dataset(self.file_paths, self.labels, public_idx, IMAGE_SIZE, self.batch_size,
                                             shuffle=True, targets=targets)
        if steps_per_epoch is not None:
            digest_dataset = digest_dataset.repeat()  # com pouco dado público, N pode passar de uma passada

        # Digest do FedMD: aproxima os logits do consenso com MAE. Treinar digest_model atualiza os pesos de self.model.
        history = self.digest_model.fit(digest_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=0)

        return [{'epoch': epoch, 'loss': loss, 'public_accuracy': public_accuracy}
                for epoch, loss in enumerate(history.history['loss'])]

    def evaluate(self):
        """Avalia o modelo no conjunto de teste.

        Returns:
            dict: Perda e acurácia no teste.
        """
        metrics = self.model.evaluate(self.test_dataset, verbose=0, return_dict=True)
        return {'test_loss': metrics['loss'], 'test_accuracy': metrics['accuracy']}

    def set_extra_test(self, paths: np.ndarray, labels: np.ndarray):
        """Define um conjunto de teste externo (ex.: PH2), avaliado junto com o teste original."""
        self.extra_test_dataset = make_client_dataset(paths, labels, np.arange(len(labels)), IMAGE_SIZE,
                                                      self.batch_size, shuffle=False)

    def extra_test_logits(self):
        """Logits do modelo no conjunto de teste externo, na ordem dos rótulos passados a set_extra_test."""
        return self.model.predict(self.extra_test_dataset, verbose=0)

    def test_logits(self):
        """Logits do modelo no conjunto de teste, na ordem de test_labels (usado para o ensemble dos clientes)."""
        return self.model.predict(self.test_dataset, verbose=0)

    def save(self, path: str):
        """Salva o modelo (arquitetura e pesos, sem o otimizador) em um arquivo .keras."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path, include_optimizer=False)


def append_csv(path: str, rows: list[dict], columns: list[str]):
    """Acrescenta linhas ao CSV, escrevendo o cabeçalho só na criação. Salvar a cada etapa evita perder
    os resultados já obtidos se o experimento cair no meio."""
    pd.DataFrame(rows).reindex(columns=columns).to_csv(path, mode='a', header=not os.path.exists(path), index=False)


def run_experiment(public_fraction: float, alpha: float, args):
    """Roda o FedMD completo para uma configuração da ablação e grava os dois CSVs."""
    config = {'seed': args.seed, 'public_fraction': public_fraction, 'alpha': 'iid' if alpha is None else alpha}
    train_csv = os.path.join(OUTPUT_DIR, 'train_history.csv')
    test_csv  = os.path.join(OUTPUT_DIR, 'test_metrics.csv')
    ensemble_csv = os.path.join(OUTPUT_DIR, 'ensemble_metrics.csv')
    extra_csv = os.path.join(OUTPUT_DIR, 'extra_test_metrics.csv')
    logging.info(f'Configuração {config}')

    test_paths, test_labels = list_files(train_data=False)
    extra_test = list_dir(args.extra_test_dir) if args.extra_test_dir else None
    if not args.private_dir:
        client_idx, public_idx, file_paths, labels = split_dataset(
            True, NUM_CLIENTS, alpha=alpha, public_fraction=public_fraction, seed=args.seed)
        public_train_idx, public_val_idx = holdout_split(public_idx, args.val_fraction, args.seed)
        private_splits = [holdout_split(idx, args.val_fraction, args.seed + cid) for cid, idx in enumerate(client_idx)]
    else:
        # Cenário do FedMD do paper: todo o dataset de treino é público, e os dados privados vêm só de um dataset
        # externo. Uma fração dele (estratificada, a mesma do centralizado) é dividida entre os clientes (IID ou
        # Dirichlet, como no cenário padrão) e vai inteira para o treino; o resto é o teste externo.
        file_paths, labels = list_files(train_data=True)
        public_idx = np.arange(len(labels))
        public_train_idx, public_val_idx = holdout_split(public_idx, args.val_fraction, args.seed)
        ext_paths, ext_labels = list_dir(args.private_dir)
        ext_private, ext_test = split_external(ext_labels, args.private_dir_fraction, args.seed)
        offset = len(labels)
        file_paths = np.concatenate([file_paths, ext_paths[ext_private]])
        labels = np.concatenate([labels, ext_labels[ext_private]])
        private_idx = np.arange(offset, len(labels))
        client_idx = [np.array(idx, dtype=int) for idx in split_clients(
            private_idx, labels, NUM_CLIENTS, alpha, np.random.default_rng(args.seed), min_samples_per_client=10)]
        private_splits = [(idx, np.array([], dtype=int)) for idx in client_idx]
        extra_test = (ext_paths[ext_test], ext_labels[ext_test])
        logging.info(f'Privado: {len(private_idx)} imagens de {args.private_dir} '
                     f'({", ".join(f"{len(i)} ({int(labels[i].sum())} malignas)" for i in client_idx)} por cliente); '
                     f'público: {len(public_idx)} imagens; teste externo: {len(ext_test)} imagens.')

    clients = [
        Client.remote(cid, CLIENT_ARCHITECTURES[cid], file_paths, labels, *private_splits[cid],
                      test_paths, test_labels, args.batch_size, args.gpu_memory_mb, MIXED_PRECISION, args.seed,
                      digest_lr=args.digest_lr, digest_temperature=args.digest_temperature,
                      digest_ce_weight=args.digest_ce_weight, private_batch_size=args.private_batch_size,
                      jit_compile=False if args.no_jit else 'auto')
        for cid in range(NUM_CLIENTS)
    ]
    if extra_test is not None:
        ray.get([client.set_extra_test.remote(*extra_test) for client in clients])
    client_info = [{'n_private': len(idx), 'private_malignant_fraction': float(labels[idx].mean())}
                   for idx in client_idx]

    def log_train(stage, round_, histories):
        append_csv(train_csv, [
            {**config, 'client_id': cid, 'stage': stage, 'round': round_, **row}
            for cid, history in enumerate(histories) for row in history
        ], TRAIN_COLUMNS)

    def log_test(stage, round_):
        results = ray.get([client.evaluate.remote() for client in clients])
        append_csv(test_csv, [
            {**config, 'client_id': cid, 'stage': stage, 'round': round_, **client_info[cid], **result}
            for cid, result in enumerate(results)
        ], TEST_COLUMNS)
        logging.info(f'{stage} rodada {round_}: acurácia de teste ' +
                     ', '.join(f'{r["test_accuracy"]:.3f}' for r in results))
        # Ensemble dos clientes: média dos logits no teste. Mede o sistema como um todo, não cada cliente.
        logits = np.mean(ray.get([client.test_logits.remote() for client in clients]), axis=0)
        ensemble_accuracy = float(np.mean(np.argmax(logits, axis=1) == test_labels))
        append_csv(ensemble_csv, [{**config, 'stage': stage, 'round': round_, 'ensemble_accuracy': ensemble_accuracy,
                                   'mean_client_accuracy': float(np.mean([r['test_accuracy'] for r in results]))}],
                   ENSEMBLE_COLUMNS)
        logging.info(f'{stage} rodada {round_}: ensemble {ensemble_accuracy:.3f}')
        if extra_test is not None:
            log_extra_test(stage, round_)

    def log_extra_test(stage, round_):
        # Teste externo: cada cliente e o ensemble (média dos logits). A acurácia balanceada é a média do recall por
        # classe; importa quando o dataset externo é desbalanceado (o PH2 tem 4 benignos para 1 melanoma).
        extra_test_labels = extra_test[1]

        def metrics(logits):
            pred = np.argmax(logits, axis=1)
            recalls = [np.mean(pred[extra_test_labels == c] == c) for c in np.unique(extra_test_labels)]
            return float(np.mean(pred == extra_test_labels)), float(np.mean(recalls))
        all_logits = ray.get([client.extra_test_logits.remote() for client in clients])
        rows = [{**config, 'stage': stage, 'round': round_, 'model': f'cliente {cid}',
                 **dict(zip(('accuracy', 'balanced_accuracy'), metrics(l)))} for cid, l in enumerate(all_logits)]
        rows.append({**config, 'stage': stage, 'round': round_, 'model': 'ensemble',
                     **dict(zip(('accuracy', 'balanced_accuracy'), metrics(np.mean(all_logits, axis=0))))})
        append_csv(extra_csv, rows, EXTRA_TEST_COLUMNS)
        logging.info(f'{stage} rodada {round_}: teste externo ' + ', '.join(f"{r['accuracy']:.3f}" for r in rows[:-1])
                     + f" | ensemble {rows[-1]['accuracy']:.3f}")

    try:
        # 1. Transfer learning nos dados públicos rotulados.
        log_train('public_init', 0, ray.get([
            client.train.remote(public_train_idx, public_val_idx, INIT_EPOCHS, learning_rate=args.public_init_lr)
            for client in clients]))
        log_test('public_init', 0)

        # 2. Treino inicial nos dados privados de cada cliente. Com --private-init-steps, é uma época de N passos:
        #    poucos passos numa classe só não chegam a apagar o que o modelo aprendeu nos dados públicos.
        private_init = ({'epochs': args.private_init_epochs} if args.private_init_steps is None
                        else {'epochs': 1, 'steps_per_epoch': args.private_init_steps})
        if args.private_init_lr is not None:
            # O revisit passa o próprio lr (--revisit-lr), então este vale só para o treino inicial privado.
            private_init['learning_rate'] = args.private_init_lr
        if args.private_init_rehearsal:
            # Mistura amostras públicas rotuladas também no treino inicial privado, para poder treinar mais forte
            # sem que clientes com uma classe só esqueçam a outra.
            private_init.update(rehearsal_idx=public_idx, rehearsal_fraction=args.rehearsal_fraction)
        log_train('private_init', 0, ray.get([
            client.train_private.remote(**private_init) for client in clients]))
        log_test('private_init', 0)

        # 3. Rodadas do FedMD. Todos os clientes recebem o mesmo subconjunto público, na mesma ordem, e as
        #    chamadas precisam ser concorrentes: o servidor só responde quando todos enviaram os logits.
        rng = np.random.default_rng(args.seed)
        round_size = int(round(PUBLIC_ROUND_FRACTION * len(public_idx)))
        for round_ in range(1, args.digest_rounds + 1):
            round_public_idx = np.sort(rng.choice(public_idx, round_size, replace=False))
            # Rehearsal no revisit: o mesmo subconjunto público da rodada, com os rótulos verdadeiros.
            rehearsal = ({'rehearsal_idx': round_public_idx, 'rehearsal_fraction': args.rehearsal_fraction}
                         if args.rehearsal_fraction > 0 else {})

            log_train('digest', round_, ray.get([
                client.fed_train.remote(round_public_idx, args.digest_epochs, args.digest_steps) for client in clients]))
            log_train('revisit', round_, ray.get([
                client.train_private.remote(args.private_epochs, learning_rate=args.revisit_lr,
                                            steps_per_epoch=args.private_steps, **rehearsal) for client in clients]))
            log_test('federated', round_)

        if args.save_models_dir:
            # Caminho absoluto: os actors rodam em uma cópia do projeto (ver list_files).
            out = os.path.join(os.path.abspath(args.save_models_dir), f"P{public_fraction:g}_alpha_{config['alpha']}")
            ray.get([client.save.remote(os.path.join(out, f'cliente_{cid}.keras')) for cid, client in enumerate(clients)])
    finally:
        for client in clients:
            ray.kill(client)


def main(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Falha logo se o servidor não estiver no ar, em vez de travar na primeira rodada federada.
    channel = grpc.insecure_channel('localhost:50051')
    try:
        grpc.channel_ready_future(channel).result(timeout=10)
    except grpc.FutureTimeoutError:
        raise SystemExit(f'Servidor de consenso não encontrado em localhost:50051. '
                         f'Suba antes com: python server.py --clients {NUM_CLIENTS}')
    finally:
        channel.close()

    alphas = [None if alpha == 'iid' else float(alpha) for alpha in args.alphas]
    # Todo o conjunto de treino é usado: o que não é público é dividido entre os clientes.
    for public_fraction in args.public_fractions:
        for alpha in alphas:
            run_experiment(public_fraction, alpha, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ablação do FedMD com 4 clientes heterogêneos.')
    parser.add_argument('--public-fractions', type=float, nargs='+', default=[0.1, 0.4, 0.8])
    parser.add_argument('--alphas', nargs='+', default=['iid', '1.0', '0.1'],
                        help="Concentração da Dirichlet; 'iid' para divisão IID. Menor = mais não IID.")
    parser.add_argument('--digest-rounds', type=int, default=20)
    parser.add_argument('--digest-epochs', type=int, default=1)
    parser.add_argument('--private-epochs', type=int, default=3, help='Máximo de épocas do revisit em cada rodada.')
    parser.add_argument('--private-dir', default=None,
                        help='Dataset externo (pasta/<classe>/<imagem>) de onde vêm os dados privados. Com esta opção, todo o '
                             'dataset de treino vira público; uma fração do externo é dividida entre os clientes e o resto é '
                             'o teste externo (substitui --extra-test-dir). Use com --public-fractions 1.0.')
    parser.add_argument('--private-dir-fraction', type=float, default=0.4,
                        help='Fração de --private-dir que vira dado privado (estratificada); o restante é o teste externo.')
    parser.add_argument('--private-batch-size', type=int, default=None,
                        help='Batch do treino privado (private_init e revisit). Padrão: o mesmo --batch-size.')
    parser.add_argument('--extra-test-dir', default=None,
                        help='Dataset externo (pasta/<classe>/<imagem>) avaliado junto com o teste, ex.: o PH2 convertido.')
    parser.add_argument('--save-models-dir', default=None,
                        help='Salva o modelo final de cada cliente (após a última rodada) em <dir>/P<fração>_alpha_<alpha>/cliente_<i>.keras.')
    parser.add_argument('--private-init-epochs', type=int, default=PRIVATE_INIT_EPOCHS,
                        help='Máximo de épocas do treino inicial privado quando --private-init-steps não é usado.')
    parser.add_argument('--private-init-steps', type=int, default=None,
                        help=f'Treino inicial privado em 1 época de N passos. Padrão: até {PRIVATE_INIT_EPOCHS} épocas completas.')
    parser.add_argument('--val-fraction', type=float, default=VAL_FRACTION,
                        help='Fração separada para validação, do público (public_init) e do privado de cada cliente. '
                             'Valores maiores deixam menos dados de treino a cada cliente.')
    parser.add_argument('--public-init-lr', type=float, default=None,
                        help='Learning rate do treino inicial nos dados públicos. Padrão: o do compile (1e-3).')
    parser.add_argument('--private-init-lr', type=float, default=None,
                        help='Learning rate do treino inicial privado. Padrão: o mesmo do public_init (1e-3).')
    parser.add_argument('--digest-steps', type=int, default=None,
                        help='Passos do digest por época, fixos em qualquer fração pública. Padrão: uma passada pelo subconjunto da rodada.')
    parser.add_argument('--no-jit', action='store_true',
                        help='Compila os modelos sem XLA. Mais lento, mas sem os picos de memória de GPU da compilação.')
    parser.add_argument('--digest-temperature', type=float, default=None,
                        help='Usa KL com temperatura T no digest (T² · KL entre softmax do consenso e do cliente). Padrão: MAE nos logits.')
    parser.add_argument('--digest-ce-weight', type=float, default=0.0,
                        help='Peso w da entropia cruzada no rótulo verdadeiro dentro do digest: (1 − w) · destilação + w · CE. '
                             '0 = só destilação (FedMD); 1 = só CE, sem destilação.')
    parser.add_argument('--digest-lr', type=float, default=1e-3,
                        help='Learning rate do digest. O digest não usa rótulos e dá ~24 a ~190 passos por rodada.')
    parser.add_argument('--private-init-rehearsal', action='store_true',
                        help='Aplica o rehearsal (com --rehearsal-fraction) também no treino inicial privado.')
    parser.add_argument('--rehearsal-fraction', type=float, default=0.0,
                        help='Fração de amostras públicas rotuladas em cada batch do revisit (rehearsal). 0 desliga.')
    parser.add_argument('--private-steps', type=int, default=None,
                        help='Passos de gradiente por época do revisit, iguais para todo cliente. Padrão: uma passada pelos dados.')
    parser.add_argument('--revisit-lr', type=float, default=5e-5,
                        help='Learning rate do revisit; os treinos iniciais usam 1e-3. Menor = menos esquecimento.')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--gpu-memory-mb', type=int, default=2560, help='Limite de memória de GPU por cliente.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    ray.init(ignore_reinit_error=True,
             runtime_env={"working_dir": os.path.dirname(os.path.abspath(__file__))})
    main(args)
