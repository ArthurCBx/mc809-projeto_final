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
from dataset_setup import CLASS_NAMES, list_files, split_dataset, make_client_dataset

BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
NUM_CLIENTS = 4

# Uma arquitetura diferente por cliente, como no FedMD: (filtros, kernel) de cada camada convolucional.
CLIENT_ARCHITECTURES = [
    [(16, 3), (32, 3), (64, 3)],
    [(32, 3), (64, 3), (128, 3)],
    [(16, 5), (32, 5), (64, 3), (128, 3)],
    [(32, 3), (32, 3), (64, 3), (64, 3), (128, 3)],
]

TRAIN_COLUMNS = ['public_fraction', 'alpha', 'client_id', 'stage', 'round', 'epoch',
                 'loss', 'accuracy', 'val_loss', 'val_accuracy', 'restored']
TEST_COLUMNS  = ['public_fraction', 'alpha', 'client_id', 'stage', 'round',
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
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.MaxPooling2D()(x)
        return x
    return f


def build_model(backbone_fn, num_classes=2, image_size=(224, 224)):
    inputs = keras.Input((*image_size, 3))
    x = keras.layers.Rescaling(1 / 255)(inputs)   # make_client_dataset entrega pixels em 0–255
    x = backbone_fn(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(128, activation="gelu")(x) # logits: é o que o FedMD troca no consenso
    outputs = keras.layers.Dense(num_classes)(outputs)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.AdamW(1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
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
                 batch_size: int=BATCH_SIZE, gpu_memory_mb: int=None, seed: int=42):
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
        """
        gpus = tf.config.list_physical_devices('GPU')
        if gpus and gpu_memory_mb:
            tf.config.set_logical_device_configuration(
                gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=gpu_memory_mb)])
        keras.utils.set_random_seed(seed + client_id)

        self.client_id         = client_id
        self.model             = build_model(small_cnn(filters), num_classes=len(CLASS_NAMES), image_size=IMAGE_SIZE)
        self.file_paths        = file_paths    # dataset inteiro; fed_train recorta o subconjunto público
        self.labels            = labels
        self.private_train_idx = private_train_idx
        self.private_val_idx   = private_val_idx
        self.batch_size        = batch_size
        self.augment           = augmentation_pipeline(IMAGE_SIZE, train=True)
        self.test_dataset      = make_client_dataset(test_paths, test_labels, np.arange(len(test_labels)),
                                                     IMAGE_SIZE, batch_size, shuffle=False)

        self.host        = 'localhost'
        self.server_port = 50051
        self.channel     = grpc.insecure_channel(f'{self.host}:{self.server_port}')
        self.stub        = pb2_grpc.FedMDLearningStub(self.channel)

    def train(self, train_idx, val_idx, epochs: int=1, patience: int=4, augment: bool=True):
        """Treina o modelo do cliente com os rótulos verdadeiros, com early stopping na validação.

        Args:
            train_idx (np.ndarray): Índices de treino, relativos a self.file_paths.
            val_idx (np.ndarray): Índices de validação, relativos a self.file_paths.
            epochs (int, optional): Número máximo de épocas. Defaults to 1.
            patience (int, optional): Número de épocas sem melhoria para parar o treinamento. Defaults to 4.
            augment (bool, optional): Aplica augmentation_pipeline nas imagens de treino. Defaults to True.

        Returns:
            list[dict]: Uma linha por época com loss e acurácia em treino e validação. 'restored' marca a
                        época cujos pesos ficaram no modelo.
        """
        train_dataset = make_client_dataset(self.file_paths, self.labels, train_idx, IMAGE_SIZE, self.batch_size)
        if augment:
            train_dataset = train_dataset.map(lambda x, y: (self.augment(x, training=True), y),
                                              num_parallel_calls=tf.data.AUTOTUNE)
        validation_dataset = make_client_dataset(self.file_paths, self.labels, val_idx, IMAGE_SIZE,
                                                 self.batch_size, shuffle=False)

        callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, min_delta=1e-4, restore_best_weights=True)]
        history = self.model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=0
        )
        h = history.history
        best = int(np.argmin(h['val_loss']))
        return [
            {'epoch': epoch, 'loss': h['loss'][epoch], 'accuracy': h['accuracy'][epoch],
             'val_loss': h['val_loss'][epoch], 'val_accuracy': h['val_accuracy'][epoch], 'restored': epoch == best}
            for epoch in range(len(h['loss']))
        ]

    def train_private(self, epochs: int=1, patience: int=4, augment: bool=True):
        """Treina com os dados privados do cliente. Mesmo retorno de train."""
        return self.train(self.private_train_idx, self.private_val_idx, epochs, patience, augment)

    def fed_train(self, public_idx: np.ndarray, epochs: int=1):
        """Treina o modelo do cliente com os dados públicos, usando o consenso do servidor como alvo.

        Args:
            public_idx (np.ndarray): Índices do subconjunto público desta rodada, relativos a self.file_paths.
            epochs (int, optional): Número de épocas. Defaults to 1.

        Returns:
            list[dict]: Uma linha por época com a perda do modelo em relação ao consenso.
        """
        # Sem shuffle: a linha i dos logits precisa corresponder a public_idx[i].
        public_dataset = make_client_dataset(self.file_paths, self.labels, public_idx, IMAGE_SIZE, self.batch_size, shuffle=False)
        logits = self.model.predict(public_dataset, verbose=0)

        # O proto só aceita floats em uma lista plana, então achata e remonta (N, num_classes).
        logits_message = pb2.ClientLogits(client_id=self.client_id, logits=logits.ravel().tolist())
        response = self.stub.SendClientLogits(logits_message)
        consensus = np.array(response.consensus_logits, dtype=np.float32).reshape(logits.shape)

        # Mesmas imagens, agora embaralhadas junto com o consenso de cada uma.
        digest_dataset = make_client_dataset(self.file_paths, self.labels, public_idx, IMAGE_SIZE, self.batch_size,
                                             shuffle=True, targets=consensus)

        # Digest do FedMD: aproxima os logits do consenso com MAE. Depois volta para a loss de classificação.
        classification_loss = self.model.loss
        # metrics=[] é necessário: sem ele o compile mantém a accuracy anterior, que não aceita o consenso como alvo.
        self.model.compile(optimizer=self.model.optimizer, loss=keras.losses.MeanAbsoluteError(), metrics=[])
        history = self.model.fit(digest_dataset, epochs=epochs, verbose=0)
        self.model.compile(optimizer=self.model.optimizer, loss=classification_loss, metrics=["accuracy"])

        return [{'epoch': epoch, 'loss': loss} for epoch, loss in enumerate(history.history['loss'])]

    def evaluate(self):
        """Avalia o modelo no conjunto de teste.

        Returns:
            dict: Perda e acurácia no teste.
        """
        metrics = self.model.evaluate(self.test_dataset, verbose=0, return_dict=True)
        return {'test_loss': metrics['loss'], 'test_accuracy': metrics['accuracy']}


def append_csv(path: str, rows: list[dict], columns: list[str]):
    """Acrescenta linhas ao CSV, escrevendo o cabeçalho só na criação. Salvar a cada etapa evita perder
    os resultados já obtidos se o experimento cair no meio."""
    pd.DataFrame(rows).reindex(columns=columns).to_csv(path, mode='a', header=not os.path.exists(path), index=False)


def run_experiment(public_fraction: float, alpha: float, args):
    """Roda o FedMD completo para uma configuração da ablação e grava os dois CSVs."""
    config = {'public_fraction': public_fraction, 'alpha': 'iid' if alpha is None else alpha}
    train_csv = os.path.join(args.output_dir, 'train_history.csv')
    test_csv  = os.path.join(args.output_dir, 'test_metrics.csv')
    logging.info(f'Configuração {config}')

    client_idx, public_idx, file_paths, labels = split_dataset(
        True, NUM_CLIENTS, alpha=alpha, public_fraction=public_fraction, seed=args.seed)
    test_paths, test_labels = list_files(train_data=False)
    public_train_idx, public_val_idx = holdout_split(public_idx, args.val_fraction, args.seed)
    private_splits = [holdout_split(idx, args.val_fraction, args.seed + cid) for cid, idx in enumerate(client_idx)]

    clients = [
        Client.remote(cid, CLIENT_ARCHITECTURES[cid], file_paths, labels, *private_splits[cid],
                      test_paths, test_labels, args.batch_size, args.gpu_memory_mb, args.seed)
        for cid in range(NUM_CLIENTS)
    ]
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

    try:
        # 1. Transfer learning nos dados públicos rotulados.
        log_train('public_init', 0, ray.get([
            client.train.remote(public_train_idx, public_val_idx, args.init_epochs, args.patience)
            for client in clients]))
        log_test('public_init', 0)

        # 2. Treino inicial nos dados privados de cada cliente.
        log_train('private_init', 0, ray.get([
            client.train_private.remote(args.init_epochs, args.patience) for client in clients]))
        log_test('private_init', 0)

        # 3. Rodadas do FedMD. Todos os clientes recebem o mesmo subconjunto público, na mesma ordem, e as
        #    chamadas precisam ser concorrentes: o servidor só responde quando todos enviaram os logits.
        rng = np.random.default_rng(args.seed)
        for round_ in range(1, args.digest_rounds + 1):
            round_size = min(args.public_round_size or len(public_idx), len(public_idx))
            round_public_idx = np.sort(rng.choice(public_idx, round_size, replace=False))

            log_train('digest', round_, ray.get([
                client.fed_train.remote(round_public_idx, args.digest_epochs) for client in clients]))
            log_train('revisit', round_, ray.get([
                client.train_private.remote(args.private_epochs, args.patience) for client in clients]))
            log_test('federated', round_)
    finally:
        for client in clients:
            ray.kill(client)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

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
    parser.add_argument('--public-fractions', type=float, nargs='+', default=[0.1, 0.2])
    parser.add_argument('--alphas', nargs='+', default=['iid', '1.0', '0.1'],
                        help="Concentração da Dirichlet; 'iid' para divisão IID. Menor = mais não IID.")
    parser.add_argument('--digest-rounds', type=int, default=3)
    parser.add_argument('--digest-epochs', type=int, default=2)
    parser.add_argument('--private-epochs', type=int, default=10, help='Máximo de épocas do revisit em cada rodada.')
    parser.add_argument('--init-epochs', type=int, default=50, help='Máximo de épocas dos treinos iniciais.')
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--public-round-size', type=int, default=None,
                        help='Amostras públicas sorteadas por rodada. Padrão: todo o conjunto público.')
    parser.add_argument('--val-fraction', type=float, default=0.2)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--gpu-memory-mb', type=int, default=2560, help='Limite de memória de GPU por cliente.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default='results')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    ray.init(ignore_reinit_error=True,
             runtime_env={"working_dir": os.path.dirname(os.path.abspath(__file__))})
    main(args)
