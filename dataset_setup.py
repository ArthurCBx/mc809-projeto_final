import numpy as np
import os

CLASS_NAMES = ['benign', 'malignant']


def list_files(train_data: bool):
    """Lista os arquivos do dataset em ordem determinística.

    A ordem retornada aqui é a que dá significado aos índices de split_dataset:
    o índice i corresponde a file_paths[i] e labels[i].

    Returns:
        file_paths (np.ndarray): Caminho completo das imagens.
        labels (np.ndarray): Rótulo inteiro de cada imagem (posição em CLASS_NAMES).
    """
    # Caminhos absolutos: os actors do Ray rodam em uma cópia do projeto que não inclui o dataset (.gitignore).
    dir = os.path.abspath('./melanoma_cancer_dataset/train' if train_data else './melanoma_cancer_dataset/test')
    file_paths, labels = [], []
    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(dir, class_name)
        files = sorted(os.listdir(class_dir))
        file_paths.extend(os.path.join(class_dir, f) for f in files)
        labels.extend([label] * len(files))
    return np.array(file_paths), np.array(labels)


def list_dir(root: str):
    """Lista um dataset externo organizado como root/<classe>/<imagem>, com as classes de CLASS_NAMES.

    Usado para avaliar os modelos em um dataset diferente do de treino (ex.: PH2).
    """
    root = os.path.abspath(root)
    file_paths, labels = [], []
    for label, class_name in enumerate(CLASS_NAMES):
        files = sorted(os.listdir(os.path.join(root, class_name)))
        file_paths.extend(os.path.join(root, class_name, f) for f in files)
        labels.extend([label] * len(files))
    return np.array(file_paths), np.array(labels)


def split_public(labels, public_fraction: float, seed: int=42):
    """Separa, de forma estratificada, os índices do conjunto público (FedMD) e do privado.

    Returns:
        private_idx (np.ndarray): Índices que serão divididos entre os clientes.
        public_idx (np.ndarray): Índices do conjunto público, compartilhado por todos os clientes.
    """
    all_indices = np.arange(len(labels))
    if not public_fraction:
        return all_indices, np.array([], dtype=int)

    from sklearn.model_selection import train_test_split
    private_idx, public_idx = train_test_split(
        all_indices, test_size=public_fraction, stratify=labels, random_state=seed
    )
    return np.sort(private_idx), np.sort(public_idx)


def split_dataset(train_data: bool, expected_num_clients: int, alpha: float=None,
                  public_fraction: float=0.1, min_samples_per_client: int=10, seed: int=42):
    """Divide o dataset original em conjuntos IID ou não IID, dependendo do parâmetro de entrada.
    Args:
        train_data (bool): Se True, divide o conjunto de treinamento entre os clientes; se False,
                           todos os clientes recebem o conjunto de teste inteiro e não há
                           conjunto público (alpha e public_fraction são ignorados).
        expected_num_clients (int): Número de clientes para dividir o dataset.
        alpha (float, optional): Parâmetro de concentração para a distribuição de Dirichlet.
                                 Se None, o dataset será dividido de forma IID.
        public_fraction (float): Fração do dataset reservada como conjunto público do FedMD.
                                 Use 0 para não separar conjunto público.
        min_samples_per_client (int): Na divisão não IID, sorteia de novo até todo cliente ter
                                      pelo menos esse número de amostras.
        seed (int, optional): Semente para reprodutibilidade.

    Returns:
        client_idx (list[list[int]]): Índices de cada cliente, relativos à ordem de list_files.
        public_idx (np.ndarray): Índices do conjunto público, relativos à ordem de list_files.
        file_paths (np.ndarray): Caminhos das imagens.
        labels (np.ndarray): Rótulos das imagens.
    """
    rng = np.random.default_rng(seed)
    file_paths, labels = list_files(train_data)

    if not train_data:
        # Teste: todos os clientes avaliam seus modelos no conjunto de teste inteiro.
        all_indices = np.arange(len(labels)).tolist()
        client_idx = [list(all_indices) for _ in range(expected_num_clients)]
        return client_idx, np.array([], dtype=int), file_paths, labels

    private_idx, public_idx = split_public(labels, public_fraction, seed)

    num_classes = len(CLASS_NAMES)
    data_length = len(private_idx)

    client_idx = [[] for _ in range(expected_num_clients)]

    if alpha is None:
        # Divisão IID
        all_indices = rng.permutation(private_idx)
        split_indices = np.array_split(all_indices, expected_num_clients)

        for client_id, indices in enumerate(split_indices):
            client_idx[client_id] = indices.tolist()

    else:
        # Divisão não IID: para cada classe, sorteia de uma Dirichlet(alpha) a fração
        # das amostras daquela classe que vai para cada cliente.
        min_size = 0
        while min_size < min_samples_per_client:
            client_idx = [[] for _ in range(expected_num_clients)]
            for k in range(num_classes):
                idx_k = rng.permutation(private_idx[labels[private_idx] == k])
                proportions = rng.dirichlet(np.repeat(alpha, expected_num_clients))
                # Não manda mais amostras para clientes que já passaram da média,
                # evitando que um único cliente concentre quase todo o dataset.
                proportions = np.array([
                    p * (len(idx) < data_length / expected_num_clients)
                    for p, idx in zip(proportions, client_idx)
                ])
                proportions = proportions / proportions.sum()
                cut_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                for client_id, part in enumerate(np.split(idx_k, cut_points)):
                    client_idx[client_id].extend(part.tolist())
            min_size = min(len(idx) for idx in client_idx)

        for client_id in range(expected_num_clients):
            client_idx[client_id] = rng.permutation(client_idx[client_id]).tolist()

    return client_idx, public_idx, file_paths, labels


def make_client_dataset(file_paths, labels, indices, image_size=(224, 224), batch_size=32, shuffle=True, targets=None):
    """Monta um tf.data.Dataset com as imagens de um cliente, pronto para model.fit do Keras.

    Args:
        targets (np.ndarray, optional): Alvos que substituem labels[indices], na mesma ordem de
                                        indices. Usado na etapa de digest do FedMD, em que o alvo
                                        de cada imagem pública é o consenso (num_classes floats).
    """
    import tensorflow as tf

    def load(path, label):
        image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        image = tf.image.resize(image, image_size)
        return image, label

    y = labels[indices] if targets is None else np.asarray(targets, dtype=np.float32)
    ds = tf.data.Dataset.from_tensor_slices((file_paths[indices], y))
    if shuffle:
        ds = ds.shuffle(len(indices), reshuffle_each_iteration=True)
    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def main(dataset_name: str='hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images', output_dir: str='./'):
    import os
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.exists(os.path.join(output_dir, "melanoma_cancer_dataset")):
        print(f"Dataset already exists in {output_dir}. Skipping download.")
        return
    
    api.dataset_download_files(dataset=dataset_name, path=output_dir, unzip=True)
    print(f"Dataset downloaded and extracted to {output_dir}.")


if __name__ == "__main__":
    main()