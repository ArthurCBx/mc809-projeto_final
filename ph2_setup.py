"""Converte o PH2 (espelho do Hugging Face, Shah1st/PH2) para ph2_dataset/<classe>/<imagem>.jpg.

O rótulo vem do prefixo da descrição ('BENIGN:' ou 'MALIGNANT:'), o mesmo binário do dataset de treino.
Uso: python ph2_setup.py <arquivo .parquet> [pasta de saída]
"""
import io
import os
import re
import sys

import pyarrow.parquet as pq
from PIL import Image

from dataset_setup import CLASS_NAMES


def main(parquet_path: str, out_dir: str = 'ph2_dataset'):
    t = pq.read_table(parquet_path).to_pandas()
    labels = t.description.str.extract(r'^\s*(BENIGN|MALIGNANT)\s*:', flags=re.I)[0].str.lower()
    subtype = t.description.str.extract(r'image of an? ([a-z ]+?(?:nevus|melanoma))', flags=re.I)[0].str.lower()
    if labels.isna().any():
        raise ValueError(f'{labels.isna().sum()} descrições sem rótulo BENIGN/MALIGNANT')
    print('classes:', labels.value_counts().to_dict(), '| subtipos:', subtype.value_counts().to_dict())
    for class_name in CLASS_NAMES:
        os.makedirs(os.path.join(out_dir, class_name), exist_ok=True)
    for (_, row), label in zip(t.iterrows(), labels):
        image = Image.open(io.BytesIO(row.image['bytes'])).convert('RGB')
        name = os.path.splitext(os.path.basename(row.filename))[0] + '.jpg'
        image.save(os.path.join(out_dir, label, name), quality=95)
    print(f'{len(t)} imagens salvas em {out_dir}')


if __name__ == '__main__':
    main(*sys.argv[1:])
