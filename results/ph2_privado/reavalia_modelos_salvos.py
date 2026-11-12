"""Reavalia os modelos finais de results/fracoes_publicas_10_40_80 (treinados sem PH2) nas mesmas 120 imagens do PH2
que são o teste do experimento com o PH2 nos dados privados (split_external com fração 0.4 e a mesma seed).

Também avalia nas 200 imagens, para conferir contra o extra_test_metrics.csv da rodada final de cada execução.
Grava results/ph2_privado/reavaliacao_modelos_salvos.csv (uma linha por execução, α e modelo, incluindo o ensemble).
Roda na CPU: CUDA_VISIBLE_DEVICES= uv run python results/ph2_privado/reavalia_modelos_salvos.py
"""
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
import keras

from client import BATCH_SIZE, IMAGE_SIZE
from dataset_setup import list_dir, make_client_dataset, split_external

ORIGEM = 'results/fracoes_publicas_10_40_80'
OUT = 'results/ph2_privado/reavaliacao_modelos_salvos.csv'
SEEDS, FRACOES, METODOS = (42, 1), ('10', '40', '80'), ('adaptado', 'original')
ALPHAS = ('iid', '1.0', '0.1')

ph2_paths, ph2_labels = list_dir('ph2_dataset')
todas = make_client_dataset(ph2_paths, ph2_labels, np.arange(len(ph2_labels)), IMAGE_SIZE, BATCH_SIZE, shuffle=False)


def metrics(logits, y):
    pred = np.argmax(logits, axis=1)
    return float(np.mean(pred == y)), float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))


rows = []
for seed in SEEDS:
    _, teste = split_external(ph2_labels, 0.4, seed)
    for fr in FRACOES:
        for metodo in METODOS:
            for alpha in ALPHAS:
                pasta = f'{ORIGEM}/modelos/{metodo}_P{fr}_seed{seed}/P0.{fr[0]}_alpha_{alpha}'
                logits = []
                for cid in range(4):
                    modelo = keras.models.load_model(f'{pasta}/cliente_{cid}.keras', compile=False)
                    logits.append(modelo.predict(todas, verbose=0))
                    keras.backend.clear_session()
                for nome, l in [(f'cliente {i}', l) for i, l in enumerate(logits)] + [('ensemble', np.mean(logits, axis=0))]:
                    acc120, bal120 = metrics(l[teste], ph2_labels[teste])
                    acc200, bal200 = metrics(l, ph2_labels)
                    rows.append({'seed': seed, 'public_fraction': int(fr) / 100, 'método': metodo, 'alpha': alpha, 'model': nome,
                                 'ph2_120_accuracy': acc120, 'ph2_120_balanced_accuracy': bal120,
                                 'ph2_200_accuracy': acc200, 'ph2_200_balanced_accuracy': bal200})
            print(f'seed {seed} {fr}% {metodo}: ok', flush=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
print('=== fim', flush=True)
