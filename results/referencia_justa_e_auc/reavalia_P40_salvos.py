"""Sensibilidade, especificidade e AUC no PH2 (200 imagens) dos modelos finais salvos de results/fracoes_publicas_10_40_80
com f = 40% (FedMD adaptado e replicado, as três divisões, seeds 42, 1 e 2). São as mesmas execuções da Tabela 8, então a
acurácia balanceada tem de bater com o extra_test_metrics.csv da última rodada de cada uma.
Roda na CPU: CUDA_VISIBLE_DEVICES= uv run python results/referencia_justa_e_auc/reavalia_P40_salvos.py
"""
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
import keras
from sklearn.metrics import roc_auc_score

from client import BATCH_SIZE, IMAGE_SIZE
from dataset_setup import list_dir, make_client_dataset

ORIGEM = 'results/fracoes_publicas_10_40_80'
OUT = 'results/referencia_justa_e_auc/reavaliacao_P40_salvos.csv'
ph2_paths, y = list_dir('ph2_dataset')
dataset = make_client_dataset(ph2_paths, y, np.arange(len(y)), IMAGE_SIZE, BATCH_SIZE, shuffle=False)


def metrics(logits):
    pred = np.argmax(logits, axis=1)
    sens, spec = float(np.mean(pred[y == 1] == 1)), float(np.mean(pred[y == 0] == 0))
    prob = keras.ops.convert_to_numpy(keras.ops.softmax(logits, axis=-1))[:, 1]
    return {'accuracy': float(np.mean(pred == y)), 'balanced_accuracy': (sens + spec) / 2, 'sensitivity': sens,
            'specificity': spec, 'auc': float(roc_auc_score(y, prob))}


rows = []
for metodo in ('adaptado', 'original'):
    for seed in (42, 1, 2):
        for alpha in ('iid', '1.0', '0.1'):
            pasta = f'{ORIGEM}/modelos/{metodo}_P40_seed{seed}/P0.4_alpha_{alpha}'
            logits = [keras.models.load_model(f'{pasta}/cliente_{c}.keras', compile=False).predict(dataset, verbose=0)
                      for c in range(4)]
            base = {'metodo': metodo, 'seed': seed, 'alpha': alpha}
            rows += [{**base, 'model': f'cliente {c}', **metrics(l)} for c, l in enumerate(logits)]
            rows.append({**base, 'model': 'ensemble', **metrics(np.mean(logits, axis=0))})
            print(rows[-1], flush=True)
            keras.backend.clear_session()
pd.DataFrame(rows).to_csv(OUT, index=False)
