"""Centralizado (modelo único ISIC + PH2 opcional) com sensibilidade, especificidade e AUC no PH2.

Repete results/test_models/centralizado_ph2.py (cenário 1: treino completo do ISIC, teste no PH2 inteiro) e
results/ph2_privado/centralizado_ph2_privado.py (cenário 2: ISIC + os 80 do PH2 da seed, teste nos outros 120), nas
mesmas condições, gravando também os logits do PH2. Uso: centralizado_metricas.py <1|2> <seeds...>
Grava results/referencia_justa_e_auc/centralizado_cenario<N>.csv e centralizado_cenario<N>_seed<S>_logits.npz.
"""
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn.metrics import roc_auc_score

from client import (BATCH_SIZE, CLIENT_ARCHITECTURES, IMAGE_SIZE, INIT_EPOCHS, PATIENCE, VAL_FRACTION,
                    augmentation_pipeline, build_model, holdout_split, small_cnn)
from dataset_setup import CLASS_NAMES, list_dir, list_files, make_client_dataset, split_external

D = 'results/referencia_justa_e_auc'
CENARIO = int(sys.argv[1])
SEEDS = [int(s) for s in sys.argv[2:]]
OUT = f'{D}/centralizado_cenario{CENARIO}.csv'

train_paths, train_labels = list_files(train_data=True)
test_paths, test_labels = list_files(train_data=False)
ph2_paths, ph2_labels = list_dir('ph2_dataset')
test_dataset = make_client_dataset(test_paths, test_labels, np.arange(len(test_labels)), IMAGE_SIZE, BATCH_SIZE, shuffle=False)
augment = augmentation_pipeline(IMAGE_SIZE, train=True)


def metrics(logits, y):
    pred = np.argmax(logits, axis=1)
    sens, spec = float(np.mean(pred[y == 1] == 1)), float(np.mean(pred[y == 0] == 0))
    prob = tf.nn.softmax(logits).numpy()[:, 1]
    return {'ph2_accuracy': float(np.mean(pred == y)), 'ph2_balanced_accuracy': (sens + spec) / 2,
            'ph2_sensitivity': sens, 'ph2_specificity': spec, 'ph2_auc': float(roc_auc_score(y, prob))}


for seed in SEEDS:
    train_idx, val_idx = holdout_split(np.arange(len(train_labels)), VAL_FRACTION, seed)
    file_paths, labels = train_paths, train_labels
    ph2_test_idx = np.arange(len(ph2_labels))
    if CENARIO == 2:
        ph2_train_idx, ph2_test_idx = split_external(ph2_labels, 0.4, seed)  # a mesma divisão do client.py
        file_paths = np.concatenate([train_paths, ph2_paths[ph2_train_idx]])
        labels = np.concatenate([train_labels, ph2_labels[ph2_train_idx]])
        train_idx = np.concatenate([train_idx, np.arange(len(train_labels), len(labels))])
    ph2_y = ph2_labels[ph2_test_idx]
    ph2_dataset = make_client_dataset(ph2_paths[ph2_test_idx], ph2_y, np.arange(len(ph2_y)), IMAGE_SIZE, BATCH_SIZE, shuffle=False)
    test_logits, ph2_logits, rows = [], [], []
    for arch_id, filters in enumerate(CLIENT_ARCHITECTURES):
        keras.utils.set_random_seed(seed + arch_id)
        model = build_model(small_cnn(filters), num_classes=len(CLASS_NAMES), image_size=IMAGE_SIZE)
        train_dataset = make_client_dataset(file_paths, labels, train_idx, IMAGE_SIZE, BATCH_SIZE).map(
            lambda x, y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = make_client_dataset(file_paths, labels, val_idx, IMAGE_SIZE, BATCH_SIZE, shuffle=False)
        model.fit(train_dataset, validation_data=val_dataset, epochs=INIT_EPOCHS, verbose=0,
                  callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, min_delta=1e-4,
                                                           restore_best_weights=True)])
        tl, pl = model.predict(test_dataset, verbose=0), model.predict(ph2_dataset, verbose=0)
        test_logits.append(tl); ph2_logits.append(pl)
        rows.append({'seed': seed, 'model': f'arquitetura {arch_id}',
                     'test_accuracy': float(np.mean(np.argmax(tl, 1) == test_labels)), **metrics(pl, ph2_y)})
        print(f'cenário {CENARIO} seed {seed}: {rows[-1]}', flush=True)
        keras.backend.clear_session()
    rows.append({'seed': seed, 'model': 'ensemble',
                 'test_accuracy': float(np.mean(np.argmax(np.mean(test_logits, 0), 1) == test_labels)),
                 **metrics(np.mean(ph2_logits, axis=0), ph2_y)})
    np.savez(f'{D}/centralizado_cenario{CENARIO}_seed{seed}_logits.npz', labels=ph2_y, logits=np.array(ph2_logits))
    pd.DataFrame(rows).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
print('=== fim', flush=True)
