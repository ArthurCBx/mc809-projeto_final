"""Modelo centralizado (limite teórico) avaliado no teste original e no PH2.

Cada arquitetura dos clientes treinada no conjunto de treino completo, nas mesmas condições de results/limite_teorico
(lr 1e-3, batch 32, aumentação, até INIT_EPOCHS épocas com early stopping PATIENCE, VAL_FRACTION para validação).
Grava results/test_models/centralizado.csv: uma linha por arquitetura e seed, mais o ensemble das 4 arquiteturas.
"""
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from client import (BATCH_SIZE, CLIENT_ARCHITECTURES, IMAGE_SIZE, INIT_EPOCHS, PATIENCE, VAL_FRACTION,
                    augmentation_pipeline, build_model, holdout_split, small_cnn)
from dataset_setup import CLASS_NAMES, list_dir, list_files, make_client_dataset

OUT = 'results/test_models/centralizado.csv'
SEEDS = [int(s) for s in sys.argv[1:]] or [42, 1, 2]

file_paths, labels = list_files(train_data=True)
test_paths, test_labels = list_files(train_data=False)
ph2_paths, ph2_labels = list_dir('ph2_dataset')
test_dataset = make_client_dataset(test_paths, test_labels, np.arange(len(test_labels)), IMAGE_SIZE, BATCH_SIZE, shuffle=False)
ph2_dataset = make_client_dataset(ph2_paths, ph2_labels, np.arange(len(ph2_labels)), IMAGE_SIZE, BATCH_SIZE, shuffle=False)
augment = augmentation_pipeline(IMAGE_SIZE, train=True)


def metrics(logits, y):
    pred = np.argmax(logits, axis=1)
    return float(np.mean(pred == y)), float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))


for seed in SEEDS:
    train_idx, val_idx = holdout_split(np.arange(len(labels)), VAL_FRACTION, seed)
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
        t_acc, _ = metrics(tl, test_labels); p_acc, p_bal = metrics(pl, ph2_labels)
        rows.append({'seed': seed, 'model': f'arquitetura {arch_id}', 'test_accuracy': t_acc,
                     'ph2_accuracy': p_acc, 'ph2_balanced_accuracy': p_bal})
        print(f'seed {seed} arquitetura {arch_id}: teste {t_acc:.3f}, PH2 {p_acc:.3f} (balanceada {p_bal:.3f})', flush=True)
        keras.backend.clear_session()
    t_acc, _ = metrics(np.mean(test_logits, axis=0), test_labels)
    p_acc, p_bal = metrics(np.mean(ph2_logits, axis=0), ph2_labels)
    rows.append({'seed': seed, 'model': 'ensemble', 'test_accuracy': t_acc, 'ph2_accuracy': p_acc, 'ph2_balanced_accuracy': p_bal})
    pd.DataFrame(rows).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
print('=== fim', flush=True)
