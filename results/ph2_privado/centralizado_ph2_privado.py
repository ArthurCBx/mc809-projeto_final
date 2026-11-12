"""Modelo centralizado com parte do PH2 nos dados de treino.

Igual a results/test_models/centralizado_ph2.py, mas a cada seed o PH2 é dividido como no client.py
(--private-dir ph2_dataset --private-dir-fraction 0.4): os 40% que lá são os dados privados dos clientes entram aqui no
treino, junto com todo o dataset de treino, e o teste externo são os outros 60%. Grava results/ph2_privado/centralizado.csv.
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
from dataset_setup import CLASS_NAMES, list_dir, list_files, make_client_dataset, split_external

OUT = 'results/ph2_privado/centralizado.csv'
PRIVATE_FRACTION = 0.4
SEEDS = [int(s) for s in sys.argv[1:]] or [42, 1]

train_paths, train_labels = list_files(train_data=True)
test_paths, test_labels = list_files(train_data=False)
ph2_paths, ph2_labels = list_dir('ph2_dataset')
test_dataset = make_client_dataset(test_paths, test_labels, np.arange(len(test_labels)), IMAGE_SIZE, BATCH_SIZE, shuffle=False)
augment = augmentation_pipeline(IMAGE_SIZE, train=True)


def metrics(logits, y):
    pred = np.argmax(logits, axis=1)
    return float(np.mean(pred == y)), float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))


for seed in SEEDS:
    ph2_train_idx, ph2_test_idx = split_external(ph2_labels, PRIVATE_FRACTION, seed)  # a mesma divisão do client.py
    file_paths = np.concatenate([train_paths, ph2_paths[ph2_train_idx]])
    labels = np.concatenate([train_labels, ph2_labels[ph2_train_idx]])
    train_idx, val_idx = holdout_split(np.arange(len(train_labels)), VAL_FRACTION, seed)
    train_idx = np.concatenate([train_idx, np.arange(len(train_labels), len(labels))])  # PH2 só no treino, como nos clientes
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
        t_acc, _ = metrics(tl, test_labels); p_acc, p_bal = metrics(pl, ph2_y)
        rows.append({'seed': seed, 'model': f'arquitetura {arch_id}', 'test_accuracy': t_acc,
                     'ph2_accuracy': p_acc, 'ph2_balanced_accuracy': p_bal})
        print(f'seed {seed} arquitetura {arch_id}: teste {t_acc:.3f}, PH2 {p_acc:.3f} (balanceada {p_bal:.3f})', flush=True)
        keras.backend.clear_session()
    t_acc, _ = metrics(np.mean(test_logits, axis=0), test_labels)
    p_acc, p_bal = metrics(np.mean(ph2_logits, axis=0), ph2_y)
    rows.append({'seed': seed, 'model': 'ensemble', 'test_accuracy': t_acc, 'ph2_accuracy': p_acc, 'ph2_balanced_accuracy': p_bal})
    pd.DataFrame(rows).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
print('=== fim', flush=True)
