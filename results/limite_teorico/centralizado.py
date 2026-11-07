"""Limite teórico: cada arquitetura dos clientes treinada de forma centralizada no conjunto de treino completo.

Mesmas condições do resto do projeto: lr 1e-3, batch 32, aumentação, até INIT_EPOCHS épocas com early stopping
(PATIENCE) e restauração dos melhores pesos, VAL_FRACTION do treino para validação. Avalia no mesmo conjunto de teste.
"""
import os
import sys
import time

sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from client import (BATCH_SIZE, CLIENT_ARCHITECTURES, IMAGE_SIZE, INIT_EPOCHS, PATIENCE, VAL_FRACTION,
                    augmentation_pipeline, build_model, holdout_split, small_cnn)
from dataset_setup import CLASS_NAMES, list_files, make_client_dataset

OUT = 'results/limite_teorico'
SEEDS = [int(s) for s in sys.argv[1:]] or [42, 1, 2]

file_paths, labels = list_files(train_data=True)
test_paths, test_labels = list_files(train_data=False)
test_dataset = make_client_dataset(test_paths, test_labels, np.arange(len(test_labels)), IMAGE_SIZE, BATCH_SIZE, shuffle=False)
augment = augmentation_pipeline(IMAGE_SIZE, train=True)

for seed in SEEDS:
    train_idx, val_idx = holdout_split(np.arange(len(labels)), VAL_FRACTION, seed)
    for arch_id, filters in enumerate(CLIENT_ARCHITECTURES):
        keras.utils.set_random_seed(seed + arch_id)  # mesma convenção dos clientes (seed + client_id)
        model = build_model(small_cnn(filters), num_classes=len(CLASS_NAMES), image_size=IMAGE_SIZE)
        train_dataset = make_client_dataset(file_paths, labels, train_idx, IMAGE_SIZE, BATCH_SIZE).map(
            lambda x, y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = make_client_dataset(file_paths, labels, val_idx, IMAGE_SIZE, BATCH_SIZE, shuffle=False)
        start = time.time()
        history = model.fit(train_dataset, validation_data=val_dataset, epochs=INIT_EPOCHS, verbose=0,
                            callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE,
                                                                     min_delta=1e-4, restore_best_weights=True)])
        test = model.evaluate(test_dataset, verbose=0, return_dict=True)
        h = history.history
        row = {'seed': seed, 'arquitetura': arch_id, 'épocas': len(h['loss']), 'melhor_época': int(np.argmin(h['val_loss'])) + 1,
               'val_accuracy': h['val_accuracy'][int(np.argmin(h['val_loss']))], 'test_loss': test['loss'],
               'test_accuracy': test['accuracy'], 'minutos': (time.time() - start) / 60}
        print(f"seed {seed} arquitetura {arch_id}: teste {row['test_accuracy']:.3f}, {row['épocas']} épocas, "
              f"{row['minutos']:.1f} min", flush=True)
        pd.DataFrame([row]).to_csv(f'{OUT}/teto.csv', mode='a', header=not os.path.exists(f'{OUT}/teto.csv'), index=False)
        pd.DataFrame({**h, 'epoch': range(len(h['loss'])), 'seed': seed, 'arquitetura': arch_id}).to_csv(
            f'{OUT}/teto_history.csv', mode='a', header=not os.path.exists(f'{OUT}/teto_history.csv'), index=False)
        keras.backend.clear_session()
print('=== fim', flush=True)
