#!/bin/bash
# Cenário 2: rodadas federadas. private_init 10 passos lr 2e-5 (cenário 1); revisit 3x10 passos lr 2e-5.
cd "/mc809 - Comp Distribuida/projeto_final"
echo "=== cenario2: início $(date +%T)"
uv run python client.py --digest-rounds 60 --private-init-steps 10 --private-init-lr 2e-5 \
  --private-epochs 3 --private-steps 10 --revisit-lr 2e-5 > results/07_revisit_fraco_sem_rehearsal/cen2.log 2>&1
echo "=== cenario2: fim $(date +%T), código $?"
mv results/test_metrics.csv results/07_revisit_fraco_sem_rehearsal/cen2_test_metrics.csv
mv results/train_history.csv results/07_revisit_fraco_sem_rehearsal/cen2_train_history.csv
