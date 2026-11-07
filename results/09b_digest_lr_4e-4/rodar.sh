#!/bin/bash
# Digest com lr menor. Resto igual a results/09a_revisit_lr_1e-4 (revisit lr 1e-4): private_init 10 passos lr 2e-5;
# revisit 3x10 passos lr 1e-4 com 50% de rehearsal; 30 rodadas.
cd "/mc809 - Comp Distribuida/projeto_final"
lr=$1
echo "=== digest lr $lr: início $(date +%T)"
uv run python client.py --digest-rounds 30 --private-init-steps 10 --private-init-lr 2e-5 \
  --private-epochs 3 --private-steps 10 --revisit-lr 1e-4 --rehearsal-fraction 0.5 --digest-lr $lr \
  > results/09b_digest_lr_4e-4/lr${lr}.log 2>&1
echo "=== digest lr $lr: fim $(date +%T), código $?"
mv results/test_metrics.csv results/09b_digest_lr_4e-4/lr${lr}_test_metrics.csv
mv results/train_history.csv results/09b_digest_lr_4e-4/lr${lr}_train_history.csv
