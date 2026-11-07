#!/bin/bash
# Revisit com lr menor, mantendo o resto do cenário 3: private_init 10 passos lr 2e-5; revisit 3x10 passos, 50% rehearsal; 30 rodadas.
cd "/mc809 - Comp Distribuida/projeto_final"
lr=$1
echo "=== revisit lr $lr: início $(date +%T)"
uv run python client.py --digest-rounds 30 --private-init-steps 10 --private-init-lr 2e-5 \
  --private-epochs 3 --private-steps 10 --revisit-lr $lr --rehearsal-fraction 0.5 > results/09a_revisit_lr_1e-4/lr${lr}.log 2>&1
echo "=== revisit lr $lr: fim $(date +%T), código $?"
mv results/test_metrics.csv results/09a_revisit_lr_1e-4/lr${lr}_test_metrics.csv
mv results/train_history.csv results/09a_revisit_lr_1e-4/lr${lr}_train_history.csv
