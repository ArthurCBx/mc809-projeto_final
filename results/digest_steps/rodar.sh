#!/bin/bash
# Digest com passos fixos por rodada. Resto igual a results/digest_lr (lr 4e-4): private_init 10 passos lr 2e-5;
# revisit 3x10 passos lr 1e-4 com 50% de rehearsal; digest lr 4e-4; 30 rodadas.
cd "/mc809 - Comp Distribuida/projeto_final"
n=$1
echo "=== digest $n passos: início $(date +%T)"
uv run python client.py --digest-rounds 30 --private-init-steps 10 --private-init-lr 2e-5 \
  --private-epochs 3 --private-steps 10 --revisit-lr 1e-4 --rehearsal-fraction 0.5 --digest-lr 4e-4 --digest-steps $n \
  > results/digest_steps/steps${n}.log 2>&1
echo "=== digest $n passos: fim $(date +%T), código $?"
mv results/test_metrics.csv results/digest_steps/steps${n}_test_metrics.csv
mv results/train_history.csv results/digest_steps/steps${n}_train_history.csv
