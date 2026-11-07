#!/bin/bash
# Cenário 1: só public_init + private_init (sem rodadas federadas), variando os passos do private_init.
cd "/mc809 - Comp Distribuida/projeto_final"
# Uso: cenario1.sh <passos> <lr> [<lr> ...]  (lr "padrao" = sem --private-init-lr)
n=$1; shift
for lr in "$@"; do
  tag=init${n}_lr${lr}; extra=""; [ "$lr" != "padrao" ] && extra="--private-init-lr $lr"
  echo "=== $tag: início $(date +%T)"
  uv run python client.py --digest-rounds 0 --private-init-steps $n $extra > results/06_private_init_10_passos_lr_2e-5/cen1_${tag}.log 2>&1
  echo "=== $tag: fim $(date +%T), código $?"
  mv results/test_metrics.csv results/06_private_init_10_passos_lr_2e-5/cen1_${tag}_test_metrics.csv
  mv results/train_history.csv results/06_private_init_10_passos_lr_2e-5/cen1_${tag}_train_history.csv
done
