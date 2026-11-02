#!/bin/bash
# Experimento final: frações públicas 0,01, 0,05 e 0,2, seeds 42, 1 e 2.
# 5% e 20% com seeds 42 e 1 já existem (results/fracoes_pequenas/F2b e F2); aqui roda a seed 2 e o 1% completo.
cd "/mc809 - Comp Distribuida/projeto_final"
D=results/final
COMUM="--private-init-steps 300 --private-init-lr 1e-4 --private-init-rehearsal --private-epochs 3 --private-steps 10 \
  --revisit-lr 1e-4 --rehearsal-fraction 0.5 --digest-lr 4e-4 --digest-steps 30"

reinicia_servidor() {
  for p in /proc/[0-9]*; do [ "$(tr '\0' ' ' < $p/cmdline 2>/dev/null)" = "/opt/venv/bin/python server.py --clients 4 " ] && kill ${p#/proc/}; done 2>/dev/null
  uv run ray stop >/dev/null 2>&1; sleep 5
  setsid nohup uv run python server.py --clients 4 >> $D/server.log 2>&1 < /dev/null &
  sleep 10
}

roda() {  # roda <nome> <timeout> <args...>
  local nome=$1 lim=$2; shift 2
  reinicia_servidor
  echo "=== $nome: início $(date +%T)"
  timeout $lim uv run python client.py "$@" > $D/$nome.log 2>&1
  local rc=$?
  echo "=== $nome: fim $(date +%T), código $rc"
  mv results/test_metrics.csv $D/${nome}_test_metrics.csv 2>/dev/null
  mv results/train_history.csv $D/${nome}_train_history.csv 2>/dev/null
}

# 1%: escolhe o lr do public_init só com as etapas iniciais.
for lr in 1e-3 3e-4; do
  roda I1_pub$lr 40m --public-fractions 0.01 --digest-rounds 0 --public-init-lr $lr $COMUM
done
uv run python $D/escolhe_1pct.py 1e-3 3e-4
read LR1 < $D/escolha_1pct.txt
echo "=== decisão 1%: public_init lr $LR1"

# Execuções completas (30 rodadas).
for seed in 42 1 2; do
  roda P01_seed$seed 150m --public-fractions 0.01 --digest-rounds 30 --public-init-lr $LR1 $COMUM --seed $seed
done
roda P05_seed2 150m --public-fractions 0.05 --digest-rounds 30 --public-init-lr 1e-3 $COMUM --seed 2
roda P20_seed2 150m --public-fractions 0.2 --digest-rounds 30 --public-init-lr 3e-4 $COMUM --seed 2
echo "=== orquestrador: fim $(date +%T)"
