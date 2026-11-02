#!/bin/bash
# Digest com KL e temperatura. Triagem T=2 e T=4 na seed 42; o melhor roda as seeds 1 e 2.
cd "/mc809 - Comp Distribuida/projeto_final"
D=results/kl
COMUM="--digest-rounds 30 --private-init-steps 300 --private-init-lr 1e-4 --private-init-rehearsal --private-epochs 3 \
  --private-steps 10 --revisit-lr 1e-4 --rehearsal-fraction 0.5 --digest-lr 4e-4 --digest-steps 30"

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

par() {  # par <T> <seed>: roda 1% e 5% (cada um com o lr de public_init do experimento final)
  roda T$1_P01_seed$2 120m --public-fractions 0.01 --public-init-lr 3e-4 --digest-temperature $1 $COMUM --seed $2
  roda T$1_P05_seed$2 120m --public-fractions 0.05 --public-init-lr 1e-3 --digest-temperature $1 $COMUM --seed $2
}

for T in 2 4; do par $T 42; done
uv run python $D/escolhe_T.py 2 4
read TB < $D/escolha_T.txt
echo "=== decisão: T=$TB"
for seed in 1 2; do par $TB $seed; done
echo "=== orquestrador: fim $(date +%T)"
