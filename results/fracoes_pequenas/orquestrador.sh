#!/bin/bash
# Frações públicas 0,05 e 0,2. Fase 1: grade de lr/passos das etapas iniciais. Fase 2: 30 rodadas, seeds 42 e 1.
cd "/mc809 - Comp Distribuida/projeto_final"
D=results/fracoes_pequenas
GRID="--public-fractions 0.05 0.2"
FED="--private-epochs 3 --private-steps 10 --revisit-lr 1e-4 --rehearsal-fraction 0.5 --digest-lr 4e-4 --digest-steps 30"

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
  timeout $lim uv run python client.py $GRID "$@" > $D/$nome.log 2>&1
  local rc=$?
  echo "=== $nome: fim $(date +%T), código $rc"
  mv results/test_metrics.csv $D/${nome}_test_metrics.csv 2>/dev/null
  mv results/train_history.csv $D/${nome}_train_history.csv 2>/dev/null
}

# Fase 1
for pub in 1e-3 3e-4; do
  for priv in "1e-4 100" "1e-4 300" "3e-4 100"; do
    set -- $priv
    roda F1_pub${pub}_priv$1_s$2 60m --digest-rounds 0 --public-init-lr $pub \
      --private-init-steps $2 --private-init-lr $1 --private-init-rehearsal $FED
  done
done
uv run --with tabulate python $D/seleciona.py
read PUB PRIV STEPS < $D/escolha.txt
echo "=== decisão: public_init lr $PUB, private_init lr $PRIV, $STEPS passos"

# Fase 2
for seed in 42 1; do
  roda F2_seed$seed 180m --digest-rounds 30 --public-init-lr $PUB --private-init-steps $STEPS \
    --private-init-lr $PRIV --private-init-rehearsal $FED --seed $seed
done
echo "=== orquestrador: fim $(date +%T)"
