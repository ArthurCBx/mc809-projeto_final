#!/bin/bash
# Fase 2 da segunda colocada, se ela ficou a < 0,005 da primeira. Espera o orquestrador (PID 9301) terminar.
cd "/mc809 - Comp Distribuida/projeto_final"
D=results/fracoes_publicas_5_e_20
while [ -d /proc/9301 ]; do sleep 30; done
[ -f $D/escolha2.txt ] || { echo "=== complemento: sem segunda colocada próxima, nada a rodar $(date +%T)"; exit 0; }
read PUB PRIV STEPS < $D/escolha2.txt
FED="--private-epochs 3 --private-steps 10 --revisit-lr 1e-4 --rehearsal-fraction 0.5 --digest-lr 4e-4 --digest-steps 30"
for seed in 42 1; do
  for p in /proc/[0-9]*; do [ "$(tr '\0' ' ' < $p/cmdline 2>/dev/null)" = "/opt/venv/bin/python server.py --clients 4 " ] && kill ${p#/proc/}; done 2>/dev/null
  uv run ray stop >/dev/null 2>&1; sleep 5
  setsid nohup uv run python server.py --clients 4 >> $D/server.log 2>&1 < /dev/null &
  sleep 10
  nome=F2b_seed$seed
  echo "=== $nome (public_init lr $PUB, private_init lr $PRIV, $STEPS passos): início $(date +%T)"
  timeout 180m uv run python client.py --public-fractions 0.05 0.2 --digest-rounds 30 --public-init-lr $PUB     --private-init-steps $STEPS --private-init-lr $PRIV --private-init-rehearsal $FED --seed $seed > $D/$nome.log 2>&1
  echo "=== $nome: fim $(date +%T), código $?"
  mv results/test_metrics.csv $D/${nome}_test_metrics.csv 2>/dev/null
  mv results/train_history.csv $D/${nome}_train_history.csv 2>/dev/null
done
echo "=== complemento: fim $(date +%T)"
