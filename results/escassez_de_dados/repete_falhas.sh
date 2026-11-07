#!/bin/bash
# Repete as execuções do orquestrador que terminaram com código != 0, com 2,8 GB de GPU por cliente e sem XLA (--no-jit).
cd "/mc809 - Comp Distribuida/projeto_final"
D=results/escassez_de_dados
while [ -d /proc/62005 ]; do sleep 30; done
COMUM="--digest-rounds 30 --private-init-steps 300 --private-init-lr 1e-4 --private-init-rehearsal --private-epochs 3 \
  --private-steps 10 --revisit-lr 1e-4 --rehearsal-fraction 0.5 --digest-lr 4e-4 --digest-steps 30 --digest-temperature 4"
for nome in $(grep -E "fim .*código [1-9]" $D/orquestrador.log | sed -E 's/=== ([^:]+):.*/\1/'); do
  val=$(echo $nome | sed -E 's/V([0-9.]+)_.*/\1/'); f=$(echo $nome | sed -E 's/.*_P([0-9]+)_.*/\1/'); seed=$(echo $nome | sed -E 's/.*_seed([0-9]+)/\1/')
  case $f in 01) frac=0.01; lr=3e-4;; 05) frac=0.05; lr=1e-3;; 20) frac=0.2; lr=3e-4;; esac
  for p in /proc/[0-9]*; do case "$(tr '\0' ' ' < $p/cmdline 2>/dev/null)" in "/opt/venv/bin/python server.py --clients 4"*) kill ${p#/proc/};; esac; done 2>/dev/null
  uv run ray stop >/dev/null 2>&1; sleep 5
  setsid nohup uv run python server.py --clients 4 --consensus all >> $D/server.log 2>&1 < /dev/null &
  sleep 10
  mv $D/$nome.log $D/${nome}_falha.log 2>/dev/null
  echo "=== repetição $nome: início $(date +%T)"
  timeout 150m uv run python client.py --val-fraction $val --public-fractions $frac --public-init-lr $lr $COMUM --seed $seed \
    --gpu-memory-mb 2800 --no-jit > $D/$nome.log 2>&1
  echo "=== repetição $nome: fim $(date +%T), código $?"
  for x in test_metrics train_history ensemble_metrics; do mv results/$x.csv $D/${nome}_$x.csv 2>/dev/null; done
done
echo "=== repetições: fim $(date +%T)"
