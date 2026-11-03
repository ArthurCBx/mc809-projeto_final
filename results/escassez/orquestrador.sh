#!/bin/bash
# Escassez de dados: 50% e 70% dos dados de cada cliente (público e privado) reservados para validação.
# Melhor configuração até agora (KL T=4, consenso padrão, ensemble registrado). 1%, 5% e 20% público, seeds 42, 1, 2.
cd "/mc809 - Comp Distribuida/projeto_final"
D=results/escassez
COMUM="--digest-rounds 30 --private-init-steps 300 --private-init-lr 1e-4 --private-init-rehearsal --private-epochs 3 \
  --private-steps 10 --revisit-lr 1e-4 --rehearsal-fraction 0.5 --digest-lr 4e-4 --digest-steps 30 --digest-temperature 4"

reinicia_servidor() {
  for p in /proc/[0-9]*; do case "$(tr '\0' ' ' < $p/cmdline 2>/dev/null)" in "/opt/venv/bin/python server.py --clients 4"*) kill ${p#/proc/};; esac; done 2>/dev/null
  uv run ray stop >/dev/null 2>&1; sleep 5
  setsid nohup uv run python server.py --clients 4 --consensus all >> $D/server.log 2>&1 < /dev/null &
  sleep 10
}

roda() {  # roda <nome> <args...>
  local nome=$1; shift
  reinicia_servidor
  echo "=== $nome: início $(date +%T)"
  timeout 150m uv run python client.py "$@" > $D/$nome.log 2>&1
  local rc=$?
  echo "=== $nome: fim $(date +%T), código $rc"
  for f in test_metrics train_history ensemble_metrics; do mv results/$f.csv $D/${nome}_$f.csv 2>/dev/null; done
}

for seed in 42 1 2; do
  for val in 0.5 0.7; do
    roda V${val}_P01_seed$seed --val-fraction $val --public-fractions 0.01 --public-init-lr 3e-4 $COMUM --seed $seed
    roda V${val}_P05_seed$seed --val-fraction $val --public-fractions 0.05 --public-init-lr 1e-3 $COMUM --seed $seed
    roda V${val}_P20_seed$seed --val-fraction $val --public-fractions 0.2 --public-init-lr 3e-4 $COMUM --seed $seed
  done
done
echo "=== orquestrador: fim $(date +%T)"
