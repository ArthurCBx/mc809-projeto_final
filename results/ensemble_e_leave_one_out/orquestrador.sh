#!/bin/bash
# Etapa final: ensemble dos clientes na inferência (A, consenso padrão) e consenso leave-one-out (B).
# 1% e 5% de dados públicos, seeds 42, 1 e 2, digest com KL T=4; resto igual ao experimento final.
cd "/mc809 - Comp Distribuida/projeto_final"
D=results/ensemble_e_leave_one_out
COMUM="--digest-rounds 30 --private-init-steps 300 --private-init-lr 1e-4 --private-init-rehearsal --private-epochs 3 \
  --private-steps 10 --revisit-lr 1e-4 --rehearsal-fraction 0.5 --digest-lr 4e-4 --digest-steps 30 --digest-temperature 4"

reinicia_servidor() {  # reinicia_servidor <all|loo>
  for p in /proc/[0-9]*; do case "$(tr '\0' ' ' < $p/cmdline 2>/dev/null)" in "/opt/venv/bin/python server.py --clients 4"*) kill ${p#/proc/};; esac; done 2>/dev/null
  uv run ray stop >/dev/null 2>&1; sleep 5
  setsid nohup uv run python server.py --clients 4 --consensus $1 >> $D/server.log 2>&1 < /dev/null &
  sleep 10
}

roda() {  # roda <nome> <consenso> <args...>
  local nome=$1 modo=$2; shift 2
  reinicia_servidor $modo
  echo "=== $nome: início $(date +%T)"
  timeout 120m uv run python client.py "$@" > $D/$nome.log 2>&1
  local rc=$?
  echo "=== $nome: fim $(date +%T), código $rc"
  for f in test_metrics train_history ensemble_metrics; do mv results/$f.csv $D/${nome}_$f.csv 2>/dev/null; done
}

for seed in 42 1 2; do
  for modo in all loo; do
    roda ${modo}_P01_seed$seed $modo --public-fractions 0.01 --public-init-lr 3e-4 $COMUM --seed $seed
    roda ${modo}_P05_seed$seed $modo --public-fractions 0.05 --public-init-lr 1e-3 $COMUM --seed $seed
  done
done
echo "=== orquestrador: fim $(date +%T)"
