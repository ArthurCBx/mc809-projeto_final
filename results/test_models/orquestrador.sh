#!/bin/bash
# Avaliação no PH2: FedMD adaptado (melhor configuração), FedMD original (replicação) e centralizado; 5% público, 3 seeds.
cd "/mc809 - Comp Distribuida/projeto_final"
D=results/test_models
ADAPTADO="--public-fractions 0.05 --public-init-lr 1e-3 --digest-rounds 30 --private-init-steps 300 --private-init-lr 1e-4 \
  --private-init-rehearsal --private-epochs 3 --private-steps 10 --revisit-lr 1e-4 --rehearsal-fraction 0.5 \
  --digest-lr 4e-4 --digest-steps 30 --digest-temperature 4 --extra-test-dir ph2_dataset"
# Replicação do FedMD: consenso uniforme, MAE nos logits, revisit de 3 épocas completas com lr 1e-3, private_init até
# convergência (early stopping), sem rehearsal nem limites de passos, 20 rodadas (como no paper e na primeira ablação).
ORIGINAL="--public-fractions 0.05 --public-init-lr 1e-3 --digest-rounds 20 --private-init-epochs 50 --private-epochs 3 \
  --revisit-lr 1e-3 --digest-lr 1e-3 --extra-test-dir ph2_dataset"

reinicia_servidor() {  # reinicia_servidor <accuracy|uniform>
  for p in /proc/[0-9]*; do case "$(tr '\0' ' ' < $p/cmdline 2>/dev/null)" in "/opt/venv/bin/python server.py --clients 4"*) kill ${p#/proc/};; esac; done 2>/dev/null
  uv run ray stop >/dev/null 2>&1; sleep 5
  setsid nohup uv run python server.py --clients 4 --weighting $1 >> $D/server.log 2>&1 < /dev/null &
  sleep 10
}

roda() {  # roda <nome> <ponderação> <args...>
  local nome=$1 peso=$2; shift 2
  reinicia_servidor $peso
  echo "=== $nome: início $(date +%T)"
  timeout 180m uv run python client.py "$@" > $D/$nome.log 2>&1
  local rc=$?
  echo "=== $nome: fim $(date +%T), código $rc"
  for f in test_metrics train_history ensemble_metrics extra_test_metrics; do mv results/$f.csv $D/${nome}_$f.csv 2>/dev/null; done
}

for seed in 42 1 2; do
  roda adaptado_seed$seed accuracy $ADAPTADO --seed $seed
  roda original_seed$seed uniform $ORIGINAL --seed $seed
done
for p in /proc/[0-9]*; do case "$(tr '\0' ' ' < $p/cmdline 2>/dev/null)" in "/opt/venv/bin/python server.py --clients 4"*) kill ${p#/proc/};; esac; done 2>/dev/null
uv run ray stop >/dev/null 2>&1
echo "=== centralizado: início $(date +%T)"
uv run python $D/centralizado_ph2.py 42 1 2 > $D/centralizado.log 2>&1
echo "=== centralizado: fim $(date +%T), código $?"
uv run --with tabulate python $D/resumo.py > $D/resumo.log 2>&1
echo "=== orquestrador: fim $(date +%T)"
