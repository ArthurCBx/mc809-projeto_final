#!/bin/bash
# 0,5% de dados públicos (~49 imagens). Escolhe o lr do public_init com as etapas iniciais e roda 3 seeds com a
# configuração da etapa final (KL T=4, consenso padrão, ensemble registrado).
cd "/mc809 - Comp Distribuida/projeto_final"
D=results/fracao_publica_0_5
COMUM="--private-init-steps 300 --private-init-lr 1e-4 --private-init-rehearsal --private-epochs 3 \
  --private-steps 10 --revisit-lr 1e-4 --rehearsal-fraction 0.5 --digest-lr 4e-4 --digest-steps 30 --digest-temperature 4"

reinicia_servidor() {
  for p in /proc/[0-9]*; do case "$(tr '\0' ' ' < $p/cmdline 2>/dev/null)" in "/opt/venv/bin/python server.py --clients 4"*) kill ${p#/proc/};; esac; done 2>/dev/null
  uv run ray stop >/dev/null 2>&1; sleep 5
  setsid nohup uv run python server.py --clients 4 --consensus all >> $D/server.log 2>&1 < /dev/null &
  sleep 10
}

roda() {  # roda <nome> <timeout> <args...>
  local nome=$1 lim=$2; shift 2
  reinicia_servidor
  echo "=== $nome: início $(date +%T)"
  timeout $lim uv run python client.py --public-fractions 0.005 "$@" > $D/$nome.log 2>&1
  local rc=$?
  echo "=== $nome: fim $(date +%T), código $rc"
  for f in test_metrics train_history ensemble_metrics; do mv results/$f.csv $D/${nome}_$f.csv 2>/dev/null; done
}

for lr in 1e-3 3e-4; do roda I1_pub$lr 40m --digest-rounds 0 --public-init-lr $lr $COMUM; done
uv run python $D/escolhe_lr.py 1e-3 3e-4
read LR < $D/escolha.txt
echo "=== decisão 0,5%: public_init lr $LR"
for seed in 42 1 2; do roda P005_seed$seed 120m --digest-rounds 30 --public-init-lr $LR $COMUM --seed $seed; done
echo "=== orquestrador: fim $(date +%T)"
