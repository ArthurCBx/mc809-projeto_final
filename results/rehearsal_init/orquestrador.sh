#!/bin/bash
# Noite de 01/11: rehearsal no private_init. Roda tudo sozinho; cada passo grava seus CSVs em results/rehearsal_init/.
cd "/mc809 - Comp Distribuida/projeto_final"
D=results/rehearsal_init
BASE="--private-epochs 3 --private-steps 10 --revisit-lr 1e-4 --rehearsal-fraction 0.5 --digest-lr 4e-4 --digest-steps 30"

reinicia_servidor() {
  for p in /proc/[0-9]*; do [ "$(tr '\0' ' ' < $p/cmdline 2>/dev/null)" = "/opt/venv/bin/python server.py --clients 4 " ] && kill ${p#/proc/}; done
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

# A: só public_init + private_init, para checar se o private_init com rehearsal quebra o modelo.
roda A1_init_lr4e-4 90m --digest-rounds 0 --private-init-steps 100 --private-init-lr 4e-4 --private-init-rehearsal $BASE
if uv run python $D/check_init.py $D/A1_init_lr4e-4_test_metrics.csv; then
  INIT_LR=4e-4
else
  roda A2_init_lr1e-4 90m --digest-rounds 0 --private-init-steps 100 --private-init-lr 1e-4 --private-init-rehearsal $BASE
  if uv run python $D/check_init.py $D/A2_init_lr1e-4_test_metrics.csv; then INIT_LR=1e-4; else INIT_LR=nenhum; fi
fi
echo "=== decisão do private_init: $INIT_LR"

ANTERIOR="--digest-rounds 30 --private-init-steps 10 --private-init-lr 2e-5 $BASE"
if [ "$INIT_LR" != "nenhum" ]; then
  NOVO="--digest-rounds 30 --private-init-steps 100 --private-init-lr $INIT_LR --private-init-rehearsal $BASE"
  # B: execução completa com o novo private_init.
  roda B_init${INIT_LR}_seed42 210m $NOVO --seed 42
  # C: robustez, seed 1 da nova configuração e da anterior (digest 30 passos).
  roda C_init${INIT_LR}_seed1 210m $NOVO --seed 1
  roda C_anterior_seed1 210m $ANTERIOR --seed 1
else
  # Nenhum private_init com rehearsal passou: mede a variância da configuração anterior.
  roda C_anterior_seed1 210m $ANTERIOR --seed 1
  roda C_anterior_seed2 210m $ANTERIOR --seed 2
fi
echo "=== orquestrador: fim $(date +%T)"
