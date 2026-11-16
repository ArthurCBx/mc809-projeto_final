#!/bin/bash
# 1. Referência centralizada justa do cenário 2 (Kaggle público, PH2 privado): cada arquitetura faz o public_init do
#    FedMD original (Kaggle inteiro, lr 1e-3) e o private_init nas 80 imagens do PH2 reunidas (batch 5, 50 épocas,
#    lr 1e-3, sem validação), sem rodadas (--pooled-private --no-digest --digest-rounds 0). Seeds 42 e 1.
# 2. Sensibilidade, especificidade e AUC no PH2 calculadas das predições (client.py agora grava essas colunas e os
#    logits): cenário 1, IID, f = 5% (argumentos de results/test_models) e 40% (de results/fracoes_publicas_10_40_80),
#    FedMD adaptado e original (o treinamento local é o private_init da mesma execução), seeds 42, 1 e 2; e os
#    centralizados dos dois cenários.
# Watchdog: log parado por 60 min → mata, guarda os parciais como *_travado<N>_* e tenta de novo sem XLA.
cd "/mc809 - Comp Distribuida/projeto_final"
D=results/referencia_justa_e_auc
CEN2="--public-fractions 1.0 --private-dir ph2_dataset --private-dir-fraction 0.4 --private-batch-size 5"
REFERENCIA="$CEN2 --public-init-lr 1e-3 --private-init-epochs 50 --pooled-private --no-digest --digest-rounds 0 --alphas iid"
ADAPTADO_5="--public-fractions 0.05 --public-init-lr 1e-3 --digest-rounds 30 --private-init-steps 300 --private-init-lr 1e-4 \
  --private-init-rehearsal --private-epochs 3 --private-steps 10 --revisit-lr 1e-4 --rehearsal-fraction 0.5 \
  --digest-lr 4e-4 --digest-steps 30 --digest-temperature 4 --extra-test-dir ph2_dataset --alphas iid"
ORIGINAL_5="--public-fractions 0.05 --public-init-lr 1e-3 --digest-rounds 20 --private-init-epochs 50 --private-epochs 3 \
  --revisit-lr 1e-3 --digest-lr 1e-3 --extra-test-dir ph2_dataset --alphas iid"
ADAPTADO_40="--public-fractions 0.4 --public-init-lr 3e-4 --digest-rounds 30 --private-init-steps 300 --private-init-lr 1e-4 \
  --private-init-rehearsal --private-epochs 3 --private-steps 10 --revisit-lr 1e-4 --rehearsal-fraction 0.5 \
  --digest-lr 4e-4 --digest-steps 30 --digest-temperature 4 --extra-test-dir ph2_dataset --alphas iid"
ORIGINAL_40="--public-fractions 0.4 --public-init-lr 1e-3 --digest-rounds 20 --private-init-epochs 50 --private-epochs 3 \
  --revisit-lr 1e-3 --digest-lr 1e-3 --extra-test-dir ph2_dataset --alphas iid"
CSVS="test_metrics train_history ensemble_metrics extra_test_metrics"
primeira=1
mata() {
  local proc
  for proc in /proc/[0-9]*; do case "$(tr '\0' ' ' < $proc/cmdline 2>/dev/null)" in "$1"*) kill ${proc#/proc/};; esac; done 2>/dev/null
}
reinicia_servidor() {  # reinicia_servidor <accuracy|uniform>
  mata "/opt/venv/bin/python server.py --clients 4"
  uv run ray stop >/dev/null 2>&1; sleep 5
  setsid nohup uv run python server.py --clients 4 --weighting $1 >> $D/server.log 2>&1 < /dev/null &
  sleep 10
}
roda() {  # roda <nome> <ponderação> <args...>
  local nome=$1 peso=$2 extra="" tentativa pid rc travou idade; shift 2
  if [ -f $D/${nome}_test_metrics.csv ]; then echo "=== $nome: já concluído, pulando"; primeira=0; return; fi
  for tentativa in 1 2; do
    reinicia_servidor $peso
    echo "=== $nome: início $(date +%T)${extra:+ ($extra)}"
    timeout 240m uv run python client.py "$@" $extra > $D/$nome.log 2>&1 &
    pid=$!; travou=0
    while kill -0 $pid 2>/dev/null; do
      sleep 60
      idade=$(( $(date +%s) - $(stat -c %Y $D/$nome.log) ))
      if [ $idade -gt 3600 ]; then  # o public_init com o Kaggle inteiro fica ~23 min sem escrever no log
        travou=1
        mata "timeout 240m uv run python client.py"; mata "uv run python client.py"; mata "/opt/venv/bin/python client.py"
        uv run ray stop >/dev/null 2>&1
        break
      fi
    done
    wait $pid; rc=$?
    if [ $travou -eq 0 ]; then
      echo "=== $nome: fim $(date +%T), código $rc"
      for f in $CSVS; do mv results/$f.csv $D/${nome}_$f.csv 2>/dev/null; done; mv results/extra_test_logits_*.npz $D/${nome}_logits.npz 2>/dev/null
      if [ $rc -ne 0 ] && [ $primeira -eq 1 ]; then echo "=== primeira execução falhou: parando a fila"; exit 1; fi
      primeira=0
      return
    fi
    echo "=== $nome: TRAVOU (log parado há 60 min) às $(date +%T), tentativa $tentativa"
    for f in $CSVS; do mv results/$f.csv $D/${nome}_travado${tentativa}_$f.csv 2>/dev/null; done; rm -f results/extra_test_logits_*.npz
    mv $D/$nome.log $D/${nome}_travado${tentativa}.log
    extra="--no-jit"
  done
  echo "=== $nome: desistindo após 2 travamentos"
}

for seed in 42 1; do roda referencia_seed$seed uniform $REFERENCIA --seed $seed; done
for seed in 42 1 2; do
  roda adaptado_P05_seed$seed accuracy $ADAPTADO_5 --seed $seed
  roda original_P05_seed$seed uniform $ORIGINAL_5 --seed $seed
  roda adaptado_P40_seed$seed accuracy $ADAPTADO_40 --seed $seed
  roda original_P40_seed$seed uniform $ORIGINAL_40 --seed $seed
done
mata "/opt/venv/bin/python server.py --clients 4"
uv run ray stop >/dev/null 2>&1
for c in 2 1; do
  [ $c = 2 ] && seeds="42 1" || seeds="42 1 2"
  if [ ! -f $D/centralizado_cenario$c.csv ]; then
    echo "=== centralizado cenário $c: início $(date +%T)"
    uv run python $D/centralizado_metricas.py $c $seeds > $D/centralizado_cenario$c.log 2>&1
    echo "=== centralizado cenário $c: fim $(date +%T), código $?"
  fi
done
echo "=== orquestrador: fim $(date +%T)"
