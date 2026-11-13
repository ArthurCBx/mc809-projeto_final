#!/bin/bash
# Sanity check sem colaboração no cenário Kaggle público + PH2 privado: a configuração do FedMD original, mas com o
# digest só em entropia cruzada no rótulo verdadeiro das imagens públicas (--digest-ce-weight 1). O consenso ainda é
# calculado (o servidor exige), mas não entra na loss. Compara com original_seed*_alpha* (o mesmo, com MAE no consenso).
# Seeds 42 e 1, IID, α 1.0 e α 0.1, um α por processo. Espera a fila de results/digest_ce_1_5 terminar.
# Watchdog: log parado por 60 min → mata, guarda os parciais como *_travado<N>_* e tenta de novo sem XLA.
cd "/mc809 - Comp Distribuida/projeto_final"
D=results/ph2_privado
until grep -q "=== orquestrador: fim" results/digest_ce_1_5/orquestrador.log 2>/dev/null; do sleep 60; done

CONTROLE="--public-fractions 1.0 --private-dir ph2_dataset --private-dir-fraction 0.4 --private-batch-size 5 \
  --public-init-lr 1e-3 --digest-rounds 20 --private-init-epochs 50 --private-epochs 3 --revisit-lr 1e-3 --digest-lr 1e-3 \
  --digest-ce-weight 1"
CSVS="test_metrics train_history ensemble_metrics extra_test_metrics"
primeira=1

mata() {
  local proc
  for proc in /proc/[0-9]*; do case "$(tr '\0' ' ' < $proc/cmdline 2>/dev/null)" in "$1"*) kill ${proc#/proc/};; esac; done 2>/dev/null
}

reinicia_servidor() {
  mata "/opt/venv/bin/python server.py --clients 4"
  uv run ray stop >/dev/null 2>&1; sleep 5
  setsid nohup uv run python server.py --clients 4 --weighting uniform >> $D/server.log 2>&1 < /dev/null &
  sleep 10
}

roda() {  # roda <nome> <args...>
  local nome=$1 extra="" tentativa pid rc travou idade; shift
  if [ -f $D/${nome}_test_metrics.csv ]; then echo "=== $nome: já concluído, pulando"; primeira=0; return; fi
  for tentativa in 1 2; do
    reinicia_servidor
    echo "=== $nome: início $(date +%T)${extra:+ ($extra)}"
    timeout 240m uv run python client.py "$@" $extra > $D/$nome.log 2>&1 &
    pid=$!; travou=0
    while kill -0 $pid 2>/dev/null; do
      sleep 60
      idade=$(( $(date +%s) - $(stat -c %Y $D/$nome.log) ))
      if [ $idade -gt 3600 ]; then
        travou=1
        mata "timeout 240m uv run python client.py"; mata "uv run python client.py"; mata "/opt/venv/bin/python client.py"
        uv run ray stop >/dev/null 2>&1
        break
      fi
    done
    wait $pid; rc=$?
    if [ $travou -eq 0 ]; then
      echo "=== $nome: fim $(date +%T), código $rc"
      for f in $CSVS; do mv results/$f.csv $D/${nome}_$f.csv 2>/dev/null; done
      if [ $rc -ne 0 ] && [ $primeira -eq 1 ]; then echo "=== primeira execução falhou: parando a fila"; exit 1; fi
      primeira=0
      return
    fi
    echo "=== $nome: TRAVOU (log parado há 60 min) às $(date +%T), tentativa $tentativa"
    for f in $CSVS; do mv results/$f.csv $D/${nome}_travado${tentativa}_$f.csv 2>/dev/null; done
    mv $D/$nome.log $D/${nome}_travado${tentativa}.log
    extra="--no-jit"
  done
  echo "=== $nome: desistindo após 2 travamentos"
}

for seed in 42 1; do
  for alpha in iid 1.0 0.1; do
    roda controle_original_seed${seed}_alpha$alpha $CONTROLE --alphas $alpha --seed $seed
  done
done
mata "/opt/venv/bin/python server.py --clients 4"
uv run ray stop >/dev/null 2>&1
echo "=== orquestrador: fim $(date +%T)"
