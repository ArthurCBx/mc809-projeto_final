#!/bin/bash
# Controle sem destilação (w = 1: o digest é só CE no rótulo verdadeiro das imagens públicas) com 1% e 5% de dados
# públicos, as frações em que o FedMD adaptado ganhava do treino local. O lado w = 0 (só KL) são as execuções
# all_P01_seed* e all_P05_seed* de results/ensemble_e_leave_one_out, com os mesmos argumentos; aqui só entra
# --digest-ce-weight 1 e a avaliação no PH2 (que não altera o treino). Seeds 42, 1 e 2.
# Watchdog: log parado por 60 min → mata, guarda os parciais como *_travado<N>_* e tenta de novo sem XLA.
cd "/mc809 - Comp Distribuida/projeto_final"
D=results/digest_ce_1_5
COMUM="--digest-rounds 30 --private-init-steps 300 --private-init-lr 1e-4 --private-init-rehearsal --private-epochs 3 \
  --private-steps 10 --revisit-lr 1e-4 --rehearsal-fraction 0.5 --digest-lr 4e-4 --digest-steps 30 --digest-temperature 4 \
  --digest-ce-weight 1 --extra-test-dir ph2_dataset"
CSVS="test_metrics train_history ensemble_metrics extra_test_metrics"
primeira=1

mata() {
  local proc
  for proc in /proc/[0-9]*; do case "$(tr '\0' ' ' < $proc/cmdline 2>/dev/null)" in "$1"*) kill ${proc#/proc/};; esac; done 2>/dev/null
}

reinicia_servidor() {
  mata "/opt/venv/bin/python server.py --clients 4"
  uv run ray stop >/dev/null 2>&1; sleep 5
  setsid nohup uv run python server.py --clients 4 --consensus all >> $D/server.log 2>&1 < /dev/null &
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

for seed in 42 1 2; do
  roda CE1.0_P01_seed$seed --public-fractions 0.01 --public-init-lr 3e-4 $COMUM --seed $seed
  roda CE1.0_P05_seed$seed --public-fractions 0.05 --public-init-lr 1e-3 $COMUM --seed $seed
done
mata "/opt/venv/bin/python server.py --clients 4"
uv run ray stop >/dev/null 2>&1
echo "=== orquestrador: fim $(date +%T)"
