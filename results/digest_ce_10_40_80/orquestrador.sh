#!/bin/bash
# Digest com âncora nos rótulos: (1 − w) · KL(T = 4) + w · CE no rótulo verdadeiro, w ∈ {0,5; 1}.
# w = 1 é o controle sem destilação (mesmos passos de digest, só treino supervisionado nas imagens públicas).
# Frações 10%, 40% e 80%, seeds 42 e 1, com avaliação no PH2. O w = 0 (só KL) são as execuções
# adaptado_P*_seed{42,1} de results/fracoes_publicas_10_40_80, com os mesmos argumentos.
# Watchdog: se o log ficar 25 min sem atualizar, mata a execução, guarda os parciais como *_travado<N>_* e tenta de
# novo uma vez, sem XLA (--no-jit).
cd "/mc809 - Comp Distribuida/projeto_final"
D=results/digest_ce_10_40_80
BASE="--public-init-lr 3e-4 --digest-rounds 30 --private-init-steps 300 --private-init-lr 1e-4 \
  --private-init-rehearsal --private-epochs 3 --private-steps 10 --revisit-lr 1e-4 --rehearsal-fraction 0.5 \
  --digest-lr 4e-4 --digest-steps 30 --digest-temperature 4 --extra-test-dir ph2_dataset"
CSVS="test_metrics train_history ensemble_metrics extra_test_metrics"

mata() {  # mata <padrão>: processos cujo cmdline começa com o padrão
  local proc
  for proc in /proc/[0-9]*; do case "$(tr '\0' ' ' < $proc/cmdline 2>/dev/null)" in "$1"*) kill ${proc#/proc/};; esac; done 2>/dev/null
}

reinicia_servidor() {
  mata "/opt/venv/bin/python server.py --clients 4"
  uv run ray stop >/dev/null 2>&1; sleep 5
  setsid nohup uv run python server.py --clients 4 --weighting accuracy >> $D/server.log 2>&1 < /dev/null &
  sleep 10
}

roda() {  # roda <nome> <args...>
  local nome=$1 extra="" tentativa pid rc travou idade; shift
  if [ -f $D/${nome}_test_metrics.csv ]; then echo "=== $nome: já concluído, pulando"; return; fi
  for tentativa in 1 2; do
    reinicia_servidor
    echo "=== $nome: início $(date +%T)${extra:+ ($extra)}"
    timeout 240m uv run python client.py "$@" $extra > $D/$nome.log 2>&1 &
    pid=$!; travou=0
    while kill -0 $pid 2>/dev/null; do
      sleep 60
      idade=$(( $(date +%s) - $(stat -c %Y $D/$nome.log) ))
      if [ $idade -gt 1500 ]; then
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
      return
    fi
    echo "=== $nome: TRAVOU (log parado há 25 min) às $(date +%T), tentativa $tentativa"
    for f in $CSVS; do mv results/$f.csv $D/${nome}_travado${tentativa}_$f.csv 2>/dev/null; done
    mv $D/$nome.log $D/${nome}_travado${tentativa}.log
    extra="--no-jit"
  done
  echo "=== $nome: desistindo após 2 travamentos"
}

for frac in 0.1 0.4 0.8; do
  fr=P${frac#0.}0
  for w in 0.5 1.0; do
    for seed in 42 1; do
      roda CE${w}_${fr}_seed$seed --public-fractions $frac $BASE --digest-ce-weight $w --seed $seed
    done
  done
done
mata "/opt/venv/bin/python server.py --clients 4"
uv run ray stop >/dev/null 2>&1
echo "=== orquestrador: fim $(date +%T)"
