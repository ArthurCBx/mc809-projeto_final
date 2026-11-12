#!/bin/bash
# FedMD no cenário do paper: todo o dataset do Kaggle (9.605 imagens) é público (public_init e digest), e os dados
# privados são 40% do PH2 (80 imagens, estratificadas e divididas entre os clientes por IID ou Dirichlet), todos no
# treino, sem validação, com batch 5 no treino privado. Teste: o do Kaggle e os outros 60% do PH2 (120 imagens).
#   - FedMD adaptado (configuração final, KL T = 4); o treino local é o private_init da mesma execução;
#   - FedMD original (replicação), consenso uniforme;
#   - centralizado: Kaggle completo + os mesmos 80 do PH2.
# Seeds 42 e 1. Watchdog: log parado por 60 min → mata, guarda os parciais como *_travado<N>_* e tenta de novo sem XLA.
cd "/mc809 - Comp Distribuida/projeto_final"
D=results/ph2_privado
CENARIO="--public-fractions 1.0 --private-dir ph2_dataset --private-dir-fraction 0.4 --private-batch-size 5"
ADAPTADO="$CENARIO --public-init-lr 3e-4 --digest-rounds 30 --private-init-steps 300 --private-init-lr 1e-4 \
  --private-init-rehearsal --private-epochs 3 --private-steps 10 --revisit-lr 1e-4 --rehearsal-fraction 0.5 \
  --digest-lr 4e-4 --digest-steps 30 --digest-temperature 4"
ORIGINAL="$CENARIO --public-init-lr 1e-3 --digest-rounds 20 --private-init-epochs 50 --private-epochs 3 \
  --revisit-lr 1e-3 --digest-lr 1e-3"
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

# Adota uma execução que ficou rodando quando o orquestrador anterior foi parado: espera o cliente terminar e guarda os
# CSVs se a execução chegou ao fim; senão, guarda como *_interrompido_*.
adota() {  # adota <nome> <última rodada>
  local nome=$1 ultima=$2 proc vivo=1
  while [ $vivo -eq 1 ]; do
    vivo=0
    for proc in /proc/[0-9]*; do case "$(tr '\0' ' ' < $proc/cmdline 2>/dev/null)" in "/opt/venv/bin/python client.py"*) vivo=1;; esac; done 2>/dev/null
    [ $vivo -eq 1 ] && sleep 60
  done
  [ -f results/test_metrics.csv ] || return
  if awk -F, -v r=$ultima 'NR>1 && $5=="federated" && $6==r {ok=1} END {exit !ok}' results/test_metrics.csv; then
    echo "=== $nome: fim (adotado) $(date +%T)"
    for f in $CSVS; do mv results/$f.csv $D/${nome}_$f.csv 2>/dev/null; done
  else
    echo "=== $nome: interrompido (adotado) $(date +%T)"
    for f in $CSVS; do mv results/$f.csv $D/${nome}_interrompido_$f.csv 2>/dev/null; done
    mv $D/$nome.log $D/${nome}_interrompido.log
  fi
}
adota adaptado_seed42_alpha1.0 30

# Um α por processo, com ray stop entre eles: a troca de configuração dentro do mesmo processo sobe os 4 clientes novos
# enquanto os anteriores ainda liberam a GPU, e o PC caiu duas vezes nesse momento.
for seed in 42 1; do
  for alpha in iid 1.0 0.1; do
    roda adaptado_seed${seed}_alpha$alpha accuracy $ADAPTADO --alphas $alpha --seed $seed
  done
  for alpha in iid 1.0 0.1; do
    roda original_seed${seed}_alpha$alpha uniform $ORIGINAL --alphas $alpha --seed $seed
  done
done
mata "/opt/venv/bin/python server.py --clients 4"
uv run ray stop >/dev/null 2>&1
if [ ! -f $D/centralizado.csv ]; then
  echo "=== centralizado: início $(date +%T)"
  uv run python $D/centralizado_ph2_privado.py 42 1 > $D/centralizado.log 2>&1
  echo "=== centralizado: fim $(date +%T), código $?"
fi
echo "=== orquestrador: fim $(date +%T)"
