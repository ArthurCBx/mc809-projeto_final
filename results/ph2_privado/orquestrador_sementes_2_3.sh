#!/bin/bash
# Item 5: sementes 2 e 3 no cenário 2 (Kaggle público, PH2 privado) para as variantes da Tabela 9: FedMD replicado,
# sem digest, digest só com CE e a referência centralizada justa (80 imagens do PH2 reunidas); também o centralizado
# diluído (ISIC + 80 do PH2), para a linha antiga da Tabela 9. Leva o n dos pares de 6 para 12.
# Depois: FedMD replicado com f = 1% (cenário 1, as três divisões, seeds 42, 1 e 2), que falta na Figura 7 e na Tabela 7.
# Um α por processo, com ray stop entre eles. Watchdog: log parado por 60 min → mata e tenta de novo sem XLA.
cd "/mc809 - Comp Distribuida/projeto_final"
D=results/ph2_privado
CEN2="--public-fractions 1.0 --private-dir ph2_dataset --private-dir-fraction 0.4 --private-batch-size 5 \
  --public-init-lr 1e-3 --private-init-epochs 50"
ORIGINAL="$CEN2 --digest-rounds 20 --private-epochs 3 --revisit-lr 1e-3 --digest-lr 1e-3"
REFERENCIA="$CEN2 --pooled-private --no-digest --digest-rounds 0 --alphas iid"
ORIGINAL_1="--public-fractions 0.01 --public-init-lr 1e-3 --digest-rounds 20 --private-init-epochs 50 --private-epochs 3 \
  --revisit-lr 1e-3 --digest-lr 1e-3 --extra-test-dir ph2_dataset"
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

for seed in 2 3; do
  for alpha in iid 1.0 0.1; do
    roda original_seed${seed}_alpha$alpha uniform $ORIGINAL --alphas $alpha --seed $seed
    roda semdigest_original_seed${seed}_alpha$alpha uniform $ORIGINAL --no-digest --alphas $alpha --seed $seed
    roda controle_original_seed${seed}_alpha$alpha uniform $ORIGINAL --digest-ce-weight 1 --alphas $alpha --seed $seed
  done
  D=results/referencia_justa_e_auc
  roda referencia_seed$seed uniform $REFERENCIA --seed $seed
  D=results/ph2_privado
done
mata "/opt/venv/bin/python server.py --clients 4"
uv run ray stop >/dev/null 2>&1
echo "=== centralizado cenário 2, seeds 2 e 3: início $(date +%T)"
uv run python results/referencia_justa_e_auc/centralizado_metricas.py 2 2 3 > results/referencia_justa_e_auc/centralizado_cenario2_seeds23.log 2>&1
echo "=== centralizado cenário 2, seeds 2 e 3: fim $(date +%T), código $?"

D=results/replicado_P01
for seed in 42 1 2; do
  for alpha in iid 1.0 0.1; do
    roda original_P01_seed${seed}_alpha$alpha uniform $ORIGINAL_1 --alphas $alpha --seed $seed
  done
done
mata "/opt/venv/bin/python server.py --clients 4"
uv run ray stop >/dev/null 2>&1
echo "=== orquestrador: fim $(date +%T)"
