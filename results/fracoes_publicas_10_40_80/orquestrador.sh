#!/bin/bash
# Hipótese: com mais dados públicos e menos dados por cliente, a colaboração do FedMD ficaria mais vantajosa.
# Frações das ablações (10%, 40%, 80%), 3 seeds cada, com os mesmos métodos da avaliação no PH2:
#   - FedMD adaptado (configuração final); a linha de base local é o private_init dessa mesma execução;
#   - FedMD original (replicação), com consenso uniforme.
# Os modelos finais de cada cliente (após a última rodada) ficam em modelos/<execução>/P<fração>_alpha_<alpha>/.
# O centralizado não depende da fração (results/test_models/centralizado.csv).
# Espera a avaliação do PH2 (results/test_models) terminar para não disputar a GPU.
cd "/mc809 - Comp Distribuida/projeto_final"
D=results/fracoes_publicas_10_40_80
until grep -q "=== orquestrador: fim" results/test_models/orquestrador.log 2>/dev/null; do sleep 60; done

# lr do public_init de 3e-4 no adaptado: foi o escolhido para 20% (frações a partir de 10% preferiram o lr menor).
ADAPTADO="--public-init-lr 3e-4 --digest-rounds 30 --private-init-steps 300 --private-init-lr 1e-4 \
  --private-init-rehearsal --private-epochs 3 --private-steps 10 --revisit-lr 1e-4 --rehearsal-fraction 0.5 \
  --digest-lr 4e-4 --digest-steps 30 --digest-temperature 4 --extra-test-dir ph2_dataset"
# Replicação do FedMD: os mesmos argumentos de results/test_models/orquestrador.sh, com lr 1e-3 em todas as etapas.
ORIGINAL="--public-init-lr 1e-3 --digest-rounds 20 --private-init-epochs 50 --private-epochs 3 \
  --revisit-lr 1e-3 --digest-lr 1e-3 --extra-test-dir ph2_dataset"

reinicia_servidor() {  # reinicia_servidor <accuracy|uniform>
  local proc
  for proc in /proc/[0-9]*; do case "$(tr '\0' ' ' < $proc/cmdline 2>/dev/null)" in "/opt/venv/bin/python server.py --clients 4"*) kill ${proc#/proc/};; esac; done 2>/dev/null
  uv run ray stop >/dev/null 2>&1; sleep 5
  setsid nohup uv run python server.py --clients 4 --weighting $1 >> $D/server.log 2>&1 < /dev/null &
  sleep 10
}

roda() {  # roda <nome> <ponderação> <args...>
  local nome=$1 peso=$2; shift 2
  if [ -f $D/${nome}_test_metrics.csv ]; then echo "=== $nome: já concluído, pulando"; return; fi
  reinicia_servidor $peso
  echo "=== $nome: início $(date +%T)"
  timeout 240m uv run python client.py "$@" --save-models-dir $D/modelos/$nome > $D/$nome.log 2>&1
  echo "=== $nome: fim $(date +%T), código $?"
  for f in test_metrics train_history ensemble_metrics extra_test_metrics; do mv results/$f.csv $D/${nome}_$f.csv 2>/dev/null; done
}

# Uma fração inteira (os dois métodos, 3 seeds) antes da próxima: 40% primeiro (cenário principal da hipótese).
for frac in 0.4 0.1 0.8; do
  fr=P${frac#0.}0
  for seed in 42 1 2; do
    roda adaptado_${fr}_seed$seed accuracy --public-fractions $frac $ADAPTADO --seed $seed
    roda original_${fr}_seed$seed uniform --public-fractions $frac $ORIGINAL --seed $seed
  done
done
for proc in /proc/[0-9]*; do case "$(tr '\0' ' ' < $proc/cmdline 2>/dev/null)" in "/opt/venv/bin/python server.py --clients 4"*) kill ${proc#/proc/};; esac; done 2>/dev/null
uv run ray stop >/dev/null 2>&1
echo "=== orquestrador: fim $(date +%T)"
