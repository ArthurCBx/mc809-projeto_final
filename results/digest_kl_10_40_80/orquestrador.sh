#!/bin/bash
# KL × MAE no digest com as frações das ablações (10%, 40%, 80%), 2 seeds, só no teste do dataset de treino.
# A escolha da KL (results/digest_kl) foi feita com 1% e 5% de dados públicos; aqui repetimos com frações maiores.
# O lado KL (T = 4) é o FedMD adaptado de results/fracoes_publicas_10_40_80 (adaptado_P*_seed{42,1}), com os mesmos
# argumentos e seeds; este orquestrador roda só o lado MAE (sem --digest-temperature).
# Espera a fila de results/fracoes_publicas_10_40_80 terminar para não disputar a GPU.
cd "/mc809 - Comp Distribuida/projeto_final"
D=results/digest_kl_10_40_80
until grep -q "=== orquestrador: fim" results/fracoes_publicas_10_40_80/orquestrador.log 2>/dev/null; do sleep 60; done

MAE="--public-init-lr 3e-4 --digest-rounds 30 --private-init-steps 300 --private-init-lr 1e-4 \
  --private-init-rehearsal --private-epochs 3 --private-steps 10 --revisit-lr 1e-4 --rehearsal-fraction 0.5 \
  --digest-lr 4e-4 --digest-steps 30"

reinicia_servidor() {
  local proc
  for proc in /proc/[0-9]*; do case "$(tr '\0' ' ' < $proc/cmdline 2>/dev/null)" in "/opt/venv/bin/python server.py --clients 4"*) kill ${proc#/proc/};; esac; done 2>/dev/null
  uv run ray stop >/dev/null 2>&1; sleep 5
  setsid nohup uv run python server.py --clients 4 --weighting accuracy >> $D/server.log 2>&1 < /dev/null &
  sleep 10
}

for frac in 0.1 0.4 0.8; do
  for seed in 42 1; do
    nome=MAE_P${frac#0.}0_seed$seed
    if [ -f $D/${nome}_test_metrics.csv ]; then echo "=== $nome: já concluído, pulando"; continue; fi
    reinicia_servidor
    echo "=== $nome: início $(date +%T)"
    timeout 240m uv run python client.py --public-fractions $frac $MAE --seed $seed > $D/$nome.log 2>&1
    echo "=== $nome: fim $(date +%T), código $?"
    for f in test_metrics train_history ensemble_metrics; do mv results/$f.csv $D/${nome}_$f.csv 2>/dev/null; done
  done
done
for proc in /proc/[0-9]*; do case "$(tr '\0' ' ' < $proc/cmdline 2>/dev/null)" in "/opt/venv/bin/python server.py --clients 4"*) kill ${proc#/proc/};; esac; done 2>/dev/null
uv run ray stop >/dev/null 2>&1
echo "=== orquestrador: fim $(date +%T)"
