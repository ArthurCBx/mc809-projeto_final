#!/bin/bash
# Refaz original_seed2_alpha0.1, que caiu na rodada 14 (DEADLINE_EXCEEDED: um cliente não enviou os logits a tempo;
# parciais guardados como original_seed2_alpha0.1_falhou1_*). Espera a fila principal terminar.
cd "/mc809 - Comp Distribuida/projeto_final"
until grep -q "orquestrador: fim" results/ph2_privado/orquestrador_sementes_2_3.log; do sleep 60; done
eval "$(sed -n '/^D=results\/ph2_privado/,/^primeira=1/p;/^mata() {/,/^}$/p;/^reinicia_servidor() {/,/^}$/p;/^roda() {/,/^}$/p' results/ph2_privado/orquestrador_sementes_2_3.sh)"
primeira=0
roda original_seed2_alpha0.1 uniform $ORIGINAL --alphas 0.1 --seed 2
mata "/opt/venv/bin/python server.py --clients 4"
uv run ray stop >/dev/null 2>&1
echo "=== retomada: fim $(date +%T)"
