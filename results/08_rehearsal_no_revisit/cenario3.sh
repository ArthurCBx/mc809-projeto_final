#!/bin/bash
# Cenário 3: rehearsal. private_init 10 passos lr 2e-5 (cenário 1); revisit 3x10 passos lr 1e-3 com 50% de
# amostras públicas rotuladas em cada batch; 30 rodadas (cenário 2: sem ancoragem nos rótulos, as curvas caem).
cd "/mc809 - Comp Distribuida/projeto_final"
# Espera o cenário 2 terminar (ele move os CSVs ao final): PID 38484 do cenario2.sh.
while [ -d /proc/38484 ]; do sleep 30; done
echo "=== cenario3: início $(date +%T)"
uv run python client.py --digest-rounds 30 --private-init-steps 10 --private-init-lr 2e-5 \
  --private-epochs 3 --private-steps 10 --revisit-lr 1e-3 --rehearsal-fraction 0.5 > results/08_rehearsal_no_revisit/cen3.log 2>&1
echo "=== cenario3: fim $(date +%T), código $?"
mv results/test_metrics.csv results/08_rehearsal_no_revisit/cen3_test_metrics.csv
mv results/train_history.csv results/08_rehearsal_no_revisit/cen3_train_history.csv
