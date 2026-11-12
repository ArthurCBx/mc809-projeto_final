#!/bin/bash
# MAE_P40_seed1 travou duas vezes no alpha 0.1 (rodada 8 do digest e depois no public_init), com um cliente preso e a
# GPU ociosa. IID e alpha 1.0 terminaram (MAE_P40_seed1_*.parcial.csv). Roda só o alpha 0.1, sem XLA (--no-jit),
# junta tudo em MAE_P40_seed1_*.csv e segue com o orquestrador (que pula as execuções já concluídas).
cd "/mc809 - Comp Distribuida/projeto_final"
D=results/digest_kl_10_40_80
MAE="--public-init-lr 3e-4 --digest-rounds 30 --private-init-steps 300 --private-init-lr 1e-4 \
  --private-init-rehearsal --private-epochs 3 --private-steps 10 --revisit-lr 1e-4 --rehearsal-fraction 0.5 \
  --digest-lr 4e-4 --digest-steps 30"
for proc in /proc/[0-9]*; do case "$(tr '\0' ' ' < $proc/cmdline 2>/dev/null)" in "/opt/venv/bin/python server.py --clients 4"*) kill ${proc#/proc/};; esac; done 2>/dev/null
uv run ray stop >/dev/null 2>&1; sleep 5
setsid nohup uv run python server.py --clients 4 --weighting accuracy >> $D/server.log 2>&1 < /dev/null &
sleep 10
echo "=== MAE_P40_seed1 (alpha 0.1, --no-jit): início $(date +%T)"
timeout 90m uv run python client.py --public-fractions 0.4 --alphas 0.1 --no-jit $MAE --seed 1 > $D/MAE_P40_seed1_alpha0.1.log 2>&1
rc=$?
echo "=== MAE_P40_seed1 (alpha 0.1, --no-jit): fim $(date +%T), código $rc"
if [ $rc -eq 0 ]; then
  for f in test_metrics train_history ensemble_metrics; do
    { cat $D/MAE_P40_seed1_$f.parcial.csv; tail -n +2 results/$f.csv; } > $D/MAE_P40_seed1_$f.csv && rm results/$f.csv $D/MAE_P40_seed1_$f.parcial.csv
  done
  exec bash $D/orquestrador.sh
fi
