"""Escolhe o lr do public_init para 0,5% público: maior acurácia média após o private_init, sem cliente perdendo > 0,05."""
import sys
import pandas as pd

k = ['public_fraction', 'alpha', 'client_id']
best = None
for lr in sys.argv[1:]:
    t = pd.read_csv(f'results/fracao_publica_0_5/I1_pub{lr}_test_metrics.csv')
    pub = t[t.stage == 'public_init'].set_index(k).test_accuracy
    pri = t[t.stage == 'private_init'].set_index(k).test_accuracy
    d = pri - pub
    quebram = int((d < -0.05).sum())
    print(f'pub lr {lr}: public_init {pub.mean():.3f}, private_init {pri.mean():.3f}, variação {d.mean():+.3f}, pior {d.min():+.3f}, quebram {quebram}')
    key = (quebram, -pri.mean())
    if best is None or key < best[0]:
        best = (key, lr)
print(f'escolha: {best[1]}')
open('results/fracao_publica_0_5/escolha.txt', 'w').write(best[1] + '\n')
