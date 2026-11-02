"""Escolhe a temperatura pela maior média de ganho (r26-30 − private_init) nos 6 cenários da seed 42."""
import sys
import pandas as pd

best = None
for T in sys.argv[1:]:
    gains = []
    for frac in ('01', '05'):
        t = pd.read_csv(f'results/kl/T{T}_P{frac}_seed42_test_metrics.csv')
        t['alpha'] = t['alpha'].astype(str)
        base = t[t.stage == 'private_init'].groupby('alpha').test_accuracy.mean()
        fin = t[(t.stage == 'federated') & (t['round'] > 25)].groupby('alpha').test_accuracy.mean()
        gains.append(fin - base)
    g = pd.concat(gains)
    print(f'T={T}: ganho médio {g.mean():+.4f} | por cenário {[round(x, 3) for x in g]}')
    if best is None or g.mean() > best[0]:
        best = (g.mean(), T)
print(f'escolha: T={best[1]}')
open('results/kl/escolha_T.txt', 'w').write(best[1] + '\n')
