"""Junta as acurácias finais dos 4 métodos no teste original e no PH2 (results/test_models/acuracia_final.csv).

- FedMD adaptado: média das rodadas 26-30 da melhor configuração.
- Local (public + private init): os mesmos clientes logo após o private_init, sem colaboração.
- FedMD original: média das rodadas 16-20 da replicação (20 rodadas).
- Centralizado: cada arquitetura treinada com todo o conjunto de treino.
Para os clientes e o ensemble (média dos logits dos 4 modelos).
"""
import os
import pandas as pd

D = 'results/test_models'
rows = []

def from_run(prefix, method, stage, rounds=None):
    for seed in (42, 1, 2):
        base = f'{D}/{prefix}_seed{seed}'
        if not os.path.exists(f'{base}_extra_test_metrics.csv'):
            continue
        t = pd.read_csv(f'{base}_test_metrics.csv'); e = pd.read_csv(f'{base}_ensemble_metrics.csv'); x = pd.read_csv(f'{base}_extra_test_metrics.csv')
        for df in (t, e, x): df['alpha'] = df['alpha'].astype(str)
        sel = lambda df: df[(df.stage == stage) & (df['round'].between(*rounds) if rounds else True)]
        t, e, x = sel(t), sel(e), sel(x)
        for alpha in ('iid', '1.0', '0.1'):
            for cid in range(4):
                xs = x[(x.alpha == alpha) & (x.model == f'cliente {cid}')]
                rows.append({'método': method, 'alpha': alpha, 'seed': seed, 'modelo': f'cliente {cid}',
                             'test_accuracy': t[(t.alpha == alpha) & (t.client_id == cid)].test_accuracy.mean(),
                             'ph2_accuracy': xs.accuracy.mean(), 'ph2_balanced_accuracy': xs.balanced_accuracy.mean()})
            xs = x[(x.alpha == alpha) & (x.model == 'ensemble')]
            rows.append({'método': method, 'alpha': alpha, 'seed': seed, 'modelo': 'ensemble',
                         'test_accuracy': e[e.alpha == alpha].ensemble_accuracy.mean(),
                         'ph2_accuracy': xs.accuracy.mean(), 'ph2_balanced_accuracy': xs.balanced_accuracy.mean()})

from_run('adaptado', 'FedMD adaptado', 'federated', (26, 30))
from_run('adaptado', 'Local (public + private init)', 'private_init')
from_run('original', 'FedMD original', 'federated', (16, 20))
if os.path.exists(f'{D}/centralizado.csv'):
    c = pd.read_csv(f'{D}/centralizado.csv')
    for _, r in c.iterrows():
        rows.append({'método': 'Centralizado', 'alpha': '—', 'seed': r.seed,
                     'modelo': r.model.replace('arquitetura', 'cliente') if r.model != 'ensemble' else 'ensemble',
                     'test_accuracy': r.test_accuracy, 'ph2_accuracy': r.ph2_accuracy, 'ph2_balanced_accuracy': r.ph2_balanced_accuracy})
r = pd.DataFrame(rows)
r.to_csv(f'{D}/acuracia_final.csv', index=False)

r['tipo'] = r.modelo.where(r.modelo == 'ensemble', 'média dos clientes')
per_seed = r.groupby(['método', 'alpha', 'tipo', 'seed'])[['test_accuracy', 'ph2_accuracy', 'ph2_balanced_accuracy']].mean()
agg = per_seed.groupby(level=[0, 1, 2]).agg(['mean', 'std'])
out = pd.DataFrame({c: [f'{m:.3f} ± {s:.3f}' for m, s in zip(agg[(c, 'mean')], agg[(c, 'std')])]
                    for c in ('test_accuracy', 'ph2_accuracy', 'ph2_balanced_accuracy')}, index=agg.index)
out.to_csv(f'{D}/resumo.csv')
print(out.to_markdown())
