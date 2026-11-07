"""Tabelas do relatório final: frações públicas 0,01, 0,05 e 0,2; seeds 42, 1 e 2.

Linha de base: acurácia após public_init + private_init. Ganho do FedMD: média das rodadas 26-30 menos a linha de base.
5% e 20% com seeds 42 e 1 vêm do experimento anterior (mesma configuração e mesmo código).
"""
import pandas as pd

K = ['public_fraction', 'alpha']
FONTES = {
    0.01: {s: f'results/experimento_final/P01_seed{s}_test_metrics.csv' for s in (42, 1, 2)},
    0.05: {42: 'results/fracoes_publicas_5_e_20/F2b_seed42_test_metrics.csv', 1: 'results/fracoes_publicas_5_e_20/F2b_seed1_test_metrics.csv',
           2: 'results/experimento_final/P05_seed2_test_metrics.csv'},
    0.2: {42: 'results/fracoes_publicas_5_e_20/F2_seed42_test_metrics.csv', 1: 'results/fracoes_publicas_5_e_20/F2_seed1_test_metrics.csv',
          2: 'results/experimento_final/P20_seed2_test_metrics.csv'},
}
ALPHAS = ['iid', '1.0', '0.1']

rows = []
for frac, por_seed in FONTES.items():
    for seed, path in por_seed.items():
        t = pd.read_csv(path)
        t['alpha'] = t['alpha'].astype(str)
        t = t[t.public_fraction == frac]
        for alpha in ALPHAS:
            c = t[t.alpha == alpha]
            fed = c[c.stage == 'federated']
            rows.append({'fração': frac, 'alpha': alpha, 'seed': seed,
                         'public_init': c[c.stage == 'public_init'].test_accuracy.mean(),
                         'baseline': c[c.stage == 'private_init'].test_accuracy.mean(),
                         'final': fed[fed['round'] > 25].test_accuracy.mean(),
                         'pior cliente': fed.test_accuracy.min()})
r = pd.DataFrame(rows)
r['ganho'] = r['final'] - r['baseline']
r['alpha'] = pd.Categorical(r['alpha'], ALPHAS)

g = r.groupby(['fração', 'alpha'], observed=True)
fmt = lambda m, s: f'{m:.3f} ± {s:.3f}'
tab = pd.DataFrame({
    'Baseline (public+private init)': [fmt(m, s) for m, s in zip(g.baseline.mean(), g.baseline.std())],
    'FedMD (r26-30)': [fmt(m, s) for m, s in zip(g.final.mean(), g.final.std())],
    'Ganho': [f'{m:+.3f} ± {s:.3f}' for m, s in zip(g.ganho.mean(), g.ganho.std())],
    'Seeds com ganho > 0': g.ganho.apply(lambda x: f'{int((x > 0).sum())}/{len(x)}').values,
    'Pior cliente': g['pior cliente'].min().round(3).values,
}, index=[f'{f:g} / {a}' for f, a in g.groups.keys()])
print('## Resultado final (média ± desvio padrão, 3 seeds)\n')
print(tab.to_markdown())
print('\n## Ganho por seed\n')
p = r.pivot_table(index=['fração', 'alpha'], columns='seed', values='ganho', observed=True)
p.index = [f'{f:g} / {a}' for f, a in p.index]
print(p.to_markdown(floatfmt='+.3f'))
print('\n## Ganho médio por fração\n')
print(r.groupby('fração').ganho.agg(['mean', 'std']).to_markdown(floatfmt='+.3f'))
r.to_csv('results/experimento_final/resumo_por_seed.csv', index=False)
