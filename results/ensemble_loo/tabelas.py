"""Tabelas da etapa final: ensemble dos clientes (consenso padrão) e consenso leave-one-out. 1% e 5%, seeds 42, 1, 2."""
import os
import pandas as pd

D = 'results/ensemble_loo'
rows = []
for modo in ('all', 'loo'):
    for frac, tag in ((0.01, 'P01'), (0.05, 'P05')):
        for seed in (42, 1, 2):
            base = f'{D}/{modo}_{tag}_seed{seed}'
            if not os.path.exists(f'{base}_test_metrics.csv'):
                continue
            t = pd.read_csv(f'{base}_test_metrics.csv'); t['alpha'] = t['alpha'].astype(str)
            e = pd.read_csv(f'{base}_ensemble_metrics.csv'); e['alpha'] = e['alpha'].astype(str)
            for alpha in ('iid', '1.0', '0.1'):
                c, ce = t[t.alpha == alpha], e[e.alpha == alpha]
                fed, fede = c[(c.stage == 'federated') & (c['round'] > 25)], ce[(ce.stage == 'federated') & (ce['round'] > 25)]
                rows.append({'consenso': modo, 'fração': frac, 'alpha': alpha, 'seed': seed,
                             'clientes_init': c[c.stage == 'private_init'].test_accuracy.mean(),
                             'clientes_fedmd': fed.test_accuracy.mean(),
                             'ensemble_init': ce[ce.stage == 'private_init'].ensemble_accuracy.mean(),
                             'ensemble_fedmd': fede.ensemble_accuracy.mean()})
r = pd.DataFrame(rows)
r['alpha'] = pd.Categorical(r['alpha'], ['iid', '1.0', '0.1'])
r.to_csv(f'{D}/resumo_por_seed.csv', index=False)
n = r.groupby(['consenso', 'fração', 'alpha'], observed=True).seed.count()
print('seeds disponíveis por cenário:', n.to_dict(), '\n')

def tab(df, cols):
    g = df.groupby(['fração', 'alpha'], observed=True)[cols]
    m, s = g.mean(), g.std()
    out = m.copy().astype(object)
    for c in cols: out[c] = [f'{a:.3f} ± {b:.3f}' for a, b in zip(m[c], s[c])]
    out.index = [f'{f:g} / {a}' for f, a in out.index]
    return out

a = r[r.consenso == 'all'].copy()
a['ganho_clientes'] = a.clientes_fedmd - a.clientes_init
a['ensemble_vs_clientes_init'] = a.ensemble_init - a.clientes_init
a['ganho_ensemble'] = a.ensemble_fedmd - a.ensemble_init
print('## A. Consenso padrão: clientes e ensemble (média ± dp)\n')
print(tab(a, ['clientes_init', 'clientes_fedmd', 'ensemble_init', 'ensemble_fedmd']).to_markdown(), '\n')
print(tab(a, ['ganho_clientes', 'ensemble_vs_clientes_init', 'ganho_ensemble']).to_markdown(), '\n')
print('médias:', a[['ganho_clientes', 'ensemble_vs_clientes_init', 'ganho_ensemble']].mean().round(4).to_dict(), '\n')

p = r.pivot_table(index=['fração', 'alpha', 'seed'], columns='consenso', values=['clientes_fedmd', 'ensemble_fedmd'], observed=True).dropna()
d = pd.DataFrame({'clientes (loo − all)': p['clientes_fedmd']['loo'] - p['clientes_fedmd']['all'],
                  'ensemble (loo − all)': p['ensemble_fedmd']['loo'] - p['ensemble_fedmd']['all']})
print('## B. Leave-one-out − padrão (média ± dp por cenário)\n')
g = d.groupby(level=[0, 1], observed=True)
out = pd.DataFrame({c: [f'{m:+.3f} ± {s:.3f}' for m, s in zip(g[c].mean(), g[c].std())] for c in d.columns},
                   index=[f'{f:g} / {a}' for f, a in g.groups.keys()])
print(out.to_markdown(), '\n')
for c in d.columns:
    print(f'{c}: média {d[c].mean():+.4f}, dp {d[c].std():.4f}, loo melhor em {int((d[c] > 0).sum())}/{len(d)}')
