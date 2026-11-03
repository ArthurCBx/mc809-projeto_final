"""0,5% de dados públicos comparado com 1% e 5% (mesma configuração: KL T=4, consenso padrão; results/ensemble_loo/all_*)."""
import os
import pandas as pd

SRC = {0.005: 'results/meio_pct/P005_seed{s}', 0.01: 'results/ensemble_loo/all_P01_seed{s}', 0.05: 'results/ensemble_loo/all_P05_seed{s}'}
rows = []
for frac, pat in SRC.items():
    for s in (42, 1, 2):
        base = pat.format(s=s)
        if not os.path.exists(f'{base}_test_metrics.csv'):
            continue
        t = pd.read_csv(f'{base}_test_metrics.csv'); t['alpha'] = t['alpha'].astype(str)
        e = pd.read_csv(f'{base}_ensemble_metrics.csv'); e['alpha'] = e['alpha'].astype(str)
        for a in ('iid', '1.0', '0.1'):
            c, ce = t[t.alpha == a], e[e.alpha == a]
            rows.append({'fração': frac, 'alpha': a, 'seed': s,
                         'public_init': c[c.stage == 'public_init'].test_accuracy.mean(),
                         'clientes_init': c[c.stage == 'private_init'].test_accuracy.mean(),
                         'clientes_fedmd': c[(c.stage == 'federated') & (c['round'] > 25)].test_accuracy.mean(),
                         'ensemble_init': ce[ce.stage == 'private_init'].ensemble_accuracy.mean(),
                         'ensemble_fedmd': ce[(ce.stage == 'federated') & (ce['round'] > 25)].ensemble_accuracy.mean()})
r = pd.DataFrame(rows)
r['ganho_clientes'] = r.clientes_fedmd - r.clientes_init
r['ganho_ensemble'] = r.ensemble_fedmd - r.ensemble_init
r['alpha'] = pd.Categorical(r['alpha'], ['iid', '1.0', '0.1'])
r.to_csv('results/meio_pct/resumo_por_seed.csv', index=False)
print('seeds:', r.groupby(['fração']).seed.nunique().to_dict(), '\n')
g = r.groupby(['fração', 'alpha'], observed=True)
cols = ['public_init', 'clientes_init', 'clientes_fedmd', 'ganho_clientes', 'ensemble_init', 'ensemble_fedmd', 'ganho_ensemble']
m, sd = g[cols].mean(), g[cols].std()
out = pd.DataFrame({c: [(f'{a:+.3f} ± {b:.3f}' if c.startswith('ganho') else f'{a:.3f} ± {b:.3f}') for a, b in zip(m[c], sd[c])] for c in cols},
                   index=[f'{f*100:g}% / {a}' for f, a in m.index])
print(out.to_markdown(), '\n')
print('Ganho médio por fração:')
print(r.groupby('fração')[['ganho_clientes', 'ganho_ensemble']].agg(['mean', 'std']).round(4).to_markdown())
print('\nseeds × cenários com ganho > 0 (clientes):', r.groupby('fração').ganho_clientes.apply(lambda x: f'{int((x > 0).sum())}/{len(x)}').to_dict())
