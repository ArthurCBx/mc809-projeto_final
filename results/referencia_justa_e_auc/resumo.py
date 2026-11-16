"""Resume as métricas do PH2 do cenário 1 (IID): treinamento local (private_init da execução do adaptado), FedMD
adaptado (média das rodadas 26–30) e replicado (16–20), média dos clientes e ensemble, 3 seeds."""
import glob
import pandas as pd

COLS = ['balanced_accuracy', 'sensitivity', 'specificity', 'auc']
rows = []
for f in sorted(glob.glob('results/referencia_justa_e_auc/*_P*_seed*_extra_test_metrics.csv')):
    nome = f.split('/')[-1].replace('_extra_test_metrics.csv', '')
    metodo, fr, seed = nome.split('_')
    e = pd.read_csv(f)
    ultimas = range(26, 31) if metodo == 'adaptado' else range(16, 21)
    sel = {'FedMD ' + metodo: e[(e.stage == 'federated') & e['round'].isin(ultimas)]}
    if metodo == 'adaptado':
        sel['Local'] = e[e.stage == 'private_init']
    for m, d in sel.items():
        for tipo, dd in (('clientes', d[d.model != 'ensemble']), ('ensemble', d[d.model == 'ensemble'])):
            rows.append({'f': fr, 'metodo': m, 'tipo': tipo, 'seed': seed, **dd[COLS].mean()})
r = pd.DataFrame(rows)
g = r.groupby(['f', 'tipo', 'metodo'])
print(pd.concat([g[COLS].mean().round(3), g.seed.count().rename('n')], axis=1).to_string())
