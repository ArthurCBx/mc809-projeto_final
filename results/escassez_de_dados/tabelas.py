"""Escassez de dados: validação de 20% (referência), 50% e 70% em 1%, 5% e 20% de dados públicos, 3 seeds.

Referência com validação de 20%: 1% e 5% em results/ensemble_e_leave_one_out/all_* (mesma configuração, KL T=4); 20% no
experimento final (MAE no digest; a diferença entre KL e MAE foi de +0,002 em média). Sem ensemble para 20%/20%.
"""
import os
import pandas as pd

def src(val, frac, seed):
    if val == 0.2:
        if frac == 0.2:
            return {42: 'results/fracoes_publicas_5_e_20/F2_seed42', 1: 'results/fracoes_publicas_5_e_20/F2_seed1', 2: 'results/experimento_final/P20_seed2'}[seed]
        return f"results/ensemble_e_leave_one_out/all_P{'01' if frac == 0.01 else '05'}_seed{seed}"
    return f"results/escassez_de_dados/V{val}_P{ {0.01: '01', 0.05: '05', 0.2: '20'}[frac] }_seed{seed}"

rows = []
for val in (0.2, 0.5, 0.7):
    for frac in (0.01, 0.05, 0.2):
        for seed in (42, 1, 2):
            base = src(val, frac, seed)
            if not os.path.exists(f'{base}_test_metrics.csv'):
                continue
            t = pd.read_csv(f'{base}_test_metrics.csv'); t['alpha'] = t['alpha'].astype(str); t = t[t.public_fraction == frac]
            e = None
            if os.path.exists(f'{base}_ensemble_metrics.csv'):
                e = pd.read_csv(f'{base}_ensemble_metrics.csv'); e['alpha'] = e['alpha'].astype(str)
            for a in ('iid', '1.0', '0.1'):
                c = t[t.alpha == a]
                row = {'val': val, 'fração': frac, 'alpha': a, 'seed': seed,
                       'public_init': c[c.stage == 'public_init'].test_accuracy.mean(),
                       'baseline': c[c.stage == 'private_init'].test_accuracy.mean(),
                       'fedmd': c[(c.stage == 'federated') & (c['round'] > 25)].test_accuracy.mean()}
                if e is not None:
                    ce = e[e.alpha == a]
                    row['ensemble_init'] = ce[ce.stage == 'private_init'].ensemble_accuracy.mean()
                    row['ensemble_fedmd'] = ce[(ce.stage == 'federated') & (ce['round'] > 25)].ensemble_accuracy.mean()
                rows.append(row)
r = pd.DataFrame(rows)
r['ganho'] = r.fedmd - r.baseline
r['alpha'] = pd.Categorical(r['alpha'], ['iid', '1.0', '0.1'])
r.to_csv('results/escassez_de_dados/resumo_por_seed.csv', index=False)
print('seeds por (val, fração):', r.groupby(['val', 'fração']).seed.nunique().to_dict(), '\n')

g = r.groupby(['fração', 'alpha', 'val'], observed=True)
m, s = g[['baseline', 'fedmd', 'ganho']].mean(), g[['baseline', 'fedmd', 'ganho']].std()
out = pd.DataFrame({'linha de base': [f'{a:.3f}' for a in m.baseline], 'FedMD': [f'{a:.3f}' for a in m.fedmd],
                    'ganho': [f'{a:+.3f} ± {b:.3f}' for a, b in zip(m.ganho, s.ganho)],
                    'seeds com ganho': g.ganho.apply(lambda x: f'{int((x > 0).sum())}/{len(x)}').values},
                   index=[f'{f*100:g}% / {a} / val {v:g}' for f, a, v in m.index])
print('## Por cenário\n'); print(out.to_markdown(), '\n')
print('## Ganho médio por fração pública e validação\n')
p = r.groupby(['fração', 'val']).agg(ganho=('ganho', 'mean'), dp=('ganho', 'std'), linha_base=('baseline', 'mean'), fedmd=('fedmd', 'mean'))
print(p.round(4).to_markdown(), '\n')
if 'ensemble_init' in r:
    q = r.dropna(subset=['ensemble_init']).groupby(['fração', 'val'])[['baseline', 'ensemble_init', 'fedmd', 'ensemble_fedmd']].mean()
    print('## Ensemble (média dos cenários)\n'); print(q.round(3).to_markdown())
