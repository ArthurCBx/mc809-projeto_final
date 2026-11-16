"""Testes das comparações pareadas do artigo: t pareado, Wilcoxon pareado (bilateral, exato) e IC de 95% por bootstrap
da média das diferenças (percentil, 10.000 reamostragens dos pares). Só lê CSVs.
Uso (na raiz): python results/estatistica_e_divisoes/testes.py
Ganho = média das rodadas 26–30 (adaptado) ou 16–20 (replicado) − private_init da mesma execução, média dos clientes.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

R = 'results'
ALPHAS = ['iid', '1.0', '0.1']
rng = np.random.default_rng(0)
linhas = []


def load(path):
    t = pd.read_csv(f'{R}/{path}')
    t['alpha'] = t['alpha'].astype(str)
    return t


def fim(d, r):
    return d[(d.stage == 'federated') & d['round'].between(*r)]


def ganho(base, frac, r=(26, 30)):
    t = load(f'{base}_test_metrics.csv')
    t = t[np.isclose(t.public_fraction, frac)]
    return {a: fim(t[t.alpha == a], r).test_accuracy.mean() - t[(t.alpha == a) & (t.stage == 'private_init')].test_accuracy.mean()
            for a in ALPHAS}


def pares(bases_a, bases_b, frac):
    d = []
    for ba, bb in zip(bases_a, bases_b):
        ga = ganho(ba, frac)
        gb = ganho(bb, frac) if bb else dict.fromkeys(ALPHAS, 0.0)
        d += [ga[a] - gb[a] for a in ALPHAS]
    return d


def teste(rotulo, d, onde):
    d = np.asarray(d, dtype=float)
    boot = rng.choice(d, (10000, len(d))).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    w = stats.wilcoxon(d, method='exact')
    linhas.append({'comparação': rotulo, 'onde no artigo': onde, 'n': len(d), 'média': d.mean(),
                   'IC 95% (bootstrap)': f'[{lo:+.4f}; {hi:+.4f}]', 'a favor': f'{(d > 0).sum()}/{len(d)}',
                   'p (t pareado)': stats.ttest_1samp(d, 0).pvalue, 'p (Wilcoxon)': w.pvalue,
                   'IC exclui 0': bool(lo > 0 or hi < 0)})


KL = {0.01: [f'ensemble_e_leave_one_out/all_P01_seed{s}' for s in (42, 1, 2)],
      0.05: [f'ensemble_e_leave_one_out/all_P05_seed{s}' for s in (42, 1, 2)],
      **{f: [f'fracoes_publicas_10_40_80/adaptado_P{int(f * 100)}_seed{s}' for s in (42, 1, 2)] for f in (0.1, 0.4, 0.8)}}
SEM = [None] * 3

# Tabela 4 / Figura 6: ganho do FedMD adaptado sobre o treinamento local (9 pares por fração).
for f in KL:
    teste(f'FedMD adaptado − local, f = {f:.0%}', pares(KL[f], SEM, f), 'Tabelas 4 e 5, Seção 6.2')

# Seção 6.2: KL − MAE com 10/40/80% (seeds 42 e 1).
MAE = {f: [f'digest_kl_10_40_80/MAE_P{int(f * 100)}_seed{s}' for s in (42, 1)] for f in (0.1, 0.4, 0.8)}
teste('KL − MAE, ganho, 10/40/80%', sum((pares(KL[f][:2], MAE[f], f) for f in MAE), []), 'Seção 6.2 (p ≈ 0,27)')

# Tabela 6 / Seção 6.3: KL (w = 0) contra os pesos com rótulos.
for f, s in ((0.01, 'P01'), (0.05, 'P05')):
    ce = [f'digest_ce_1_5/CE1.0_{s}_seed{x}' for x in (42, 1, 2)]
    teste(f'w = 0 − w = 1, ganho, f = {f:.0%}', pares(KL[f], ce, f), 'Tabela 6, Seção 6.3')
for w in ('1.0', '0.5'):
    d = sum((pares([f'digest_ce_10_40_80/CE{w}_P{int(f * 100)}_seed{x}' for x in (42, 1)], KL[f][:2], f)
             for f in (0.1, 0.4, 0.8)), [])
    teste(f'w = {w} − w = 0, ganho, 10/40/80%', d, 'Tabela 6, Seção 6.3')

# Tabela 9 / Seção 6.6: cenário 2, PH2 bal. média dos clientes (6 pares: 2 seeds × 3 α).
def ph2(prefixo, etapa, seed, alpha):
    x = load(f'ph2_privado/{prefixo}_seed{seed}_alpha{alpha}_extra_test_metrics.csv')
    x = x[x.model != 'ensemble']
    x = fim(x, (16, 20)) if etapa == 'final' else x[x.stage == etapa]
    return x.balanced_accuracy.mean()


# Usa todas as seeds em que as variantes já terminaram (42 e 1; 2 e 3 quando a fila do item 5 acabar).
SEEDS_C2 = [s for s in (42, 1, 2, 3) if all(os.path.exists(f'{R}/ph2_privado/{p}_seed{s}_alpha{a}_extra_test_metrics.csv')
                                            for p in ('original', 'semdigest_original', 'controle_original') for a in ALPHAS)]


def c2(a, b):
    return [ph2(*a, s, al) - ph2(*b, s, al) for s in SEEDS_C2 for al in ALPHAS]


FED, LOC, SD, CE = ('original', 'final'), ('original', 'private_init'), ('semdigest_original', 'final'), ('controle_original', 'final')
teste('Cenário 2: FedMD − sem digest (PH2 bal.)', c2(FED, SD), 'Seção 6.6 (p ≈ 0,02)')
teste('Cenário 2: FedMD − local (PH2 bal.)', c2(FED, LOC), 'Seção 6.6')
teste('Cenário 2: FedMD − digest só CE (PH2 bal.)', c2(FED, CE), 'Tabela 9')

t = pd.DataFrame(linhas)
pd.set_option('display.width', 250)
fmt = t.copy()
for c in ('média',):
    fmt[c] = fmt[c].map('{:+.4f}'.format)
for c in ('p (t pareado)', 'p (Wilcoxon)'):
    fmt[c] = fmt[c].map('{:.3f}'.format)
print(fmt.to_string(index=False))
t.to_csv('results/estatistica_e_divisoes/testes.csv', index=False)
