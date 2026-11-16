"""Distribuição dos dados privados por cliente, refeita com as mesmas funções e sementes do client.py (não treina nada).

Cenário 1: split_dataset (fração pública f estratificada; o resto dividido IID ou por Dirichlet(α)), f ∈ {1, 5, 10, 40,
80}%, seeds 42, 1 e 2. Cenário 2: split_external(PH2, 0.4, seed) + split_clients, seeds 42 e 1. As contagens incluem os
20% que o cenário 1 separa para a parada antecipada (como a coluna n_private dos CSVs de resultado).
Confere com as colunas n_private e private_malignant_fraction dos test_metrics.csv das execuções.
Uso (na raiz): PYTHONPATH=. python results/estatistica_e_divisoes/divisoes.py
"""
import glob

import numpy as np
import pandas as pd

from dataset_setup import list_dir, list_files, split_clients, split_dataset, split_external

D = 'results/estatistica_e_divisoes'
ALPHAS = {'iid': None, '1.0': 1.0, '0.1': 0.1}
rows = []

for f in (0.01, 0.05, 0.1, 0.4, 0.8):
    for seed in (42, 1, 2):
        for nome, alpha in ALPHAS.items():
            client_idx, _, _, labels = split_dataset(True, 4, alpha=alpha, public_fraction=f, seed=seed)
            for cid, idx in enumerate(client_idx):
                y = labels[np.asarray(idx, dtype=int)]
                rows.append({'cenario': 1, 'f': f, 'alpha': nome, 'seed': seed, 'cliente': cid,
                             'n': len(y), 'melanomas': int(y.sum())})

ph2_paths, ph2_labels = list_dir('ph2_dataset')
train_paths, train_labels = list_files(train_data=True)
for seed in (42, 1):
    priv, _ = split_external(ph2_labels, 0.4, seed)
    labels = np.concatenate([train_labels, ph2_labels[priv]])
    private_idx = np.arange(len(train_labels), len(labels))
    for nome, alpha in ALPHAS.items():
        client_idx = split_clients(private_idx, labels, 4, alpha, np.random.default_rng(seed), 10)
        for cid, idx in enumerate(client_idx):
            y = labels[np.asarray(idx, dtype=int)]
            rows.append({'cenario': 2, 'f': 1.0, 'alpha': nome, 'seed': seed, 'cliente': cid,
                         'n': len(y), 'melanomas': int(y.sum())})

r = pd.DataFrame(rows)
r['frac_melanoma'] = r.melanomas / r.n
r['frac_classe_majoritaria'] = np.maximum(r.frac_melanoma, 1 - r.frac_melanoma)
r.to_csv(f'{D}/divisoes_por_cliente.csv', index=False)

# Conferência com os CSVs das execuções (n_private e private_malignant_fraction no private_init).
fontes = {1: ['ensemble_e_leave_one_out/all_P0[15]_seed*', 'fracoes_publicas_10_40_80/adaptado_P*_seed*'],
          2: ['ph2_privado/original_seed*_alpha*']}
erros, conferidos = 0, 0
for cen, pads in fontes.items():
    for pad in pads:
        for arq in glob.glob(f'results/{pad}_test_metrics.csv'):
            t = pd.read_csv(arq)
            if 'travado' in arq or 'interrompido' in arq or 'client_id' not in t:
                continue
            t = t.query("stage == 'private_init'")
            for _, l in t.iterrows():
                m = r[(r.cenario == cen) & np.isclose(r.f, l.public_fraction) & (r.alpha == str(l.alpha))
                      & (r.seed == l.seed) & (r.cliente == l.client_id)]
                conferidos += 1
                if len(m) != 1 or m.n.iloc[0] != l.n_private or not np.isclose(m.frac_melanoma.iloc[0], l.private_malignant_fraction):
                    erros += 1
print(f'conferência com os CSVs das execuções: {conferidos} clientes, {erros} divergências')

# Resumo do cenário 1 por fração e divisão (3 seeds × 4 clientes = 12 clientes).
c1 = r[r.cenario == 1]
res = c1.groupby(['f', 'alpha']).agg(
    n_min=('n', 'min'), n_max=('n', 'max'),
    melanoma_min=('frac_melanoma', 'min'), melanoma_max=('frac_melanoma', 'max'),
    acima_94=('frac_classe_majoritaria', lambda s: int((s > 0.94).sum())),
    uma_classe=('frac_classe_majoritaria', lambda s: int((s == 1).sum())))
print(res.reindex(list(ALPHAS), level='alpha').round(3).to_string())
print('\ncenário 1, f = 40%, seed 42 (desenvolvimento incremental):')
print(c1[(c1.f == 0.4) & (c1.seed == 42)].pivot_table(index='alpha', columns='cliente', values='frac_classe_majoritaria').round(3))
print('\ncenário 2 (imagens / melanomas por cliente):')
c2 = r[r.cenario == 2].copy()
c2['cel'] = c2.n.astype(str) + ' / ' + c2.melanomas.astype(str)
print(c2.pivot_table(index=['seed', 'alpha'], columns='cliente', values='cel', aggfunc='first').to_string())
