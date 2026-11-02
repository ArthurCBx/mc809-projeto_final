"""Tabelas do relatório da madrugada: média de acurácia de teste dos 4 clientes por configuração."""
import pandas as pd

K = ['public_fraction', 'alpha']
ORDEM = [(0.1, 'iid'), (0.1, '1.0'), (0.1, '0.1'), (0.4, 'iid'), (0.4, '1.0'), (0.4, '0.1'), (0.8, 'iid'), (0.8, '1.0'), (0.8, '0.1')]


def load(path):
    t = pd.read_csv(path)
    t['alpha'] = t['alpha'].astype(str)
    return t


def fed(t, a, b):
    f = t[(t.stage == 'federated') & t['round'].between(a, b)]
    return f.groupby(K).test_accuracy.mean()


def stage(t, s):
    return t[t.stage == s].groupby(K).test_accuracy.mean()


def ordered(df):
    return df.reindex(pd.MultiIndex.from_tuples(ORDEM, names=K))


def md(df):
    df = df.copy()
    df.index = [f'{p} / {a}' for p, a in df.index]
    return df.to_markdown(floatfmt='.3f')


ref = load('results/test_metrics_3_private_epoch.csv')                      # uniforme, revisit completo lr 1e-3
w = load('results/test_metrics_3_private_epoch_weighted.csv')               # ponderado, revisit completo lr 1e-3
c2 = load('results/madrugada/cen2_test_metrics.csv')
c3 = load('results/madrugada/cen3_test_metrics.csv')

pr3 = c3[c3.stage == 'federated'].groupby(K + ['round']).test_accuracy.mean()
comp = pd.DataFrame({
    'Uniforme (r16-20)': fed(ref, 16, 20),
    'Ponderado (r16-20)': fed(w, 16, 20),
    'Cen. 2 (r1-2)': fed(c2, 1, 2),
    'Cen. 2 (r16-20)': fed(c2, 16, 20),
    'Cen. 3 private_init': stage(c3, 'private_init'),
    'Cen. 3 (r16-20)': fed(c3, 16, 20),
    'Cen. 3 (r26-30)': fed(c3, 26, 30),
    'Cen. 3 pico': pr3.groupby(level=[0, 1]).max(),
})
print('## Comparação geral\n')
print(md(ordered(comp)))

print('\n## Cenário 3 por cliente (média r26-30)\n')
f = c3[(c3.stage == 'federated') & (c3['round'] > 25)]
pc = f.pivot_table(index=K, columns='client_id', values='test_accuracy')
info = c3[c3.stage == 'private_init'].pivot_table(index=K, columns='client_id', values='private_malignant_fraction')
pc.columns = [f'cliente {c}' for c in pc.columns]
info.columns = [f'% malig. {c}' for c in info.columns]
print(md(ordered(pd.concat([pc, info], axis=1))))

print('\n## Cenário 3: curva da média por rodada\n')
curve = pr3.unstack('round')[[1, 5, 10, 15, 20, 25, 30]]
curve.columns = [f'r{c}' for c in curve.columns]
print(md(ordered(curve)))
