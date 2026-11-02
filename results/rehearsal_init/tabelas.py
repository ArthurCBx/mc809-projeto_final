"""Tabelas do relatório de 01/11: private_init com rehearsal vs anterior, seeds 42 e 1."""
import pandas as pd

K = ['public_fraction', 'alpha']
ORDEM = [(0.1, 'iid'), (0.1, '1.0'), (0.1, '0.1'), (0.4, 'iid'), (0.4, '1.0'), (0.4, '0.1'), (0.8, 'iid'), (0.8, '1.0'), (0.8, '0.1')]
RUNS = {
    'Anterior s42': 'results/digest_steps/steps30_test_metrics.csv',
    'Anterior s1': 'results/rehearsal_init/C_anterior_seed1_test_metrics.csv',
    'Rehearsal init s42': 'results/rehearsal_init/B_init1e-4_seed42_test_metrics.csv',
    'Rehearsal init s1': 'results/rehearsal_init/C_init1e-4_seed1_test_metrics.csv',
}


def load(p):
    t = pd.read_csv(p)
    t['alpha'] = t['alpha'].astype(str)
    return t


def md(df):
    df = df.reindex(pd.MultiIndex.from_tuples(ORDEM, names=K)).copy()
    df.index = [f'{p} / {a}' for p, a in df.index]
    return df.to_markdown(floatfmt='+.3f' if (df.abs() < 0.2).all().all() else '.3f')


T = {k: load(v) for k, v in RUNS.items()}
fin = {k: t[(t.stage == 'federated') & (t['round'] > 25)].groupby(K).test_accuracy.mean() for k, t in T.items()}
pub = {k: t[t.stage == 'public_init'].groupby(K).test_accuracy.mean() for k, t in T.items()}
pri = {k: t[t.stage == 'private_init'].groupby(K).test_accuracy.mean() for k, t in T.items()}

print('## Média r26-30\n'); print(md(pd.DataFrame(fin)))
print('\n## Diferença r26-30 − public_init (mesma execução)\n'); d = pd.DataFrame({k: fin[k] - pub[k] for k in T}); print(md(d))
print('\n## Diferença private_init − public_init\n'); print(md(pd.DataFrame({k: pri[k] - pub[k] for k in T})))
print('\n## Resumo\n')
s = pd.DataFrame({'média r26-30': {k: fin[k].mean() for k in T}, 'vs public_init': {k: d[k].mean() for k in T},
                  'configs >= public_init': {k: int((d[k] >= 0).sum()) for k in T},
                  'pior cliente': {k: T[k][T[k].stage == 'federated'].test_accuracy.min() for k in T}})
print(s.to_markdown(floatfmt='.3f'))
print('\n## Média das duas seeds, vs public_init\n')
m = pd.DataFrame({'Anterior': (d['Anterior s42'] + d['Anterior s1']) / 2, 'Rehearsal init': (d['Rehearsal init s42'] + d['Rehearsal init s1']) / 2})
print(md(m)); print('\nmédias:', m.mean().round(3).to_dict())
