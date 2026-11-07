import sys, pandas as pd
k = ['public_fraction', 'alpha', 'client_id']
rows = []
for tag in sys.argv[1:]:
    t = pd.read_csv(f'results/06_private_init_10_passos_lr_2e-5/cen1_{tag}_test_metrics.csv'); t['alpha'] = t['alpha'].astype(str)
    pub = t[t.stage == 'public_init'].set_index(k).test_accuracy
    pri = t[t.stage == 'private_init'].set_index(k)
    df = pd.concat([pri[['private_malignant_fraction']], (pri.test_accuracy - pub).rename('queda')], axis=1).reset_index()
    df['teste'] = tag
    rows.append(df)
a = pd.concat(rows)
a['uma_classe'] = (a.private_malignant_fraction <= 0.02) | (a.private_malignant_fraction >= 0.98)
s = a.groupby('teste', sort=False).agg(quebram=('queda', lambda q: int((q < -0.05).sum())), queda_media=('queda', 'mean'), pior=('queda', 'min'))
s['queda_media_uma_classe'] = a[a.uma_classe].groupby('teste', sort=False).queda.mean()
s['queda_media_duas_classes'] = a[~a.uma_classe].groupby('teste', sort=False).queda.mean()
print(s.round(3).to_string())
print()
print(a[a.queda < -0.05].pivot_table(index=k, columns='teste', values='queda').round(2).to_string())
