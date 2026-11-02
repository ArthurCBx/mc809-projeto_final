"""Critério do private_init: nenhum cliente perde mais de 0,05 de acurácia de teste em relação ao próprio public_init."""
import sys
import pandas as pd

t = pd.read_csv(sys.argv[1])
t['alpha'] = t['alpha'].astype(str)
k = ['public_fraction', 'alpha', 'client_id']
pub = t[t.stage == 'public_init'].set_index(k).test_accuracy
pri = t[t.stage == 'private_init'].set_index(k).test_accuracy
d = (pri - pub)
print(f'{sys.argv[1]}: clientes que quebram={int((d < -0.05).sum())}/{len(d)}, variação média={d.mean():+.3f}, '
      f'pior={d.min():+.3f}, melhor={d.max():+.3f}')
sys.exit(0 if (d < -0.05).sum() == 0 else 1)
