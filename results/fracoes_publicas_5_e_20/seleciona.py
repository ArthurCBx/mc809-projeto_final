"""Fase 1: escolhe a combinação de lr do public_init e (lr, passos) do private_init.

Elimina combinações em que algum cliente perde > 0,05 no private_init em relação ao public_init e escolhe,
entre as restantes, a de maior acurácia média após o private_init. Se a segunda ficar a < 0,005, grava escolha2.txt. Imprime a tabela e grava a escolha em
escolha.txt no formato 'PUB_LR PRIV_LR PRIV_STEPS'.
"""
import glob, os, re
import pandas as pd

D = os.path.dirname(os.path.abspath(__file__))
k = ['public_fraction', 'alpha', 'client_id']
rows = []
for f in sorted(glob.glob(f'{D}/F1_*_test_metrics.csv')):
    pub_lr, priv_lr, steps = re.search(r'F1_pub(\S+?)_priv(\S+?)_s(\d+)_test', os.path.basename(f)).groups()
    t = pd.read_csv(f)
    pub = t[t.stage == 'public_init'].set_index(k).test_accuracy
    pri = t[t.stage == 'private_init'].set_index(k).test_accuracy
    d = pri - pub
    rows.append({'pub_lr': pub_lr, 'priv_lr': priv_lr, 'passos': int(steps), 'public_init': pub.mean(),
                 'private_init': pri.mean(), 'variação média': d.mean(), 'pior': d.min(), 'quebram': int((d < -0.05).sum())})
r = pd.DataFrame(rows).sort_values(['quebram', 'private_init'], ascending=[True, False])
print(r.round(3).to_string(index=False))
ok = r[r.quebram == 0]
best = (ok if len(ok) else r).iloc[0]
open(f'{D}/escolha.txt', 'w').write(f"{best.pub_lr} {best.priv_lr} {best.passos}\n")
# Se a segunda colocada ficar a menos de 0,005 da primeira, ela também vai para a fase 2 (decisão pelo resultado final).
cands = ok if len(ok) > 1 else r
second = cands.iloc[1] if len(cands) > 1 else None
if second is not None and best.private_init - second.private_init < 0.005:
    open(f'{D}/escolha2.txt', 'w').write(f"{second.pub_lr} {second.priv_lr} {second.passos}\n")
    print(f"segunda colocada a {best.private_init - second.private_init:.3f}: public_init lr {second.pub_lr}, "
          f"private_init lr {second.priv_lr}, {second.passos} passos -> também roda a fase 2")
elif os.path.exists(f'{D}/escolha2.txt'):
    os.remove(f'{D}/escolha2.txt')
print(f"\nescolha: public_init lr {best.pub_lr}, private_init lr {best.priv_lr}, {best.passos} passos"
      + ('' if len(ok) else ' (nenhuma combinação sem quebra; escolhida a com menos quebras)'))
