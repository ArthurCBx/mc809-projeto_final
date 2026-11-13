"""Números da Parte B de results/RESPOSTAS_ARTIGO.md, recalculados a partir dos CSVs em results/.

Uso (na raiz do projeto): CUDA_VISIBLE_DEVICES= PYTHONPATH=. python results/respostas_artigo.py
Só lê CSVs; não treina nada.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

R = 'results'
ALPHAS = ['iid', '1.0', '0.1']
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 30)


def load(path):
    t = pd.read_csv(f'{R}/{path}')
    t['alpha'] = t['alpha'].astype(str)
    if 'seed' not in t:
        t['seed'] = 42
    return t


def fim(d, r):
    return d[(d.stage == 'federated') & d['round'].between(*r)]


def testes(d, rotulo):
    """Teste t pareado (= t de uma amostra nas diferenças), Wilcoxon e sinal, todos bilaterais."""
    d = np.asarray(d, dtype=float)
    t = stats.ttest_1samp(d, 0)
    w = stats.wilcoxon(d)
    s = stats.binomtest(int((d > 0).sum()), len(d))
    print(f'{rotulo}: n = {len(d)}, média = {d.mean():+.4f}, dp = {d.std(ddof=1):.4f}, positivos = {(d > 0).sum()}/{len(d)} | '
          f't = {t.statistic:.3f}, p(t pareado) = {t.pvalue:.4f} | p(Wilcoxon) = {w.pvalue:.4f} | p(sinal) = {s.pvalue:.4f}')


# ---------------------------------------------------------------------------------------------------------------- B1
print('\n===== B1: dispersão entre clientes (FedMD adaptado, KL T = 4) =====')
FONTES_KL = {0.01: [f'ensemble_e_leave_one_out/all_P01_seed{s}' for s in (42, 1, 2)],
             0.05: [f'ensemble_e_leave_one_out/all_P05_seed{s}' for s in (42, 1, 2)],
             0.1: [f'fracoes_publicas_10_40_80/adaptado_P10_seed{s}' for s in (42, 1, 2)],
             0.4: [f'fracoes_publicas_10_40_80/adaptado_P40_seed{s}' for s in (42, 1, 2)],
             0.8: [f'fracoes_publicas_10_40_80/adaptado_P80_seed{s}' for s in (42, 1, 2)]}
linhas = []
for frac, bases in FONTES_KL.items():
    for base in bases:
        t = load(f'{base}_test_metrics.csv')
        for alpha in ALPHAS:
            a = t[t.alpha == alpha]
            loc = a[a.stage == 'private_init'].set_index('client_id').test_accuracy.sort_index()
            fed = fim(a, (26, 30)).groupby('client_id').test_accuracy.mean().sort_index()
            melhor, pior = loc.idxmax(), loc.idxmin()
            linhas.append({'fração': frac, 'alpha': alpha, 'seed': a.seed.iloc[0],
                           'local': loc.mean(), 'FedMD': fed.mean(),
                           'Δ melhor': fed[melhor] - loc[melhor], 'Δ pior': fed[pior] - loc[pior],
                           'dp local': loc.std(ddof=1), 'dp FedMD': fed.std(ddof=1),
                           'dp0 local': loc.std(ddof=0), 'dp0 FedMD': fed.std(ddof=0),
                           'amp local': loc.max() - loc.min(), 'amp FedMD': fed.max() - fed.min()})
b1 = pd.DataFrame(linhas)
print(b1.groupby('fração').mean(numeric_only=True).drop(columns='seed').round(3).to_string())
print('execuções por fração:', b1.groupby('fração').size().to_dict())

# ---------------------------------------------------------------------------------------------------------------- B2
print('\n===== B2: PH2 privado =====')
D8 = 'ph2_privado'
VARIANTES = [('public_init', 'original', 'public_init'), ('Treinamento local', 'original', 'private_init'),
             ('Sem digest', 'semdigest_original', 'final'), ('Digest só com CE', 'controle_original', 'final'),
             ('FedMD replicado', 'original', 'final')]
linhas = []
for seed in (42, 1):
    for alpha in ALPHAS:
        for nome, prefixo, etapa in VARIANTES:
            base = f'{D8}/{prefixo}_seed{seed}_alpha{alpha}'
            t, e, x = (load(f'{base}_{f}.csv') for f in ('test_metrics', 'ensemble_metrics', 'extra_test_metrics'))
            sel = (lambda d: fim(d, (16, 20))) if etapa == 'final' else (lambda d: d[d.stage == etapa])
            t, e, x = sel(t), sel(e), sel(x)
            linhas.append({'seed': seed, 'alpha': alpha, 'variante': nome,
                           'validação': t.test_accuracy.mean(), 'validação ens.': e.ensemble_accuracy.mean(),
                           'PH2 bal. clientes': x[x.model != 'ensemble'].balanced_accuracy.mean(),
                           'PH2 bal. ensemble': x[x.model == 'ensemble'].balanced_accuracy.mean(),
                           'PH2 acc clientes': x[x.model != 'ensemble'].accuracy.mean()})
b2 = pd.DataFrame(linhas)
ordem = [v[0] for v in VARIANTES]
cols = ['validação', 'PH2 bal. clientes', 'PH2 bal. ensemble', 'validação ens.']
g = b2.groupby(['alpha', 'variante'])[cols]
tab = g.mean().round(3).astype(str) + ' ± ' + g.std().round(3).astype(str)
print(tab.reindex(pd.MultiIndex.from_product([ALPHAS, ordem])).to_string())
print('\nmédia dos 3 α (média de 6 execuções), ± dp entre as 6:')
g = b2.groupby('variante')[cols]
print((g.mean().round(3).astype(str) + ' ± ' + g.std().round(3).astype(str)).reindex(ordem).to_string())

# public_init idêntico entre as três variantes da mesma seed/α?
pi = []
for seed in (42, 1):
    for alpha in ALPHAS:
        for prefixo in ('original', 'semdigest_original', 'controle_original'):
            t = load(f'{D8}/{prefixo}_seed{seed}_alpha{alpha}_test_metrics.csv')
            for st in ('public_init', 'private_init'):
                pi.append({'seed': seed, 'alpha': alpha, 'prefixo': prefixo, 'stage': st,
                           'acc': t[t.stage == st].test_accuracy.mean()})
pi = pd.DataFrame(pi).pivot_table(index=['seed', 'alpha', 'stage'], columns='prefixo', values='acc')
print('\npublic_init / private_init (validação) por prefixo:\n', pi.round(4).to_string())

c8 = pd.read_csv(f'{R}/{D8}/centralizado.csv')
u, en = c8[c8.model != 'ensemble'], c8[c8.model == 'ensemble']
print('\nteto centralizado PH2 privado: modelo único validação %.3f ± %.3f, PH2 bal %.3f ± %.3f; '
      'ensemble validação %.3f ± %.3f, PH2 bal %.3f ± %.3f' % (
          u.test_accuracy.mean(), u.test_accuracy.std(), u.ph2_balanced_accuracy.mean(), u.ph2_balanced_accuracy.std(),
          en.test_accuracy.mean(), en.test_accuracy.std(), en.ph2_balanced_accuracy.mean(), en.ph2_balanced_accuracy.std()))

# ---------------------------------------------------------------------------------------------------------------- B3
print('\n===== B3 =====')
p = b2.pivot_table(index=['alpha', 'seed'], columns='variante', values='PH2 bal. clientes')
p['Δ rodadas (FedMD − local)'] = p['FedMD replicado'] - p['Treinamento local']
p['Δ rodadas (sem digest − local)'] = p['Sem digest'] - p['Treinamento local']
p['Δ rodadas (CE − local)'] = p['Digest só com CE'] - p['Treinamento local']
p['FedMD − sem digest'] = p['FedMD replicado'] - p['Sem digest']
print(p.round(4).to_string())
print(p.groupby('alpha').mean().round(4).reindex(ALPHAS).to_string())
print('média dos 3 α:\n', p.mean().round(4).to_string())
# O "local" de cada variante é o private_init da própria execução; confere também esse Δ.
for nome, prefixo in (('sem digest', 'semdigest_original'), ('CE', 'controle_original')):
    dd = []
    for seed in (42, 1):
        for alpha in ALPHAS:
            x = load(f'{D8}/{prefixo}_seed{seed}_alpha{alpha}_extra_test_metrics.csv')
            x = x[x.model != 'ensemble']
            dd.append(fim(x, (16, 20)).balanced_accuracy.mean() - x[x.stage == 'private_init'].balanced_accuracy.mean())
    print(f'Δ rodadas {nome} contra o private_init da própria execução: média {np.mean(dd):+.4f}')
print('\nPH2 bal. após private_init (cada execução com o seu): ', end='')
for prefixo in ('original', 'semdigest_original', 'controle_original'):
    v = [load(f'{D8}/{prefixo}_seed{s}_alpha{a}_extra_test_metrics.csv').query("stage == 'private_init' and model != 'ensemble'")
         .balanced_accuracy.mean() for s in (42, 1) for a in ALPHAS]
    print(f'{prefixo} {np.mean(v):.3f}; ', end='')
print()

# ---------------------------------------------------------------------------------------------------------------- B4
print('\n===== B4: testes =====')
testes(p['FedMD − sem digest'], 'FedMD − sem digest, PH2 bal. clientes')


def ganho(base, frac, r=(26, 30)):
    t = load(f'{base}_test_metrics.csv')
    t = t[t.public_fraction == frac]
    out = {}
    for alpha in ALPHAS:
        a = t[t.alpha == alpha]
        out[alpha] = fim(a, r).test_accuracy.mean() - a[a.stage == 'private_init'].test_accuracy.mean()
    return out


def pares(fonte_a, fonte_b, frac):
    d = []
    for (ba, bb) in zip(fonte_a, fonte_b):
        ga, gb = ganho(ba, frac), ganho(bb, frac)
        d += [ga[a] - gb[a] for a in ALPHAS]
    return d


CE15 = {0.01: [f'digest_ce_1_5/CE1.0_P01_seed{s}' for s in (42, 1, 2)],
        0.05: [f'digest_ce_1_5/CE1.0_P05_seed{s}' for s in (42, 1, 2)]}
testes(pares(FONTES_KL[0.01], CE15[0.01], 0.01), 'KL − CE (w = 1), 1%')
testes(pares(FONTES_KL[0.05], CE15[0.05], 0.05), 'KL − CE (w = 1), 5%')

MAE = {0.01: [f'experimento_final/P01_seed{s}' for s in (42, 1, 2)],
       0.05: ['fracoes_publicas_5_e_20/F2b_seed42', 'fracoes_publicas_5_e_20/F2b_seed1', 'experimento_final/P05_seed2'],
       0.1: [f'digest_kl_10_40_80/MAE_P10_seed{s}' for s in (42, 1)],
       0.4: [f'digest_kl_10_40_80/MAE_P40_seed{s}' for s in (42, 1)],
       0.8: [f'digest_kl_10_40_80/MAE_P80_seed{s}' for s in (42, 1)]}
d_10_80 = sum((pares(FONTES_KL[f][:2], MAE[f], f) for f in (0.1, 0.4, 0.8)), [])
testes(d_10_80, 'KL − MAE, 10/40/80% (KL = fracoes_publicas_10_40_80 seeds 42 e 1)')
d_1_5 = sum((pares(FONTES_KL[f], MAE[f], f) for f in (0.01, 0.05)), [])
testes(d_1_5, 'KL − MAE, 1% e 5% (KL = ensemble_e_leave_one_out/all_*)')
KL_ANTIGO = {0.01: [f'digest_kl/T4_P01_seed{s}' for s in (42, 1, 2)], 0.05: [f'digest_kl/T4_P05_seed{s}' for s in (42, 1, 2)]}
d_1_5_antigo = sum((pares(KL_ANTIGO[f], MAE[f], f) for f in (0.01, 0.05)), [])
testes(d_1_5_antigo, 'KL − MAE, 1% e 5% (KL = digest_kl/T4_*, a fonte do p ≈ 0,08 original)')

# ---------------------------------------------------------------------------------------------------------------- B5
print('\n===== B5: passos por rodada no PH2 privado =====')
for prefixo in ('original', 'controle_original', 'semdigest_original'):
    h = load(f'{D8}/{prefixo}_seed42_alphaiid_train_history.csv')
    print(prefixo, h.groupby('stage').size().to_dict())
    pi_ep = h[h.stage == 'private_init'].groupby('client_id').size().to_dict()
    rv = h[h.stage == 'revisit'].groupby(['round', 'client_id']).size().describe()[['min', 'max']].to_dict()
    print('   épocas no private_init por cliente:', pi_ep, '| épocas do revisit por rodada/cliente:', rv)
t = load(f'{D8}/original_seed42_alphaiid_test_metrics.csv')
for seed in (42, 1):
    for alpha in ALPHAS:
        t = load(f'{D8}/original_seed{seed}_alpha{alpha}_test_metrics.csv')
        n = t[t.stage == 'public_init'].set_index('client_id').n_private.to_dict()
        print(f'n_private seed {seed} α {alpha}: {n} → passos/época do revisit (batch 5):',
              {k: int(np.ceil(v / 5)) for k, v in n.items()})
n_pub = 9605
print('digest no PH2 privado: subconjunto da rodada = round(0,8 ·', n_pub, ') =', round(0.8 * n_pub),
      '→ passos =', int(np.ceil(round(0.8 * n_pub) / 32)))
h = load('fracoes_publicas_10_40_80/original_P10_seed42_train_history.csv')
pi_ep = h[(h.stage == 'private_init')].groupby(['alpha', 'client_id']).size()
print('cenário padrão (original_P10_seed42): épocas do private_init por cliente:', pi_ep.to_dict())
rv = h[h.stage == 'revisit'].groupby(['alpha', 'round', 'client_id']).size()
print('   épocas do revisit:', rv.min(), 'a', rv.max(), '| restored != última época no revisit:',
      int((h[h.stage == 'revisit'].groupby(['alpha', 'round', 'client_id']).apply(
          lambda g: g.sort_values('epoch').restored.iloc[-1] == False)).sum()), 'de', len(rv))

# ---------------------------------------------------------------------------------------------------------------- B10
print('\n===== B10 =====')
from dataset_setup import list_files, split_public  # noqa: E402

_, labels = list_files(True)
for f in (0.01, 0.05, 0.1, 0.4, 0.8, 1.0):
    pub = len(labels) if f == 1.0 else len(split_public(labels, f, 42)[1])
    n = round(0.8 * pub)
    valores = 2 * n
    payload = 4 * valores
    # protobuf: client_id (tag + varint ≤ 2 B), logits empacotados (tag 1 B + varint de comprimento + 4 B/valor),
    # public_accuracy (tag 1 B + 4 B)
    comp = payload
    varint = len(np.base_repr(comp, 2)) // 7 + 1
    total = 2 + 1 + varint + payload + 5
    print(f'f = {f:g}: público {pub}, subconjunto da rodada {n}, {valores} floats, payload {payload} B, '
          f'mensagem ≈ {total} B ({total / 1024:.1f} KiB)')

# B10: parâmetros das 4 arquiteturas (mesmo build_model do client.py)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
from client import CLIENT_ARCHITECTURES, build_model, small_cnn  # noqa: E402

tot = 0
for cid, arq in enumerate(CLIENT_ARCHITECTURES):
    m = build_model(small_cnn(arq))
    n_tr = sum(int(np.prod(w.shape)) for w in m.trainable_weights)
    n_all = m.count_params()
    tot += n_all
    print(f'arquitetura {cid} {arq}: {n_all} parâmetros ({n_tr} treináveis), float32 = {4 * n_all} B '
          f'({4 * n_all / 2**20:.2f} MiB)')
print('total das 4:', tot)

# ---------------------------------------------------------------------------------------------------------------- B11
print('\n===== B11: sensibilidade e especificidade no PH2 (200 imagens, 160 benignas / 40 melanomas), IID =====')


def se_es(acc, bal):
    e = (acc - 0.4 * bal) / 0.6
    return 2 * bal - e, e


FONTES_B11 = {0.05: 'test_models/{m}_seed{s}', 0.4: 'fracoes_publicas_10_40_80/{m}_P40_seed{s}'}
for frac, padrao in FONTES_B11.items():
    for nome, stage, r in (('Treinamento local', 'private_init', None), ('FedMD adaptado', 'federated', (26, 30)),
                           ('FedMD replicado', 'federated', (16, 20))):
        prefixo = 'original' if nome == 'FedMD replicado' else 'adaptado'
        v = []
        for s in (42, 1, 2):
            x = load(padrao.format(m=prefixo, s=s) + '_extra_test_metrics.csv')
            x = x[(x.alpha == 'iid') & (x.model != 'ensemble') & (x.stage == stage)]
            if r:
                x = x[x['round'].between(*r)]
            x = x.groupby('model')[['accuracy', 'balanced_accuracy']].mean()  # média das rodadas por cliente
            for _, row in x.iterrows():
                v.append((*se_es(row.accuracy, row.balanced_accuracy), row.accuracy, row.balanced_accuracy))
        v = np.array(v)
        print(f'{frac:.0%} {nome}: S = {v[:, 0].mean():.3f}, E = {v[:, 1].mean():.3f} '
              f'(acc {v[:, 2].mean():.3f}, bal {v[:, 3].mean():.3f}; {len(v)} clientes×seeds)')
c = pd.read_csv(f'{R}/test_models/centralizado.csv')
for nome, cc in (('Teto centralizado (modelo único)', c[c.model != 'ensemble']), ('Teto centralizado (ensemble)', c[c.model == 'ensemble'])):
    s, e = se_es(cc.ph2_accuracy, cc.ph2_balanced_accuracy)
    print(f'{nome}: S = {s.mean():.3f}, E = {e.mean():.3f} (acc {cc.ph2_accuracy.mean():.3f}, bal {cc.ph2_balanced_accuracy.mean():.3f})')
