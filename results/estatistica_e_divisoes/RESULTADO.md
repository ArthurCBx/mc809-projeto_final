# Distribuição por cliente e testes estatísticos refeitos (15/11)

Nenhum treino: os dois scripts só refazem divisões ou leem CSVs.

## 3. Distribuição dos dados privados por cliente (`divisoes.py`)

Refaz as divisões com as mesmas funções e sementes do `client.py` (`split_dataset`; no cenário 2, `split_external` +
`split_clients`) e grava `divisoes_por_cliente.csv` (uma linha por cenário, f, α, seed e cliente: imagens e melanomas).
Conferência: as contagens batem com as colunas `n_private` e `private_malignant_fraction` de 204 clientes das execuções
(0 divergências). Contagens incluem os 20% separados para a parada antecipada no cenário 1.

**Cenário 1** (3 seeds × 4 clientes = 12 clientes por linha):

| f | Divisão | Imagens por cliente | Fração de melanomas (mín.–máx.) | Clientes com > 94% de uma classe | Clientes com uma classe só |
|---|---|---|---|---|---|
| 1% | IID | 2.377 | 0,47–0,49 | 0 | 0 |
| | α = 1,0 | 1.262–3.250 | 0,05–0,86 | 1 | 0 |
| | α = 0,1 | 26–6.400 | 0,00–1,00 | 8 | 5 |
| 5% | IID | 2.281 | 0,47–0,50 | 0 | 0 |
| | α = 1,0 | 568–4.267 | 0,00–0,98 | 4 | 2 |
| | α = 0,1 | 47–4.663 | 0,00–1,00 | 12 | 7 |
| 10% | IID | 2.161 | 0,47–0,49 | 0 | 0 |
| | α = 1,0 | 686–4.500 | 0,00–0,86 | 2 | 1 |
| | α = 0,1 | 59–3.897 | 0,00–1,00 | 10 | 6 |
| 40% | IID | 1.440–1.441 | 0,46–0,50 | 0 | 0 |
| | α = 1,0 | 443–2.961 | 0,00–0,95 | 5 | 2 |
| | α = 0,1 | 138–2.884 | 0,00–1,00 | 11 | 6 |
| 80% | IID | 480–481 | 0,45–0,52 | 0 | 0 |
| | α = 1,0 | 38–950 | 0,00–0,94 | 3 | 3 |
| | α = 0,1 | 41–1.000 | 0,00–1,00 | 10 | 8 |

**Correção no texto (Seção 6.1):** com f = 40%, α = 1,0 e seed 42 (a do desenvolvimento incremental), a classe
majoritária de cada cliente tem 96,7%, 93,8%, 100% e 95,3% das amostras. Um cliente fica abaixo de 94%; a frase deve
ser "todos os clientes ficaram com mais de 93% das amostras de uma única classe". O CSV do desenvolvimento
(`01_replicacao_fedmd`) dá a mesma divisão.

**Cenário 2** (imagens / melanomas por cliente; 80 imagens do PH2, 16 melanomas):

| Seed | Divisão | Cliente 0 | Cliente 1 | Cliente 2 | Cliente 3 |
|---|---|---|---|---|---|
| 42 | IID | 20 / 0 | 20 / 7 | 20 / 3 | 20 / 6 |
| | α = 1,0 | 18 / 1 | 15 / 4 | 25 / 11 | 22 / 0 |
| | α = 0,1 | 16 / 3 | 15 / 13 | 25 / 0 | 24 / 0 |
| 1 | IID | 20 / 3 | 20 / 5 | 20 / 3 | 20 / 5 |
| | α = 1,0 | 24 / 9 | 23 / 5 | 20 / 1 | 13 / 1 |
| | α = 0,1 | 10 / 1 | 45 / 0 | 12 / 12 | 13 / 3 |

Sustenta as frases do texto: na divisão IID da seed 42 um cliente não tem nenhum melanoma; com α = 0,1 há clientes
com dezenas de imagens e nenhum melanoma (45 / 0, 25 / 0, 24 / 0) e um só com melanomas (12 / 12). O revisit de
"6 a 27 passos" corresponde a 10 e 45 imagens (3 × ⌈n / 5⌉).

## 4. Testes estatísticos refeitos (`testes.py`, `testes.csv`)

Para cada comparação pareada: teste t pareado bilateral, Wilcoxon pareado bilateral exato e IC de 95% por bootstrap
da média das diferenças (percentil, 10.000 reamostragens dos pares, semente fixa).

| Comparação | Onde | n | Média | IC 95% (bootstrap) | A favor | p (t) | p (Wilcoxon) |
|---|---|---|---|---|---|---|---|
| FedMD adaptado − local, 1% | Tab. 4–5 | 9 | +0,0095 | [+0,0068; +0,0124] | 9/9 | < 0,001 | 0,004 |
| FedMD adaptado − local, 5% | Tab. 4–5 | 9 | +0,0107 | [+0,0077; +0,0137] | 9/9 | < 0,001 | 0,004 |
| FedMD adaptado − local, 10% | Tab. 4 | 9 | +0,0015 | [−0,0019; +0,0059] | 6/9 | 0,50 | 0,73 |
| FedMD adaptado − local, 40% | Tab. 4 | 9 | −0,0093 | [−0,0146; −0,0028] | 1/9 | 0,020 | 0,055 |
| FedMD adaptado − local, 80% | Tab. 4 | 9 | −0,0155 | [−0,0225; −0,0095] | 0/9 | 0,003 | 0,004 |
| KL − MAE, 10/40/80% | Seção 6.2 | 18 | +0,0021 | [−0,0012; +0,0057] | 13/18 | 0,27 | 0,25 |
| w = 0 − w = 1, 1% | Tab. 6 | 9 | +0,0030 | [−0,0011; +0,0072] | 6/9 | 0,22 | 0,25 |
| w = 0 − w = 1, 5% | Tab. 6 | 9 | −0,0067 | [−0,0123; −0,0014] | 3/9 | 0,053 | 0,055 |
| w = 0,5 − w = 0, 1% | Tab. 6 | 9 | +0,0002 | [−0,0032; +0,0031] | 6/9 | 0,92 | 0,65 |
| w = 0,5 − w = 1, 1% | Tab. 6 | 9 | +0,0032 | [+0,0000; +0,0080] | 6/9 | 0,21 | 0,16 |
| w = 0,5 − w = 0, 5% | Tab. 6 | 9 | +0,0064 | [+0,0022; +0,0100] | 8/9 | 0,016 | 0,039 |
| w = 0,5 − w = 1, 5% | Tab. 6 | 9 | −0,0003 | [−0,0043; +0,0038] | 5/9 | 0,89 | 0,82 |
| w = 1 − w = 0, 10/40/80% | Tab. 6 | 18 | +0,0214 | [+0,0182; +0,0248] | 18/18 | < 0,001 | < 0,001 |
| w = 0,5 − w = 0, 10/40/80% | Tab. 6 | 18 | +0,0162 | [+0,0130; +0,0195] | 18/18 | < 0,001 | < 0,001 |
| Cenário 2: FedMD − sem digest (PH2 bal.), 4 seeds | Seção 6.6 | 12 | +0,041 | [+0,015; +0,066] | 9/12 | 0,014 | 0,027 |
| Cenário 2: FedMD − local (PH2 bal.), 4 seeds | Seção 6.6 | 12 | +0,065 | [+0,036; +0,093] | 11/12 | 0,001 | 0,002 |
| Cenário 2: FedMD − digest só CE (PH2 bal.), 4 seeds | Tab. 9 | 12 | +0,105 | [+0,060; +0,148] | 10/12 | 0,001 | 0,005 |
| (antes, 2 seeds) FedMD − sem digest | | 6 | +0,056 | [+0,025; +0,086] | 5/6 | 0,023 | 0,062 |

**As conclusões se mantêm**, com duas ressalvas de redação:

1. **Cenário 2, FedMD − sem digest:** o t (p = 0,023) e o bootstrap (IC exclui 0) concordam, mas o Wilcoxon fica em
   0,062. Com 6 pares e 5 a favor, o menor p exato possível do Wilcoxon é 0,0625; o teste não tem como chegar a 0,05.
   **Com as seeds 2 e 3 (n = 12), os dois testes passam: p = 0,014 (t) e 0,027 (Wilcoxon).**
2. **5%, controle sem destilação melhor que o FedMD:** é marginal nos dois testes (p ≈ 0,05), embora o IC por
   bootstrap exclua 0. "A partir de 5% o controle supera o FedMD" é sustentado de forma firme a partir de 10%; com 5%
   convém "supera, no limite da significância".

O ganho negativo com 40% passa a ter teste (t: p = 0,020; Wilcoxon: 0,055), assim como o positivo com 1% e 5%
(p = 0,004 nos dois testes).

**Texto sugerido** no lugar de "os p-valores reportados devem ser lidos como indicativos" (Seção 5.6):

> As comparações pareadas usam o teste t pareado bilateral e o teste de postos sinalizados de Wilcoxon pareado
> (bilateral, exato), com intervalo de confiança de 95% da diferença média obtido por bootstrap dos pares (10.000
> reamostragens). Os pares são as combinações de semente e divisão dos dados, avaliadas com a mesma partição em cada
> método.

Vale manter meia frase de que pares de mesma semente compartilham a divisão pública e o sorteio do PH2, e por isso não
são totalmente independentes: o bootstrap por pares não modela essa dependência.
