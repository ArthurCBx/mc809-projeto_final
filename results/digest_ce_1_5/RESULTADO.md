# Controle sem destilação com 1% e 5% de dados públicos (12–13/11)

## Configuração

O mesmo controle de `results/digest_ce_10_40_80` (w = 1: o digest é só entropia cruzada no rótulo verdadeiro das
imagens públicas; o consenso é calculado mas não entra na loss), nas frações em que o FedMD adaptado ganhava do treino
local. O lado w = 0 (só KL, T = 4) são as execuções `all_P01_seed*` e `all_P05_seed*` de
`results/ensemble_e_leave_one_out`, com os mesmos argumentos; aqui só entram `--digest-ce-weight 1` e a avaliação no PH2
(que não altera o treino). Seeds 42, 1 e 2.

## Resultado

Média dos 4 clientes, 3 seeds × 3 α. Ganho = média das rodadas 26–30 menos o `private_init` da mesma execução.

| Fração | Digest | Local | Final | Ganho | Pares com ganho | Ensemble |
|---|---|---|---|---|---|---|
| 1% | KL (w = 0) | 0,859 | 0,869 | **+0,009 ± 0,005** | 9/9 | 0,877 |
| | Só CE (w = 1) | 0,859 | 0,866 | +0,006 ± 0,009 | 6/9 | 0,883 |
| 5% | KL (w = 0) | 0,868 | 0,878 | +0,011 ± 0,005 | 9/9 | 0,885 |
| | Só CE (w = 1) | 0,869 | 0,886 | **+0,017 ± 0,008** | 9/9 | **0,897** |

Diferença pareada KL − CE no ganho: **1%: +0,003 ± 0,007** (KL melhor em 6/9, p = 0,22); **5%: −0,007 ± 0,009**
(KL melhor em 3/9, p = 0,05).

## Conclusões

1. **Com 1%, a destilação tem uma pequena vantagem, mas sem significância.** O ganho da KL é mais estável (9/9 pares
   positivos contra 6/9), o que é coerente com a hipótese de que, com só ~96 imagens públicas, treinar mais nos rótulos
   delas sobreajusta e o consenso funciona como um alvo mais suave. A diferença (+0,003) está dentro do ruído.
2. **Com 5%, o controle sem destilação já é melhor** (+0,017 contra +0,011), como em 10%, 40% e 80%.
3. **Junto com `results/digest_ce_10_40_80`, o ganho do FedMD adaptado vem do treino extra com rótulos públicos em
   todas as frações a partir de 5%.** Só com 1% a destilação empata ou supera o controle, e por pouco.

## w = 0,5 com 1% e 5% (20/11)

`orquestrador_w05.sh`: o mesmo FedMD adaptado, com os mesmos argumentos das execuções de w = 0 e w = 1 (configuração
final; `public_init` com lr 3e-4 em 1% e 1e-3 em 5%), só com `--digest-ce-weight 0.5`, avaliação no PH2 e seeds 42, 1
e 2 (`CE0.5_P0{1,5}_seed*`; os logits do PH2 ficam em `CE0.5_*_extra_test_logits_*.npz`). A sessão caiu durante
`CE0.5_P05_seed42`, que foi refeito do zero (parciais em `CE0.5_P05_seed42_interrompido_*`).

### Ganho sobre o treinamento local (formato da Tabela 6)

Média dos 4 clientes, 3 seeds × 3 divisões; entre parênteses, os pares de seed e divisão com ganho.

| f | w = 0 (FedMD) | w = 0,5 | w = 1 (sem destilação) |
|---|---|---|---|
| 1% | +0,009 (9/9) | **+0,010** (7/9) | +0,006 (6/9) |
| 5% | +0,011 (9/9) | **+0,017** (9/9) | **+0,017** (9/9) |

Por divisão:

| f | Divisão | w = 0 | w = 0,5 | w = 1 |
|---|---|---|---|---|
| 1% | IID | +0,006 | +0,004 | +0,004 |
| | α = 1,0 | +0,009 | +0,007 | +0,006 |
| | α = 0,1 | +0,013 | +0,017 | +0,009 |
| 5% | IID | +0,009 | +0,016 | +0,013 |
| | α = 1,0 | +0,011 | +0,013 | +0,015 |
| | α = 0,1 | +0,012 | +0,022 | +0,024 |

Acurácia final e ensemble (validação) e PH2 (acurácia balanceada, média dos clientes):

| f | w | Local | Final | Ensemble | PH2 (local → final) |
|---|---|---|---|---|---|
| 1% | 0 | 0,859 | 0,869 | 0,877 | não avaliado |
| | 0,5 | 0,859 | 0,869 | 0,879 | 0,645 → 0,587 |
| | 1 | 0,859 | 0,866 | 0,883 | 0,644 → 0,615 |
| 5% | 0 | 0,868 | 0,878 | 0,885 | não avaliado |
| | 0,5 | 0,868 | 0,885 | 0,894 | 0,635 → 0,590 |
| | 1 | 0,869 | 0,886 | 0,897 | 0,634 → 0,585 |

### Testes pareados (por seed e divisão, n = 9; `results/estatistica_e_divisoes/testes.py`)

| Comparação (ganho) | Média | IC 95% (bootstrap) | A favor | p (t) | p (Wilcoxon) |
|---|---|---|---|---|---|
| 1%: w = 0,5 − w = 0 | +0,0002 | [−0,0032; +0,0031] | 6/9 | 0,92 | 0,65 |
| 1%: w = 0,5 − w = 1 | +0,0032 | [+0,0000; +0,0080] | 6/9 | 0,21 | 0,16 |
| 5%: w = 0,5 − w = 0 | +0,0064 | [+0,0022; +0,0100] | 8/9 | 0,016 | 0,039 |
| 5%: w = 0,5 − w = 1 | −0,0003 | [−0,0043; +0,0038] | 5/9 | 0,89 | 0,82 |

### Leitura

1. **Com 1%, os três pesos empatam.** w = 0,5 fica igual à KL pura (+0,010 contra +0,009) e um pouco acima do controle
   sem destilação (+0,003, sem significância; o IC encosta em zero e nem o t nem o Wilcoxon confirmam). Com ~96 imagens
   públicas, misturar o consenso aos rótulos não atrapalha, mas também não traz ganho mensurável.
2. **Com 5%, w = 0,5 se comporta como o controle sem destilação** (+0,017 nos dois; diferença −0,0003, p ≈ 0,9) e
   supera a KL pura (+0,006, 8/9 pares, p = 0,016 no t e 0,039 no Wilcoxon). Assim que os rótulos entram no digest, o
   ganho passa a ser o do treino supervisionado.
3. **O padrão das outras frações se estende a 1% e 5%:** quanto mais peso nos rótulos, maior o ganho, exceto com 1%,
   em que todos ficam em ~+0,01. A destilação nunca supera de forma significativa o uso dos rótulos públicos.
4. **No PH2, nenhum peso passa do treinamento local** (0,64 → 0,59 com w = 0,5), como nas demais frações.
