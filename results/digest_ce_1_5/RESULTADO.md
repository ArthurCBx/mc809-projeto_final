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
