# KL × MAE no digest com 10%, 40% e 80% de dados públicos (08–09/11)

## Por que

A escolha da KL com T = 4 (`results/digest_kl`) foi feita só com 1% e 5% de dados públicos, e a diferença ficou
dentro do ruído. Aqui repetimos a comparação nas frações das ablações, só no teste do dataset de treino.

## Configuração

FedMD adaptado (configuração final, `public_init` com lr 3e-4), seeds 42 e 1.

- **Lado KL (T = 4):** as execuções `adaptado_P{10,40,80}_seed{42,1}` de `results/fracoes_publicas_10_40_80`.
- **Lado MAE:** este experimento (`MAE_P{10,40,80}_seed{42,1}`). Os argumentos são idênticos, exceto que não há
  `--digest-temperature` nem `--extra-test-dir`. A avaliação no PH2 não altera o treino.

O ganho é a média das rodadas 26–30 menos a média logo após o `private_init`, na mesma execução. O treino local
das duas execuções pareadas não é idêntico (até 0,004 de diferença, por não determinismo da GPU). Por isso a
comparação usa o ganho de cada execução sobre o próprio local.

## Resultado

Ganho sobre o treino local (média dos 4 clientes, média das 2 seeds):

| Fração | Digest | IID | α = 1.0 | α = 0.1 | Média |
|---|---|---|---|---|---|
| 10% | KL | +0,008 | −0,002 | −0,001 | +0,002 |
| | MAE | +0,003 | −0,002 | −0,002 | 0,000 |
| 40% | KL | −0,007 | 0,000 | −0,018 | −0,008 |
| | MAE | −0,005 | −0,017 | −0,016 | −0,013 |
| 80% | KL | −0,010 | −0,019 | −0,009 | −0,013 |
| | MAE | −0,006 | −0,014 | −0,015 | −0,012 |

Diferença pareada KL − MAE (mesma fração, α e seed):

| Fração | KL − MAE | Desvio | KL melhor |
|---|---|---|---|
| 10% | +0,0024 | 0,0036 | 5/6 |
| 40% | +0,0048 | 0,0106 | 4/6 |
| 80% | −0,0010 | 0,0075 | 4/6 |
| **10% a 80%** | **+0,0021** | 0,0077 | 13/18 (p = 0,27, teste t; p = 0,25, Wilcoxon) |

## Conclusões

1. **A KL continua ligeiramente à frente, mas sem significância**: +0,002 em média, melhor em 13 dos 18 pares.
   É o mesmo tamanho de efeito da comparação com 1% e 5% (+0,0019).
2. **A troca da loss não muda a conclusão principal.** Com 40% e 80%, as duas variantes perdem do treino local
   (−0,008 a −0,013). Com 10%, as duas empatam com ele.
3. **A decisão de usar KL se mantém**, mas como escolha neutra. Ela não explica nenhum dos resultados do estudo.

## Observações

- `MAE_P40_seed1` travou duas vezes no α = 0.1: na rodada 8 do digest e depois no `public_init`. Em ambas, um
  cliente ficou preso com a GPU ociosa. O IID e o α = 1.0 da segunda tentativa foram aproveitados, e o α = 0.1
  foi refeito sem XLA (`--no-jit`, `retoma_P40_seed1.sh`). Os logs e CSVs parciais das tentativas estão em
  `MAE_P40_seed1_travado*`.
- O α = 0.1 de `MAE_P80_seed1` ficou ~5× mais lento por um trecho (GPU em 99%, memória perto do limite), mas
  terminou normalmente.

## Arquivos

- `orquestrador.sh`, `orquestrador.log`, `server.log` e um `.log` por execução.
- `MAE_P{10,40,80}_seed{42,1}_{test_metrics,train_history,ensemble_metrics}.csv`.
- Tabelas e gráfico: seção 4 de `result_exploration.ipynb` (`results/figuras/4_ganho_por_fracao.png`).
