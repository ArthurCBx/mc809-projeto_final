# Digest com âncora nos rótulos: (1 − w) · KL + w · CE (10–11/11)

## Por que

O digest do FedMD treina cada cliente para imitar o consenso dos colegas. Como o conjunto público é rotulado,
testamos misturar a destilação com a entropia cruzada no rótulo verdadeiro (a loss de Hinton et al., 2015):

**L = (1 − w) · T² · KL(consenso ‖ cliente) + w · CE(rótulo)**, com T = 4.

- **w = 0:** FedMD adaptado atual (só KL).
- **w = 0,5:** destilação ancorada nos rótulos.
- **w = 1:** controle **sem destilação**: os mesmos passos de digest, mas só com treino supervisionado nas imagens
  públicas. O consenso ainda é calculado (o servidor exige), mas não entra na loss.

## Configuração

FedMD adaptado (configuração final, `public_init` com lr 3e-4), frações 10%, 40% e 80%, seeds 42 e 1, com
avaliação no PH2. O lado w = 0 são as execuções `adaptado_P*_seed{42,1}` de `results/fracoes_publicas_10_40_80`,
com os mesmos argumentos; aqui só muda `--digest-ce-weight`. O ganho é a média das rodadas 26–30 menos a média
logo após o `private_init`, na mesma execução.

## Resultado

Média dos 4 clientes, 2 seeds × 3 α:

| Fração | w | Local | Final | Ganho | Pares com ganho | Ensemble local | Ensemble final | PH2 bal. local | PH2 bal. final |
|---|---|---|---|---|---|---|---|---|---|
| 10% | 0 | 0,878 | 0,880 | +0,002 ± 0,008 | 4/6 | 0,892 | 0,884 | 0,650 | 0,622 |
| | 0,5 | 0,876 | 0,893 | +0,017 ± 0,007 | 6/6 | 0,893 | 0,897 | 0,637 | 0,593 |
| | 1 | 0,877 | **0,897** | **+0,020 ± 0,007** | 6/6 | 0,893 | **0,903** | 0,646 | 0,590 |
| 40% | 0 | 0,890 | 0,882 | −0,008 ± 0,011 | 1/6 | 0,901 | 0,888 | 0,609 | 0,584 |
| | 0,5 | 0,888 | 0,896 | +0,007 ± 0,007 | 5/6 | 0,900 | 0,900 | 0,617 | 0,591 |
| | 1 | 0,889 | **0,902** | **+0,013 ± 0,007** | 6/6 | 0,898 | **0,907** | 0,615 | 0,594 |
| 80% | 0 | 0,895 | 0,882 | −0,013 ± 0,008 | 0/6 | 0,904 | 0,886 | 0,606 | 0,549 |
| | 0,5 | 0,895 | 0,901 | +0,006 ± 0,004 | 6/6 | 0,905 | 0,902 | 0,612 | 0,593 |
| | 1 | 0,892 | **0,905** | **+0,012 ± 0,003** | 6/6 | 0,904 | **0,908** | 0,610 | 0,596 |

Ganho médio nas três frações: **w = 0: −0,006; w = 0,5: +0,010; w = 1: +0,015.**

Referência: centralizado (modelo único) 0,901; ensemble centralizado 0,904 (`results/test_models/centralizado.csv`).

## Conclusões

1. **Ancorar o digest nos rótulos transforma a perda do FedMD em ganho** em todas as frações (de −0,006 para
   +0,010 em média) e leva os clientes a ~0,90, o nível do centralizado.
2. **Mas o controle sem destilação (w = 1) é ainda melhor**, em todas as frações e em 18 de 18 pares. O ganho vem
   de treinar mais com os rótulos públicos, não do consenso dos colegas: quanto mais peso a KL tem, menor o ganho.
   No nosso cenário (dados públicos rotulados e do mesmo domínio), a destilação do FedMD não acrescenta nada além do
   que os próprios rótulos públicos já dão.
3. **O ensemble com w = 1 passa o centralizado** (0,903–0,908 contra 0,904), sem nenhuma troca de logits que
   importe para o treino.
4. **No PH2, o treino extra no domínio de origem não ajuda e às vezes piora.** Com 10%, a acurácia balanceada cai
   de 0,622 (w = 0) para 0,590 (w = 1). Com 40% e 80%, as variantes com CE ficam um pouco acima do w = 0, mas abaixo do
   próprio local. Nenhuma passa do local, que continua o melhor no PH2.

## Observações

- A divisão em `public_init` e `private_init` é a mesma nas três variantes; o local difere até ~0,003 entre execuções
  pareadas por não determinismo da GPU, por isso a comparação usa o ganho de cada execução sobre o próprio local.
- `CE0.5_P80_seed1` foi interrompido por um reinício da máquina logo no começo (`CE0.5_P80_seed1_interrompido.log`) e
  refeito do zero.
- Implementação: `--digest-ce-weight` e `with_label_anchor` em `client.py`.

## Arquivos

- `orquestrador.sh` (com watchdog de travamento), `orquestrador.log`, `server.log` e um `.log` por execução.
- `CE{0.5,1.0}_P{10,40,80}_seed{42,1}_{test_metrics,train_history,ensemble_metrics,extra_test_metrics}.csv`.
