# FedMD no cenário do paper: Kaggle público, PH2 privado (11–12/11)

## Configuração

- **Público:** todo o conjunto de treino do Kaggle (9.605 imagens), usado no `public_init` e no digest (batch 32).
- **Privado:** 40% do PH2 (80 imagens: 64 benignas, 16 melanomas), divididas entre os 4 clientes por IID ou Dirichlet
  (α = 1.0 e 0.1), como no cenário padrão. Todas vão para o treino, sem validação (e sem early stopping), com
  **batch 5** no `private_init` e no revisit. lr e número de passos iguais aos da configuração final.
- **Teste:** o teste do Kaggle (1.000 imagens) e os outros 60% do PH2 (120 imagens, 24 melanomas). A divisão do PH2
  depende só da seed (`split_external`), então todos os métodos usam as mesmas 120 imagens.
- **Métodos:** FedMD adaptado (KL T = 4, `public_init` com lr 3e-4; o treino local é o `private_init` da mesma execução),
  FedMD original (consenso uniforme, lr 1e-3, revisit de 3 épocas completas, 20 rodadas) e centralizado (Kaggle
  completo + os mesmos 80 do PH2). Seeds 42 e 1.
- Comando: `results/ph2_privado/orquestrador.sh` (opções `--public-fractions 1.0 --private-dir ph2_dataset
  --private-dir-fraction 0.4 --private-batch-size 5`).

Com tão poucas imagens, mesmo a divisão IID (aleatória, não estratificada) deixa clientes sem nenhum melanoma; com
α = 0.1, há clientes com 0% ou 100% de melanomas.

## Resultado

Média dos 4 clientes, 2 seeds × 3 α (FedMD: média das últimas 5 rodadas). "Bal." é a acurácia balanceada.

| Método | Teste Kaggle | Kaggle (ensemble) | PH2 bal. (clientes) | PH2 bal. (ensemble) |
|---|---|---|---|---|
| Só `public_init` (Kaggle) | **0,903** | **0,907** | 0,605 ± 0,034 | 0,617 |
| Local (+ `private_init` no PH2) | 0,879 | 0,894 | 0,632 ± 0,031 | 0,612 |
| FedMD adaptado | 0,850 | 0,858 | 0,664 ± 0,051 | 0,677 |
| FedMD original | 0,610 | 0,620 | **0,733 ± 0,089** | **0,800** |
| Centralizado (modelo único) | 0,895 | | 0,653 ± 0,058 | |
| Centralizado (ensemble) | 0,901 | | | 0,630 |

Por α (média das 2 seeds), PH2 balanceada dos clientes / do ensemble e teste no Kaggle:

| α | Local | FedMD adaptado | FedMD original |
|---|---|---|---|
| IID | 0,630 / 0,651 — Kaggle 0,879 | 0,679 / 0,695 — Kaggle 0,843 | **0,807 / 0,873** — Kaggle 0,625 |
| 1.0 | 0,646 / 0,589 — Kaggle 0,878 | 0,660 / 0,674 — Kaggle 0,844 | **0,772 / 0,845** — Kaggle 0,655 |
| 0.1 | 0,620 / 0,596 — Kaggle 0,881 | 0,653 / 0,662 — Kaggle 0,862 | 0,621 / 0,682 — Kaggle 0,551 |

Referência nas mesmas 120 imagens: o FedMD do cenário padrão, treinado **sem** PH2 (modelos salvos de
`results/fracoes_publicas_10_40_80`, `reavaliacao_modelos_salvos.csv`), fica entre 0,54 e 0,64 de acurácia balanceada.

## Conclusões

1. **Neste cenário, a colaboração ajuda no domínio privado quando os clientes têm as duas classes (IID e α = 1.0).** Os dois FedMD superam o treino local no PH2: o adaptado
   em +0,03 por cliente e +0,07 no ensemble; o original em +0,10 por cliente e +0,19 no ensemble. É o primeiro
   experimento do estudo em que o FedMD ganha do local e do centralizado no teste que importa para o cliente.
2. **O centralizado não aproveita os dados privados.** As 80 imagens do PH2 são menos de 1% do seu treino, e ele fica em
   0,65 no PH2, abaixo dos dois FedMD. Treinar com batch pequeno e muitos passos sobre os dados privados (o que o FedMD
   faz no revisit) é o que adapta o modelo ao novo domínio.
3. **Há uma troca entre os dois domínios.** Quanto mais o método se especializa no PH2, mais esquece o Kaggle: o FedMD
   original chega a 0,87–0,90 no ensemble do PH2 (IID e α = 1.0, o melhor resultado de todo o estudo), mas cai para
   ~0,61 no Kaggle; o adaptado fica no meio (0,66–0,68 no PH2, 0,85 no Kaggle).
4. **Com α = 0.1, a especialização deixa de compensar.** Clientes que só veem uma classe do PH2 herdam esse viés, e o
   FedMD original fica no nível do local (0,621 contra 0,620 por cliente) e cai para 0,55 no Kaggle. Só o FedMD
   adaptado, que se especializa menos, continua acima do local (0,653). A vantagem do FedMD no domínio privado depende,
   portanto, de cada cliente ter as duas classes.

## Sanity check: FedMD original sem colaboração (12–13/11)

`orquestrador_controle.sh`: a mesma configuração do FedMD original, mas com o digest só em entropia cruzada no rótulo
verdadeiro das imagens do Kaggle (`--digest-ce-weight 1`). O consenso é calculado, mas não entra na loss. Seeds 42 e 1,
IID, α = 1.0 e 0.1 (`controle_original_seed*_alpha*`).

| α | Método | Teste Kaggle | PH2 bal. (clientes) | PH2 bal. (ensemble) |
|---|---|---|---|---|
| IID | FedMD original (consenso) | 0,625 | **0,807** | **0,873** |
| | Sem colaboração (só CE) | **0,826** | 0,616 | 0,596 |
| 1.0 | FedMD original (consenso) | 0,655 | **0,772** | **0,845** |
| | Sem colaboração (só CE) | **0,797** | 0,598 | 0,568 |
| 0.1 | FedMD original (consenso) | 0,551 | **0,621** | **0,682** |
| | Sem colaboração (só CE) | **0,756** | 0,598 | 0,579 |

Sem o consenso, o PH2 volta ao nível do treino local (~0,60) e o Kaggle cai bem menos (0,76–0,83 contra 0,55–0,66).

**Controle sem digest** (`orquestrador_semdigest.sh`, opção `--no-digest`): a mesma configuração, mas cada rodada é só
o revisit nos dados privados, sem troca de logits nem treino no Kaggle depois do `public_init`
(`semdigest_original_seed*_alpha*`). Separa o efeito do consenso do efeito de reancorar no Kaggle.

| Variante (configuração do FedMD original) | Kaggle | PH2 bal. após `private_init` | PH2 bal. final (clientes) | Δ PH2 nas rodadas | PH2 bal. final (ensemble) |
|---|---|---|---|---|---|
| Consenso (FedMD original) | 0,610 | 0,660 | **0,733** | **+0,074** | **0,800** |
| Sem digest (só revisit no PH2) | 0,599 | 0,677 | 0,678 | 0,000 | 0,680 |
| Digest só CE (reancora no Kaggle) | **0,793** | 0,673 | 0,604 | −0,069 | 0,581 |

(Média de 2 seeds × 3 α; por α, o Δ do consenso é +0,104 (IID), +0,083 (α = 1.0) e +0,034 (α = 0.1).)

- **O consenso acrescenta de fato.** Sem digest, as 20 rodadas de revisit não mudam nada no PH2 (Δ = 0,000): a adaptação
  vem toda do `private_init`. Com o consenso, os clientes ganham +0,074 a mais. Consenso − sem digest no PH2:
  **+0,056 por cliente, melhor em 5 de 6 pares (p = 0,02)**; no ensemble, 0,800 contra 0,680.
- **A perda no Kaggle não vem do consenso**, e sim do treino privado (50 épocas, lr 1e-3, nas ~20 imagens): sem digest,
  o Kaggle fica em 0,60, igual ao do consenso (0,61).
- **O controle com CE reancora no Kaggle**: recupera o Kaggle (0,79) e apaga a adaptação ao PH2 (−0,069).
- **Conclusão:** neste cenário, o consenso transfere entre os clientes o que cada um aprendeu do PH2, pela única via que
  o FedMD permite (os logits sobre as imagens públicas). É o primeiro resultado do estudo em que a colaboração, isolada
  dos outros efeitos, melhora os clientes. O ganho diminui quando os clientes têm uma classe só (α = 0.1).

## Limitações

- 2 seeds, e o teste do PH2 tem só 24 melanomas (cada um vale ~0,04 de sensibilidade): os desvios são altos
  (até ±0,09).
- Não há validação privada; o número de passos foi mantido da configuração final, sem ajuste para este cenário.

## Observações sobre a execução

- O PC caiu duas vezes no momento em que os 4 clientes de uma nova configuração subiam na GPU (o Windows ocupava
  ~3 GB da placa). O orquestrador passou a rodar um α por processo, com `ray stop` entre eles; a configuração IID do
  `adaptado_seed42`, já completa, foi aproveitada (`adaptado_seed42_alphaiid_*`).
- O watchdog de travamento foi ajustado de 25 para 60 min, porque o `public_init` com o Kaggle inteiro fica ~23 min
  sem escrever no log.

## Arquivos

- `orquestrador.sh`, `orquestrador.log`, `server.log`, `centralizado_ph2_privado.py`, `centralizado.csv`,
  `centralizado.log`.
- `{adaptado,original}_seed{42,1}_alpha{iid,1.0,0.1}_{test_metrics,train_history,ensemble_metrics,extra_test_metrics}.csv`
  e um `.log` por execução.
- `reavalia_modelos_salvos.py` e `reavaliacao_modelos_salvos.csv`: FedMD do cenário padrão reavaliado nas mesmas
  120 imagens.

## Seeds 2 e 3: Tabela 9 com 4 seeds (15–16/11)

`orquestrador_sementes_2_3.sh`: as mesmas variantes (FedMD replicado, sem digest, digest só com CE) com as seeds 2 e 3,
nas três divisões, mais a referência centralizada justa (`results/referencia_justa_e_auc/referencia_seed{2,3}_*`) e o
centralizado diluído. `original_seed2_alpha0.1` caiu na rodada 14 (DEADLINE_EXCEEDED: um cliente não enviou os logits
em 10 min; parciais em `original_seed2_alpha0.1_falhou1_*`) e foi refeito por `retoma_original_seed2_alpha0.1.sh`.

Média de 4 seeds × 3 divisões (12 execuções por variante):

| Variante | Validação | PH2 bal. (clientes) | PH2 bal. (ensemble) |
|---|---|---|---|
| Só `public_init` | 0,901 | 0,650 | 0,628 |
| Treinamento local (fim do `private_init`) | 0,610 | 0,636 | 0,616 |
| Sem digest (só revisit) | 0,607 | 0,660 | 0,671 |
| Digest só com CE (w = 1) | 0,793 | 0,596 | 0,574 |
| FedMD (consenso) | 0,624 | 0,701 | 0,735 |
| Referência centralizada justa (80 do PH2 reunidas) | 0,700 | 0,820 | 0,831 |
| Centralizado diluído (ISIC + 80 do PH2) | 0,896 | 0,646 | 0,639 |

Por divisão (PH2 bal., clientes / ensemble):

| Divisão | Local | Sem digest | Só CE | FedMD | FedMD − local | FedMD − sem digest |
|---|---|---|---|---|---|---|
| IID | 0,676 / 0,736 | 0,704 / 0,740 | 0,609 / 0,601 | **0,778 / 0,814** | +0,102 | +0,074 |
| α = 1,0 | 0,654 / 0,592 | 0,684 / 0,693 | 0,588 / 0,578 | **0,731 / 0,781** | +0,077 | +0,048 |
| α = 0,1 | 0,577 / 0,521 | 0,593 / 0,582 | 0,592 / 0,542 | 0,595 / 0,611 | +0,017 | +0,001 |

Testes (`results/estatistica_e_divisoes/testes.py`, n = 12):

| Comparação (PH2 bal., clientes) | Média | IC 95% (bootstrap) | A favor | p (t) | p (Wilcoxon) |
|---|---|---|---|---|---|
| FedMD − sem digest | +0,041 | [+0,015; +0,066] | 9/12 | 0,014 | 0,027 |
| FedMD − local | +0,065 | [+0,036; +0,093] | 11/12 | 0,001 | 0,002 |
| FedMD − digest só CE | +0,105 | [+0,060; +0,148] | 10/12 | 0,001 | 0,005 |

**Leitura.** O resultado se mantém com o dobro de pares e agora passa nos dois testes: o consenso acrescenta +0,041 sobre
as rodadas sem digest (com 2 seeds era +0,056) e o efeito vem das divisões em que os clientes têm as duas classes
(+0,074 na IID, +0,048 com α = 1,0, +0,001 com α = 0,1). Os pares contrários são seed 2 com α = 1,0 e α = 0,1, e seed 1
com α = 0,1. A referência justa (0,820 / 0,831) fica acima do FedMD: por cliente, o FedMD percorre ~35% do caminho
entre o local e o teto ((0,701 − 0,636) / (0,820 − 0,636)); no ensemble, ~55% ((0,735 − 0,616) / (0,831 − 0,616)),
e na divisão IID o ensemble do FedMD (0,814) chega perto do teto. A referência varia muito entre seeds (0,90, 0,84,
0,78 e 0,77 por cliente), um reflexo do teste com 24 melanomas. Os números de 2 seeds acima (0,733 / 0,800, +0,056)
ficam como histórico; os do artigo devem passar a ser estes.
