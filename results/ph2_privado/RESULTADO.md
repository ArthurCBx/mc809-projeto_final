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
