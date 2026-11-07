# Um Estudo da Destilação de Modelos Federados para Detecção de Melanoma

Projeto final da disciplina **MC809 – Computação Distribuída**.

Este projeto avalia o **FedMD** (*Federated Learning via Model Distillation*, Li & Wang, 2019) na classificação
binária de lesões de pele (benigna × maligna) com 4 clientes, cada um com uma arquitetura de CNN diferente e
com dados privados que podem ser IID ou não IID. Nesse esquema, os clientes não compartilham pesos. Eles
trocam previsões (logits) sobre um conjunto público pequeno, e um servidor calcula o consenso dessas
previsões.

O trabalho tem três partes:

1. Reproduzir o FedMD original num cenário de domínio único: os dados públicos e privados vêm do mesmo
   dataset, e há clientes que só têm imagens de uma classe.
2. Identificar por que ele falha nesse cenário e propor adaptações, avaliadas uma a uma em ablações.
3. Comparar a versão adaptada com o treino só local, com o FedMD original e com um modelo centralizado. A
   comparação é feita no conjunto de teste e num dataset externo (PH2).

---

## Sumário

- [Visão geral do método](#visão-geral-do-método)
- [Dados](#dados)
- [Implementação](#implementação)
- [Instalação e preparação](#instalação-e-preparação)
- [Um experimento simples](#um-experimento-simples)
- [Configurações apresentadas no estudo](#configurações-apresentadas-no-estudo)
- [Principais resultados](#principais-resultados)
- [Conclusões](#conclusões)
- [Limitações](#limitações)
- [Estrutura dos resultados](#estrutura-dos-resultados)
- [Referências](#referências)

---

## Visão geral do método

Cada experimento roda 4 clientes (actors do Ray, que compartilham uma GPU) e um servidor de consenso (gRPC).
O treino tem quatro etapas:

| Etapa | O que acontece | Parâmetros principais |
|---|---|---|
| `public_init` | Cada cliente treina no conjunto público rotulado (transfer learning), com early stopping | `--public-init-lr` |
| `private_init` | Cada cliente treina nos próprios dados privados | `--private-init-steps`, `--private-init-lr`, `--private-init-epochs`, `--private-init-rehearsal` |
| Rodadas federadas | A cada rodada, um **digest** seguido de um **revisit** | `--digest-rounds` |
| ↳ digest | Os clientes enviam ao servidor os logits sobre 80% do conjunto público (sorteados a cada rodada). O servidor devolve o consenso, e cada cliente treina para se aproximar dele (MAE ou KL) | `--digest-steps`, `--digest-lr`, `--digest-temperature` |
| ↳ revisit | Cada cliente volta a treinar nos dados privados, opcionalmente misturando amostras públicas rotuladas (*rehearsal*) | `--private-epochs`, `--private-steps`, `--revisit-lr`, `--rehearsal-fraction` |

Os dados privados são divididos entre os clientes de forma IID ou não IID, por Dirichlet com parâmetro α
(`--alphas`). Com α = 0.1 e duas classes, vários clientes ficam com imagens de uma classe só. Ao fim de cada
etapa e de cada rodada, cada cliente é avaliado no conjunto de teste. O mesmo vale para o **ensemble** (a média
dos logits dos 4 clientes) e, se configurado, para um dataset externo.

---

## Dados

| Dataset | Uso | Origem |
|---|---|---|
| [Melanoma Skin Cancer Dataset of 10000 Images](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images) | Treino: 9.605 imagens (5.000 benignas, 4.605 malignas), divididas em público e privado. Teste: 1.000 imagens (balanceadas) | Kaggle |
| [PH²](https://www.fc.up.pt/addi/ph2%20database.html) | Avaliação externa: 200 imagens dermoscópicas (160 benignas, entre nevos comuns e atípicos, e 40 melanomas) | Espelho [Shah1st/PH2](https://huggingface.co/datasets/Shah1st/PH2) no Hugging Face |

> **Licença do PH2:** a distribuição oficial exige cadastro no site do ADDI e proíbe redistribuição. O espelho
> usado aqui não é oficial. Para uso acadêmico, prefira a fonte oficial e cite Mendonça et al. (2013).

A fração pública (`--public-fractions`) é sorteada de forma estratificada do conjunto de treino. O restante é
dividido entre os 4 clientes. Cada cliente separa `--val-fraction` (padrão 0,2) dos próprios dados para a
validação do early stopping.

---

## Implementação

| Arquivo | Para que serve |
|---|---|
| `client.py` | Ponto de entrada dos experimentos. Define as 4 arquiteturas de CNN (`CLIENT_ARCHITECTURES`) e o actor `Client` do Ray, que treina, faz o digest, faz o revisit e avalia. Também orquestra a grade de experimentos (frações públicas × α) e grava os resultados em CSV. |
| `server.py` | Servidor gRPC de consenso. Espera os logits de todos os clientes em cada rodada, calcula o consenso (média uniforme ou ponderada pela acurácia pública; `loo` exclui o próprio cliente) e devolve o resultado a cada um. |
| `dataset_setup.py` | Baixa o dataset do Kaggle, lista os arquivos, separa público e privado e divide o privado entre os clientes (IID ou Dirichlet). Também monta os `tf.data.Dataset`. `list_dir` lista datasets externos no formato `pasta/<classe>/<imagem>`. |
| `ph2_setup.py` | Converte o PH2 (parquet do Hugging Face) para `ph2_dataset/{benign,malignant}/*.jpg`. |
| `proto_config/` | Definição gRPC (`message.proto`) e o código gerado (`message_pb2*.py`). |
| `results/*/` | Scripts de orquestração (`orquestrador.sh`), análise (`tabelas.py`, `resumo.py`) e os resultados de cada experimento. |

### Saídas de uma execução do `client.py`

Os arquivos ficam em `results/` e recebem **novas linhas a cada execução**. Renomeie ou mova esses arquivos
entre experimentos diferentes; os orquestradores fazem isso automaticamente.

| Arquivo | Conteúdo |
|---|---|
| `test_metrics.csv` | Acurácia e perda de cada cliente no teste, por etapa e rodada |
| `train_history.csv` | Histórico de treino por época (loss, accuracy, val_loss, val_accuracy) |
| `ensemble_metrics.csv` | Acurácia do ensemble dos 4 clientes e a média dos clientes, por etapa e rodada |
| `extra_test_metrics.csv` | Acurácia e acurácia balanceada no dataset externo, só quando `--extra-test-dir` é usado |

---

## Instalação e preparação

Requisitos: Python ≥ 3.12, [uv](https://docs.astral.sh/uv/) e uma GPU NVIDIA. Os experimentos rodaram numa RTX
5070 com 12 GB, no WSL2 com 24 GB de RAM.

```bash
uv sync                                   # instala as dependências do pyproject.toml

# Dataset de treino (precisa de credenciais do Kaggle: `kaggle auth login` ou KAGGLE_API_TOKEN)
uv run python dataset_setup.py            # cria ./melanoma_cancer_dataset/{train,test}/{benign,malignant}

# PH2 (opcional, só para a avaliação externa): baixe o parquet do espelho e converta
curl -L -o ph2.parquet "https://huggingface.co/datasets/Shah1st/PH2/resolve/main/data/train-00000-of-00001-9f7fbbd66f09fd25.parquet"
uv run --with pyarrow python ph2_setup.py ph2.parquet   # cria ./ph2_dataset/{benign,malignant}
```

O código gRPC já está gerado em `proto_config/`. Para recompilar depois de alterar o `message.proto`, rode
dentro de `proto_config/`:

```bash
python -m grpc_tools.protoc -I=. --python_out=. --grpc_python_out=. message.proto
# depois, em message_pb2_grpc.py, troque "import message_pb2" por "import proto_config.message_pb2"
```

---

## Um experimento simples

Você precisa de dois terminais, os dois na raiz do projeto. O servidor tem que estar rodando antes do cliente.

```bash
# terminal 1: servidor de consenso para 4 clientes
uv run python server.py --clients 4

# terminal 2: FedMD adaptado com 5% de dados públicos, divisão IID, 30 rodadas
uv run python client.py --public-fractions 0.05 --alphas iid --public-init-lr 1e-3 --digest-rounds 30 \
  --private-init-steps 300 --private-init-lr 1e-4 --private-init-rehearsal \
  --private-epochs 3 --private-steps 10 --revisit-lr 1e-4 --rehearsal-fraction 0.5 \
  --digest-lr 4e-4 --digest-steps 30 --digest-temperature 4
```

Cada configuração (uma fração pública × um α) leva de ~10 a 15 minutos na GPU citada acima. O log mostra a
acurácia de teste de cada cliente e do ensemble em cada etapa. Os resultados vão para `results/*.csv`.

**Observações práticas:**

- Os 4 clientes dividem a GPU. Por padrão, cada um usa 2,5 GB (`--gpu-memory-mb`) com batch 32.
- Se aparecer `ResourceExhaustedError` (picos de memória da compilação XLA, vistos com conjuntos muito
  pequenos), use `--no-jit` e `--gpu-memory-mb 2800`.
- Para várias seeds e configurações, use os scripts `results/*/orquestrador.sh` como modelo. Eles reiniciam
  o servidor antes de cada execução, aplicam um timeout e movem os CSVs para a pasta do experimento.

---

## Configurações apresentadas no estudo

Todos os comandos supõem um servidor no ar (`uv run python server.py --clients 4`, com as opções indicadas)
e geram as 3 divisões, IID, α = 1.0 e α = 0.1 (o padrão de `--alphas`). Para repetir com outras seeds, use
`--seed 42`, `--seed 1` e `--seed 2`.

### 1. FedMD original (replicação)

Consenso uniforme, MAE nos logits, `private_init` até convergir, revisit de 3 épocas completas com lr 1e-3,
sem rehearsal e sem limite de passos. São 20 rodadas, como no paper.

```bash
uv run python server.py --clients 4 --weighting uniform
uv run python client.py --public-fractions 0.05 --public-init-lr 1e-3 --digest-rounds 20 \
  --private-init-epochs 50 --private-epochs 3 --revisit-lr 1e-3 --digest-lr 1e-3
```

### 2. Treino só local (public + private init, sem colaboração)

É a linha de base do estudo. Ela corresponde à etapa `private_init` do FedMD adaptado, e para obtê-la basta
zerar as rodadas federadas:

```bash
uv run python server.py --clients 4
uv run python client.py --public-fractions 0.05 --public-init-lr 1e-3 --digest-rounds 0 \
  --private-init-steps 300 --private-init-lr 1e-4 --private-init-rehearsal --rehearsal-fraction 0.5
```

### 3. FedMD adaptado (melhor configuração)

```bash
uv run python server.py --clients 4                       # consenso ponderado pela acurácia pública (padrão)
uv run python client.py --public-fractions 0.05 --public-init-lr 1e-3 --digest-rounds 30 \
  --private-init-steps 300 --private-init-lr 1e-4 --private-init-rehearsal \
  --private-epochs 3 --private-steps 10 --revisit-lr 1e-4 --rehearsal-fraction 0.5 \
  --digest-lr 4e-4 --digest-steps 30 --digest-temperature 4
```

O lr do `public_init` foi escolhido por fração pública, olhando o resultado após o `private_init`:

| Fração pública | `--public-init-lr` |
|---|---|
| 1% (`0.01`) | `3e-4` |
| 5% (`0.05`) | `1e-3` |
| 20% (`0.2`) | `3e-4` |

### 4. Modelo centralizado (limite teórico)

As 4 arquiteturas treinadas com todo o conjunto de treino, com lr 1e-3, batch 32, aumentação e early stopping.

```bash
uv run python results/limite_teorico/centralizado.py 42 1 2          # avalia no teste original
uv run python results/test_models/centralizado_ph2.py 42 1 2   # avalia no teste original e no PH2
```

### 5. Avaliação no dataset externo (PH2)

Acrescente `--extra-test-dir ph2_dataset` a qualquer um dos comandos acima. O experimento completo do
relatório (FedMD adaptado, treino local, FedMD original e centralizado, com 5% de dados públicos e 3 seeds)
está em:

```bash
results/test_models/orquestrador.sh                 # gera results/test_models/acuracia_final.csv e resumo.csv
```

### Ablações (variações sobre a configuração 3)

| Ablação | Como reproduzir | Pasta |
|---|---|---|
| Fração pública | `--public-fractions 0.01 0.05 0.2` (0,4 e 0,8 nas primeiras ablações) | `results/experimento_final` |
| Loss do digest: MAE × KL | sem `--digest-temperature` (MAE) × `--digest-temperature 2` ou `4` | `results/digest_kl` |
| Ponderação do consenso | `server.py --weighting uniform` × `accuracy` | `results/04_consenso_ponderado` |
| Consenso leave-one-out | `server.py --consensus loo` | `results/ensemble_e_leave_one_out` |
| lr e passos do digest | `--digest-lr 1e-3/4e-4`, `--digest-steps 30` | `results/09b_digest_lr_4e-4`, `results/09c_digest_30_passos` |
| lr do revisit | `--revisit-lr 1e-3/1e-4/5e-5` | `results/09a_revisit_lr_1e-4` |
| Rehearsal no `private_init` | com e sem `--private-init-rehearsal` | `results/09d_rehearsal_private_init` |
| Escassez de dados por cliente | `--val-fraction 0.5` ou `0.7` | `results/escassez_de_dados` |
| Número de rodadas | `--digest-rounds 60` (revisit com lr 5e-5) | `results/03_revisit_lr_5e-5` |

---

## Principais resultados

Os números abaixo são acurácias no conjunto de teste do dataset de treino, em média ± desvio padrão de 3
seeds, salvo indicação. A **linha de base** é o treino local (`public_init` + `private_init`). O **ganho** do
FedMD é a média das rodadas 26–30 menos a linha de base.

### FedMD original × FedMD adaptado (casos não IID, α = 0.1)

No FedMD original, os clientes com dados de uma classe só esquecem a outra classe no treino privado e
colapsam para 0,50. O consenso não consegue recuperá-los.

| Fração pública / α | FedMD original | FedMD adaptado |
|---|---|---|
| 10% / 0.1 | 0,627 | 0,842 |
| 40% / 0.1 | 0,503 | 0,832–0,873 |
| 80% / 0.1 | 0,475 | 0,854–0,877 |

(Uma seed por configuração, primeiras ablações com 10%, 40% e 80% de dados públicos.)

### Ganho do FedMD adaptado sobre o treino local

| Fração pública | IID | α = 1.0 | α = 0.1 | Ganho médio | Seeds × cenários com ganho |
|---|---|---|---|---|---|
| 1% | +0,009 | +0,006 | +0,014 | +0,010 | 8/9 |
| 5% | +0,011 | +0,011 | +0,008 | +0,010 | 9/9 |
| 20% | −0,007 | −0,006 | −0,022 | −0,011 | 1/9 |

Origem dos números: `results/experimento_final` (digest com MAE).

### Limite teórico e ensemble

- **Modelo centralizado:** 0,898 ± 0,004, na média das 4 arquiteturas. Com 20% de dados públicos, o treino
  local já chega a 0,894 no caso IID, praticamente no teto.
- **Ensemble dos clientes sem colaboração** (média dos logits após o `private_init`): ganha **+0,020** sobre o
  cliente médio, o dobro do ganho do FedMD por cliente (+0,010). Com 1% de dados públicos, o FedMD **reduz** a
  acurácia do ensemble (−0,004 a −0,013), porque deixa os modelos parecidos entre si. Com 5%, o efeito é neutro.

### Ablações que não mudaram o resultado

- **KL no lugar de MAE no digest:** +0,002 em média, sem significância (p ≈ 0,08).
- **Consenso leave-one-out:** −0,0004 ± 0,0048.
- **Escassez de dados por cliente** (validação de 50% ou 70%): a linha de base quase não cai. O `private_init`
  com rehearsal continua chegando perto da capacidade das CNNs.

### Dataset externo (PH2)

*Experimento em andamento.* Os resultados vão ficar em `results/test_models/acuracia_final.csv` (por método,
α, seed e modelo) e em `results/test_models/resumo.csv`. No PH2, 80% das imagens são benignas, então prever
sempre "benigno" já dá 0,80 de acurácia. Por isso, os resultados também trazem a acurácia balanceada, que é a
média do recall de cada classe.

---

## Conclusões

1. **O FedMD original não funciona neste cenário.** Com os dados públicos e privados vindo do mesmo domínio e
   clientes que só têm uma classe, o treino privado com lr 1e-3 apaga o conhecimento do `public_init`, e os
   clientes colapsam para 0,50. A distância em relação ao paper vem do cenário, não do método: no paper, o
   conjunto público é de outra tarefa, e todo participante vê todas as classes.
2. **Três adaptações resolvem o colapso:**
   - `private_init` com lr baixo (1e-4) e um número fixo de passos;
   - *rehearsal*, que mistura amostras públicas rotuladas no treino privado;
   - equilíbrio entre o digest e o revisit: lr 4e-4 e 30 passos no digest, para que a destilação não domine
     quando há muitos dados públicos.
3. **Com essas adaptações, o FedMD melhora cada cliente em cerca de +0,01 sempre que a linha de base está
   longe do teto** (de 1% a 5% de dados públicos). O maior ganho é no caso mais desbalanceado (+0,014 com
   1% e α = 0.1). Quando a linha de base já está perto do limite (20% de dados públicos), a destilação
   atrapalha.
4. **O que compensa a falta de dados públicos é o treino privado, não a colaboração.** Contra o `public_init`
   sozinho, o ganho parece crescer quando a fração pública diminui. Contra a linha de base justa (treino
   local completo), ele fica estável em ~+0,01.
5. **Se a inferência pode combinar os modelos, um ensemble de clientes independentes é melhor e mais barato
   que o FedMD.** O FedMD se justifica quando cada cliente precisa de um modelo próprio para inferência local,
   que é o cenário federado típico.
6. **Nem a loss do digest nem a forma de calcular o consenso foram o gargalo.** O que limita o ganho é a
   quantidade de dados no digest e a proximidade do teto de capacidade das arquiteturas.

---

## Limitações

- **Os hiperparâmetros foram escolhidos olhando o conjunto de teste.** A validação privada dos clientes que
  só têm uma classe não serve para essa escolha. As grades foram pequenas (2 ou 3 valores por parâmetro).
- **Usamos 3 seeds.** O desvio padrão fica entre ~0,005 e 0,01 por cenário, então diferenças menores que
  ~0,01 não são conclusivas.
- **Falta um controle "rehearsal sem digest".** Parte do ganho do FedMD adaptado pode vir do rehearsal com
  rótulos públicos, e não da destilação.
- **O estudo cobre só 4 clientes, CNNs pequenas e imagens de 224×224.** O teto de 0,898 limita o espaço para
  ganhos.
- **O ensemble tem custo de comunicação na inferência**, que não foi medido.
- **O PH2 usado veio de um espelho não oficial.**

---

## Estrutura dos resultados

As pastas numeradas seguem a sequência do desenvolvimento incremental, na mesma ordem da tabela de etapas do
relatório. As de 01 a 05 usam frações públicas de 10%, 40% e 80%, com uma seed.

| Pasta | Etapa | Conteúdo |
|---|---|---|
| `results/01_replicacao_fedmd/` | 1 | Replicação do FedMD: consenso uniforme, revisit em 3 épocas completas, lr 1e-3 em tudo. Inclui o log do servidor das etapas 1 a 3 |
| `results/02_revisit_1_epoca/` | 2 | Revisit com 1 época |
| `results/03_revisit_lr_5e-5/` | 3 | Revisit com lr 5e-5, em 20 e 60 rodadas |
| `results/04_consenso_ponderado/` | 4 | Consenso ponderado pela acurácia pública |
| `results/05_limite_passos_revisit_private_init/` | 5 | Revisit limitado a 10 passos e `private_init` a 5 épocas. Inclui a execução retomada depois de um travamento |
| `results/06_private_init_10_passos_lr_2e-5/` | 6 | Varredura de passos e lr do `private_init` (só as etapas iniciais). Inclui uma execução descartada |
| `results/07_revisit_fraco_sem_rehearsal/` | 7 | Revisit com lr 2e-5 e sem rehearsal, 60 rodadas |
| `results/08_rehearsal_no_revisit/` | 8 | Rehearsal de 50% no revisit. O `RELATORIO.md` cobre as etapas 6 a 8 |
| `results/09a_revisit_lr_1e-4/` | 9 | Revisit com lr 1e-4 (`RESULTADO.md`) |
| `results/09b_digest_lr_4e-4/` | 9 | Digest com lr 4e-4 (`RESULTADO.md`) |
| `results/09c_digest_30_passos/` | 9 | Digest limitado a 30 passos por rodada (`RESULTADO.md`) |
| `results/09d_rehearsal_private_init/` | 9 | Rehearsal no `private_init`, com 2 seeds (`RELATORIO.md`) |

Avaliações da configuração final:

| Pasta | Conteúdo |
|---|---|
| `results/fracoes_publicas_5_e_20/` | Frações públicas de 5% e 20% e escolha dos hiperparâmetros (`RELATORIO.md`) |
| `results/experimento_final/` | **Experimento final**: frações de 1%, 5% e 20% com 3 seeds, mais o resumo de todos os extras (`RELATORIO.md`) |
| `results/limite_teorico/` | Modelo centralizado (limite teórico) |
| `results/digest_kl/` | KL × MAE no digest (`RESULTADO.md`) |
| `results/ensemble_e_leave_one_out/` | Ensemble dos clientes e consenso leave-one-out (`RELATORIO.md`) |
| `results/escassez_de_dados/` | Validação de 50% e 70% (`RELATORIO.md`) |
| `results/test_models/` | Avaliação no PH2: FedMD adaptado, treino local, FedMD original e centralizado |

Os PNGs `results/acc_increase_table*.png` continuam na raiz de `results/`.

O relatório que consolida tudo é o `results/experimento_final/RELATORIO.md`.

---

## Referências

- D. Li, J. Wang. **FedMD: Heterogenous Federated Learning via Model Distillation.** NeurIPS Workshop on
  Federated Learning for Data Privacy and Confidentiality, 2019. [arXiv:1910.03581](https://arxiv.org/abs/1910.03581).
  Código de referência: [diogenes0319/FedMD_clean](https://github.com/diogenes0319/FedMD_clean).
- T. Mendonça, P. M. Ferreira, J. S. Marques, A. R. S. Marcal, J. Rozeira. **PH² – A dermoscopic image database
  for research and benchmarking.** 35th Annual International Conference of the IEEE EMBS, 2013.
- G. Hinton, O. Vinyals, J. Dean. **Distilling the Knowledge in a Neural Network.** arXiv:1503.02531, 2015.
- M. H. Javed. **Melanoma Skin Cancer Dataset of 10000 Images.** Kaggle, 2022.
