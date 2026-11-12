# Um Estudo da Destilação de Modelos Federados para Detecção de Melanoma

Projeto final da disciplina **MC809 – Computação Distribuída**.

Este projeto avalia o **FedMD** (*Federated Learning via Model Distillation*, Li & Wang, 2019) na classificação
binária de lesões de pele (benigna × maligna) com 4 clientes, cada um com uma arquitetura de CNN diferente e
com dados privados que podem ser IID ou não IID. Nesse esquema, os clientes não compartilham pesos. Eles
trocam previsões (logits) sobre um conjunto público pequeno, e um servidor calcula o consenso dessas
previsões.

O trabalho tem quatro partes:

1. Reproduzir o FedMD original num cenário de domínio único: os dados públicos e privados vêm do mesmo
   dataset, e há clientes que só têm imagens de uma classe.
2. Identificar por que ele falha nesse cenário e propor adaptações, avaliadas uma a uma em ablações.
3. Comparar a versão adaptada com o treino só local, com o FedMD original e com um modelo centralizado, em
   frações públicas de 1% a 80%. A comparação é feita no conjunto de teste, num dataset externo (PH2) e também
   com o ensemble dos 4 clientes.
4. Testar o FedMD no cenário para o qual ele foi pensado: todo o dataset do Kaggle como conjunto público e um
   punhado de imagens de outro domínio (PH2) como dados privados de cada cliente.

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

Outras opções do `client.py` usadas nos experimentos finais: `--digest-ce-weight w` (mistura a CE no rótulo verdadeiro
ao digest), `--private-dir`, `--private-dir-fraction` e `--private-batch-size` (dados privados vindos de um dataset
externo, com todo o dataset de treino como público).

Com `--save-models-dir <pasta>`, o modelo final de cada cliente (após a última rodada) também é salvo em
`<pasta>/P<fração>_alpha_<alpha>/cliente_<i>.keras`, sem o otimizador (~0,7 MB cada). Para carregar:
`keras.models.load_model(caminho)`.

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
| 10%, 20%, 40% e 80% (`0.1`, `0.2`, `0.4`, `0.8`) | `3e-4` (escolhido para 20% e repetido nas outras, sem varredura) |

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

A mesma comparação com 10%, 40% e 80% de dados públicos (FedMD adaptado, que também dá o treino local, e
FedMD original, 3 seeds, salvando os modelos finais) está em:

```bash
results/fracoes_publicas_10_40_80/orquestrador.sh   # pula as execuções que já têm _test_metrics.csv
```

Os gráficos do relatório são gerados por `result_exploration.ipynb` (só CPU) e salvos em `results/figuras/`.

### 6. FedMD no cenário do paper: Kaggle público, PH2 privado

Todo o treino do Kaggle vira público (`--public-fractions 1.0`), e os dados privados são 40% do PH2 (80 imagens,
~20 por cliente), divididos por IID ou Dirichlet e treinados com batch 5, sem validação. O teste externo são os
outros 60% do PH2.

```bash
uv run python server.py --clients 4
uv run python client.py --public-fractions 1.0 --private-dir ph2_dataset --private-dir-fraction 0.4 \
  --private-batch-size 5 --public-init-lr 3e-4 --digest-rounds 30 --private-init-steps 300 --private-init-lr 1e-4 \
  --private-init-rehearsal --private-epochs 3 --private-steps 10 --revisit-lr 1e-4 --rehearsal-fraction 0.5 \
  --digest-lr 4e-4 --digest-steps 30 --digest-temperature 4
results/ph2_privado/orquestrador.sh   # experimento completo: FedMD adaptado e original, 2 seeds, mais o centralizado
```

### Ablações (variações sobre a configuração 3)

| Ablação | Como reproduzir | Pasta |
|---|---|---|
| Fração pública | `--public-fractions 0.01 0.05 0.1 0.2 0.4 0.8` | `results/experimento_final`, `results/fracoes_publicas_10_40_80` |
| Loss do digest: MAE × KL | sem `--digest-temperature` (MAE) × `--digest-temperature 2` ou `4` | `results/digest_kl` (1% e 5%), `results/digest_kl_10_40_80` (10%, 40% e 80%) |
| Digest ancorado nos rótulos: (1 − w) · KL + w · CE | `--digest-ce-weight 0.5` ou `1` (w = 1: controle sem destilação) | `results/digest_ce_10_40_80` |
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

Média dos 4 clientes, 3 seeds. Digest com KL (T = 4), exceto em 20%, que só tem a execução com MAE.

| Fração pública | IID | α = 1.0 | α = 0.1 | Ganho médio | Seeds × cenários com ganho |
|---|---|---|---|---|---|
| 1% | +0,006 | +0,009 | +0,013 | **+0,009** | 9/9 |
| 5% | +0,009 | +0,011 | +0,012 | **+0,011** | 9/9 |
| 10% | +0,007 | −0,002 | 0,000 | +0,002 | 6/9 |
| 20% (MAE) | −0,007 | −0,006 | −0,022 | −0,011 | 1/9 |
| 40% | −0,008 | −0,002 | −0,018 | −0,009 | 1/9 |
| 80% | −0,012 | −0,015 | −0,019 | −0,015 | 0/9 |

Origem: `results/ensemble_e_leave_one_out` (1% e 5%), `results/experimento_final` (20%) e
`results/fracoes_publicas_10_40_80` (10%, 40% e 80%). Gráfico: `results/figuras/4_ganho_por_fracao.png`.

O ganho cai à medida que a fração pública aumenta e vira negativo a partir de 20%. Os dados públicos têm rótulo
e são do mesmo domínio, então mais dados públicos deixam o `public_init` mais forte (0,66 com 1%, 0,85 com 5%,
~0,89 com 40% e ~0,90 com 80%), e sobra menos espaço para a colaboração. Isso vale mesmo quando cada cliente
fica com poucos dados próprios: com 80%, são ~480 imagens por cliente, e o treino local já chega a 0,896.

### Limite teórico e ensemble

- **Modelo centralizado** (um modelo único treinado com todo o conjunto de treino): 0,898 ± 0,004 na média das
  4 arquiteturas (0,901 na repetição com avaliação no PH2). Com 20% de dados públicos ou mais, o treino local
  já chega a ~0,89–0,90, praticamente no teto.
- **Ensemble dos clientes** (cada um prevê sozinho e a resposta é a média dos 4 logits): o ensemble dos
  clientes treinados **só localmente** empata ou supera o ensemble dos mesmos clientes depois do FedMD em
  todas as frações, no teste e no PH2. A destilação deixa os modelos parecidos entre si, e o ensemble perde a
  diversidade que o fazia bom.

| Fração pública | Teste: ensemble local | Teste: ensemble FedMD adaptado | Teste: ensemble FedMD original | PH2 bal.: ensemble local | PH2 bal.: ensemble FedMD adaptado | PH2 bal.: ensemble FedMD original |
|---|---|---|---|---|---|---|
| 5% | 0,883 | 0,884 | 0,702 | **0,659** | 0,620 | 0,555 |
| 10% | **0,892** | 0,883 | 0,746 | **0,635** | 0,604 | 0,550 |
| 40% | **0,900** | 0,887 | 0,705 | **0,594** | 0,575 | 0,552 |
| 80% | **0,903** | 0,885 | 0,715 | **0,592** | 0,544 | 0,559 |
| *Centralizado* | *modelo único 0,901; ensemble 0,904* | | | *modelo único 0,665; ensemble 0,656* | | |

(Média dos 3 α e das 3 seeds. Gráficos: `results/figuras/6_ensemble_*.png`.) Com 40% e 80%, o ensemble local
alcança um modelo único treinado com todos os dados, sem nenhuma troca de informação entre os clientes.

### Ablações que não mudaram o resultado

- **KL no lugar de MAE no digest:** +0,002 em média, sem significância, com 1% e 5% (p ≈ 0,08) e também com
  10%, 40% e 80% (+0,0021, KL melhor em 13 de 18 pares, p ≈ 0,27; `results/digest_kl_10_40_80/RESULTADO.md`).
  Nas frações maiores, as duas variantes perdem do treino local.
- **Consenso leave-one-out:** −0,0004 ± 0,0048.
- **Escassez de dados por cliente** (validação de 50% ou 70%): a linha de base quase não cai. O `private_init`
  com rehearsal continua chegando perto da capacidade das CNNs.

### Digest ancorado nos rótulos: a destilação não acrescenta nada aos rótulos públicos

O digest misturando a destilação com a entropia cruzada no rótulo verdadeiro das imagens públicas,
**(1 − w) · KL + w · CE** (`results/digest_ce_10_40_80/RESULTADO.md`, 10%, 40% e 80%, 2 seeds):

| Peso da CE | 10% | 40% | 80% | Acurácia final | Ensemble |
|---|---|---|---|---|---|
| w = 0 (só KL, FedMD) | +0,002 | −0,008 | −0,013 | 0,880–0,882 | 0,884–0,888 |
| w = 0,5 (KL + CE) | +0,017 | +0,007 | +0,006 | 0,893–0,901 | 0,897–0,902 |
| w = 1 (só CE, **sem destilação**) | **+0,020** | **+0,013** | **+0,012** | **0,897–0,905** | **0,903–0,908** |

(Ganho sobre o treino local; `results/figuras/7_*.png`.) Ancorar nos rótulos tira o FedMD da perda, mas o
controle sem destilação é o melhor em todas as frações e em 18 de 18 pares: o ganho vem de treinar mais com os
rótulos públicos, não do consenso. No PH2, nenhuma variante passa do treino local.

### Dataset externo (PH2)

5% de dados públicos, 3 seeds, média dos 4 clientes (± desvio padrão entre as seeds). No PH2, 80% das imagens
são benignas, então prever sempre "benigno" já dá 0,80 de acurácia. Por isso, a tabela também traz a acurácia
balanceada (média do recall de cada classe), que dá 0,5 para esse classificador trivial.

| Método | α | Teste (dataset de treino) | PH2: acurácia | PH2: acurácia balanceada |
|---|---|---|---|---|
| Local (public + private init) | IID | 0,879 ± 0,009 | 0,691 ± 0,064 | 0,644 ± 0,027 |
| | 1.0 | 0,871 ± 0,009 | 0,702 ± 0,072 | 0,640 ± 0,056 |
| | 0.1 | 0,857 ± 0,010 | 0,660 ± 0,066 | 0,628 ± 0,059 |
| FedMD original | IID | 0,897 ± 0,001 | 0,766 ± 0,020 | 0,646 ± 0,022 |
| | 1.0 | 0,783 ± 0,111 | 0,721 ± 0,061 | 0,616 ± 0,089 |
| | 0.1 | 0,528 ± 0,034 | 0,505 ± 0,009 | 0,503 ± 0,005 |
| FedMD adaptado | IID | 0,888 ± 0,003 | 0,745 ± 0,061 | 0,632 ± 0,054 |
| | 1.0 | 0,879 ± 0,017 | 0,756 ± 0,041 | 0,611 ± 0,062 |
| | 0.1 | 0,865 ± 0,004 | 0,718 ± 0,036 | 0,605 ± 0,035 |
| Centralizado | — | 0,901 ± 0,003 | 0,795 ± 0,017 | 0,665 ± 0,021 |

- **No dataset de treino**, o resultado repete o do experimento final: o FedMD adaptado fica ~+0,01 acima do treino
  local em todos os α, e o FedMD original só funciona no caso IID (com α = 0.1, os clientes colapsam para ~0,50).
- **No PH2, nenhum método generaliza bem**: a acurácia balanceada fica entre 0,60 e 0,67 para todos, inclusive o
  centralizado, e as diferenças entre local, FedMD adaptado e centralizado ficam dentro do desvio padrão. O PH2 tem
  só 40 imagens malignas, então um melanoma a mais ou a menos muda a sensibilidade em 0,025.
- **O FedMD desloca as previsões para "benigno" no PH2**: a acurácia sobe (0,66–0,70 → 0,72–0,76) e a acurácia
  balanceada cai um pouco (−0,01 a −0,03). Em detecção de melanoma, isso significa mais falsos negativos.

Arquivos: `results/test_models/acuracia_final.csv` (por método, α, seed e modelo, incluindo o ensemble) e
`results/test_models/resumo.csv`. Os gráficos por cliente estão em `results/figuras/1_*.png` (gerados por
`result_exploration.ipynb`).

**Em outras frações públicas** (média dos 4 clientes, 3 seeds, média dos 3 α; `results/figuras/5_*.png`):

| Fração pública | Teste: local | Teste: FedMD adaptado | Teste: FedMD original | PH2 bal.: local | PH2 bal.: FedMD adaptado | PH2 bal.: FedMD original |
|---|---|---|---|---|---|---|
| 5% | 0,869 | **0,878** | 0,736 | **0,637** | 0,616 | 0,588 |
| 10% | 0,877 | **0,878** | 0,774 | **0,631** | 0,605 | 0,590 |
| 40% | **0,891** | 0,882 | 0,733 | **0,597** | 0,571 | 0,579 |
| 80% | **0,896** | 0,881 | 0,722 | 0,593 | 0,544 | **0,600** |

- No PH2, **todos os métodos pioram quando a fração pública aumenta**: com mais dados do mesmo domínio, os
  modelos ficam mais presos a ele e deixam passar mais melanomas.
- O FedMD adaptado **fica abaixo do treino local na acurácia balanceada do PH2 em todas as frações**, e a
  diferença cresce com a fração (−0,02 com 5% e −0,05 com 80%).
- O FedMD original tem a melhor acurácia balanceada no PH2 nos casos IID com 40% e 80% (~0,65), mas colapsa nos
  casos não IID.

### FedMD no cenário do paper: Kaggle público, PH2 privado

Todo o Kaggle como conjunto público e 80 imagens do PH2 como dados privados (~20 por cliente, batch 5), 2 seeds × 3 α,
teste do PH2 nas outras 120 imagens (`results/ph2_privado/RESULTADO.md`, `results/figuras/8_ph2_privado_troca.png`):

| Método | Teste do Kaggle | PH2 bal. (clientes) | PH2 bal. (ensemble) |
|---|---|---|---|
| Só `public_init` (Kaggle) | **0,903** | 0,605 | 0,617 |
| Local (+ `private_init` no PH2) | 0,879 | 0,632 | 0,612 |
| FedMD adaptado | 0,850 | 0,664 | 0,677 |
| FedMD original | 0,610 | **0,733** | **0,800** |
| Centralizado (Kaggle + os 80 do PH2; modelo único / ensemble) | 0,895 / 0,901 | 0,653 | 0,630 |
| FedMD do cenário padrão, sem PH2 no treino (mesmas 120 imagens) | — | 0,54–0,64 | |

Por divisão dos dados privados (média das 2 seeds; PH2 balanceada dos clientes / do ensemble, e teste do Kaggle):

| α | Local | FedMD adaptado | FedMD original |
|---|---|---|---|
| IID | 0,630 / 0,651 — Kaggle 0,879 | 0,679 / 0,695 — Kaggle 0,843 | **0,807 / 0,873** — Kaggle 0,625 |
| 1.0 | 0,646 / 0,589 — Kaggle 0,878 | 0,660 / 0,674 — Kaggle 0,844 | **0,772 / 0,845** — Kaggle 0,655 |
| 0.1 | 0,620 / 0,596 — Kaggle 0,881 | **0,653 / 0,662** — Kaggle 0,862 | 0,621 / 0,682 — Kaggle 0,551 |

Com só ~20 imagens por cliente, mesmo a divisão IID (aleatória) deixa clientes sem nenhum melanoma; com α = 0.1, há
clientes com 25 imagens só benignas ou 12 só malignas.

- **Quando os clientes têm as duas classes (IID e α = 1.0), a colaboração ajuda no domínio privado.** Os dois FedMD
  superam o local e o centralizado no PH2, e o FedMD original fica muito à frente (+0,13 a +0,18 por cliente sobre o
  local; 0,85–0,87 no ensemble, o melhor resultado de todo o estudo).
- **Com clientes de uma classe só (α = 0.1), a vantagem do FedMD original desaparece**: ele empata com o local por
  cliente (0,621 contra 0,620) e perde ainda mais no Kaggle (0,55). O mesmo problema do cenário padrão volta: o revisit
  agressivo especializa cada cliente na classe que ele tem. O FedMD adaptado, que se especializa menos, fica à frente
  (0,653).
- **O centralizado não aproveita os dados privados**: as 80 imagens são menos de 1% do seu treino.
- **Há uma troca entre os domínios.** Quanto mais o método se especializa no PH2, mais esquece o Kaggle.

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
3. **Com essas adaptações, o FedMD melhora cada cliente em cerca de +0,01 só quando há pouco dado público**
   (1% e 5%, positivo em 18 de 18 combinações). Com 10%, empata. De 20% a 80%, a destilação atrapalha (−0,009
   a −0,015). A hipótese de que mais dados públicos e menos dados por cliente favoreceriam a colaboração não se
   confirmou: como os dados públicos têm rótulo e são do mesmo domínio, eles deixam a inicialização local mais
   forte.
4. **O que compensa a falta de dados públicos é o treino privado, não a colaboração.** Contra o `public_init`
   sozinho, o ganho parece crescer quando a fração pública diminui. Contra a linha de base justa (treino
   local completo), ele fica estável em ~+0,01.
5. **Treinar os clientes de forma independente e combinar as previsões é tão bom ou melhor que o FedMD, e
   mais barato.** O ensemble dos clientes treinados só localmente empata ou supera o ensemble pós-FedMD em
   todas as frações, no teste e no PH2, e com 40–80% alcança um modelo único centralizado. Isso está em linha
   com a literatura de ensembles em aprendizado federado (Guha et al., 2019; Allouah et al., 2024). O FedMD só
   se justifica quando cada cliente precisa de um modelo próprio para inferência local, e mesmo assim só com
   pouco dado público.
6. **Fora do domínio (PH2), nenhum método generaliza bem, e o FedMD piora a sensibilidade.** A acurácia
   balanceada fica entre 0,54 e 0,67 para todos, inclusive o centralizado. O FedMD desloca as previsões para
   "benigno": a acurácia total sobe e a balanceada cai, o que em detecção de melanoma significa mais falsos
   negativos.
7. **Nem a loss do digest nem a forma de calcular o consenso foram o gargalo.** O que limita o ganho é a
   quantidade de dados no digest e a proximidade do teto de capacidade das arquiteturas.
8. **Com dados públicos rotulados e do mesmo domínio, a destilação não acrescenta nada aos próprios rótulos.**
   Trocar o consenso pelo rótulo verdadeiro no digest (o controle sem destilação) dá o melhor resultado em todas as
   frações (+0,012 a +0,020 sobre o local, contra −0,013 a +0,002 da KL pura).
9. **No cenário para o qual o FedMD foi pensado, ele funciona, desde que cada cliente tenha as duas classes.** Com todo
   o Kaggle como conjunto público e poucas imagens de outro domínio como dados privados, os dois FedMD adaptam os
   clientes ao PH2 melhor que o treino local e que o centralizado nas divisões IID e α = 1.0; o FedMD original chega a
   0,85–0,87 no ensemble, ao custo de cair para ~0,63 no domínio público. Com clientes de uma classe só (α = 0.1), a
   vantagem do original desaparece, e só o adaptado, mais conservador, continua acima do local. O valor do FedMD
   aparece quando os dados privados são de outro domínio, escassos e com as duas classes; quando público e privado
   são do mesmo domínio, a inicialização pública já resolve.

---

## Limitações

- **Os hiperparâmetros foram escolhidos olhando o conjunto de teste.** A validação privada dos clientes que
  só têm uma classe não serve para essa escolha. As grades foram pequenas (2 ou 3 valores por parâmetro).
- **Usamos 3 seeds.** O desvio padrão fica entre ~0,005 e 0,01 por cenário, então diferenças menores que
  ~0,01 não são conclusivas.
- **O controle sem destilação foi feito só com 2 seeds** (`results/digest_ce_10_40_80`, w = 1). Ele indica que o
  ganho do FedMD adaptado vem dos rótulos públicos (no digest e no rehearsal), e não do consenso.
- **O cenário PH2 privado usa só 2 seeds e 24 melanomas no teste**, e o número de passos do treino privado não
  foi ajustado para 20 imagens por cliente.
- **O estudo cobre só 4 clientes, CNNs pequenas e imagens de 224×224.** O teto de 0,898 limita o espaço para
  ganhos.
- **O ensemble tem custo de comunicação na inferência**, que não foi medido.
- **O PH2 usado veio de um espelho não oficial**, e tem só 40 melanomas: a acurácia balanceada varia muito
  entre seeds (até ±0,09).
- **Com 10%, 40% e 80%, o lr do `public_init` não foi varrido** (usamos o de 20%).
- **Travamentos ocasionais:** um cliente às vezes fica preso com a GPU ociosa. Em uma execução da comparação
  KL × MAE (`MAE_P40_seed1`), o α = 0.1 precisou ser refeito sem XLA (`--no-jit`); os logs dos travamentos
  ficam na pasta do experimento. Os orquestradores mais recentes têm um watchdog que reinicia a execução
  travada, e o de `ph2_privado` roda um α por processo, porque o PC caiu duas vezes na troca de configuração
  com a GPU perto do limite.

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
| `results/test_models/` | Avaliação no PH2: FedMD adaptado, treino local, FedMD original e centralizado (5% público) |
| `results/fracoes_publicas_10_40_80/` | Mesma comparação com 10%, 40% e 80% de dados públicos, 3 seeds, com avaliação no PH2 e os modelos finais em `modelos/` |
| `results/digest_kl_10_40_80/` | KL × MAE no digest com 10%, 40% e 80%, 2 seeds (lado MAE; o lado KL é o de `fracoes_publicas_10_40_80`) |
| `results/digest_ce_10_40_80/` | Digest ancorado nos rótulos, (1 − w) · KL + w · CE, com w = 0,5 e 1 (controle sem destilação), 2 seeds (`RESULTADO.md`) |
| `results/ph2_privado/` | FedMD no cenário do paper: Kaggle público, 40% do PH2 privado; FedMD adaptado, original e centralizado, 2 seeds, mais a reavaliação dos modelos salvos nas mesmas imagens (`RESULTADO.md`) |
| `results/figuras/` | Figuras do relatório, geradas por `result_exploration.ipynb` |

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
- N. Guha, A. Talwalkar, V. Smith. **One-Shot Federated Learning.** arXiv:1902.11175, 2019.
- Y. Allouah et al. **Revisiting Ensembling in One-Shot Federated Learning.** NeurIPS, 2024. arXiv:2411.07182.
- T. Lin, L. Kong, S. U. Stich, M. Jaggi. **Ensemble Distillation for Robust Model Fusion in Federated Learning.**
  NeurIPS, 2020. arXiv:2006.07242.
- B. Lakshminarayanan, A. Pritzel, C. Blundell. **Simple and Scalable Predictive Uncertainty Estimation using
  Deep Ensembles.** NeurIPS, 2017.
