# Respostas para fechar o artigo (figuras e conferências)

Tudo aqui sai de resultados que já estavam em `results/`. Nenhum experimento novo foi rodado, e o processamento usou só
CPU (`CUDA_VISIBLE_DEVICES=`). Os números da Parte B saem de `results/respostas_artigo.py`; para reproduzir, rode na
raiz `CUDA_VISIBLE_DEVICES= PYTHONPATH=. python results/respostas_artigo.py`. Desvio padrão = amostral (ddof = 1),
salvo indicação.

---

## Parte A: figuras

Todas as figuras de `results/figuras/` foram regeneradas pelo `result_exploration.ipynb`:

- **A1.** As figuras de três painéis agora têm `figsize=(11, 3,2)`, fonte 14 e a legenda abaixo dos painéis, e não mais
  à direita, que roubava largura. A paleta e o estilo continuam os mesmos. Eixos, títulos e legendas estão em
  português, com vírgula decimal no eixo y e nos rótulos de α. A terminologia segue a do artigo: validação, FedMD
  replicado, treinamento local e teto centralizado. Para não colidirem em escala log, os rótulos de fração ficaram
  `1 5 10 40 80` com o eixo "fração pública (%)", e as etapas ganharam mais espaço entre `pub` e `priv`.
  A `2_etapa_06` (um painel) passou para 9 × 3,8.
- **A2.** `4_ganho_por_fracao.png`: só 1%, 5%, 10%, 40% e 80%, nas duas linhas (KL e MAE). A fonte de 20% foi retirada
  do notebook.
- **A3.** `7_ce_ganho.png`: inclui 1% e 5% (`results/digest_ce_1_5/CE1.0_*` para w = 1 e
  `ensemble_e_leave_one_out/all_P01_*` e `all_P05_*` para w = 0, 3 seeds); a linha de w = 0,5 começa em 10%.
  **`7_ce_ph2.png`: as execuções de w = 1 com 1% e 5% gravaram o PH2 (`*_extra_test_metrics.csv`), mas as de w = 0
  (`all_P01_*`, `all_P05_*`) não.** Na figura do PH2, portanto, a linha w = 1 vai de 1% a 80% e as linhas w = 0 e
  w = 0,5 começam em 10%. A referência do treinamento local nessa figura vem do `private_init` das execuções w = 0
  (10/40/80%) e das execuções w = 1 (1% e 5%), já que o `private_init` não depende do digest. (O pedido mandava
  registrar isso em B10, mas o anoto aqui, porque B10 trata do custo de comunicação.)
- **A4.** Figura nova `9_ph2_privado_controles.png`, em duas linhas de painéis (acurácia balanceada no PH2 em cima,
  validação embaixo) × 3 divisões. Cada variante tem duas barras (média dos clientes e ensemble), com desvio entre as
  seeds 42 e 1. As linhas tracejadas são o teto centralizado deste cenário, na cor da barra que cada uma referencia
  (modelo único para a média dos clientes, ensemble para o ensemble). Os rótulos do eixo x são curtos ("Local", "Sem
  digest", "Só CE", "FedMD replicado"); a legenda da figura no artigo deve dizer que "Só CE" é o digest só com CE
  (w = 1).

---

## Parte B: conferências

### B1. Dispersão entre clientes (Tabela 7)

**A coluna "Dispersão" é a amplitude (máx − mín) entre os 4 clientes**, não o desvio padrão. Recalculada com
3 seeds × 3 α (9 execuções por fração), FedMD adaptado com KL (T = 4). Local = `private_init`; FedMD = média das
rodadas 26–30 de cada cliente; cada coluna é a média das 9 execuções.

| f | Local | FedMD | Δ melhor | Δ pior | Amplitude (máx − mín) | Desvio padrão (ddof = 1) |
|---|---|---|---|---|---|---|
| 1% | 0,859 | 0,869 | −0,005 | +0,028 | 0,046 → 0,017 | 0,020 → 0,008 |
| 5% | 0,868 | 0,878 | −0,002 | +0,031 | 0,036 → 0,013 | 0,016 → 0,006 |
| 10% | 0,877 | 0,878 | −0,011 | +0,020 | 0,037 → 0,011 | 0,016 → 0,005 |
| 40% | 0,891 | 0,882 | −0,019 | +0,003 | 0,028 → 0,013 | 0,013 → 0,006 |
| 80% | 0,896 | 0,881 | −0,019 | −0,008 | 0,019 → 0,015 | 0,008 → 0,007 |

A tabela do artigo bate em todos os valores. **Correção sugerida:** renomear a coluna para "Amplitude (máx − mín)",
ou explicar na legenda que é a diferença entre o melhor e o pior cliente, média de 9 execuções.

Fontes: `ensemble_e_leave_one_out/all_P01_seed{42,1,2}_test_metrics.csv`, `all_P05_*` e
`fracoes_publicas_10_40_80/adaptado_P{10,40,80}_seed{42,1,2}_test_metrics.csv` (coluna `test_accuracy`).

### B2. Cenário PH2 privado: tabela completa por α

Média das seeds 42 e 1 ± desvio padrão entre elas. Validação = teste do Kaggle (média dos clientes); PH2 = 120
imagens fora do treino. Variantes federadas: média das rodadas 16–20. "Treinamento local" é o fim do `private_init`
das execuções `original_*`. A última coluna (validação do ensemble) não foi pedida, mas sai dos mesmos arquivos.

| α | Variante | Validação | PH2 bal. (clientes) | PH2 bal. (ensemble) | Validação (ensemble) |
|---|---|---|---|---|---|
| IID | public_init | 0,898 ± 0,008 | 0,650 ± 0,030 | 0,630 ± 0,000 | 0,901 ± 0,005 |
| IID | Treinamento local | 0,593 ± 0,004 | 0,702 ± 0,030 | 0,750 ± 0,007 | 0,621 ± 0,062 |
| IID | Sem digest | 0,631 ± 0,014 | 0,720 ± 0,015 | 0,780 ± 0,002 | 0,663 ± 0,008 |
| IID | Digest só com CE | 0,826 ± 0,009 | 0,616 ± 0,009 | 0,596 ± 0,012 | 0,869 ± 0,017 |
| IID | FedMD replicado | 0,625 ± 0,000 | 0,807 ± 0,030 | 0,873 ± 0,042 | 0,636 ± 0,005 |
| 1,0 | public_init | 0,900 ± 0,000 | 0,673 ± 0,017 | 0,654 ± 0,011 | 0,904 ± 0,000 |
| 1,0 | Treinamento local | 0,609 ± 0,001 | 0,689 ± 0,031 | 0,583 ± 0,118 | 0,516 ± 0,013 |
| 1,0 | Sem digest | 0,603 ± 0,013 | 0,703 ± 0,005 | 0,664 ± 0,013 | 0,570 ± 0,074 |
| 1,0 | Digest só com CE | 0,797 ± 0,007 | 0,598 ± 0,003 | 0,568 ± 0,004 | 0,842 ± 0,004 |
| 1,0 | FedMD replicado | 0,655 ± 0,023 | 0,772 ± 0,007 | 0,845 ± 0,029 | 0,664 ± 0,020 |
| 0,1 | public_init | 0,896 ± 0,005 | 0,665 ± 0,051 | 0,651 ± 0,044 | 0,902 ± 0,003 |
| 0,1 | Treinamento local | 0,550 ± 0,001 | 0,587 ± 0,046 | 0,521 ± 0,029 | 0,569 ± 0,009 |
| 0,1 | Sem digest | 0,563 ± 0,016 | 0,610 ± 0,015 | 0,595 ± 0,082 | 0,620 ± 0,000 |
| 0,1 | Digest só com CE | 0,756 ± 0,067 | 0,598 ± 0,022 | 0,579 ± 0,009 | 0,859 ± 0,022 |
| 0,1 | FedMD replicado | 0,551 ± 0,091 | 0,621 ± 0,003 | 0,682 ± 0,009 | 0,560 ± 0,091 |

**Média dos 3 α** (6 execuções; ± desvio entre as 6):

| Variante | Validação | PH2 bal. (clientes) | PH2 bal. (ensemble) | Artigo hoje |
|---|---|---|---|---|
| public_init | 0,898 ± 0,004 | 0,663 ± 0,029 | 0,645 ± 0,023 | |
| Treinamento local | 0,584 ± 0,027 | 0,660 ± 0,063 | 0,618 ± 0,119 | |
| Sem digest | 0,599 ± 0,033 | 0,678 ± 0,054 | 0,680 ± 0,091 | 0,599 / 0,678 / 0,680 ✔ |
| Digest só com CE | 0,793 ± 0,043 | 0,604 ± 0,014 | 0,581 ± 0,014 | 0,793 / 0,604 / 0,581 ✔ |
| FedMD replicado | 0,610 ± 0,063 | 0,733 ± 0,089 | 0,800 ± 0,095 | 0,610 / 0,733 / 0,800 ✔ |

Os três trios do artigo estão confirmados.

**Teto centralizado do cenário** (`ph2_privado/centralizado.csv`, Kaggle + os mesmos 80 do PH2, seeds 42 e 1):

- Modelo único (média das 4 arquiteturas × 2 seeds): validação **0,895 ± 0,009**, PH2 bal. **0,653 ± 0,058**.
- Ensemble: validação **0,901 ± 0,006**, PH2 bal. **0,630 ± 0,044**.

Os valores do artigo (0,895 e 0,653; 0,901 e 0,630) estão confirmados.

**Atenção (possível correção no texto):** o `RESULTADO.md` desta pasta tem uma linha "Local (+ `private_init` no
PH2)" com 0,879 na validação e 0,632 no PH2. Essa linha é o `private_init` do **FedMD adaptado**
(`adaptado_*`: lr 3e-4, 300 passos com rehearsal). No protocolo do FedMD replicado (50 épocas, lr 1e-3, sem
rehearsal), o treinamento local já derruba a validação para **0,584** e fica em **0,660** no PH2. Se o artigo compara
o FedMD replicado com o "local", o local certo é o desta tabela.

Notas:
- O `public_init` e o `private_init` não são idênticos entre `original_*`, `semdigest_original_*` e
  `controle_original_*` da mesma seed (a GPU não é determinística). No PH2 balanceado após o `private_init`, eles
  dão 0,660, 0,677 e 0,673 na média dos 6. O "Δ PH2 nas rodadas" do `RESULTADO.md` (+0,074 / 0,000 / −0,069) usa o
  `private_init` de cada execução; com o `private_init` comum das `original_*`, o sem digest ficaria em +0,018 e o CE
  em −0,056.
- Fontes: `ph2_privado/{original,semdigest_original,controle_original}_seed{42,1}_alpha{iid,1.0,0.1}_{test_metrics,ensemble_metrics,extra_test_metrics}.csv`.

### B3. Significado de +0,104, +0,083 e +0,034

**Confirmado.** São o ganho do FedMD replicado ao longo das rodadas no PH2 balanceado (média dos clientes): média das
rodadas 16–20 menos o fim do `private_init`, na mesma execução (`original_*`), média das seeds 42 e 1:

| α | FedMD (final − `private_init`) | FedMD − sem digest (final × final) |
|---|---|---|
| IID | **+0,104** (seeds: +0,104, +0,104) | **+0,087** (+0,119, +0,054) |
| 1,0 | **+0,083** (+0,066, +0,101) | **+0,069** (+0,077, +0,061) |
| 0,1 | **+0,034** (+0,003, +0,064) | **+0,011** (+0,024, −0,002) |
| Média | **+0,074** (0,0737) | **+0,056** (0,0556) |

Na coluna "FedMD − sem digest", os dois lados são a média das rodadas 16–20 da mesma seed e α. Fontes:
`ph2_privado/original_*_extra_test_metrics.csv` e `semdigest_original_*_extra_test_metrics.csv` (coluna
`balanced_accuracy`, `model != 'ensemble'`).

### B4. Testes estatísticos

Todos os valores-p citados vêm de um **teste t pareado bilateral** (equivalente ao t de uma amostra nas diferenças).
Para comparar, também calculei Wilcoxon pareado (bilateral, exato) e o teste do sinal.

| Comparação | Pares | Média da diferença ± dp | A favor | t | **p (t pareado)** | p (Wilcoxon) | p (sinal) |
|---|---|---|---|---|---|---|---|
| FedMD − sem digest, PH2 privado (PH2 bal. clientes) | 6 (2 seeds × 3 α) | +0,056 ± 0,042 | 5/6 | 3,25 | **0,023** | 0,063 | 0,22 |
| KL − CE (w = 1), 1%, ganho na validação | 9 (3 seeds × 3 α) | +0,003 ± 0,007 | 6/9 | 1,33 | **0,221** | 0,25 | 0,51 |
| KL − CE (w = 1), 5%, ganho na validação | 9 | −0,007 ± 0,009 | 3/9 | −2,27 | **0,053** | 0,055 | 0,51 |
| KL − MAE, 10/40/80%, ganho na validação | 18 (3 frações × 3 α × 2 seeds) | +0,002 ± 0,008 | 13/18 | 1,15 | **0,266** | 0,246 | 0,096 |
| KL − MAE, 1% e 5%, ganho na validação (KL de `digest_kl/T4_*`) | 18 (2 frações × 3 α × 3 seeds) | +0,0019 ± 0,0043 | 12/18 | 1,85 | **0,082** | 0,067 | 0,24 |

**Correção importante no p ≈ 0,08 (KL − MAE com 1% e 5%):** ele vem das execuções KL de `results/digest_kl/T4_*`,
e não das `ensemble_e_leave_one_out/all_*` que a Figura 4 usa como KL em 1% e 5%. Os argumentos são os mesmos, mas as
execuções são outras. Com as `all_*`, a diferença KL − MAE nessas frações cai para **+0,0004 ± 0,0034, 8/18 a favor,
p = 0,62** (t pareado). Sugestão: citar o teste com as mesmas execuções da figura (p = 0,62, "sem diferença") ou
dizer explicitamente que o p = 0,08 vem de uma rodada anterior (`digest_kl`). Nas duas versões, a conclusão de que a
loss do digest não importa continua de pé.

Para todos os outros, sugiro escrever "teste t pareado bilateral, n = …" junto do valor-p. Com 6 pares, o Wilcoxon
não chega a 0,05 (o menor p exato possível com 5/6 positivos é 0,063); vale citar o t como o teste usado.

Fontes: `ph2_privado/*` (linha 1); `ensemble_e_leave_one_out/all_P0{1,5}_*` × `digest_ce_1_5/CE1.0_P0{1,5}_*`
(linhas 2 e 3); `fracoes_publicas_10_40_80/adaptado_*` (seeds 42 e 1) × `digest_kl_10_40_80/MAE_*` (linha 4);
`digest_kl/T4_*` × `experimento_final/P01_*`, `fracoes_publicas_5_e_20/F2b_seed{42,1}` e `experimento_final/P05_seed2`
(linha 5). Ganho = média das rodadas 26–30 − `private_init`, média dos clientes.

### B5. Protocolo do FedMD replicado

- **O `digest` percorre todo o subconjunto público da rodada, com lotes de 32? Sim.** O FedMD replicado não passa
  `--digest-steps` (`ph2_privado/orquestrador.sh:15`, `fracoes_publicas_10_40_80/orquestrador.sh:18`,
  `test_models/orquestrador.sh:10`), então `steps_per_epoch=None` e o `fit` faz uma passada (`client.py:304-310`) com
  `batch_size = BATCH_SIZE = 32` (`client.py:14`, `:289`, `:304`), 1 época (`--digest-epochs 1`, `client.py:516`). O
  subconjunto da rodada é `round(0,8 · |público|)` (`client.py:22`, `:466-468`). Os logits enviados ao servidor
  cobrem esse subconjunto inteiro. (O FedMD adaptado usa `--digest-steps 30`.)
- **O `private_init`: até 50 épocas com parada antecipada no cenário padrão, 50 completas no PH2 privado? Sim.**
  `--private-init-epochs 50` nos três orquestradores acima. No cenário padrão, o cliente tem validação privada (20%,
  `client.py:364`), e `train()` usa `EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)`
  (`client.py:248`). Em `fracoes_publicas_10_40_80/original_P10_seed42`, os clientes pararam entre 5 e 13 épocas.
  No PH2 privado, a validação privada é vazia (`client.py:380`), não há callback (`client.py:240`, `:248-249`) e os
  4 clientes rodam as 50 épocas (200 linhas `private_init` em cada `*_train_history.csv`).
- **O `revisit` faz 3 épocas completas, sem rehearsal? Sim, com uma ressalva.** `--private-epochs 3`, sem
  `--private-steps` (cada época é uma passada pelos dados privados) e sem `--rehearsal-fraction` (padrão 0,
  `client.py:556`, então `rehearsal = {}` em `client.py:470-471`). O `train_history` mostra 3 épocas em todas as
  rodadas. **Ressalva para o cenário padrão:** como há validação privada, o `EarlyStopping` com
  `restore_best_weights=True` continua ligado. Com patience 4 ele nunca interrompe as 3 épocas, mas, no Keras 3.15
  (`EarlyStopping.on_train_end`), restaura no fim os pesos da época de menor `val_loss`. Em
  `original_P10_seed42`, o modelo ficou com os pesos de uma época anterior à última em 144 dos 240 revisits. No PH2
  privado (sem validação), os pesos ficam os da 3ª época.
- **Passos por rodada no PH2 privado:**
  - `digest`: subconjunto = round(0,8 × 9.605) = 7.684 imagens → ⌈7.684 / 32⌉ = **241 passos** (240 lotes cheios + 1 de 4).
  - `revisit`: 3 × ⌈n_k / 5⌉ (batch 5, `--private-batch-size 5`). Com IID, n_k = 20 → **3 × 4 = 12 passos**. Nas
    divisões Dirichlet, n_k varia de 10 a 45 (`n_private` em `*_test_metrics.csv`), ou seja, de 6 a 27 passos.
  - Sugestão de texto: "~240 passos de digest e ~12 de revisit (IID; de 6 a 27 nas divisões não IID)".

### B6. Perda do digest

- **O KL é multiplicado por T²? Sim:** `client.py:91-101`, `return temperature ** 2 * keras.ops.sum(target * (log_target - log_pred), axis=-1)`,
  com `target = softmax(consenso / T)` e `log_pred = log_softmax(cliente / T)`.
- **Na perda mista, o T² entra só no KL? Sim:** `with_label_anchor` (`client.py:104-114`) calcula
  `(1 − w) · distill_loss(consenso, logits) + w · CE(rótulo, logits)`, e o T² está dentro de `distill_loss`. A CE
  (`sparse_categorical_crossentropy`, `from_logits=True`) não é escalada.
- Nota: o FedMD replicado não usa KL. Sem `--digest-temperature`, a perda é `mean_absolute_error` nos logits
  (`client.py:175-176`). No controle "Digest só com CE" do PH2 privado (`controle_original_*`), w = 1 zera o termo
  de destilação, que é a MAE.

### B7. Acurácia `a_k` usada no consenso ponderado

**Confirmado.** Em `fed_train` (`client.py:288-296`), os logits vêm de `self.model.predict(public_dataset)` sobre
`round_public_idx`, o subconjunto sorteado na rodada (`client.py:468`, `:475`), sem embaralhar e sem aumentação.
`public_accuracy = mean(argmax(logits) == rótulo)` é calculada **desses mesmos logits** e vai na mesma mensagem
(`ClientLogits.public_accuracy`). O modelo está no estado em que começa a rodada: depois do revisit da rodada
anterior (ou do `private_init`, na rodada 1) e antes do digest. O servidor usa `max(a_k − 0,5, 0)` normalizado como
peso (`server.py:65-76`). Isso só vale com `--weighting accuracy` (FedMD adaptado). O FedMD replicado roda com
`--weighting uniform`, e aí `a_k` é enviada mas ignorada.

### B8. Aumentação e pré-processamento

- **Onde a aumentação é aplicada:** `train()` (`client.py:237-239`, `augment=True` por padrão) aplica
  `augmentation_pipeline(train=True)` (flip horizontal, `RandomRotation(0.1)` = ±10% de 2π ≈ ±36°,
  `RandomZoom(0.1)`, `client.py:77-88`) no **`public_init`** (`client.py:443-445`), no **`private_init`** e no
  **`revisit`** (`train_private`, `client.py:268-272`). Quando há rehearsal, as amostras públicas misturadas também
  são aumentadas, porque a aumentação é aplicada depois da mistura.
- **No `digest`, não há aumentação:** `fed_train` monta `digest_dataset` com `make_client_dataset` e o passa direto ao
  `fit` (`client.py:304-310`), sem `self.augment`. **Os logits do consenso são calculados sem aumentação (sim), e o
  cliente também treina com as imagens sem aumentação (não com imagens aumentadas).** Se o artigo diz que o digest
  usa imagens aumentadas, é preciso corrigir.
- **Normalização:** sim, `keras.layers.Rescaling(1 / 255)` é a primeira camada do modelo (`client.py:62`) e vale em
  todas as etapas e na inferência. `make_client_dataset` só decodifica o JPEG e redimensiona para 224 × 224, com
  pixels em 0–255 (`dataset_setup.py`, `make_client_dataset`). Não há padronização por média/desvio.

### B9. Taxas de aprendizado do `public_init` (`--public-init-lr`)

| Fração | FedMD adaptado | FedMD replicado | Fonte |
|---|---|---|---|
| 1% | **3e-4** | — (não rodado) | `ensemble_e_leave_one_out/orquestrador.sh:28`, `digest_ce_1_5/orquestrador.sh:62`; MAE: `experimento_final/orquestrador.sh:37` (escolhido em `escolha_1pct.txt`: 3e-4) |
| 5% | **1e-3** | 1e-3 | `ensemble_e_leave_one_out/orquestrador.sh:29`, `digest_ce_1_5/orquestrador.sh:63`, `test_models/orquestrador.sh:5` e `:10`; MAE: `fracoes_publicas_5_e_20/escolha2.txt` (F2b) e `experimento_final/orquestrador.sh:39` |
| 10%, 40%, 80% | **3e-4** | 1e-3 | `fracoes_publicas_10_40_80/orquestrador.sh:14` e `:18`; `digest_kl_10_40_80/orquestrador.sh:11`; `digest_ce_10_40_80/orquestrador.sh:10` |
| PH2 privado (100% do Kaggle) | 3e-4 | 1e-3 (inclusive nos controles) | `ph2_privado/orquestrador.sh:12` e `:15`, `orquestrador_controle.sh:11`, `orquestrador_semdigest.sh:11` |

Como não é monotônico (3e-4, 1e-3, 3e-4), vale explicar no texto que o lr do `public_init` foi escolhido por fração
com as etapas iniciais (`experimento_final/escolhe_1pct.py`, `fracoes_publicas_5_e_20/seleciona.py`).

### B10. Custo de comunicação (Seção 5.3)

`proto_config/message.proto`: `ClientLogits { int32 client_id; repeated float logits; float public_accuracy; }`.
`float` é float32, e `repeated float` em proto3 é empacotado: **4 bytes por valor**. Cada mensagem leva
N × 2 logits (2 classes), com N = round(0,8 × |público|) (`client.py:22`, `:466`). Os outros campos somam ~10 bytes.
A resposta `ServerConsensus` tem o mesmo tamanho.

| f | |público| | N (subconjunto da rodada) | floats | Tamanho da mensagem por cliente por rodada |
|---|---|---|---|---|
| 1% | 97 | 78 | 156 | 0,6 KiB (634 B) |
| 5% | 481 | 385 | 770 | 3,0 KiB (3.090 B) |
| **10%** | 961 | 769 | 1.538 | **6,0 KiB (6.152 B de payload, ~6.162 B no total)** |
| 40% | 3.842 | 3.074 | 6.148 | 24,0 KiB |
| **80%** | 7.684 | 6.147 | 12.294 | **48,0 KiB (49.176 B de payload, ~49.187 B no total)** |
| PH2 privado (100%) | 9.605 | 7.684 | 15.368 | 60,0 KiB |

|público| = `split_public` estratificado (`dataset_setup.py`) sobre as 9.605 imagens de treino do Kaggle.

Parâmetros das 4 arquiteturas (`build_model(small_cnn(...))`, `client.py:27-32`, `:42-74`; inclui as médias móveis
do BatchNorm, que o FedAvg também enviaria):

| Arquitetura | Camadas conv. (filtros, kernel) | Parâmetros (treináveis) | float32 |
|---|---|---|---|
| 0 | (16,3) (32,3) (64,3) | 32.610 (32.386) | 130.440 B ≈ 127 KiB |
| 1 | (32,3) (64,3) (128,3) | 110.914 (110.466) | 443.656 B ≈ 433 KiB |
| 2 | (16,5) (32,5) (64,3) (128,3) | 124.130 (123.650) | 496.520 B ≈ 485 KiB |
| 3 | (32,3) ×2 (64,3) ×2 (128,3) | 157.474 (156.834) | 629.896 B ≈ 615 KiB |

Comparação: com f = 10%, a mensagem de logits (6 KiB) é ~21× a 102× menor que os pesos de um cliente. Com f = 80%
(48 KiB), é ~2,7× a 13× menor. Ressalva: o FedAvg exige arquiteturas iguais, então a comparação é por ordem de
grandeza.

### B11. Sensibilidade no PH2 (Seção 6.5)

PH2 completo (200 imagens: 160 benignas, 40 melanomas), IID, média dos 4 clientes × 3 seeds (12 modelos). A fórmula
`E = (acc − 0,4·bal)/0,6`, `S = 2·bal − E` é exata para essa proporção de 80/20 e foi aplicada por cliente antes da
média. Fontes: `test_models/{adaptado,original}_seed*_extra_test_metrics.csv` (5%),
`fracoes_publicas_10_40_80/{adaptado,original}_P40_seed*_extra_test_metrics.csv` (40%) e
`test_models/centralizado.csv`.

| f | Modelo | Sensibilidade | Especificidade | Artigo |
|---|---|---|---|---|
| 5% | Treinamento local | **0,567** | 0,722 | 0,57 ✔ |
| 5% | FedMD adaptado | **0,445** (0,4446) | 0,820 | 0,44 ✔ |
| 5% | (FedMD replicado) | 0,445 | 0,847 | |
| 40% | Treinamento local | **0,248** | 0,960 | 0,24 → **corrigir para 0,25** |
| 40% | FedMD adaptado | **0,176** | 0,971 | 0,18 ✔ |
| 40% | (FedMD replicado) | 0,419 | 0,878 | |
| — | Teto centralizado (modelo único) | **0,448** | 0,882 | 0,45 ✔ |
| — | Teto centralizado (ensemble) | 0,400 | 0,913 | |

O teto centralizado não depende da fração. Única correção: 40% local, de 0,24 para 0,25.
