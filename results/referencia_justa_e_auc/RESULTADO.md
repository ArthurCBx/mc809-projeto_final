# Referência centralizada justa no cenário 2 e sensibilidade, especificidade e AUC no PH2 (14–15/11)

Dois experimentos para fechar pontas do artigo. Nada do cenário 2 estava salvo (nem pesos do `public_init`, nem
predições do PH2), então tudo foi retreinado. O `client.py` passou a gravar sensibilidade, especificidade e AUC no
`extra_test_metrics.csv` e os logits do teste externo em `*_logits.npz`. A AUC usa a probabilidade softmax de melanoma
e não depende do limiar.

## 1. Referência centralizada justa no cenário 2 (Tabela 9)

Cada uma das 4 arquiteturas faz o `public_init` do FedMD replicado (ISIC inteiro, η = 10⁻³, parada antecipada) e
depois o `private_init` nas **80 imagens do PH2 reunidas**, com o protocolo do cenário 2 (lotes de 5, 50 épocas,
η = 10⁻³, sem validação), sem rodadas. Opções: `--pooled-private --no-digest --digest-rounds 0`. Seeds 42 e 1, mesmo
sorteio do PH2 de cada seed. "Validação" é o conjunto de 1.000 imagens do ISIC; PH2 são as 120 imagens fora do treino.

| Variante (média de 2 seeds) | Validação | PH2 bal. (clientes) | PH2 bal. (ensemble) | AUC PH2 (clientes / ensemble) |
|---|---|---|---|---|
| Treinamento local (fim do `private_init`, Tabela 9) | 0,584 | 0,660 | 0,618 | |
| FedMD (consenso, Tabela 9) | 0,610 | 0,733 | 0,800 | |
| **Referência centralizada justa** | **0,686** | **0,867** | **0,883** | 0,955 / 0,975 |
| Centralizado antigo (ISIC + 80 do PH2 diluídas), refeito | 0,899 | 0,672 | 0,654 | 0,810 / 0,861 |

Por seed, a referência justa dá 0,897 e 0,837 por cliente e 0,932 e 0,833 no ensemble (sensibilidade 0,80,
especificidade 0,94, por cliente). O `public_init` dessas execuções chega a 0,898 na validação e 0,661 no PH2, como o do
FedMD replicado.

**Leitura.** A referência fica acima do FedMD e passa a ser um teto de verdade. Por cliente, o FedMD percorre cerca de
35% do caminho entre o treinamento local e o teto ((0,733 − 0,660) / (0,867 − 0,660)); no ensemble, cerca de 70%
((0,800 − 0,618) / (0,883 − 0,618)), e na divisão IID o ensemble do FedMD (0,873) fica perto do teto. O teto também
perde a validação (0,686): a queda vem de treinar 50 épocas com η = 10⁻³ em poucas imagens, não da colaboração. O teto
não depende de α (reúne as mesmas 80 imagens), então a distância com α = 0,1 inclui o custo de separar as classes entre
clientes. O centralizado refeito reproduz o da Tabela 9 (0,895 / 0,653 no artigo).

**Atualização com 4 seeds (16/11):** com as seeds 2 e 3, a referência justa fica em 0,700 na validação, 0,820 por cliente e
0,831 no ensemble (0,90, 0,84, 0,78 e 0,77 por seed); o FedMD, com 4 seeds, em 0,701 / 0,735. A tabela completa
está em `results/ph2_privado/RESULTADO.md` (seção "Seeds 2 e 3").

## 2. Sensibilidade, especificidade e AUC no PH2, cenário 1 (Seção 6.5, nota 3, Tabela 8)

Divisão IID, f = 5% (argumentos de `results/test_models`) e 40% (de `results/fracoes_publicas_10_40_80`), seeds 42, 1
e 2. Treinamento local = `private_init` da execução do adaptado. FedMD: média das rodadas 26–30 (adaptado) ou 16–20
(replicado). PH2 inteiro (200 imagens). Entre parênteses, o valor do artigo (Tabela 8 e nota 3), para conferir a
reprodução.

Média dos clientes:

| f | Modelo | Bal. | Sensibilidade | Especificidade | AUC |
|---|---|---|---|---|---|
| 5% | Treinamento local | 0,651 (0,644) | 0,581 (0,57) | 0,720 | 0,731 |
| | FedMD adaptado | 0,627 (0,632) | 0,417 (0,44) | 0,837 | 0,742 |
| | FedMD replicado | 0,623 (0,646) | 0,388 | 0,857 | 0,733 |
| 40% | Treinamento local | 0,603 (0,604) | 0,246 (0,25) | 0,959 | 0,727 |
| | FedMD adaptado | 0,567 (0,574) | 0,157 (0,18) | 0,977 | 0,632 |
| | FedMD replicado | 0,629 (0,649) | 0,358 | 0,899 | 0,783 |
| — | Teto centralizado (modelo único) | 0,673 (0,665) | 0,452 (0,45) | 0,894 | 0,824 |

Ensemble dos 4 clientes:

| f | Modelo | Bal. | Sensibilidade | Especificidade | AUC |
|---|---|---|---|---|---|
| 5% | Treinamento local | 0,678 | 0,592 | 0,765 | 0,765 |
| | FedMD adaptado | 0,619 | 0,383 | 0,854 | 0,760 |
| | FedMD replicado | 0,611 | 0,313 | 0,908 | 0,784 |
| 40% | Treinamento local | 0,593 | 0,217 | 0,969 | 0,736 |
| | FedMD adaptado | 0,574 | 0,163 | 0,984 | 0,634 |
| | FedMD replicado | 0,617 | 0,312 | 0,923 | 0,825 |
| — | Teto centralizado (ensemble) | 0,643 | 0,367 | 0,919 | 0,864 |

**Leitura.**
- **Com 5%, a queda de sensibilidade do FedMD adaptado é do limiar.** A AUC fica igual à do local (0,742 contra
  0,731): o modelo ordena os casos tão bem quanto antes, mas desloca o ponto de decisão para a classe benigna.
- **Com 40%, a perda é do modelo.** A AUC do adaptado cai para 0,632 contra 0,727 do local; mudar o limiar não a
  recupera.
- **Nenhum método passa de 0,83 de AUC por modelo no PH2**, e o teto centralizado é o melhor (0,824). Todos têm AUC bem
  acima do que a acurácia balanceada sugere, ou seja, parte da má generalização é de calibração do limiar no novo
  domínio.
- A reprodução bate com o artigo dentro de ~0,02 (não determinismo da GPU). O replicado fica ~0,02 abaixo da Tabela 8
  na acurácia balanceada; os valores do artigo continuam válidos, e os novos servem para as métricas que faltavam.

Conferência extra: os modelos finais salvos de f = 40% (as mesmas execuções da Tabela 8) reavaliados com as métricas
novas (`reavaliacao_P40_salvos.csv`, rodada final apenas) dão, com divisão IID, AUC 0,63 no adaptado e 0,78 no
replicado, iguais às dos retreinos.

## Arquivos

- `orquestrador.sh`, `orquestrador.log`, `server.log`, um `.log` por execução.
- `referencia_seed{42,1}_*`: referência justa (CSVs e logits).
- `{adaptado,original}_P{05,40}_seed{42,1,2}_*`: cenário 1, IID; `resumo.py` gera as tabelas da Seção 2.
- `centralizado_metricas.py`, `centralizado_cenario{1,2}.csv` e `_seed*_logits.npz`: centralizados com AUC.
- `reavalia_P40_salvos.py`, `reavaliacao_P40_salvos.csv`: modelos salvos de f = 40% reavaliados.
