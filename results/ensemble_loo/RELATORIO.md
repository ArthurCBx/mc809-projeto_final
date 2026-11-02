# Etapa final: ensemble dos clientes e consenso leave-one-out (02/11, 01:11–06:37)

12 execuções, todas sem erros: 1% e 5% de dados públicos × seeds 42, 1 e 2 × duas formas de consenso.

**Configuração.** A mesma do experimento final, com KL e T = 4 no digest:

- `private_init` com 300 passos, lr 1e-4 e rehearsal.
- Revisit com 3×10 passos, lr 1e-4 e 50% de rehearsal.
- Digest com 30 passos e lr 4e-4.
- 30 rodadas e consenso ponderado.

**Variações:**

- **A (consenso padrão):** todos os clientes recebem o mesmo consenso. Em cada avaliação, o ensemble dos 4
  clientes é calculado como a média dos logits no teste.
- **B (leave-one-out):** cada cliente recebe o consenso calculado só com os outros três.

## A. Ensemble dos clientes (consenso padrão, média ± desvio padrão de 3 seeds)

| Fração / alpha | Clientes após private_init | Clientes após FedMD | **Ensemble após private_init** | Ensemble após FedMD |
|---|---|---|---|---|
| 1% / iid | 0,875 ± 0,003 | 0,881 ± 0,004 | **0,887 ± 0,004** | 0,883 ± 0,005 |
| 1% / 1.0 | 0,862 ± 0,005 | 0,871 ± 0,009 | **0,885 ± 0,004** | 0,881 ± 0,003 |
| 1% / 0.1 | 0,840 ± 0,002 | 0,853 ± 0,004 | **0,881 ± 0,003** | 0,868 ± 0,008 |
| 5% / iid | 0,879 ± 0,007 | 0,889 ± 0,005 | 0,885 ± 0,009 | **0,892 ± 0,005** |
| 5% / 1.0 | 0,868 ± 0,008 | 0,878 ± 0,014 | 0,881 ± 0,007 | 0,882 ± 0,019 |
| 5% / 0.1 | 0,856 ± 0,004 | 0,868 ± 0,003 | 0,880 ± 0,006 | 0,881 ± 0,010 |

Ensemble significa a média dos logits dos 4 clientes. As colunas "FedMD" usam as rodadas 26–30.

| Fração / alpha | Ganho do FedMD por cliente | Ganho do ensemble sem colaboração | Efeito do FedMD sobre o ensemble |
|---|---|---|---|
| 1% / iid | +0,006 | +0,012 | −0,004 |
| 1% / 1.0 | +0,009 | +0,023 | −0,004 |
| 1% / 0.1 | +0,013 | **+0,041** | **−0,013** |
| 5% / iid | +0,009 | +0,006 | +0,007 |
| 5% / 1.0 | +0,011 | +0,014 | +0,001 |
| 5% / 0.1 | +0,012 | +0,024 | +0,001 |
| **Média** | **+0,010** | **+0,020** | **−0,002** |

- O **ganho do FedMD por cliente** compara os clientes após FedMD com os clientes após o `private_init`.
- O **ganho do ensemble sem colaboração** compara o ensemble após o `private_init` com os clientes após o
  `private_init`.
- O **efeito do FedMD sobre o ensemble** compara o ensemble após FedMD com o ensemble após o
  `private_init`.

## B. Consenso leave-one-out − consenso padrão (média ± desvio padrão de 3 seeds)

| Fração / alpha | Clientes | Ensemble |
|---|---|---|
| 1% / iid | +0,001 ± 0,000 | +0,001 ± 0,001 |
| 1% / 1.0 | −0,004 ± 0,011 | −0,007 ± 0,011 |
| 1% / 0.1 | +0,002 ± 0,004 | +0,004 ± 0,011 |
| 5% / iid | −0,001 ± 0,001 | −0,001 ± 0,003 |
| 5% / 1.0 | −0,002 ± 0,003 | −0,002 ± 0,001 |
| 5% / 0.1 | +0,001 ± 0,003 | −0,001 ± 0,003 |
| **Média (18 pares)** | −0,0004 ± 0,0048 (melhor em 11/18) | −0,0008 ± 0,0067 (melhor em 7/18) |

## Conclusões

1. **Combinar as previsões de clientes que nunca colaboraram rende o dobro do FedMD.** Fazer a média dos 4
   clientes logo após o `private_init`, sem nenhuma rodada federada, já dá **+0,020** sobre o cliente
   médio. O FedMD melhora cada cliente em **+0,010**. Com 1% de dados públicos e alpha 0.1, o ensemble sem
   colaboração chega a +0,041. Clientes treinados em dados diferentes erram em imagens diferentes, e a
   média compensa esses erros.
2. **O FedMD reduz a diversidade entre os modelos.**
   - **Com 1% de dados públicos, o ensemble piora com o FedMD** (−0,004 a −0,013). Ao destilar todos em
     direção ao mesmo consenso, os modelos ficam parecidos e o ensemble perde o que o fazia funcionar. O
     efeito é maior com alpha 0.1, onde a diversidade inicial é maior.
   - **Com 5% de dados públicos, o efeito fica entre neutro e levemente positivo** (+0,001 a +0,007). O
     melhor resultado de sistema aparece aqui: 0,892 em 5% / iid, perto do teto (0,898).
3. **Qual abordagem usar depende de como o modelo será usado:**
   - Se for possível combinar os modelos na inferência (por exemplo, consultando os 4 clientes para cada
     amostra), o **ensemble sem colaboração** basta e é mais barato: 0,880–0,887, sem rodadas federadas.
   - Se cada cliente precisa de **um modelo próprio** para usar sozinho, como no cenário federado típico
     (inferência local, sem consultar os outros), o **FedMD é o que melhora esse modelo** (+0,010).
4. **O consenso leave-one-out não tem efeito.** A diferença média é −0,0004 nos clientes e −0,0008 no
   ensemble, bem dentro do ruído, com 11 e 7 de 18 pares a favor. A hipótese de que cada cliente reforça o
   próprio conhecimento ao destilar um consenso que o inclui não se confirmou. Com 4 clientes, o peso de
   cada um no consenso já é pequeno.

## Limitações

- **O ensemble tem um custo que o FedMD não tem.** Ele exige reunir os modelos, ou as previsões deles para
  cada amostra, no momento da inferência. Isso tem custo de comunicação e pode não ser permitido em
  cenários com restrições de privacidade.
- **O ensemble é uma média simples, com pesos iguais.** Uma ponderação pela acurácia pública, como no
  consenso, poderia melhorá-lo, mas não foi testada.
- **Mesmas ressalvas do experimento final:** os hiperparâmetros foram escolhidos olhando o conjunto de
  teste, e o desvio padrão entre seeds fica em torno de 0,005–0,01.

## Mudanças no código

- **`client.py`:**
  - Novo método `Client.test_logits()`.
  - Em cada avaliação, o ensemble dos clientes vai para `results/ensemble_metrics.csv`, com as colunas
    `ensemble_accuracy` e `mean_client_accuracy`.
- **`server.py`:** nova opção `--consensus {all,loo}`.

## Arquivos (`results/ensemble_loo/`)

- `{all,loo}_P{01,05}_seed{42,1,2}_{test_metrics,train_history,ensemble_metrics}.csv` e um `.log` por
  execução.
- `resumo_por_seed.csv`. As tabelas são geradas com `uv run --with tabulate python results/ensemble_loo/tabelas.py`.
- `smoke_*`: teste rápido do caminho novo antes da noite (1 rodada). Não entra nas tabelas.
