# Revisit com lr 1e-4 (31/10, 18:49–21:17)

Mesma configuração do cenário 3 (`results/madrugada/RELATORIO.md`), mudando só o lr do revisit:
`private_init` com 10 passos e lr 2e-5, digest com lr 1e-3, revisit com 3×10 passos e 50% de rehearsal,
30 rodadas e seed 42.

| Configuração | public_init | lr 1e-3 (r26–30) | **lr 1e-4 (r26–30)** | Diferença | 1e-4 vs public_init | Pior cliente com 1e-3 | **Pior cliente com 1e-4** |
|---|---|---|---|---|---|---|---|
| 0.1 / iid | 0,861 | 0,889 | 0,859 | −0,031 | −0,002 | 0,842 | 0,835 |
| 0.1 / 1.0 | 0,878 | 0,874 | 0,859 | −0,015 | −0,018 | 0,700 | 0,844 |
| 0.1 / 0.1 | 0,868 | 0,842 | 0,847 | +0,006 | −0,020 | 0,557 | 0,810 |
| 0.4 / iid | 0,885 | 0,891 | 0,886 | −0,006 | +0,001 | 0,827 | 0,863 |
| 0.4 / 1.0 | 0,873 | 0,860 | 0,853 | −0,008 | −0,020 | 0,606 | 0,842 |
| 0.4 / 0.1 | 0,883 | 0,832 | 0,854 | +0,022 | −0,029 | 0,565 | 0,827 |
| 0.8 / iid | 0,887 | 0,897 | 0,890 | −0,007 | +0,003 | 0,797 | 0,879 |
| 0.8 / 1.0 | 0,884 | 0,881 | 0,886 | +0,005 | +0,002 | 0,500 | 0,863 |
| 0.8 / 0.1 | 0,888 | 0,854 | 0,877 | +0,023 | −0,011 | 0,508 | 0,850 |
| **Média** | | **0,869** | **0,868** | −0,001 | **−0,011** | | |

"Pior cliente" é a menor acurácia de teste de qualquer cliente em qualquer rodada federada.

## Decisão

**Fica o lr 1e-4. Não rodei o teste com 2e-5.**

- **A média geral é a mesma:** 0,868 contra 0,869. A diferença está bem abaixo da variação entre execuções.
- **Nas configurações com alpha 0.1, o lr 1e-4 é melhor:** +0,006, +0,022 e +0,023.
- **A estabilidade melhorou bastante:**
  - Com lr 1e-3, algum cliente caía para 0,50–0,70 em 6 das 9 configurações.
  - Com lr 1e-4, nenhum cliente fica abaixo de 0,81 em nenhuma rodada.
  - A queda na rodada 1 desapareceu.
- **O custo aparece nas configurações IID com pouco dado público:** em 0.1 / iid, a perda é de 0,031. Com
  revisit fraco, o digest pesa mais e o modelo aprende menos com os dados privados.
- **Por que não testar 2e-5:** nos dados IID, o 2e-5 tende a piorar mais. Esse padrão apareceu no cenário 2,
  sem rehearsal: com revisit fraco, os modelos perdem a ancoragem nos rótulos. A estabilidade nos casos não
  IID, que era o problema, já foi resolvida com 1e-4.

## Ainda sem ganho sobre o public_init

A média fica 0,011 abaixo do modelo treinado só nos dados públicos. Ela só fica acima, e por muito pouco
(no máximo +0,003), nas configurações 0.4 / iid, 0.8 / iid e 0.8 / 1.0. O treino federado agora é estável,
mas ainda não melhora o modelo inicial.

## Próximos passos possíveis

- **Fração de rehearsal (0,25 e 0,75)**, já planejada.
- **lr do digest (1e-4).** É o único treino de cada rodada que não usa rótulos. Ele dá entre ~24 e ~190
  passos por rodada com lr 1e-3, contra 30 passos do revisit, agora em 1e-4. O desequilíbrio entre os dois
  é o candidato mais provável para explicar por que o modelo não passa do `public_init`.
- **Controle sem digest**, para medir a contribuição do FedMD isoladamente.

Arquivos: `lr1e-4_test_metrics.csv`, `lr1e-4_train_history.csv` e `lr1e-4.log`.
