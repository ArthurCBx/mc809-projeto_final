# Digest com lr 4e-4 (31/10, 21:21–23:50)

## Configuração

- `private_init`: 10 passos, lr 2e-5.
- Revisit: 3 épocas de 10 passos, lr 1e-4, com 50% de rehearsal.
- **Digest: lr 4e-4.** Nos testes anteriores era 1e-3.
- 30 rodadas, seed 42, consenso ponderado.

A coluna *Cen. 3* usa lr 1e-3 no revisit e no digest (`results/08_rehearsal_no_revisit/RELATORIO.md`). A coluna
*Revisit 1e-4* usa lr 1e-4 no revisit e 1e-3 no digest (`results/09a_revisit_lr_1e-4/RESULTADO.md`).

## Resultados

Média dos 4 clientes nas rodadas 26 a 30:

| Configuração | Cen. 3 | Revisit 1e-4 | **Digest 4e-4** | public_init | Digest 4e-4 − public_init | Pior cliente (qualquer rodada) |
|---|---|---|---|---|---|---|
| 0.1 / iid | 0,889 | 0,859 | 0,879 | 0,868 | **+0,011** | 0,846 |
| 0.1 / 1.0 | 0,874 | 0,859 | 0,873 | 0,870 | **+0,002** | 0,846 |
| 0.1 / 0.1 | 0,842 | 0,847 | 0,869 | 0,866 | **+0,003** | 0,830 |
| 0.4 / iid | 0,891 | 0,886 | 0,888 | 0,884 | **+0,004** | 0,874 |
| 0.4 / 1.0 | 0,860 | 0,853 | 0,859 | 0,882 | −0,022 | 0,824 |
| 0.4 / 0.1 | 0,832 | 0,854 | 0,873 | 0,882 | −0,008 | 0,852 |
| 0.8 / iid | 0,897 | 0,890 | 0,888 | 0,896 | −0,008 | 0,880 |
| 0.8 / 1.0 | 0,881 | 0,886 | 0,886 | 0,899 | −0,013 | 0,870 |
| 0.8 / 0.1 | 0,854 | 0,877 | 0,875 | 0,894 | −0,019 | 0,811 |
| **Média** | 0,869 | 0,868 | **0,877** | | **−0,006** | |

## Conclusões

- **É a melhor configuração até agora.** A média é 0,877, contra 0,868–0,869 nas duas anteriores.
- **Também é estável.** Nenhum cliente fica abaixo de 0,81 em nenhuma rodada, mesmo nos que só têm uma
  classe nos dados privados.
- **Com 10% de dados públicos, o treino federado supera o `public_init` nas 3 configurações** (+0,002 a
  +0,011). É a primeira vez que isso acontece com dados não IID. O ganho ainda é pequeno e foi medido com
  uma seed só.
- **Com mais dados públicos, ainda fica abaixo do `public_init`** (40%: −0,022 a +0,004; 80%: −0,008 a
  −0,019). O `public_init` já é forte nessas configurações (~0,88–0,90). Além disso, o número de passos
  do digest cresce com a fração pública: ~24, ~96 e ~190 por rodada. Com lr 4e-4, isso ainda é bem mais
  que os 30 passos do revisit.

## Próximo passo sugerido

**Limitar o número de passos do digest por rodada.** Por exemplo, usar 30 passos, o mesmo do revisit, ou
outro valor fixo para todas as frações públicas. Hoje o peso do digest depende da fração pública, e as
configurações em que o resultado fica abaixo do `public_init` são justamente as que têm mais dados
públicos. Com um limite fixo, o equilíbrio entre digest e revisit fica igual em todas as configurações.

Arquivos: `lr4e-4_test_metrics.csv`, `lr4e-4_train_history.csv` e `lr4e-4.log`.
