# Relatório: cenários de private_init, rodadas e rehearsal (31/10)

Execução de 01:25 a 18:32, com uma interrupção de 03:10 a 11:07 (ver "Incidentes").
Todas as execuções usaram seed 42, 4 clientes, batch 32, digest de 1 época com lr 1e-3
e consenso ponderado pela acurácia no conjunto público.

## Resumo

1. **O colapso dos clientes com uma classe só vinha do treino privado com lr 1e-3.** Com o `private_init`
   em 10 passos e **lr 2e-5**, nenhum dos 36 clientes perde mais de 0,025 de acurácia em relação ao
   `public_init`. Antes, os clientes com uma classe só caíam para 0,50.
2. **Sem ancoragem nos rótulos, o FedMD degrada todos os modelos, inclusive com dados IID.** No cenário 2
   (revisit com lr 2e-5), a média cai de forma contínua em todas as 9 configurações, e o melhor ponto é
   sempre a rodada 1 ou 2.
3. **O rehearsal resolve a degradação e o colapso ao mesmo tempo.** No cenário 3 (revisit com lr 1e-3 e 50%
   de amostras públicas rotuladas), as curvas ficam estáveis ou sobem. As configurações não IID passam de
   0,47–0,65 (execuções anteriores) para 0,83–0,88.
4. **Ressalva importante:** comparado ao modelo treinado só nos dados públicos (`public_init`), o resultado
   final do cenário 3 é, em média, **0,014 pior**. Nas configurações não IID, a colaboração ainda não
   supera a linha de base de transfer learning. O que o cenário 3 conseguiu foi *não destruir* o modelo
   inicial. Ainda não há ganho atribuível ao FedMD.

## Cenário 1: passos e lr do private_init

O cenário rodou só `public_init` e `private_init` (`--digest-rounds 0`), comparando cada cliente com o
próprio `public_init`. O critério era que nenhum cliente perdesse mais de 0,05.

| private_init | Clientes que quebram (de 36) | Queda média | Uma classe | Duas classes | Pior queda |
|---|---|---|---|---|---|
| 10 passos, lr 1e-3 | 22 | −0,189 | −0,348 | −0,099 | −0,46 |
| 5 passos, lr 1e-3 | 20 | −0,105 | −0,208 | −0,046 | −0,41 |
| 3 passos, lr 1e-3 | 7 | −0,039 | −0,084 | −0,013 | −0,29 |
| 10 passos, lr 1e-4 | 4 | −0,012 | −0,032 | −0,001 | −0,16 |
| **10 passos, lr 2e-5** | **0** | **−0,005** | **−0,013** | **0,000** | **−0,025** |

**Decisão: 10 passos com lr 2e-5.** Reduzir o número de passos tinha um limite: com 3 passos, o modelo quase
não aprende, e ainda havia quedas por ruído, provavelmente do BatchNorm atualizando as médias móveis com
poucos batches. Com AdamW, cada passo desloca os pesos em aproximadamente o valor do lr, então baixar o lr
foi a alavanca certa.

O lado negativo é que, com 10 passos a lr 2e-5, o `private_init` praticamente não altera o modelo: a
acurácia fica quase igual à do `public_init`.

## Cenário 2: número de rodadas (revisit com 3×10 passos e lr 2e-5, 60 rodadas)

Média dos 4 clientes por rodada:

| Configuração | private_init | r1 | r5 | r10 | r20 | r40 | r60 | Pico (rodada) |
|---|---|---|---|---|---|---|---|---|
| 0.1 / iid | 0,879 | 0,873 | 0,867 | 0,855 | 0,855 | 0,848 | 0,850 | 0,873 (1) |
| 0.1 / 1.0 | 0,873 | 0,866 | 0,856 | 0,840 | 0,843 | 0,840 | 0,839 | 0,866 (1) |
| 0.1 / 0.1 | 0,866 | 0,852 | 0,833 | 0,766 | 0,731 | 0,639 | 0,589 | 0,859 (2) |
| 0.4 / iid | 0,882 | 0,877 | 0,871 | 0,865 | 0,863 | 0,849 | 0,863 | 0,877 (1) |
| 0.4 / 1.0 | 0,859 | 0,856 | 0,795 | 0,697 | 0,670 | 0,608 | 0,574 | 0,856 (1) |
| 0.4 / 0.1 | 0,873 | 0,865 | 0,846 | 0,844 | 0,823 | 0,793 | 0,719 | 0,865 (1) |
| 0.8 / iid | 0,899 | 0,897 | 0,882 | 0,854 | 0,816 | 0,745 | 0,695 | 0,897 (1) |
| 0.8 / 1.0 | 0,891 | 0,898 | 0,890 | 0,870 | 0,833 | 0,758 | 0,660 | 0,899 (2) |
| 0.8 / 0.1 | 0,892 | 0,853 | 0,747 | 0,623 | 0,553 | 0,495 | 0,560 | 0,853 (1) |

**Decisão: com revisit fraco, o número ótimo de rodadas é de 1 a 2. Em todas as configurações, mais rodadas
só pioram o resultado.**

A queda nas configurações IID, em que não existe colapso para uma classe, mostrou a causa. O digest dá
~150 passos por rodada com lr 1e-3 em direção a um consenso que é a média dos próprios modelos. Sem um
revisit forte nos rótulos verdadeiros, os modelos se destilam entre si e perdem qualidade aos poucos. O
número de rodadas não é o problema. O problema é a falta de ancoragem.

## Cenário 3: rehearsal (revisit com 3×10 passos, lr 1e-3 e 50% de amostras públicas rotuladas, 30 rodadas)

### Comparação com as execuções anteriores (média dos 4 clientes)

| Configuração | Uniforme (r16–20) | Ponderado (r16–20) | Cen. 2 (r16–20) | Cen. 3 public_init | **Cen. 3 (r26–30)** | Cen. 3 pico |
|---|---|---|---|---|---|---|
| 0.1 / iid | 0,898 | 0,899 | 0,858 | 0,867 | **0,889** | 0,897 |
| 0.1 / 1.0 | 0,892 | 0,886 | 0,844 | 0,879 | **0,874** | 0,884 |
| 0.1 / 0.1 | 0,627 | 0,608 | 0,737 | 0,863 | **0,842** | 0,854 |
| 0.4 / iid | 0,891 | 0,895 | 0,867 | 0,887 | **0,891** | 0,896 |
| 0.4 / 1.0 | 0,653 | 0,654 | 0,675 | 0,882 | **0,860** | 0,877 |
| 0.4 / 0.1 | 0,503 | 0,544 | 0,838 | 0,884 | **0,832** | 0,870 |
| 0.8 / iid | 0,892 | 0,897 | 0,816 | 0,896 | **0,897** | 0,902 |
| 0.8 / 1.0 | 0,778 | 0,774 | 0,831 | 0,897 | **0,881** | 0,897 |
| 0.8 / 0.1 | 0,475 | 0,530 | 0,561 | 0,894 | **0,854** | 0,880 |

As colunas "Uniforme" e "Ponderado" são as execuções de 25/10, com revisit completo e lr 1e-3 (sem limite
de passos).

### Por cliente (média r26–30) e fração de imagens malignas nos dados privados

| Configuração | Cliente 0 | Cliente 1 | Cliente 2 | Cliente 3 | % malignas (0 / 1 / 2 / 3) |
|---|---|---|---|---|---|
| 0.1 / 0.1 | 0,825 | 0,848 | 0,824 | 0,869 | 100 / 66 / 100 / 0 |
| 0.4 / 1.0 | 0,868 | 0,868 | 0,847 | 0,858 | 3 / 6 / 0 / 95 |
| 0.4 / 0.1 | 0,778 | 0,849 | 0,829 | 0,872 | 100 / 1 / 100 / 0 |
| 0.8 / 1.0 | 0,895 | 0,883 | 0,844 | 0,902 | 57 / 86 / 0 / 39 |
| 0.8 / 0.1 | 0,839 | 0,845 | 0,867 | 0,865 | 100 / 100 / 100 / 0 |

Nenhum cliente com dados de uma classe só termina abaixo de 0,77. Antes, todos esses clientes ficavam em 0,50.

### Curva da média por rodada

| Configuração | r1 | r5 | r10 | r15 | r20 | r25 | r30 |
|---|---|---|---|---|---|---|---|
| 0.1 / iid | 0,859 | 0,867 | 0,876 | 0,876 | 0,882 | 0,891 | 0,897 |
| 0.1 / 1.0 | 0,807 | 0,852 | 0,863 | 0,873 | 0,865 | 0,879 | 0,884 |
| 0.1 / 0.1 | 0,708 | 0,832 | 0,803 | 0,741 | 0,851 | 0,854 | 0,838 |
| 0.4 / iid | 0,889 | 0,869 | 0,878 | 0,888 | 0,896 | 0,896 | 0,888 |
| 0.4 / 1.0 | 0,723 | 0,848 | 0,864 | 0,852 | 0,854 | 0,862 | 0,874 |
| 0.4 / 0.1 | 0,659 | 0,786 | 0,859 | 0,848 | 0,850 | 0,845 | 0,812 |
| 0.8 / iid | 0,873 | 0,889 | 0,895 | 0,896 | 0,893 | 0,894 | 0,895 |
| 0.8 / 1.0 | 0,774 | 0,871 | 0,861 | 0,894 | 0,882 | 0,884 | 0,892 |
| 0.8 / 0.1 | 0,739 | 0,821 | 0,771 | 0,872 | 0,874 | 0,857 | 0,877 |

Dois padrões aparecem:

- **Há uma queda na rodada 1 das configurações não IID.** A média cai para 0,66–0,77 e depois se recupera.
  O primeiro revisit com lr 1e-3 desloca o modelo antes que o rehearsal compense.
- **Há oscilação entre rodadas.** Clientes com uma classe só caem por algumas rodadas para ~0,6–0,7 e se
  recuperam. Isso acontece mesmo com rehearsal de 50%.

## Conclusões

- **Configuração recomendada até agora:** `private_init` com 10 passos e lr 2e-5. Revisit com 3×10 passos,
  lr 1e-3 e 50% de rehearsal. Consenso ponderado. Entre 20 e 30 rodadas.
- **O ganho em relação às execuções anteriores vem de não destruir o modelo do `public_init`, e não da
  colaboração.** Comparado ao `public_init`, o resultado das rodadas 26–30 fica assim:

  | Configuração | public_init | Cen. 3 (r26–30) | Diferença |
  |---|---|---|---|
  | 0.1 / iid | 0,867 | 0,889 | +0,023 |
  | 0.1 / 1.0 | 0,879 | 0,874 | −0,004 |
  | 0.1 / 0.1 | 0,863 | 0,842 | −0,021 |
  | 0.4 / iid | 0,887 | 0,891 | +0,004 |
  | 0.4 / 1.0 | 0,882 | 0,860 | −0,022 |
  | 0.4 / 0.1 | 0,884 | 0,832 | −0,052 |
  | 0.8 / iid | 0,896 | 0,897 | +0,001 |
  | 0.8 / 1.0 | 0,897 | 0,881 | −0,016 |
  | 0.8 / 0.1 | 0,894 | 0,854 | −0,040 |

  A diferença média é **−0,014**. Só a configuração 0.1 / iid mostra ganho claro.
- **O rehearsal mistura dois efeitos.** Com o rehearsal, o revisit passa a treinar também com os rótulos
  públicos. Parte do resultado pode vir disso, e não do digest. Para separar os dois efeitos, falta um
  experimento de controle (ver abaixo).
- **Todas as execuções usaram uma seed só.** Em execuções anteriores, a variação entre execuções idênticas
  chegou a ~0,02–0,05 nas configurações não IID. Diferenças desse tamanho nesta tabela não são conclusivas.

## Sugestões para os próximos experimentos

1. **Controle sem digest:** rodar o cenário 3 com o digest desligado, mantendo só o revisit com rehearsal.
   Se o resultado for igual, a contribuição do FedMD é nula neste cenário.
2. **Seeds 1 e 2** com a configuração recomendada, para medir a variância e tornar as comparações válidas.
3. **Fração de rehearsal:** testar 0,25 e 0,75. Com 0,5, ainda há oscilação nos clientes com uma classe
   só. Uma fração maior pode estabilizar o treino, e uma menor deixaria mais espaço para os dados privados.
4. **lr do revisit com rehearsal:** testar 1e-4. A queda na rodada 1 e a oscilação sugerem que lr 1e-3 é
   alto mesmo com rehearsal.
5. **Escolha da rodada por validação:** hoje o resultado é medido na última rodada. Escolher a rodada pela
   validação pública evitaria terminar num ponto de oscilação.
6. **Proteção contra travamentos:** colocar timeout no `stub.SendClientLogits` e nos `ray.get` do
   `run_experiment`. Assim, um cliente travado vira erro em vez de parar a execução por horas.

## Mudanças no código (`client.py`)

| Argumento | Função |
|---|---|
| `--private-init-steps N` | `private_init` em 1 época de N passos. Sem o argumento, até `PRIVATE_INIT_EPOCHS = 5` épocas completas. |
| `--private-init-lr` | lr do `private_init`. O revisit continua usando `--revisit-lr`. |
| `--rehearsal-fraction` | Fração de amostras públicas rotuladas em cada batch do revisit. O padrão, 0, desliga o rehearsal. Sem `--private-steps`, cada época tem os passos de uma passada pelos dados privados. |

Os CSVs agora têm a coluna `seed`.

Comando do cenário 3:

```
python server.py --clients 4
python client.py --digest-rounds 30 --private-init-steps 10 --private-init-lr 2e-5 \
  --private-epochs 3 --private-steps 10 --revisit-lr 1e-3 --rehearsal-fraction 0.5
```

## Incidentes

- **Os cenários 2 e 3 deveriam ter rodado de madrugada, mas ficaram parados de 03:10 a 11:07.** Eu tinha
  deixado uma única espera em segundo plano como gatilho para continuar, e o aviso dela não chegou.
  Provavelmente a sessão do Claude Code foi encerrada. Depois disso, o cenário 3 passou a ser disparado
  por um script (`cenario3.sh`) que espera o PID do cenário 2, sem depender de mim.
- **Não houve travamento nesta série.** O alarme de log parado (15 min) ficou ativo o tempo todo e só deu
  um falso positivo: era o monitor de um teste que já tinha terminado.

## Arquivos (`results/08_rehearsal_no_revisit/`)

- **Cenário 1:** `cen1_init{10,5,3}_*` (lr 1e-3) e `cen1_init10_lr{1e-4,2e-5}_*`. Análise em `analise_cen1.py`.
- **Cenário 2:** `cen2_test_metrics.csv`, `cen2_train_history.csv` e `cen2.log`.
- **Cenário 3:** `cen3_test_metrics.csv`, `cen3_train_history.csv` e `cen3.log`.
- **Tabelas deste relatório:** `uv run --with tabulate python results/08_rehearsal_no_revisit/tabelas_relatorio.py`.
- **Servidor:** `server.log`, com os pesos do consenso em cada rodada.
- **Descartado:** `descartado_init10steps_*`, a execução interrompida antes do plano.
