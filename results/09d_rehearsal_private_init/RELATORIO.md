# Relatório da noite de 01/11: rehearsal no private_init

A noite inteira rodou num único script (`orquestrador.sh`), das 02:12 às 08:40, sem erros nem travamentos.
As decisões seguiram regras fixas, definidas antes de começar. O registro está em `orquestrador.log`.

## Resumo

1. **Rehearsal no `private_init` com lr 1e-4 e 100 passos.** Com essa configuração, o modelo aprende com os
   dados privados sem quebrar. Nenhum dos 36 clientes perdeu mais de 0,024. Os clientes IID chegam a
   melhorar até +0,021 em relação ao `public_init`. O lr 4e-4 quebrou um cliente (−0,083) e foi descartado.
2. **O resultado final não mudou com isso.** Com duas seeds, as duas configurações ficam em média
   **0,000** em relação ao `public_init`. O ganho do `private_init` some ao longo das 30 rodadas.
3. **Esta foi a primeira comparação com duas seeds, e ela confirma um ganho real do FedMD com 10% de dados
   públicos.** Nas configurações 0.1 / iid e 0.1 / 1.0, o resultado final fica **+0,017 a +0,021** acima do
   `public_init`, nas duas seeds e nas duas configurações. É o resultado mais consistente até agora.
4. **Com mais dados públicos ou dados muito desbalanceados, o resultado fica levemente abaixo do
   `public_init`** (−0,007 a −0,020) nas configurações 0.4 / 1.0, 0.4 / 0.1, 0.8 / 1.0 e 0.8 / 0.1.
5. **O ruído entre seeds é grande.** Na mesma configuração, a diferença entre seeds chega a ~0,03 (0.4 / 0.1
   anterior: +0,004 com a seed 42 e −0,024 com a seed 1). Diferenças menores que ~0,015 não são
   conclusivas.

## A. Teste do private_init (sem rodadas federadas)

Configuração comum: rehearsal de 50%, 100 passos. O critério é que nenhum cliente perca mais de 0,05 em
relação ao próprio `public_init`.

| private_init | Clientes que quebram (de 36) | Variação média | Pior | Melhor |
|---|---|---|---|---|
| 10 passos, lr 2e-5, sem rehearsal (anterior) | 0 | −0,005 | −0,025 | ~0 |
| 100 passos, lr 4e-4, rehearsal (A1) | 1 | −0,005 | −0,083 | +0,027 |
| **100 passos, lr 1e-4, rehearsal (A2)** | **0** | **+0,002** | **−0,024** | **+0,025** |

**Decisão do orquestrador: lr 1e-4.** O A1 falhou por um único cliente: o cliente 2 da configuração
0.1 / 1.0, que tem 14% de imagens malignas, ou seja, não é um cliente de uma classe só.

## B e C. Execuções completas (30 rodadas)

Configuração comum: revisit com 3×10 passos, lr 1e-4 e 50% de rehearsal; digest com 30 passos e lr 4e-4;
consenso ponderado.

- **Anterior:** `private_init` com 10 passos, lr 2e-5, sem rehearsal. A seed 42 é a execução
  `results/09c_digest_30_passos/steps30`.
- **Rehearsal init:** `private_init` com 100 passos, lr 1e-4 e rehearsal.

### Média das rodadas 26–30

| Configuração | Anterior s42 | Anterior s1 | Rehearsal init s42 | Rehearsal init s1 |
|---|---|---|---|---|
| 0.1 / iid | 0,877 | 0,885 | 0,885 | 0,878 |
| 0.1 / 1.0 | 0,880 | 0,866 | 0,884 | 0,871 |
| 0.1 / 0.1 | 0,860 | 0,862 | 0,860 | 0,862 |
| 0.4 / iid | 0,883 | 0,884 | 0,889 | 0,878 |
| 0.4 / 1.0 | 0,874 | 0,875 | 0,865 | 0,869 |
| 0.4 / 0.1 | 0,876 | 0,865 | 0,878 | 0,872 |
| 0.8 / iid | 0,896 | 0,887 | 0,893 | 0,891 |
| 0.8 / 1.0 | 0,882 | 0,873 | 0,879 | 0,868 |
| 0.8 / 0.1 | 0,877 | 0,877 | 0,884 | 0,874 |

### Diferença para o public_init da mesma execução

| Configuração | Anterior s42 | Anterior s1 | Rehearsal init s42 | Rehearsal init s1 | **Média anterior** | **Média rehearsal init** |
|---|---|---|---|---|---|---|
| 0.1 / iid | +0,014 | +0,022 | +0,011 | +0,031 | **+0,018** | **+0,021** |
| 0.1 / 1.0 | +0,012 | +0,022 | +0,012 | +0,021 | **+0,017** | **+0,017** |
| 0.1 / 0.1 | −0,000 | +0,007 | −0,010 | +0,007 | +0,003 | −0,001 |
| 0.4 / iid | +0,007 | −0,004 | +0,007 | +0,005 | +0,001 | +0,006 |
| 0.4 / 1.0 | −0,007 | −0,014 | −0,021 | −0,020 | −0,011 | −0,020 |
| 0.4 / 0.1 | +0,004 | −0,024 | −0,003 | −0,020 | −0,010 | −0,012 |
| 0.8 / iid | +0,001 | −0,002 | +0,007 | +0,009 | −0,001 | +0,008 |
| 0.8 / 1.0 | −0,002 | −0,014 | −0,012 | −0,003 | −0,008 | −0,008 |
| 0.8 / 0.1 | −0,010 | −0,013 | −0,005 | −0,008 | −0,011 | −0,007 |
| **Média** | +0,002 | −0,002 | −0,002 | +0,002 | **0,000** | **0,000** |

| Execução | Média r26–30 | Configurações ≥ public_init | Pior cliente (qualquer rodada) |
|---|---|---|---|
| Anterior s42 | 0,878 | 5 / 9 | 0,819 |
| Anterior s1 | 0,875 | 3 / 9 | 0,812 |
| Rehearsal init s42 | 0,880 | 4 / 9 | 0,837 |
| Rehearsal init s1 | 0,874 | 5 / 9 | 0,836 |

## Interpretação

- **O rehearsal no `private_init` funciona como pretendido, mas não muda o resultado final.** Os clientes
  saem do `private_init` melhores, com até +0,021 nos casos IID, e o pior cliente nas rodadas fica um pouco
  mais alto (0,836 contra 0,812–0,819). Mesmo assim, as 30 rodadas de digest e revisit levam os modelos ao
  mesmo patamar, qualquer que seja o ponto de partida.
- **O FedMD ajuda onde o `public_init` é fraco e os dados privados trazem informação nova.** Com 10% de
  dados públicos, o `public_init` fica em ~0,86–0,87. Nos casos IID e com alpha 1.0, os clientes têm as duas
  classes, e a colaboração soma **~+0,02** de forma consistente.
- **Onde não ajuda, há duas causas:**
  - **Muitos dados públicos (40% e 80%):** o `public_init` já está em ~0,88–0,90, perto do que estas CNNs
    conseguem. Sobra pouco espaço para melhorar.
  - **Dados muito desbalanceados (alpha 0.1):** os clientes só têm uma classe nos dados privados, então não
    têm informação nova sobre a outra. O ganho nesse caso é apenas não colapsar.

## Sugestões para os próximos experimentos

1. **Controle sem digest:** `private_init` e revisit com rehearsal, sem digest. Agora que existe um ganho
   consistente (+0,02 com 10% público), esse controle é o que diz se o ganho vem da destilação ou do
   rehearsal. É o experimento mais importante para o artigo.
2. **Frações públicas pequenas:** 0,05 e 0,2. O ganho aparece justamente quando há pouco dado público.
   Mapear essa faixa mostra até onde o FedMD compensa.
3. **Teto centralizado:** cada arquitetura treinada com todo o conjunto de treino. Isso mede quanto espaço
   de melhora existe com 40% e 80% de dados públicos.
4. **Destilação com KL e temperatura** no lugar do MAE nos logits. Pode transferir melhor a confiança
   relativa entre as classes, principalmente com arquiteturas diferentes.
5. **Mais seeds (≥3) na configuração final.** Com o ruído atual (~0,03 entre seeds na mesma configuração),
   duas seeds bastam para o ganho de +0,02, mas não para diferenças menores.

**Configuração recomendada até agora** (as duas variantes do `private_init` empatam; o rehearsal deixa o
pior cliente um pouco mais alto):

```
python client.py --digest-rounds 30 --private-init-steps 100 --private-init-lr 1e-4 --private-init-rehearsal \
  --private-epochs 3 --private-steps 10 --revisit-lr 1e-4 --rehearsal-fraction 0.5 --digest-lr 4e-4 --digest-steps 30
```

## Mudança no código

- **`client.py`:** novo argumento `--private-init-rehearsal`, que aplica o rehearsal (com
  `--rehearsal-fraction`) também no `private_init`. As amostras públicas vêm de todo o conjunto público,
  com os rótulos verdadeiros.

## Arquivos (`results/09d_rehearsal_private_init/`)

- **Testes de `private_init`:** `A1_init_lr4e-4_*` e `A2_init_lr1e-4_*`. O critério está em `check_init.py`.
- **Execuções completas:** `B_init1e-4_seed42_*`, `C_init1e-4_seed1_*` e `C_anterior_seed1_*`.
- **Tabelas deste relatório:** `uv run --with tabulate python results/09d_rehearsal_private_init/tabelas.py`.
- **Logs:** `orquestrador.log` (decisões), `server.log` e um `.log` por execução.
