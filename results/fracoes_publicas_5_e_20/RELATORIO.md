# Frações públicas pequenas: 0,05 e 0,2 (01/11, 11:54–16:59)

Rodamos 6 configurações: frações públicas de 0,05 e 0,2, cada uma com divisão IID, alpha 1.0 e alpha 0.1.
Todas as execuções terminaram sem erros. O registro está em `orquestrador.log` e `complemento.log`.

## Resumo

1. **Com 5% de dados públicos, o FedMD dá o maior ganho de todos os experimentos: +0,03 a +0,04 sobre o
   `public_init`** nas configurações IID e alpha 1.0. O ganho se repete nas duas seeds e nas duas
   combinações de hiperparâmetros testadas.
2. **Pela primeira vez houve ganho no caso mais desbalanceado.** Com 5% público e alpha 0.1, a combinação
   com `public_init` em lr 1e-3 chega a **+0,017** (+0,010 com a seed 42, +0,024 com a seed 1).
3. **Com 20% de dados públicos, o ganho fica menor ou some:** +0,011 a +0,016 no IID, ~0 com alpha 1.0 e
   negativo com alpha 0.1.
4. **A sua hipótese se confirmou em parte.** Um `public_init` menos forte (lr 1e-3) deixa mais espaço para
   o FedMD: o ganho médio vai de +0,009 para +0,016. A acurácia final, porém, é praticamente a mesma nas
   duas combinações (0,877 contra 0,875). O que muda é o ponto de partida, não o ponto de chegada.

## Fase 1: hiperparâmetros das etapas iniciais

Só `public_init` e `private_init`, com `private_init` usando rehearsal de 50%. O critério era que nenhum
cliente perdesse mais de 0,05, e a ordenação é pela acurácia média depois do `private_init`.

| lr do public_init | private_init | Após public_init | Após private_init | Variação média | Pior | Quebram |
|---|---|---|---|---|---|---|
| **3e-4** | lr 1e-4, 300 passos | **0,876** | **0,881** | +0,005 | −0,046 | 0 |
| **1e-3** | lr 1e-4, 300 passos | 0,867 | 0,876 | **+0,010** | −0,026 | 0 |
| 3e-4 | lr 1e-4, 100 passos | 0,873 | 0,873 | −0,001 | −0,025 | 0 |
| 1e-3 | lr 1e-4, 100 passos | 0,867 | 0,867 | 0,000 | −0,033 | 0 |
| 1e-3 | lr 3e-4, 100 passos | 0,863 | 0,867 | +0,003 | −0,069 | 1 |
| 3e-4 | lr 3e-4, 100 passos | 0,874 | 0,863 | −0,011 | −0,086 | 1 |

- **Com pouco dado público, o lr menor no `public_init` deixa a inicialização mais forte** (0,876 contra
  0,867). Com lr 1e-3, o modelo provavelmente se ajusta demais às poucas imagens antes de o early stopping
  agir.
- **Partindo do `public_init` com lr 1e-3, o modelo aprende mais no `private_init`** (+0,010 contra +0,005).
- **lr 3e-4 no `private_init` é alto demais** com qualquer `public_init`: as duas combinações quebram 1
  cliente.
- **Com lr 1e-4, 300 passos no `private_init` funcionam melhor que 100.**

As duas primeiras combinações ficaram a menos de 0,005 uma da outra, então as duas foram para a fase 2.

## Fase 2: 30 rodadas, seeds 42 e 1

O resto da configuração é o mesmo de antes:

- Revisit: 3×10 passos, lr 1e-4, 50% de rehearsal.
- Digest: 30 passos, lr 4e-4.
- Consenso ponderado.
- A: `public_init` com lr 3e-4. B: `public_init` com lr 1e-3. As duas com `private_init` em lr 1e-4, 300
  passos e rehearsal.

### Média das duas seeds

| Configuração | A: public_init | A: final (r26–30) | **A: ganho** | B: public_init | B: final (r26–30) | **B: ganho** |
|---|---|---|---|---|---|---|
| 0.05 / iid | 0,859 | 0,891 | **+0,032** | 0,849 | 0,889 | **+0,040** |
| 0.05 / 1.0 | 0,856 | 0,885 | **+0,029** | 0,852 | 0,886 | **+0,034** |
| 0.05 / 0.1 | 0,858 | 0,854 | −0,004 | 0,851 | 0,868 | **+0,017** |
| 0.2 / iid | 0,877 | 0,888 | +0,011 | 0,867 | 0,884 | +0,016 |
| 0.2 / 1.0 | 0,880 | 0,880 | 0,000 | 0,869 | 0,867 | −0,002 |
| 0.2 / 0.1 | 0,878 | 0,862 | −0,016 | 0,863 | 0,854 | −0,009 |
| **Média** | | **0,877** | **+0,009** | | **0,875** | **+0,016** |

### Ganho por seed (final − public_init)

| Configuração | A s42 | A s1 | B s42 | B s1 |
|---|---|---|---|---|
| 0.05 / iid | +0,026 | +0,038 | +0,031 | +0,049 |
| 0.05 / 1.0 | +0,019 | +0,039 | +0,025 | +0,043 |
| 0.05 / 0.1 | −0,007 | −0,002 | +0,010 | +0,024 |
| 0.2 / iid | +0,009 | +0,012 | +0,020 | +0,013 |
| 0.2 / 1.0 | −0,005 | +0,005 | +0,004 | −0,008 |
| 0.2 / 0.1 | −0,014 | −0,019 | +0,001 | −0,018 |

## Interpretação

- **O ganho do FedMD cresce quando há menos dado público.** Juntando todos os experimentos com a
  configuração adaptada, o ganho no caso IID é aproximadamente:

  | Fração pública | 5% | 10% | 20% | 40% | 80% |
  |---|---|---|---|---|---|
  | Ganho no IID | +0,03–0,04 | +0,02 | +0,01–0,02 | ~0 | ~0 |

  Quanto menos dado público, mais fraco é o `public_init` e mais o conhecimento dos dados privados
  acrescenta.
- **A escolha da linha de base muda a conclusão.** O ganho de B é maior em parte porque o `public_init`
  dele é pior. Uma comparação mais rigorosa usa como referência o melhor `public_init` disponível, que é o
  de A:
  - Com 5% de dados públicos, o resultado final de B continua acima: +0,030 (IID), +0,030 (alpha 1.0) e
    +0,010 (alpha 0.1). **O ganho com 5% se mantém nessa comparação.**
  - Com 20%, o resultado final de B fica abaixo do `public_init` de A com alpha 1.0 (−0,013) e com alpha
    0.1 (−0,024).
- **Recomendação por fração:**
  - **5%:** `public_init` com lr 1e-3 (B), que é melhor com alpha 0.1 (0,868 contra 0,854) e empata nas
    outras.
  - **20%:** `public_init` com lr 3e-4 (A), que é melhor com alpha 1.0 (0,880 contra 0,867) e com alpha 0.1
    (0,862 contra 0,854).

  A fronteira faz sentido: quanto menos dados públicos, menos o `public_init` deve "fechar" o modelo.
- **Ressalvas:**
  - Os hiperparâmetros foram escolhidos olhando o conjunto de teste.
  - Há duas seeds. A variação entre elas chega a ~0,02 por configuração, então só ganhos de ~0,02 ou mais
    (os casos de 5% público) devem ser considerados robustos.

## Sugestões

1. **Controle sem digest**, agora com 5% de dados públicos, onde o ganho é maior e mais claro. É o teste
   que diz se o +0,03–0,04 vem da destilação ou do rehearsal com dados públicos.
2. **Uma seed a mais (seed 2)** nas configurações recomendadas para 5% e 20%, para fechar os números do
   artigo com 3 seeds.
3. **Frações ainda menores (0,02)**, para ver se o ganho continua crescendo ou se o `public_init` fica
   fraco demais.
4. **Destilação com KL e temperatura** no lugar do MAE nos logits.

## Mudança no código

- **`client.py`:** novo argumento `--public-init-lr`, com o lr do treino inicial nos dados públicos. O
  padrão continua sendo o do compile (1e-3).

## Arquivos (`results/fracoes_publicas_5_e_20/`)

- **Fase 1:** `F1_pub{lr}_priv{lr}_s{passos}_*`. Seleção em `seleciona.py`, escolhas em `escolha.txt` e
  `escolha2.txt`.
- **Fase 2:** `F2_seed{42,1}_*` (A, `public_init` com lr 3e-4) e `F2b_seed{42,1}_*` (B, `public_init` com
  lr 1e-3).
- **Logs:** `orquestrador.log`, `complemento.log`, `server.log` e um `.log` por execução.

**Comandos (5% de dados públicos, configuração B):**

```
python server.py --clients 4
python client.py --public-fractions 0.05 --digest-rounds 30 --public-init-lr 1e-3 \
  --private-init-steps 300 --private-init-lr 1e-4 --private-init-rehearsal \
  --private-epochs 3 --private-steps 10 --revisit-lr 1e-4 --rehearsal-fraction 0.5 --digest-lr 4e-4 --digest-steps 30
```

Para 20% de dados públicos (configuração A), use `--public-fractions 0.2 --public-init-lr 3e-4` e mantenha
o resto.
