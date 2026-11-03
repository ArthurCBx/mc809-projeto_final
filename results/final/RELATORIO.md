# Experimento final: frações públicas 1%, 5% e 20% com 3 seeds (01/11)

## Configuração

| Etapa | Configuração |
|---|---|
| `public_init` | lr **3e-4** com 1% e 20% de dados públicos; lr **1e-3** com 5%. Até 50 épocas, early stopping com patience 4 |
| `private_init` | 300 passos, lr 1e-4, rehearsal de 50% com amostras públicas rotuladas |
| Digest | 30 passos por rodada, lr 4e-4, MAE contra o consenso ponderado pela acurácia pública |
| Revisit | 3 épocas de 10 passos, lr 1e-4, 50% de rehearsal |
| Rodadas | 30, sobre 80% do conjunto público sorteado a cada rodada |

- Rodamos 4 clientes com arquiteturas diferentes, com 3 divisões dos dados privados: IID, alpha 1.0 e
  alpha 0.1. Usamos as seeds 42, 1 e 2.
- **Linha de base:** a acurácia após `public_init` + `private_init`, ou seja, cada cliente sozinho, usando
  os dados públicos e os próprios dados privados.
- **Ganho do FedMD:** a média das rodadas 26 a 30 menos a linha de base.
- As seeds 42 e 1 com 5% e 20% de dados públicos vêm de `results/fracoes_pequenas/` (`F2b` e `F2`). A
  configuração e o código são os mesmos.
- **Escolha do lr do `public_init` para 1%:** testamos só as etapas iniciais com lr 1e-3 e lr 3e-4. O lr
  3e-4 deu 0,861 após o `private_init`, contra 0,857 com 1e-3, e nenhum cliente quebrou nos dois casos.

## Resultado (média ± desvio padrão, 3 seeds)

| Fração / alpha | Linha de base (public + private init) | FedMD (r26–30) | **Ganho** | Seeds com ganho | Pior cliente |
|---|---|---|---|---|---|
| 1% / iid | 0,873 ± 0,004 | 0,882 ± 0,003 | **+0,009 ± 0,003** | 3/3 | 0,837 |
| 1% / 1.0 | 0,865 ± 0,004 | 0,871 ± 0,010 | +0,006 ± 0,008 | 2/3 | 0,784 |
| 1% / 0.1 | 0,838 ± 0,003 | 0,852 ± 0,002 | **+0,014 ± 0,005** | 3/3 | 0,793 |
| 5% / iid | 0,877 ± 0,009 | 0,888 ± 0,004 | **+0,011 ± 0,008** | 3/3 | 0,852 |
| 5% / 1.0 | 0,868 ± 0,009 | 0,878 ± 0,013 | **+0,011 ± 0,007** | 3/3 | 0,837 |
| 5% / 0.1 | 0,859 ± 0,009 | 0,866 ± 0,006 | **+0,008 ± 0,004** | 3/3 | 0,825 |
| 20% / iid | 0,894 ± 0,002 | 0,888 ± 0,002 | −0,007 ± 0,003 | 0/3 | 0,868 |
| 20% / 1.0 | 0,884 ± 0,003 | 0,878 ± 0,005 | −0,006 ± 0,007 | 1/3 | 0,827 |
| 20% / 0.1 | 0,879 ± 0,006 | 0,858 ± 0,012 | −0,022 ± 0,010 | 0/3 | 0,813 |

### Ganho por seed

| Fração / alpha | Seed 42 | Seed 1 | Seed 2 |
|---|---|---|---|
| 1% / iid | +0,006 | +0,012 | +0,008 |
| 1% / 1.0 | −0,002 | +0,013 | +0,008 |
| 1% / 0.1 | +0,009 | +0,015 | +0,018 |
| 5% / iid | +0,011 | +0,019 | +0,003 |
| 5% / 1.0 | +0,010 | +0,018 | +0,004 |
| 5% / 0.1 | +0,004 | +0,007 | +0,012 |
| 20% / iid | −0,009 | −0,003 | −0,009 |
| 20% / 1.0 | −0,012 | +0,003 | −0,007 |
| 20% / 0.1 | −0,013 | −0,020 | −0,032 |

### Ganho médio por fração

| Fração pública | Ganho médio | Desvio padrão | Seeds × configurações com ganho |
|---|---|---|---|
| 1% | +0,010 | 0,006 | 8/9 |
| 5% | +0,010 | 0,006 | 9/9 |
| 20% | −0,011 | 0,010 | 1/9 |

### Média de cada etapa

| Fração / alpha | public_init | + private_init | + FedMD |
|---|---|---|---|
| 1% / iid | 0,661 | 0,873 | 0,882 |
| 1% / 1.0 | 0,660 | 0,865 | 0,871 |
| 1% / 0.1 | 0,659 | 0,838 | 0,852 |
| 5% / iid | 0,850 | 0,877 | 0,888 |
| 5% / 1.0 | 0,854 | 0,868 | 0,878 |
| 5% / 0.1 | 0,854 | 0,859 | 0,866 |
| 20% / iid | 0,880 | 0,894 | 0,888 |
| 20% / 1.0 | 0,883 | 0,884 | 0,878 |
| 20% / 0.1 | 0,881 | 0,879 | 0,858 |

## Conclusões

1. **Com pouco dado público (1% e 5%), o FedMD supera a linha de base em todas as configurações, por
   ~+0,01.** O ganho é positivo em 17 das 18 combinações de seed e configuração. O desvio padrão (~0,006) é
   menor que o ganho, então o resultado é consistente, embora pequeno.
2. **Com 20% de dados públicos, o FedMD fica abaixo da linha de base** (−0,006 a −0,022, positivo em só 1
   das 9 combinações). Os clientes já saem bem treinados do `public_init` + `private_init` (~0,88–0,89), e
   as rodadas de destilação puxam os modelos para o consenso em vez de melhorá-los.
3. **O caso mais desbalanceado ganha mais justamente com 1% de dados públicos** (+0,014, positivo nas 3
   seeds). É o único cenário em que a colaboração ajuda de forma clara um cliente que só tem uma classe nos
   dados privados. A linha de base nesse caso é a mais baixa (0,838), então há mais a ganhar com o
   conhecimento dos outros clientes.
4. **Com 1% de dados públicos, o `public_init` praticamente não funciona** (~0,66). Com cerca de 77
   imagens de treino e 19 de validação, o early stopping tende a parar o treino antes de o modelo sair do
   acaso: em geral, só um dos quatro clientes aprende. Quem carrega o modelo é o `private_init`, de 0,66
   para 0,84–0,87. Esse é o cenário mais próximo do que o FedMD supõe: o conhecimento está nos dados
   privados, e o conjunto público serve só de canal para trocá-lo.
5. **O ganho depende da linha de base usada.** Contra o `public_init` sozinho, o ganho com 5% de dados
   públicos era de ~+0,03 no experimento anterior. Contra `public_init` + `private_init`, cai para ~+0,01.
   Ou seja, cerca de dois terços do ganho vêm do treino com os dados privados de cada cliente, e um terço
   vem da colaboração.

## Limite teórico: treino centralizado

Cada uma das 4 arquiteturas foi treinada de forma centralizada, com o conjunto de treino completo, nas
seeds 42, 1 e 2 (`results/teto/`). O treino usou lr 1e-3, batch 32 e a mesma aumentação. Foram até 50
épocas, com early stopping de patience 4 e 20% dos dados para validação. A avaliação foi feita no mesmo
conjunto de teste.

| Arquitetura | Seed 42 | Seed 1 | Seed 2 | Média |
|---|---|---|---|---|
| 0 | 0,902 | 0,896 | 0,908 | 0,902 |
| 1 | 0,902 | 0,880 | 0,898 | 0,893 |
| 2 | 0,907 | 0,904 | 0,890 | 0,900 |
| 3 | 0,898 | 0,899 | 0,897 | 0,898 |
| **Média** | 0,902 | 0,895 | 0,898 | **0,898 ± 0,004** |

Fração da distância até o teto (0,898) que o FedMD recupera, calculada como
(FedMD − linha de base) / (teto − linha de base):

| Fração / alpha | Linha de base | FedMD | Teto | Distância recuperada |
|---|---|---|---|---|
| 1% / iid | 0,873 | 0,882 | 0,898 | 35% |
| 1% / 1.0 | 0,865 | 0,871 | 0,898 | 18% |
| 1% / 0.1 | 0,838 | 0,852 | 0,898 | 23% |
| 5% / iid | 0,877 | 0,888 | 0,898 | **51%** |
| 5% / 1.0 | 0,868 | 0,878 | 0,898 | 34% |
| 5% / 0.1 | 0,859 | 0,866 | 0,898 | 19% |
| 20% / iid | 0,894 | 0,888 | 0,898 | — (a linha de base já está no teto) |
| 20% / 1.0 | 0,884 | 0,878 | 0,898 | — |
| 20% / 0.1 | 0,879 | 0,858 | 0,898 | — |

- **Com 20% de dados públicos, a linha de base já está praticamente no teto** (0,894 contra 0,898 no IID).
  Não sobra espaço para o FedMD ganhar, o que explica o ganho negativo nesse regime.
- **Com 1% e 5%, o FedMD recupera de 18% a 51% da distância até o teto.** O melhor caso é 5% / iid.
- **O teto pode estar um pouco subestimado.** Em 4 dos 12 treinos, o early stopping parou em 8 épocas ou
  menos, com a melhor época entre a 2 e a 4. O pior caso foi a arquitetura 1 com a seed 1, com 0,880. O
  melhor modelo individual chegou a 0,908. Com uma patience maior, o teto provavelmente ficaria entre
  0,90 e 0,91. Mantivemos patience 4 para usar as mesmas condições de treino do resto do projeto.

## Variante: KL com temperatura no digest

Trocamos a loss do digest de MAE nos logits para T² · KL(softmax(consenso/T) ‖ softmax(cliente/T)), com
T = 4, nos cenários de 1% e 5% de dados públicos e 3 seeds (`results/kl/RESULTADO.md`). O ganho médio passou
de +0,0097 com MAE para **+0,0115** com KL. Comparando por par (mesma seed e mesmo cenário), a KL ficou
+0,0019 ± 0,0043 acima, melhor em 12 dos 18 pares (p ≈ 0,08). A vantagem é pequena, aparece mais nos
cenários desbalanceados e não muda a conclusão: **a loss do digest não é o gargalo**.

## Extra: ensemble dos clientes e consenso leave-one-out

Experimento com 1% e 5% de dados públicos, 3 seeds e KL com T = 4 (`results/ensemble_loo/RELATORIO.md`).

- **Ensemble:** fazer a média dos 4 clientes logo após o `private_init`, sem nenhuma rodada federada, já
  dá **+0,020** sobre o cliente médio, o dobro do ganho do FedMD por cliente (+0,010). O efeito chega a
  +0,041 com 1% de dados públicos e alpha 0.1.
- **O FedMD reduz a diversidade entre os modelos:** com 1% de dados públicos, ele piora o ensemble (−0,004
  a −0,013). Com 5%, o efeito é neutro ou levemente positivo, e o melhor resultado de sistema é 0,892, em
  5% / iid.
- **Leitura:** o FedMD é a escolha certa quando cada cliente precisa de um modelo próprio para inferência
  local. Se for possível combinar os modelos na inferência, o ensemble sem colaboração é melhor e mais
  barato.
- **Leave-one-out** (cada cliente recebe o consenso dos outros três): sem efeito (−0,0004 ± 0,0048).

## Extra: 0,5% de dados públicos

Rodamos o cenário com 49 imagens públicas na configuração da etapa final: KL com T = 4, consenso padrão e
3 seeds (`results/meio_pct/RESULTADO.md`).

- **O `public_init` não aprende nada** (~0,51). Todo o aprendizado vem do `private_init`.
- **O ganho do FedMD por cliente é +0,013 em média**, positivo nas 9 combinações de seed e cenário. O maior
  ganho é com alpha 0.1: +0,019.
- **Contra a linha de base `public_init` + `private_init`, o ganho fica estável em ~+0,01 entre 0,5% e 5%**
  (0,5%: +0,013, 1%: +0,010, 5%: +0,011). Ele só "cresce" quando a comparação é com o `public_init`
  sozinho. Quem compensa a falta de dados públicos é o `private_init`. O FedMD acrescenta cerca de 1 ponto
  sempre que a linha de base está longe do teto.

## Extra: escassez de dados (validação de 50% e 70%)

Testamos 1%, 5% e 20% de dados públicos, com 3 seeds, reservando mais dados para validação
(`results/escassez/RELATORIO.md`). Mesmo treinando com só 30% dos próprios dados, **a linha de base quase não
muda** (1%: 0,857 contra 0,859; 5%: 0,869 contra 0,868). O `private_init` com 300 passos e rehearsal ainda
chega ao limite de capacidade dessas CNNs.

Como a linha de base não mudou, o ganho do FedMD continua em ~+0,01 com 1% e 5% de dados públicos. Com 20%,
a linha de base caiu um pouco (0,886 → 0,880) e o FedMD foi de −0,011 para −0,005. No cenário IID, ele ficou
positivo (+0,003, nas 3 seeds). Para testar de fato uma inicialização mais fraca, o controle certo é reduzir
os passos do `private_init` ou a capacidade do modelo.

## Limitações

- **Os hiperparâmetros foram escolhidos olhando o conjunto de teste.** A validação privada dos clientes
  que só têm uma classe não serve para isso. O efeito esperado é pequeno, porque as escolhas foram
  grosseiras (2 a 3 valores por parâmetro), mas o artigo deve mencionar.
- **Falta um controle sem digest.** O revisit usa rehearsal com rótulos públicos, então parte do ganho
  pode vir dele e não da destilação. Esse controle não foi rodado.
- **Com 1% de dados públicos, a destilação acontece em só ~77 imagens por rodada**, e o `public_init` é
  instável.
- **Três seeds dão um desvio padrão de ~0,005–0,01** por configuração. Isso basta para separar os regimes
  de 1–5% e de 20%, mas não para comparar configurações com diferenças menores que ~0,01.

## Reprodução

```
python server.py --clients 4
# 1% (troque por --public-fractions 0.2 para 20%; para 5%, use --public-fractions 0.05 --public-init-lr 1e-3)
python client.py --public-fractions 0.01 --digest-rounds 30 --public-init-lr 3e-4 \
  --private-init-steps 300 --private-init-lr 1e-4 --private-init-rehearsal \
  --private-epochs 3 --private-steps 10 --revisit-lr 1e-4 --rehearsal-fraction 0.5 \
  --digest-lr 4e-4 --digest-steps 30 --seed 42
```

Tabelas: `uv run --with tabulate python results/final/tabelas.py`. O resumo por seed está em
`results/final/resumo_por_seed.csv`.

## Arquivos (`results/final/`)

- `I1_pub{1e-3,3e-4}_*`: escolha do lr do `public_init` para 1% (`escolhe_1pct.py`, `escolha_1pct.txt`).
- `P01_seed{42,1,2}_*`: 1% de dados públicos.
- `P05_seed2_*`: 5%, seed 2. As seeds 42 e 1 estão em `results/fracoes_pequenas/F2b_*`.
- `P20_seed2_*`: 20%, seed 2. As seeds 42 e 1 estão em `results/fracoes_pequenas/F2_*`.
- `orquestrador.log`, `server.log` e um `.log` por execução.
