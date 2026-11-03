# Escassez de dados: validação de 50% e 70% (02–03/11)

**Pergunta:** se cada cliente tiver menos dados para treinar, a inicialização fica mais fraca e o FedMD passa
a ajudar mais?

**Como testamos:** aumentamos a fração de validação (`--val-fraction`) de 0,2 para 0,5 e 0,7. Isso vale para
o público usado no `public_init` e para o privado de cada cliente. Com isso, cada cliente treina com 50% ou
30% dos próprios dados, em vez de 80%. Por exemplo, um cliente IID com 1% de dados públicos passa de ~1730
para ~1080 ou ~650 imagens de treino.

**Configuração:** a melhor até agora.

- `private_init`: 300 passos, lr 1e-4, com rehearsal.
- Revisit: 3×10 passos, lr 1e-4, com 50% de rehearsal.
- Digest: KL com T = 4, 30 passos, lr 4e-4.
- Consenso padrão, 30 rodadas.
- lr do `public_init`: 3e-4 com 1% e 20% de dados públicos, 1e-3 com 5%.

Foram 3 frações públicas × 2 validações × 3 seeds = 18 execuções. A referência com validação de 0,2 vem de
`results/ensemble_loo/all_*` (1% e 5%, mesma configuração) e do experimento final (20%, com MAE no digest).

**Incidente:** as 3 execuções com validação de 0,7 e 1% de dados públicos caíram por falta de memória na
GPU no `private_init`. O problema veio de um pico de memória da compilação XLA. Repetimos essas três sem
XLA (`--no-jit`) e com 2,8 GB por cliente. O algoritmo é o mesmo, então os números são comparáveis. O log
de cada tentativa que falhou ficou em `*_falha.log`.

## Resultado (média de 3 seeds)

| Fração pública | Validação | Linha de base (public + private init) | FedMD (r26–30) | **Ganho** | Desvio padrão |
|---|---|---|---|---|---|
| 1% | 0,2 | 0,859 | 0,869 | **+0,010** | 0,005 |
| 1% | 0,5 | 0,857 | 0,867 | **+0,010** | 0,007 |
| 1% | 0,7 | 0,857 | 0,868 | **+0,011** | 0,006 |
| 5% | 0,2 | 0,868 | 0,878 | **+0,011** | 0,005 |
| 5% | 0,5 | 0,874 | 0,878 | +0,004 | 0,008 |
| 5% | 0,7 | 0,869 | 0,876 | +0,007 | 0,007 |
| 20% | 0,2 | 0,886 | 0,875 | −0,011 | 0,010 |
| 20% | 0,5 | 0,882 | 0,872 | −0,010 | 0,011 |
| 20% | 0,7 | 0,880 | 0,874 | −0,005 | 0,009 |

### Por cenário

| Cenário | Val 0,2 | Val 0,5 | Val 0,7 |
|---|---|---|---|
| 1% / iid | +0,006 | +0,012 | +0,011 |
| 1% / 1.0 | +0,009 | +0,010 | +0,013 |
| 1% / 0.1 | +0,013 | +0,008 | +0,009 |
| 5% / iid | +0,009 | +0,010 | +0,011 |
| 5% / 1.0 | +0,011 | +0,005 | +0,005 |
| 5% / 0.1 | +0,012 | −0,001 | +0,005 |
| 20% / iid | −0,007 | −0,001 | **+0,003 (3/3 seeds)** |
| 20% / 1.0 | −0,006 | −0,007 | −0,002 |
| 20% / 0.1 | −0,022 | −0,023 | −0,017 |

### Ensemble (média dos cenários)

| Fração / validação | Linha de base | Ensemble após private_init | FedMD | Ensemble após FedMD |
|---|---|---|---|---|
| 1% / 0,2 | 0,859 | 0,884 | 0,869 | 0,877 |
| 1% / 0,5 | 0,857 | 0,879 | 0,867 | 0,877 |
| 1% / 0,7 | 0,857 | 0,882 | 0,868 | 0,878 |
| 5% / 0,2 | 0,868 | 0,882 | 0,878 | 0,885 |
| 5% / 0,5 | 0,874 | 0,887 | 0,878 | 0,883 |
| 5% / 0,7 | 0,869 | 0,883 | 0,876 | 0,882 |
| 20% / 0,5 | 0,882 | 0,896 | 0,872 | 0,878 |
| 20% / 0,7 | 0,880 | 0,894 | 0,874 | 0,881 |

## Conclusões

1. **A manipulação não conseguiu enfraquecer a inicialização.** Mesmo treinando com só 30% dos próprios
   dados, os clientes chegam praticamente à mesma linha de base: 0,857 contra 0,859 com 1%, 0,869 contra
   0,868 com 5% e 0,880 contra 0,886 com 20%.

   O `private_init` usa 300 passos fixos com rehearsal, e ~650 imagens por cliente ainda bastam para essas
   CNNs chegarem a ~0,86–0,88, perto da capacidade delas (o teto centralizado é 0,898). O que limita o
   modelo é a capacidade da arquitetura, não a quantidade de dados.

2. **Como a linha de base não mudou, o ganho do FedMD também não mudou.** Com 1% e 5% de dados públicos,
   ele continua em ~+0,01 nas três validações. As diferenças entre 0,2, 0,5 e 0,7 (por exemplo, +0,011 e
   +0,004 com 5%) estão dentro do desvio padrão.

3. **Houve um efeito pequeno e coerente com a hipótese com 20% de dados públicos.** Esse é o único caso em
   que a linha de base caiu um pouco, de 0,886 para 0,880. Ali, o FedMD passou de −0,011 para −0,005 em
   média. No cenário IID, ficou positivo com validação de 0,7 (+0,003, nas 3 seeds), quando antes era
   −0,007. Isso reforça a leitura anterior: o FedMD só ajuda quando a linha de base está abaixo do teto.

4. **O ensemble continua igual:** sem colaboração, ele fica entre 0,879 e 0,896. Com 1% de dados públicos, o
   FedMD continua reduzindo a diversidade entre os clientes.

## Para testar de fato a hipótese

O que segura a inicialização é o `private_init`, não o volume de dados. Opções para enfraquecê-la:

- **Reduzir os passos do `private_init`**, de 300 para 50 ou 100. Esse é o controle mais direto da força da
  inicialização privada.
- **Validação muito maior**, de 0,9 ou 0,95. Isso deixa ~100–200 imagens por cliente, região em que o
  tamanho do conjunto deve começar a pesar.
- **Reduzir a capacidade do modelo**, com menos filtros. Isso abaixa o teto e a linha de base juntos e cria
  espaço para a colaboração.

## Mudanças no código

- `client.py`, `--val-fraction`: fração de validação, do público e do privado. O padrão continua 0,2.
- `client.py`, `--no-jit`: compila os modelos sem XLA, evitando os picos de memória de GPU da compilação. O
  padrão continua com XLA.

## Arquivos (`results/escassez/`)

- `V{0.5,0.7}_P{01,05,20}_seed{42,1,2}_{test_metrics,train_history,ensemble_metrics}.csv` e um `.log` por
  execução. As tentativas que falharam estão em `V0.7_P01_seed*_falha.log`.
- `orquestrador.sh`, `repete_falhas.sh` e os respectivos logs, mais o `server.log`.
- `resumo_por_seed.csv`. As tabelas são geradas por `uv run --with tabulate python results/escassez/tabelas.py`.
