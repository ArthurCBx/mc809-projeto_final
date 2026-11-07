# Cenário extremo: 0,5% de dados públicos (02/11, 16:03–17:30)

O conjunto público tem **49 imagens** (47% malignas). Delas, ~39 são usadas no treino do `public_init`, ~10 na
validação, e ~39 são sorteadas por rodada para o digest e o rehearsal.

A configuração é a mesma da etapa final (`results/ensemble_e_leave_one_out/`, consenso padrão):

- `private_init` com 300 passos, lr 1e-4 e rehearsal.
- Revisit com 3×10 passos, lr 1e-4 e 50% de rehearsal.
- Digest com KL (T = 4), 30 passos por rodada e lr 4e-4.
- 30 rodadas.
- Seeds 42, 1 e 2.

O lr do `public_init` foi escolhido rodando só as etapas iniciais, como no cenário de 1%:

| lr do `public_init` | public_init | private_init |
|---|---|---|
| 1e-3 (escolhido) | 0,500 | 0,848 |
| 3e-4 | 0,500 | 0,847 |

Com só ~39 imagens de treino, o `public_init` não aprende nada nos dois casos (prevê sempre a mesma classe),
então a escolha do lr não faz diferença.

## Resultado (média ± desvio padrão, 3 seeds), comparado com 1% e 5% na mesma configuração

| Fração / alpha | public_init | **Linha de base** (public + private init) | FedMD (r26–30) | **Ganho por cliente** | Ensemble após private_init | Ensemble após FedMD | Efeito do FedMD sobre o ensemble |
|---|---|---|---|---|---|---|---|
| **0,5% / iid** | 0,522 | 0,872 ± 0,007 | 0,880 ± 0,005 | **+0,009 ± 0,004** | 0,880 | 0,885 | +0,004 |
| **0,5% / 1.0** | 0,521 | 0,864 ± 0,010 | 0,875 ± 0,003 | **+0,011 ± 0,008** | 0,880 | 0,886 | +0,006 |
| **0,5% / 0.1** | 0,508 | 0,821 ± 0,011 | 0,841 ± 0,013 | **+0,019 ± 0,007** | 0,859 | 0,862 | +0,003 |
| 1% / iid | 0,662 | 0,875 ± 0,003 | 0,881 ± 0,004 | +0,006 ± 0,003 | 0,887 | 0,883 | −0,004 |
| 1% / 1.0 | 0,659 | 0,862 ± 0,005 | 0,871 ± 0,009 | +0,009 ± 0,004 | 0,885 | 0,881 | −0,004 |
| 1% / 0.1 | 0,660 | 0,840 ± 0,002 | 0,853 ± 0,004 | +0,013 ± 0,005 | 0,881 | 0,868 | −0,013 |
| 5% / iid | 0,847 | 0,879 ± 0,007 | 0,889 ± 0,005 | +0,009 ± 0,006 | 0,885 | 0,892 | +0,007 |
| 5% / 1.0 | 0,848 | 0,868 ± 0,008 | 0,878 ± 0,014 | +0,011 ± 0,008 | 0,881 | 0,882 | +0,001 |
| 5% / 0.1 | 0,854 | 0,856 ± 0,004 | 0,868 ± 0,003 | +0,012 ± 0,001 | 0,880 | 0,881 | +0,001 |

| Fração pública | Ganho médio por cliente | Desvio padrão | Seeds × cenários com ganho | Efeito médio sobre o ensemble |
|---|---|---|---|---|
| **0,5%** | **+0,013** | 0,008 | 9/9 | +0,005 |
| 1% | +0,010 | 0,005 | 9/9 | −0,007 |
| 5% | +0,011 | 0,005 | 9/9 | +0,003 |

Ganho por seed com 0,5% de dados públicos:

| alpha | seed 42 | seed 1 | seed 2 |
|---|---|---|---|
| iid | +0,009 | +0,012 | +0,005 |
| 1.0 | +0,006 | +0,021 | +0,007 |
| 0.1 | +0,014 | **+0,027** | +0,017 |

## Conclusões

1. **Com 0,5% de dados públicos, o FedMD dá o maior ganho médio por cliente: +0,013**, positivo nas 9
   combinações de seed e cenário. O maior ganho é no caso mais desbalanceado (alpha 0.1): +0,019 em
   média, chegando a +0,027 em uma seed.
2. **A hipótese "menos dados públicos, mais ganho" precisa de um ajuste.**
   - **Comparado só com o `public_init`, o ganho realmente cresce** à medida que a fração pública diminui,
     porque o `public_init` fica mais fraco: 0,85 com 5%, 0,66 com 1% e 0,51 com 0,5%.
   - **Comparado com a linha de base justa (`public_init` + `private_init`), o ganho fica praticamente
     constante, em torno de +0,01, entre 0,5% e 5%**: +0,013, +0,010 e +0,011. A tendência de aumento com
     0,5% está dentro do desvio padrão. Com 20%, o ganho fica negativo, porque a linha de base já está no
     teto.
   - **Ou seja:** quem compensa a falta de dados públicos é o `private_init`, não o FedMD. Enquanto a linha
     de base estiver longe do teto, o FedMD acrescenta cerca de 1 ponto.
3. **O ganho cresce com o desbalanceamento:** com alpha 0.1, chega a +0,019 (0,5%), +0,013 (1%) e +0,012
   (5%). É onde cada cliente sozinho é mais fraco (linha de base de 0,82–0,86) e onde a colaboração
   compensa mais.
4. **O nível absoluto cai um pouco com 0,5%.** A linha de base com alpha 0.1 fica em 0,821, contra 0,840
   com 1%. Provavelmente o rehearsal, que agora mistura só ~49 imagens públicas, protege menos os clientes
   que têm uma classe só. O FedMD recupera parte disso (0,841), mas não tudo.
5. **Com 0,5%, o FedMD não piora o ensemble** (+0,005), ao contrário do que acontece com 1% (−0,007). A
   diferença está dentro do ruído. O ensemble sem colaboração continua acima dos clientes individuais
   (0,859–0,880 contra 0,821–0,872).

## Arquivos (`results/fracao_publica_0_5/`)

- `I1_pub{1e-3,3e-4}_*`: escolha do lr do `public_init` (`escolhe_lr.py`, `escolha.txt`).
- `P005_seed{42,1,2}_{test_metrics,train_history,ensemble_metrics}.csv` e um `.log` por execução.
- `resumo_por_seed.csv`. As tabelas são geradas por `uv run --with tabulate python results/fracao_publica_0_5/tabelas.py`, que
  também compara com 1% e 5% (`results/ensemble_e_leave_one_out/all_*`).
