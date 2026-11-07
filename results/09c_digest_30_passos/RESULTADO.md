# Digest limitado a 30 passos por rodada (01/11, 23:57–01:53)

## Configuração

Igual a `results/09b_digest_lr_4e-4/` (digest com lr 4e-4), mais `--digest-steps 30`:

- **`private_init`**: 10 passos, lr 2e-5.
- **Revisit**: 3×10 passos, lr 1e-4, 50% de rehearsal.
- **Digest**: lr 4e-4, **30 passos por rodada** em qualquer fração pública. Sem o limite, eram ~24, ~96 e
  ~190 passos com 10%, 40% e 80% de dados públicos.
- 30 rodadas, seed 42, consenso ponderado.

## Resultado

Média dos 4 clientes nas rodadas 26–30:

| Configuração | Cen. 3 | Digest 4e-4 | **Digest 4e-4, 30 passos** | public_init | 30 passos − public_init | Pior cliente (qualquer rodada) |
|---|---|---|---|---|---|---|
| 0.1 / iid | 0,889 | 0,879 | 0,877 | 0,863 | **+0,014** | 0,849 |
| 0.1 / 1.0 | 0,874 | 0,873 | 0,880 | 0,868 | **+0,012** | 0,855 |
| 0.1 / 0.1 | 0,842 | 0,869 | 0,860 | 0,860 | 0,000 | 0,819 |
| 0.4 / iid | 0,891 | 0,888 | 0,883 | 0,876 | **+0,007** | 0,862 |
| 0.4 / 1.0 | 0,860 | 0,859 | 0,874 | 0,881 | −0,007 | 0,853 |
| 0.4 / 0.1 | 0,832 | 0,873 | 0,876 | 0,871 | **+0,004** | 0,827 |
| 0.8 / iid | 0,897 | 0,888 | 0,896 | 0,895 | +0,001 | 0,858 |
| 0.8 / 1.0 | 0,881 | 0,886 | 0,882 | 0,884 | −0,002 | 0,853 |
| 0.8 / 0.1 | 0,854 | 0,875 | 0,877 | 0,887 | −0,010 | 0,856 |
| **Média** | 0,869 | 0,877 | **0,878** | | **+0,002** | |

O `public_init` varia um pouco de uma execução para outra (até ~0,005) por não determinismo da GPU. Por
isso, a comparação com o `public_init` usa sempre o valor da mesma execução.

## Conclusões

- **Pela primeira vez, a média fica no nível do `public_init`** (+0,002). Cinco das nove configurações
  ficam iguais ou acima dele, e a pior fica 0,010 abaixo. Na execução sem limite, a média era −0,006 e o
  pior caso, −0,022.
- **O ganho vem das configurações que mais perdiam no digest.** Em 0.4 / 1.0, a diferença para o
  `public_init` foi de −0,022 para −0,007. Em 0.8 / iid, de −0,008 para +0,001. Em 0.4 / 0.1, de −0,008
  para +0,004. Isso confirma que o número de passos do digest pesava nas configurações com mais dados
  públicos.
- **A média geral praticamente não mudou** (0,877 → 0,878). Algumas configurações perderam um pouco, como
  0.1 / 0.1 (−0,009) e 0.4 / iid (−0,005). As diferenças por configuração estão dentro do ruído de uma
  seed.
- **O treino continua estável.** Nenhum cliente fica abaixo de 0,82 em nenhuma rodada.

## Situação geral

O FedMD adaptado agora fica estável e no nível do `public_init`. Com 10% de dados públicos, chega a
superar o `public_init` em +0,012 a +0,014. Com mais dados públicos, fica no mesmo nível. Os ganhos ainda
são pequenos e vêm de uma única seed.

## Próximos passos possíveis

1. **Seeds 1 e 2 com esta configuração.** Ela é a candidata a final, e é preciso saber se os ±0,01 em
   relação ao `public_init` são reais.
2. **Testar outros números de passos no digest** (15 e 60) para ver se 30 é um bom ponto.
3. **Fração de rehearsal** (0,25 e 0,75).
4. **Controle sem digest**, quando for a hora de medir quanto a destilação contribui.

Arquivos: `steps30_test_metrics.csv`, `steps30_train_history.csv` e `steps30.log`.
