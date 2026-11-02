# Digest com KL e temperatura no lugar de MAE nos logits (01–02/11)

## Configuração

Igual ao experimento final (`results/final/RELATORIO.md`). A única mudança é a loss do digest:
**T² · KL(softmax(consenso/T) ‖ softmax(cliente/T))**. O servidor continua fazendo a média ponderada dos
logits.

- **Triagem com a seed 42:** comparamos T = 2 e T = 4. O ganho médio foi de +0,0064 com T = 2 e de
  +0,0071 com T = 4. Escolhemos **T = 4**, embora a diferença esteja dentro do ruído.
- **Com T = 4, rodamos as seeds 42, 1 e 2**, com 1% e 5% de dados públicos.
- A comparação com MAE usa as mesmas seeds do experimento final.

## Resultado (média de 3 seeds)

O ganho é a média das rodadas 26–30 menos a linha de base (`public_init` + `private_init`).

| Fração / alpha | Final com MAE | Final com KL T=4 | Ganho com MAE | Ganho com KL T=4 |
|---|---|---|---|---|
| 1% / iid | 0,882 | 0,882 | +0,009 ± 0,003 | +0,009 ± 0,004 |
| 1% / 1.0 | 0,871 | 0,873 | +0,006 ± 0,008 | +0,010 ± 0,006 |
| 1% / 0.1 | 0,852 | 0,856 | +0,014 ± 0,005 | **+0,018 ± 0,006** |
| 5% / iid | 0,888 | 0,890 | +0,011 ± 0,008 | +0,013 ± 0,005 |
| 5% / 1.0 | 0,878 | 0,878 | +0,011 ± 0,007 | +0,009 ± 0,013 |
| 5% / 0.1 | 0,866 | 0,869 | +0,008 ± 0,004 | +0,010 ± 0,008 |
| **Média** | | | **+0,0097** | **+0,0115** |

Comparando por par (mesma seed e mesmo cenário), a KL fica em média **+0,0019 ± 0,0043** acima da MAE e é
melhor em **12 dos 18** pares.

## Conclusão

- **A KL dá uma vantagem pequena e não significativa sobre a MAE:** +0,002 em média, com 12 de 18 pares a
  favor. Um teste t pareado aproximado dá t ≈ 1,9 (p ≈ 0,08). Com 3 seeds, isso é uma tendência, não um
  resultado firme.
- **A vantagem aparece mais nos cenários desbalanceados:** +0,004 em 1% / alpha 0.1, +0,004 em 1% / alpha
  1.0 e +0,002 em 5% / alpha 0.1. Isso faz sentido: com a temperatura, o digest transmite a confiança
  relativa entre as classes, o que ajuda mais os clientes que só viram uma classe.
- **A loss do digest não é o gargalo.** O ganho do FedMD continua em ~+0,01 com as duas losses. O que
  limita é a quantidade de dados no digest (~77 imagens por rodada com 1% de dados públicos) e a
  proximidade do teto (0,898).

**Recomendação:** as duas losses podem ser usadas. Para o artigo, dá para manter a MAE, que é a do FedMD
original, e citar a KL como variante testada, com uma vantagem pequena nos cenários desbalanceados. Outra
opção é adotar a KL com T = 4 como configuração final, avisando que a diferença não é significativa.

## Arquivos

- `T{2,4}_P{01,05}_seed42_*`: triagem da temperatura (`escolhe_T.py`, `escolha_T.txt`).
- `T4_P{01,05}_seed{1,2}_*`: seeds adicionais com T = 4.
- `comparacao_por_seed.csv`: ganho por loss, cenário e seed.
- `orquestrador.log`, `server.log` e um `.log` por execução.
