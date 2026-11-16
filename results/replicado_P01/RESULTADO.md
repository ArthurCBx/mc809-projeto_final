# FedMD replicado com f = 1% (15–16/11)

Fecha a lacuna do 1% na Figura 7 e na Tabela 7. Mesmo protocolo do FedMD replicado das outras frações (consenso
uniforme, MAE, revisit de 3 épocas completas, η = 10⁻³, 20 rodadas, private init até 50 épocas com parada antecipada),
com avaliação no PH2 (200 imagens). Seeds 42, 1 e 2, as três divisões, um α por processo
(`results/ph2_privado/orquestrador_sementes_2_3.sh`). Resultado = média das rodadas 16–20.

| Divisão | Local (fim do private init) | Validação (clientes) | Validação (ensemble) | PH2 bal. (clientes) | PH2 bal. (ensemble) |
|---|---|---|---|---|---|
| IID | 0,882 | 0,900 ± 0,002 | 0,906 | 0,643 | 0,637 |
| α = 1,0 | 0,854 | 0,878 ± 0,016 | 0,903 | 0,620 | 0,650 |
| α = 0,1 | 0,583 | 0,639 ± 0,067 | 0,483 | 0,524 | 0,458 |
| Média | 0,773 | 0,806 | 0,764 | 0,596 | 0,582 |

O padrão é o das outras frações: no nível do teto com dados IID e colapso com α = 0,1 (clientes de uma classe só). Com
α = 0,1, o ensemble fica abaixo do acaso na validação (0,483), porque os clientes colapsados preveem classes opostas
com logits grandes. Linha para a Tabela 7: validação 0,764 e PH2 0,582 (ensemble do replicado com 1%).

Arquivos: `original_P01_seed{42,1,2}_alpha{iid,1.0,0.1}_*` (CSVs, logits do PH2 e logs).
