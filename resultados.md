# Classificação (dataset_)

## Dataset

- Amostras: **18316**
- Atributos preditores no arquivo original: **102**
- Atributos usados após o tratamento: **62** (**169** colunas numéricas finais, após one-hot dos categóricos)
- Atributo alvo: `price_label`
- Valores ausentes nos preditores usados (imputados pela média do treino): 71006
- Classes (10): 0: 1524 (8.3%), 1: 1672 (9.1%), 2: 1753 (9.6%), 3: 1587 (8.7%), 4: 871 (4.8%), 5: 2279 (12.4%), 6: 1934 (10.6%), 7: 1626 (8.9%), 8: 1731 (9.5%), 9: 3339 (18.2%)
- Atributos descartados: **40**
  - texto livre / identificador (> 40 valores distintos) (31): `listing_url`, `name`, `summary`, `space`, `description`, `neighborhood_overview`, `notes`, `transit`, `access`, `interaction`, `house_rules`, `picture_url`, `host_url`, `host_name`, `host_since`, `host_location`, `host_about`, `host_response_rate`, `host_thumbnail_url`, `host_picture_url`, `host_neighborhood`, `host_verifications`, `street`, `neighborhood`, `suburb`, `zipcode`, `smart_location`, `amenities`, `calendar_updated`, `first_review`, `last_review`
  - constante (5): `scrape_id`, `country_code`, `country`, `has_availability`, `requires_license`
  - ignorada por escolha (vazamento do alvo / identificador) (4): `host_id`, `price`, `weekly_price`, `monthly_price`

## Validação cruzada

- 5 folds, semente 42; tamanhos dos folds de teste: [3664, 3663, 3663, 3663, 3663]
- k do kNN: 5

## Tabela comparativa (média ± desvio padrão dos folds)

| Classificador | Acurácia | Precisão | Recall | F1-Score | Tempo Treino (s) | Tempo Teste (s) |
|---|---|---|---|---|---|---|
| K-vizinhos (Dist. Euclidiana) | 0.3131 ± 0.0031 | 0.2797 ± 0.0024 | 0.2836 ± 0.0034 | 0.2768 ± 0.0024 | 0.0036 ± 0.0010 | 1.8934 ± 0.1345 |
| K-vizinhos (Dist. Manhattan) | 0.3342 ± 0.0039 | 0.2974 ± 0.0046 | 0.3008 ± 0.0042 | 0.2945 ± 0.0041 | 0.0057 ± 0.0003 | 8.8599 ± 0.0953 |
| Bayesiano (Univariado) | 0.1703 ± 0.0126 | 0.2532 ± 0.0147 | 0.2095 ± 0.0148 | 0.1492 ± 0.0107 | 0.0550 ± 0.0081 | 0.0754 ± 0.0177 |
| Bayesiano (Multivariado) | 0.1763 ± 0.0085 | 0.2979 ± 0.0153 | 0.2146 ± 0.0110 | 0.1598 ± 0.0102 | 0.3413 ± 0.1045 | 0.0719 ± 0.0066 |

## Análise

- Melhor Acurácia: **K-vizinhos (Dist. Manhattan)** (0.3342 ± 0.0039).
- Melhor F1-Score: **K-vizinhos (Dist. Manhattan)** (0.2945 ± 0.0041).
- Mais rápido no treino: **K-vizinhos (Dist. Euclidiana)** (0.0036 s).
- Mais rápido no teste: **Bayesiano (Multivariado)** (0.0719 s).
- Mais rápido no total (treino + teste): **Bayesiano (Univariado)** (0.1305 s).

**Critério de equilíbrio:** entre os modelos cujo F1-Score médio está a no máximo 0.01 do melhor, escolhe-se o de menor tempo total (treino + teste).
**Melhor equilíbrio desempenho × tempo: K-vizinhos (Dist. Manhattan)** (F1-Score = 0.2945, tempo total = 8.8655 s).
O modelo mais preciso também é o de melhor equilíbrio.


# Regressão (Sberbank_Russian_Housing_Market_dataset)

## Dataset

- Amostras: **30471**
- Atributos preditores no arquivo original: **289**
- Atributos usados após o tratamento: **289** (**438** colunas numéricas finais, após one-hot dos categóricos)
- Atributo alvo: `price_doc`
- Valores ausentes nos preditores usados (imputados pela média do treino): 261026
- Alvo: média=7123035.2777, desvio=4780032.8919, min=100000.0000, mediana=6274411.0000, max=111111112.0000

## Validação cruzada

- 5 folds, semente 42; tamanhos dos folds de teste: [6095, 6094, 6094, 6094, 6094]
- k do kNN: 5

## Tabela comparativa (média ± desvio padrão dos folds)

| Regressor | R² | R² ajustado | MAE | Tempo Treino (s) | Tempo Teste (s) |
|---|---|---|---|---|---|
| K-vizinhos (Dist. Euclidiana) | 0.4681 ± 0.0066 | 0.4269 ± 0.0071 | 1876107.1091 ± 32806.2815 | 0.0117 ± 0.0007 | 5.7022 ± 0.7164 |
| K-vizinhos (Dist. Manhattan) | 0.4165 ± 0.0087 | 0.3713 ± 0.0093 | 1956778.0061 ± 31145.3934 | 0.0225 ± 0.0047 | 57.5805 ± 2.8591 |
| Regressão Linear Múltipla | 0.5636 ± 0.0230 | 0.5298 ± 0.0248 | 1806397.7715 ± 26348.1697 | 7.6025 ± 3.2028 | 0.0108 ± 0.0024 |

## Análise

- Melhor R²: **Regressão Linear Múltipla** (0.5636 ± 0.0230).
- Melhor MAE: **Regressão Linear Múltipla** (1806397.7715 ± 26348.1697).
- Mais rápido no treino: **K-vizinhos (Dist. Euclidiana)** (0.0117 s).
- Mais rápido no teste: **Regressão Linear Múltipla** (0.0108 s).
- Mais rápido no total (treino + teste): **K-vizinhos (Dist. Euclidiana)** (5.7139 s).

**Critério de equilíbrio:** entre os modelos cujo R² médio está a no máximo 0.01 do melhor, escolhe-se o de menor tempo total (treino + teste).
**Melhor equilíbrio desempenho × tempo: Regressão Linear Múltipla** (R² = 0.5636, tempo total = 7.6133 s).
O modelo mais preciso também é o de melhor equilíbrio.
