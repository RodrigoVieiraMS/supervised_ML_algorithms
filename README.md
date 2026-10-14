 Classificação e Regressão (kNN, Bayes, Regressão Linear)
**Datasets (OpenML):** classificação = **Melbourne Airbnb** (`price_label`) · regressão = **Sberbank Russian Housing Market** (`price_doc`).

---

# PARTE 1 – Como utilizar o código (VS Code)

## 1.1 Instalar o numpy (única dependência)
No terminal do VS Code (`Ctrl+'`):

```bash
pip install numpy
```

Se `pip` não for reconhecido: `python -m pip install numpy` (Windows: `py -m pip install numpy`).
Não usa scikit-learn nem pandas (regra do PDF): leitor de ARFF, tratamento dos dados, algoritmos, métricas e validação cruzada foram escritos à mão.

## 1.2 Arquivos

```
projeto_ia_av2/
├── main.py          # TODO o código (CONFIG, dados, folds, métricas, algoritmos, tabelas)
├── README.md        # este arquivo (uso + slides)
├── resultados.md    # gerado ao rodar (tabelas e análise)
└── data/            # coloque aqui os 2 datasets
```

## 1.3 Passo a passo
1. Coloque os datasets (ARFF do OpenML) em `data/` com um destes nomes:
   - Classificação (Melbourne Airbnb): `melbourne_airbnb.arff` ou `dataset_`
   - Regressão (Sberbank Housing): `sberbank.arff` ou `Sberbank_Russian_Housing_Market_dataset`
   (outro nome? edite a lista `arquivos` no bloco `CONFIG` do `main.py`).
2. Abra o `main.py` no VS Code e clique em **▶ Run Python File**.
3. O terminal mostra, para cada tarefa, a descrição do dataset, o tratamento, os folds e a **tabela no formato da Tabela 1 do PDF** (média ± desvio padrão, tempos de treino e teste) com a análise. Tudo também é salvo em `resultados.md`.

Leva cerca de 5 minutos (a maior parte é o kNN-Manhattan da regressão). Para um teste rápido, edite em `CONFIG`: `"folds": 3`.

## 1.4 Onde está cada coisa no `main.py`
`1. CONFIG` · `2. Leitura e tratamento dos dados` · `3. Divisão em folds (método único)` · `4. Métricas` · `5. Algoritmos` · `6. Validação cruzada e tabelas` · `7. Análise` · `8. Execução`

| Exigência do PDF | Onde (no `main.py`) |
|---|---|
| kNN Euclidiana e Manhattan (classificação e regressão) | `class KNN` |
| Bayesiano univariado / multivariado (`np.cov`, Σ⁻¹, \|Σ\|) | `class BayesUnivariado` / `class BayesMultivariado` |
| Regressão Linear Múltipla | `class RegressaoLinearMultipla` |
| Accuracy, F1, Precision, Recall, R², R² ajustado, MAE | seção 4 |
| Média e desvio de todos os folds; tabela como no PDF | `tabela_console` |
| Tempo de treino e de teste | `validacao_cruzada` |
| Cada fold é teste 1x; método único de divisão | `criar_folds` + `validar_folds` |
| > 10 atributos e > 1000 instâncias | 102 / 18.316 e 289 / 30.471 |

## 1.5 Divisão em folds (método único)
`criar_folds()` é chamada **uma vez por tarefa**; todos os algoritmos usam os **mesmos folds**. Classificação: k-fold **estratificado**; regressão: aleatório. `validar_folds()` garante folds disjuntos que cobrem todas as amostras, ou seja, **cada fold é teste exatamente 1 vez** (k = 5, semente 42).

## 1.6 Tratamento dos dados (seção 2 do `main.py`)

| Etapa | O que faz | Por quê |
|---|---|---|
| Leitura ARFF própria | tokenizador com aspas simples/duplas, `?` = ausente; detecta ARFF pelo conteúdo (arquivo sem extensão) | sem pandas |
| Alvo | classificação: `price_label` (10 classes); regressão: `price_doc` | |
| Vazamento do alvo | Melbourne: `price`, `weekly_price`, `monthly_price` descartados | `price_label` é o preço em faixas |
| Identificador | `host_id` descartado | não descreve o imóvel |
| Texto livre / IDs | coluna textual com > 40 (Melbourne) ou > 150 (Sberbank) valores distintos é descartada | one-hot inviável |
| Colunas constantes | descartadas | sem informação |
| Strings numéricas | ex.: `"100%"` vira número | |
| Categóricos | one-hot; atributo binário vira 1 coluna 0/1 | evita colunas espelhadas |
| Valores ausentes | média do treino do fold | sem vazamento |
| Outliers | winsorização: corta cada coluna contínua nos percentis 1 e 99 do treino | z-score e distâncias eram dominados por extremos |
| Escala | z-score com média/desvio do treino | kNN e Bayes dependem da escala |

Todo o pré-processamento é **ajustado só no treino de cada fold** (sem *data leakage*).

## 1.7 Hiperparâmetros e detalhes
- kNN: **k = 5**. Bayes univariado: piso de variância **0,1**; Bayes multivariado: Σ + **0,1**·I (escolhidos em holdout interno do treino do fold 1, sem usar os folds de teste). Regressão Linear: pseudo-inversa por SVD com corte **1e-6**.
- Euclidiana via ‖x−y‖² = ‖x‖² + ‖y‖² − 2x·y (float64); Manhattan acumulada por atributo em float32 (mesmos vizinhos que float64 nos testes). Empate no voto: menor índice de classe. Precisão/Recall/F1: média **macro**. Desvio padrão populacional (`np.std`). Tempos: só `fit` e `predict`.
- Os tempos abaixo foram medidos em um ambiente com 1 núcleo de CPU. **No seu computador os valores absolutos mudam** (as métricas não): use os números do seu terminal nos slides.

