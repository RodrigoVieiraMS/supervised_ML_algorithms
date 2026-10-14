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

---

# PARTE 2 – Apresentação de slides

Os números abaixo vêm da execução real (5 folds, semente 42). Se rodar de novo no seu computador, as métricas ficam iguais; só os tempos mudam.

---

## Slide 1 – Capa
**Avaliação de Algoritmos de Aprendizado Supervisionado: kNN, Classificadores Bayesianos e Regressão Linear**
Inteligência Artificial Computacional · Profa. Dra. Cynthia Moreira Maia · [nome(s)] · UNIFOR – CCT · 15/10/2026

---

## Slide 2 – Introdução
- **Objetivo:** comparar algoritmos clássicos de aprendizado supervisionado em **classificação** (faixa de preço de anúncios Airbnb) e **regressão** (preço de imóveis na Rússia), com datasets do OpenML.
- **O que será comparado:**
  - Classificação: kNN (Euclidiana), kNN (Manhattan), Bayesiano univariado, Bayesiano multivariado.
  - Regressão: kNN (Euclidiana), kNN (Manhattan), Regressão Linear Múltipla.
- **Critérios:** qualidade (Accuracy, Precision, Recall, F1, R², R² ajustado, MAE) e custo (tempo de treino e de teste).
- **Por que importa:** não existe algoritmo universalmente melhor. É preciso equilibrar **precisão e custo computacional**.
- **Rigor:** algoritmos implementados do zero (apenas `numpy`), validação cruzada de 5 folds.

---

## Slide 3 – Justificativa dos datasets

| Critério do PDF | Classificação: Melbourne Airbnb | Regressão: Sberbank Housing |
|---|---|---|
| Mais de 10 atributos | 102 preditores | 289 preditores |
| Mais de 1000 instâncias | 18.316 | 30.471 |
| Alvo | `price_label` (10 faixas de preço) | `price_doc` (preço de venda) |

- **Melbourne Airbnb:** problema real e difícil (10 classes quase balanceadas), com atributos numéricos, categóricos e de texto. Exige decisões de tratamento (texto, vazamento do alvo), o que enriquece a análise. Mostra bem o contraste entre métodos de **distância** (kNN) e **probabilísticos** (Bayes).
- **Sberbank:** muitos atributos (289), com ausentes, outliers e variáveis colineares. Permite comparar um modelo **paramétrico linear** com um **não paramétrico** (kNN) e discutir robustez numérica.
- Ambos são do OpenML e atendem aos critérios do PDF.

---

## Slide 4 – Banco de dados

| | Classificação | Regressão |
|---|---|---|
| Dataset | `melbourne_airbnb` | Sberbank Russian Housing Market |
| Amostras | **18.316** | **30.471** |
| Atributos (preditores) | **102** originais → 62 usados (169 colunas após one-hot) | **289** → 289 usados (438 colunas após one-hot) |
| O que representa | Anúncios do Airbnb em Melbourne (InsideAirbnb, 2018): tipo de imóvel, quartos, banheiros, localização, notas, disponibilidade, perfil do anfitrião, textos | Transações de imóveis na Rússia (ago/2011–jun/2015): área, andar, ano, e características do entorno (distâncias, contagens de serviços) |
| Alvo | faixa de preço 0–9 | preço de venda (média 7,12 mi; mediana 6,27 mi; desvio 4,78 mi) |
| Distribuição | classes de 4,8% (classe 4) a 18,2% (classe 9) | assimétrica (máx. 111 mi) |

---

## Slide 5 – Tratamento dos dados
(resumo da tabela da seção 1.6)
1. **Leitura manual do ARFF** (sem pandas).
2. **Vazamento do alvo:** `price`, `weekly_price`, `monthly_price` removidos do Melbourne (o `price_label` é derivado do preço).
3. **Colunas sem informação:** 31 textos livres/IDs (URLs, descrições...) e 5 constantes removidas → 62 atributos usados.
4. **Categóricos → one-hot** (binários em 1 coluna).
5. **Ausentes → média do treino** (71 mil no Melbourne, 261 mil no Sberbank).
6. **Winsorização (percentis 1–99 do treino)** contra outliers.
7. **z-score:** $x'=\dfrac{x-\mu}{\sigma}$ com $\mu,\sigma$ do treino.
8. **Sem data leakage:** tudo é reajustado em cada fold.

*Efeito medido (holdout interno, só dados de treino):* a winsorização elevou o R² da regressão linear de ≈ 0,40 para ≈ 0,55.

---

## Slide 6 – Validação cruzada (método único)
- Uma função `criar_folds()` gera k = 5 folds disjuntos, semente fixa. **Estratificado** na classificação, aleatório na regressão.
- **Mesmos folds para todos os algoritmos.** Cada fold é teste **exatamente 1 vez** (`validar_folds`).
- Tamanho dos folds de teste: Melbourne [3664, 3663, 3663, 3663, 3663]; Sberbank [6095, 6094, 6094, 6094, 6094].

$$\bar m=\frac1k\sum_{i=1}^{k}m_i \qquad s=\sqrt{\frac1k\sum_{i=1}^{k}(m_i-\bar m)^2}$$

---

## Slide 7 – kNN
**Conceito:** método baseado em instâncias (não paramétrico). Guarda o treino e, para um ponto novo $\mathbf{x}$, usa os $k$ exemplos mais próximos.

$$d_{Euclidiana}(X,Y)=\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}\qquad d_{Manhattan}(X,Y)=\sum_{i=1}^{n}|x_i-y_i|$$

- Classificação (voto): $\hat y=\arg\max_c\sum_{i\in N_k(\mathbf{x})}\mathbb{1}[y_i=c]$ · Regressão (média): $\hat y=\frac1k\sum_{i\in N_k(\mathbf{x})}y_i$
- **Custo:** treino $O(1)$; teste $O(n\cdot d)$ por consulta. $k=5$.
- Manhattan é menos sensível a valores grandes isolados (não eleva ao quadrado).

---

## Slide 8 – Bayesiano univariado
$$p(x)=\frac{1}{\sqrt{2\pi}\,\sigma}\exp\!\left[-\frac12\left(\frac{x-\mu}{\sigma}\right)^2\right]\qquad p(\mathbf{x}|\omega_c)=\prod_{j=1}^{d}p(x_j|\omega_c)$$

- Regra de Bayes + normal **por atributo e por classe**, atributos assumidos **independentes**.
- **Treino:** priors $P(\omega_c)$, $\mu_{cj}$, $\sigma_{cj}^2$. **Decisão:** $\hat y=\arg\max_c[\log P(\omega_c)+\sum_j\log p(x_j|\omega_c)]$.
- Piso de 0,1 na variância (evita variância ≈ 0 em atributos binários raros).

---

## Slide 9 – Bayesiano multivariado
$$p(\mathbf{x})=\frac{1}{(2\pi)^{d/2}|\boldsymbol\Sigma|^{1/2}}\exp\!\left[-\frac12(\mathbf{x}-\boldsymbol\mu)^t\boldsymbol\Sigma^{-1}(\mathbf{x}-\boldsymbol\mu)\right]$$

- $\mathbf{x}$: vetor coluna com $d$ componentes · $\boldsymbol\mu$: vetor de médias · $\boldsymbol\Sigma$: covariâncias $d\times d$ (`np.cov`) · $|\boldsymbol\Sigma|$ determinante · $\boldsymbol\Sigma^{-1}$ inversa.
- $g_c(\mathbf{x})=-\tfrac12(\mathbf{x}-\boldsymbol\mu_c)^t\boldsymbol\Sigma_c^{-1}(\mathbf{x}-\boldsymbol\mu_c)-\tfrac12\ln|\boldsymbol\Sigma_c|+\ln P(\omega_c)$. O termo quadrático é a **distância de Mahalanobis**; a fronteira é **quadrática**.
- Captura **correlação** entre atributos. Custo de treino $O(nd^2+Cd^3)$. $\boldsymbol\Sigma_c$ pode ser singular → $\boldsymbol\Sigma_c+0{,}1\,I$.

---

## Slide 10 – Regressão Linear Múltipla
$$\hat y=\beta_0+\beta_1x_1+\dots+\beta_dx_d\qquad \hat{\boldsymbol\beta}=\arg\min\sum_i(y_i-\mathbf{x}_i^t\boldsymbol\beta)^2=(A^tA)^{-1}A^t\mathbf{y}$$

- Modelo **paramétrico**: combinação linear dos atributos. Treino $O(nd^2+d^3)$; teste $O(d)$ por amostra (muito rápido).
- Só capta relações **lineares**.

---

## Slide 11 – Robustez numérica da Regressão Linear (achado do projeto)
- No Sberbank há atributos **exatamente dependentes** (ex.: `cafe_avg_price_500` = (`..._max_price_avg` + `..._min_price_avg`)/2): número de condição de $A$ ≈ $10^{18}$.
- Com a **equação normal** $(A^tA)^{-1}A^ty$ o condicionamento é elevado ao quadrado: no fold 4 os coeficientes chegaram a $10^{10}$ e o **R² foi −123**.
- **Solução:** pseudo-inversa de Moore-Penrose por **SVD**, $\hat{\boldsymbol\beta}=A^+\mathbf{y}$, com $A=USV^t\Rightarrow A^+=VS^{-1}U^t$, descartando valores singulares $<10^{-6}\,s_{max}$. Os 5 folds ficaram estáveis (R² entre 0,52 e 0,58).
- Mesma solução da equação normal quando $A^tA$ é inversível, mas estável quando não é.

---

## Slide 12 – Métricas
$$\text{Accuracy}=\tfrac{VP+VN}{VP+VN+FP+FN}\quad P=\tfrac{VP}{VP+FP}\quad R=\tfrac{VP}{VP+FN}\quad F1=\tfrac{2PR}{P+R}\ (\text{média macro})$$
$$R^2=1-\tfrac{\sum(y_i-\hat y_i)^2}{\sum(y_i-\bar y)^2}\qquad R^2_{aj}=1-(1-R^2)\tfrac{n-1}{n-p-1}\qquad MAE=\tfrac1n\sum|y_i-\hat y_i|$$
- Accuracy: fração de acertos. Precision: dos previstos como a classe, quantos eram. Recall: das da classe, quantas foram achadas. F1: média harmônica. R²: variância explicada. R² ajustado: penaliza nº de preditores $p$. MAE: erro médio na unidade do alvo (rublos).
- Referência: o chute da classe majoritária acerta **18,2%**; o aleatório, ≈ 10%.

---

## Slide 13 – Resultados: Classificação (Melbourne, 5 folds)

| Classificador | Acurácia | Precisão | Recall | F1-Score | Tempo Treino (s) | Tempo Teste (s) |
|---|---|---|---|---|---|---|
| K-vizinhos (Dist. Euclidiana) | 0.31 ± 0.00 | 0.28 ± 0.00 | 0.28 ± 0.00 | 0.28 ± 0.00 | 0.00 ± 0.00 | 0.93 ± 0.07 |
| K-vizinhos (Dist. Manhattan) | **0.33 ± 0.00** | **0.30 ± 0.00** | **0.30 ± 0.00** | **0.29 ± 0.00** | 0.00 ± 0.00 | 3.86 ± 0.13 |
| Bayesiano (Univariado) | 0.17 ± 0.01 | 0.25 ± 0.01 | 0.21 ± 0.01 | 0.15 ± 0.01 | 0.04 ± 0.01 | 0.04 ± 0.01 |
| Bayesiano (Multivariado) | 0.18 ± 0.01 | 0.30 ± 0.02 | 0.21 ± 0.01 | 0.16 ± 0.01 | 0.05 ± 0.00 | 0.06 ± 0.00 |

*(valores com 4 casas em `resultados.md`: kNN-Manhattan: acurácia 0,3342 ± 0,0039, F1 0,2945 ± 0,0041.)*

---

## Slide 14 – Resultados: Regressão (Sberbank, 5 folds)

| Regressor | R² | R² ajustado | MAE (rublos) | Tempo Treino (s) | Tempo Teste (s) |
|---|---|---|---|---|---|
| K-vizinhos (Dist. Euclidiana) | 0.47 ± 0.01 | 0.43 ± 0.01 | 1.876.096 ± 32.884 | 0.01 ± 0.00 | 4.03 ± 0.38 |
| K-vizinhos (Dist. Manhattan) | 0.42 ± 0.01 | 0.37 ± 0.01 | 1.956.845 ± 31.215 | 0.01 ± 0.00 | 26.77 ± 0.77 |
| Regressão Linear Múltipla | **0.56 ± 0.02** | **0.53 ± 0.02** | **1.806.398 ± 26.348** | 1.09 ± 0.06 | 0.01 ± 0.01 |

O MAE da melhor solução (≈ 1,8 mi) equivale a ≈ 25% do preço médio (7,12 mi).

---

## Slide 15 – Análise comparativa (classificação)
- **Acurácia e F1:** o kNN-Manhattan foi o melhor (0,33 / 0,29), à frente do kNN-Euclidiana (0,31 / 0,28). Os desvios (≈ 0,004) são pequenos frente às diferenças, então a vantagem do Manhattan é consistente nos 5 folds. Os Bayes ficaram em ≈ 0,17–0,18 de acurácia, **no nível do chute da classe majoritária (0,182)**, com F1 ≈ 0,15–0,16.
- **Por que o Bayes foi mal:** a independência é violada (atributos como `availability_30/60/90/365` e `review_scores_*` são muito correlacionados) e há muitos atributos pouco informativos, que somam ruído na verossimilhança. Em experimento exploratório (holdout interno, não é CV), usando só 7 atributos-chave (`accommodates`, `bedrooms`, `bathrooms`, `beds`, `room_type`, latitude/longitude, `cleaning_fee`, `guests_included`), o mesmo Bayes subiu de ≈ 0,16 para ≈ 0,31, confirmando que a implementação está correta e o problema é a hipótese do modelo. O multivariado (modela correlações) foi um pouco melhor, mas sofre com covariâncias quase singulares.
- **Tempos:** kNN praticamente não treina (≈ 0,00 s) mas **testa lento** (≈ 1 s Euclidiana, ≈ 4 s Manhattan); os Bayes treinam em ≈ 0,04–0,05 s e testam em ≈ 0,04–0,06 s, **≈ 15 a 100 vezes mais rápidos no teste**. Manhattan custa ≈ 4× o tempo da Euclidiana (esta usa multiplicação de matrizes otimizada).
- **Desempenho × eficiência:** o Bayes univariado é o mais rápido (0,07 s no total), mas tem F1 pela metade do melhor. O kNN-Manhattan paga ≈ 4 s para dobrar o F1.

---

## Slide 16 – Análise comparativa (regressão)
- **R² e MAE:** a **Regressão Linear** foi a melhor (R² 0,56; MAE ≈ 1,81 mi), seguida do kNN-Euclidiana (0,47; 1,88 mi) e kNN-Manhattan (0,42; 1,96 mi). O R² ajustado ficou próximo do R² (poucas amostras de teste por atributo não é problema: 6 mil amostras).
- **Tempos:** a linear é a única que "treina de verdade" (≈ 1,1 s, SVD de 24 mil × 439) mas prevê em ≈ 0,01 s. O kNN testa em ≈ 4 s (Euclidiana) e ≈ **27 s (Manhattan)** por fold: centenas de vezes mais lento que a linear na previsão.
- **Pontos fortes e limitações:**

| Algoritmo | Pontos fortes | Limitações observadas |
|---|---|---|
| kNN | simples, sem hipótese sobre os dados, captura não linearidade | teste lento (cresce com $n$ e $d$), sensível à escala e a atributos irrelevantes, escolha de $k$ |
| Bayes univariado | muito rápido | independência violada, assume normalidade, desempenho ≈ baseline aqui |
| Bayes multivariado | modela correlações | covariância $d\times d$ por classe, quase singular, precisa de regularização |
| Regressão Linear | rápida na predição, interpretável, melhor R² | só relações lineares, sensível a outliers (exigiu winsorização) e colinearidade (exigiu SVD) |

---

## Slide 17 – Conclusões gerais
- **Melhor classificador:** **kNN com distância Manhattan** (acurácia 0,33; F1 0,29). **Melhor regressor:** **Regressão Linear Múltipla** (R² 0,56; MAE 1,81 mi).
- **Mais rápido:** classificação, **Bayes univariado** (0,07 s no total); regressão, **Regressão Linear** (≈ 1,1 s no total, contra ≈ 4 e 27 s do kNN).
- **Vale a pena o mais preciso ou o mais rápido?** Critério adotado: entre os modelos a até 0,01 do melhor (F1 ou R²), escolher o de menor tempo. **Melhor equilíbrio — classificador: kNN-Manhattan** (nenhum outro chega perto do F1, e 4 s por ≈ 3.700 amostras é aceitável em uso offline); **regressor: Regressão Linear** (é ao mesmo tempo a mais precisa e a mais rápida na predição). Para classificação em tempo real, o custo de teste do kNN seria o gargalo.
- **Limitações:** um dataset por tarefa; $k$ fixo (5) e sem busca de hiperparâmetros por CV; Bayes assume normalidade (atributos binários/assimétricos violam); colunas de texto do Melbourne descartadas (informação textual não usada); tempos dependem da máquina; kNN por força bruta.
- **Trabalhos futuros:** seleção de atributos, busca de $k$, transformar texto/datas em atributos, regularização (Ridge) na regressão.

---

## Slide 18 – Referências
1. DUDA, R. O.; HART, P. E.; STORK, D. G. **Pattern Classification**. 2. ed. Wiley, 2001.
2. COVER, T.; HART, P. Nearest neighbor pattern classification. **IEEE Trans. on Information Theory**, v. 13, n. 1, 1967.
3. HASTIE, T.; TIBSHIRANI, R.; FRIEDMAN, J. **The Elements of Statistical Learning**. 2. ed. Springer, 2009.
4. BISHOP, C. M. **Pattern Recognition and Machine Learning**. Springer, 2006.
5. GOLUB, G. H.; VAN LOAN, C. F. **Matrix Computations**. 4. ed. Johns Hopkins, 2013. (SVD e pseudo-inversa.)
6. VANSCHOREN, J. et al. OpenML: networked science in machine learning. **SIGKDD Explorations**, v. 15, n. 2, 2014.
7. SHI, X. et al. Benchmarking multimodal AutoML for tabular data with text fields. arXiv:2111.02705, 2021. (Origem do `melbourne_airbnb` no OpenML, conforme o cabeçalho do arquivo.)
8. InsideAirbnb / Kaggle: *Melbourne Airbnb Open Data* (tylerx). https://www.kaggle.com/tylerx/melbourne-airbnb-open-data
9. Sberbank. *Sberbank Russian Housing Market*, Kaggle competition. OpenML: Sberbank_Russian_Housing_Market.
10. NumPy documentation. https://numpy.org/doc/
11. Enunciado do projeto AV2, Profa. Dra. Cynthia Moreira Maia, UNIFOR, 2026.
