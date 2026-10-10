"""PROJETO IA – AV2: kNN, Classificadores Bayesianos e Regressão Linear (tudo em 1 arquivo).

COMO USAR (VS Code):
  1. pip install numpy
  2. Coloque os 2 datasets (ARFF do OpenML) na pasta  data/  (nomes em CONFIG abaixo).
  3. Clique em  Run Python File (▶).  A tabela de resultados aparece no terminal
     e também é salva em  resultados.md.

Somente numpy + biblioteca padrão (sem scikit-learn, sem pandas), como pede o PDF.
Estrutura deste arquivo:
  1. CONFIG        2. Leitura e tratamento dos dados   3. Divisão em folds (método único)
  4. Métricas      5. Algoritmos                        6. Validação cruzada e tabelas
  7. Análise       8. Execução
"""
import csv
import os
import re
import time
from dataclasses import dataclass, field

import numpy as np

# ============================================================================
# 1. CONFIGURAÇÃO (edite aqui se os arquivos tiverem outro nome)
# ============================================================================
PASTA = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "classificacao": {   # Melbourne Airbnb (OpenML) -> prever a faixa de preço (10 classes)
        "arquivos": ["data/melbourne_airbnb.arff", "data/melbourne_airbnb", "data/dataset_.arff", "data/dataset_"],
        "alvo": "price_label",
        # price/weekly/monthly_price vazam o alvo (price_label é o preço em faixas); host_id é identificador
        "ignorar": ["price", "weekly_price", "monthly_price", "host_id"],
        "max_categorias": 40,     # texto com mais de 40 valores distintos = texto livre/ID -> descartado
    },
    "regressao": {       # Sberbank Russian Housing Market (OpenML) -> prever price_doc
        "arquivos": ["data/sberbank.arff", "data/Sberbank_Russian_Housing_Market_dataset.arff",
                     "data/Sberbank_Russian_Housing_Market_dataset"],
        "alvo": "price_doc",
        "ignorar": [],
        "max_categorias": 150,
    },
    "folds": 5,               # nº de folds da validação cruzada
    "k": 5,                   # vizinhos do kNN
    "semente": 42,
    "winsorizar": (1, 99),    # corte de outliers nos percentis do treino (None desliga)
    "saida": "resultados.md",
}


# ============================================================================
# 2. LEITURA E TRATAMENTO DOS DADOS (ARFF/CSV, one-hot, imputação, winsorização, z-score)
# ============================================================================
TOKENS_AUSENTES = {"", "?", "na", "nan", "null", "none"}
_POSITIVOS = {"true", "yes", "y", "t", "1"}
_REGEX_ATRIBUTO = re.compile(
    r"""^@attribute\s+('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"|\S+)\s+(.*)$""",
    re.IGNORECASE,
)


# ----------------------------------------------------------------------------
# Utilitários
# ----------------------------------------------------------------------------
def _limpar_nome(nome):
    nome = nome.strip()
    if len(nome) >= 2 and nome[0] == nome[-1] and nome[0] in "'\"":
        nome = nome[1:-1]
    return nome


def _ausente(valor):
    return valor.strip().lower() in TOKENS_AUSENTES


def _para_float(valor):
    return np.nan if _ausente(valor) else float(valor)



# Tokenizador de linhas de dados ARFF: aceita campos entre aspas simples OU duplas,
# vírgulas dentro das aspas e escapes com barra invertida.
_CAMPO = re.compile(
    r"""\s*(?:"([^"\\]*(?:\\.[^"\\]*)*)"|'([^'\\]*(?:\\.[^'\\]*)*)'|([^,]*))""")


def _desescapar(s):
    return re.sub(r"\\(.)", r"\1", s) if "\\" in s else s


def _dividir_linha_arff(linha):
    campos, pos, n = [], 0, len(linha)
    while True:
        m = _CAMPO.match(linha, pos)
        g1, g2, g3 = m.groups()
        if g1 is not None:
            campos.append(_desescapar(g1))
        elif g2 is not None:
            campos.append(_desescapar(g2))
        else:
            campos.append(g3.strip())
        pos = m.end()
        while pos < n and linha[pos] in " \t":
            pos += 1
        if pos >= n:
            break
        if linha[pos] != ",":
            raise ValueError(f"Campo malformado na linha de dados: ...{linha[max(0, pos-30):pos+30]}...")
        pos += 1
        if pos >= n:
            campos.append("")
            break
    return campos


# ----------------------------------------------------------------------------
# Leitores
# ----------------------------------------------------------------------------
def _ler_arff(caminho):
    """Retorna (nomes, tipos, registros). tipo = 'numerico' | 'texto' | [categorias]."""
    nomes, tipos, linhas = [], [], []
    em_dados = False
    with open(caminho, encoding="utf-8", errors="replace") as f:
        for bruta in f:
            linha = bruta.strip()
            if not linha or linha.startswith("%"):
                continue
            if em_dados:
                linhas.append(linha)
                continue
            baixo = linha.lower()
            if baixo.startswith("@attribute"):
                m = _REGEX_ATRIBUTO.match(linha)
                if not m:
                    raise ValueError(f"Linha @attribute inválida: {linha}")
                nomes.append(_limpar_nome(m.group(1)))
                tipo = m.group(2).strip()
                if tipo.startswith("{"):
                    interior = tipo[tipo.index("{") + 1: tipo.rindex("}")]
                    cats = next(csv.reader([interior], skipinitialspace=True, quotechar="'"))
                    tipos.append([c.strip() for c in cats])
                elif tipo.lower().split()[0] in ("numeric", "real", "integer"):
                    tipos.append("numerico")
                else:
                    tipos.append("texto")
            elif baixo.startswith("@data"):
                em_dados = True
    if linhas and linhas[0].startswith("{"):
        raise ValueError("ARFF esparso não é suportado.")
    registros = [_dividir_linha_arff(l) for l in linhas]
    for r in registros:
        if len(r) != len(nomes):
            raise ValueError("Linha de dados com número de colunas diferente do cabeçalho.")
    return nomes, tipos, registros


def _eh_arff(caminho):
    """ARFF pela extensão ou pelo conteúdo (arquivos do OpenML podem vir sem extensão)."""
    if caminho.lower().endswith(".arff"):
        return True
    with open(caminho, encoding="utf-8", errors="replace") as f:
        for _ in range(400):
            linha = f.readline().strip().lower()
            if linha.startswith(("@relation", "@attribute", "@data")):
                return True
    return False


def _ler_csv(caminho):
    with open(caminho, newline="", encoding="utf-8", errors="replace") as f:
        amostra = f.read(4096)
        f.seek(0)
        try:
            delim = csv.Sniffer().sniff(amostra, delimiters=",;\t").delimiter
        except csv.Error:
            delim = ","
        linhas = [l for l in csv.reader(f, delimiter=delim) if l]
    primeira = linhas[0]
    try:
        [float(v) for v in primeira if not _ausente(v)]
        cabecalho = False
    except ValueError:
        cabecalho = True
    if cabecalho:
        nomes, registros = [n.strip() for n in primeira], linhas[1:]
    else:
        nomes, registros = [f"col_{i}" for i in range(len(primeira))], linhas
    tipos = []
    for j in range(len(nomes)):
        try:
            [float(r[j]) for r in registros if not _ausente(r[j])]
            tipos.append("numerico")
        except ValueError:
            tipos.append("texto")
    return nomes, tipos, registros


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
@dataclass
class Dataset:
    X: np.ndarray                  # (n, d) float, pode conter NaN (imputado depois, por fold)
    y: np.ndarray                  # classes codificadas (int) ou alvo (float)
    nomes_features: list
    nome_alvo: str
    tarefa: str
    classes: np.ndarray = None     # nomes originais das classes (classificação)
    n_atributos_originais: int = 0
    n_atributos_usados: int = 0
    n_ausentes: int = 0
    n_removidas_sem_alvo: int = 0
    descartadas: list = field(default_factory=list)   # [(coluna, motivo)]

    def resumo(self):
        n, d = self.X.shape
        linhas = [
            f"- Amostras: **{n}**",
            f"- Atributos preditores no arquivo original: **{self.n_atributos_originais}**",
            f"- Atributos usados após o tratamento: **{self.n_atributos_usados}** "
            f"(**{d}** colunas numéricas finais, após one-hot dos categóricos)",
            f"- Atributo alvo: `{self.nome_alvo}`",
            f"- Valores ausentes nos preditores usados (imputados pela média do treino): {self.n_ausentes}",
        ]
        if self.n_removidas_sem_alvo:
            linhas.append(f"- Linhas removidas por alvo ausente: {self.n_removidas_sem_alvo}")
        if self.tarefa == "classificacao":
            cont = np.bincount(self.y, minlength=len(self.classes))
            dist = ", ".join(f"{c}: {q} ({100*q/n:.1f}%)" for c, q in zip(self.classes, cont))
            linhas.append(f"- Classes ({len(self.classes)}): {dist}")
        else:
            linhas.append(
                f"- Alvo: média={self.y.mean():.4f}, desvio={self.y.std():.4f}, "
                f"min={self.y.min():.4f}, mediana={np.median(self.y):.4f}, max={self.y.max():.4f}"
            )
        if self.descartadas:
            motivos = {}
            for nome, motivo in self.descartadas:
                motivos.setdefault(motivo, []).append(nome)
            linhas.append(f"- Atributos descartados: **{len(self.descartadas)}**")
            for motivo, nomes in motivos.items():
                linhas.append(f"  - {motivo} ({len(nomes)}): " + ", ".join(f"`{x}`" for x in nomes))
        return "\n".join(linhas)


def _norm(v):
    """Normaliza um valor textual: tira espaços, aspas envolventes e barras de escape."""
    v = v.strip().replace("\\", "")
    while len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
        v = v[1:-1].strip()
    return v


def _tentar_numerico(valores):
    """Se todos os valores não ausentes forem números (aceita sufixo %), devolve array float."""
    saida = np.empty(len(valores))
    for i, v in enumerate(valores):
        if _ausente(v):
            saida[i] = np.nan
            continue
        v = v.strip()
        if v.endswith("%"):
            v = v[:-1]
        try:
            saida[i] = float(v)
        except ValueError:
            return None
    return saida


def carregar_dataset(caminho, alvo=None, tarefa="classificacao", ignorar=(), max_categorias=150):
    """Carrega ARFF/CSV e devolve um Dataset numérico.

    alvo           : nome ou índice da coluna alvo (padrão: última).
    ignorar        : nomes de colunas a descartar (ex.: colunas que vazam o alvo, IDs).
    max_categorias : coluna textual com mais valores distintos que isso é tratada como
                     texto livre/identificador e descartada (one-hot inviável).
    Também são descartadas colunas constantes. Colunas textuais só com números
    (ex.: '100%') viram numéricas; as demais categóricas viram one-hot.
    """
    if _eh_arff(caminho):
        nomes, tipos, registros = _ler_arff(caminho)
    else:
        nomes, tipos, registros = _ler_csv(caminho)

    if alvo is None:
        t = len(nomes) - 1
    elif alvo in nomes:
        t = nomes.index(alvo)
    elif str(alvo).lstrip("-").isdigit():
        t = int(alvo)
    else:
        raise ValueError(f"Coluna alvo '{alvo}' não encontrada. Colunas: {nomes}")

    ignorar = set(ignorar or ())
    desconhecidas = ignorar - set(nomes)
    if desconhecidas:
        raise ValueError(f"Colunas a ignorar não existem no arquivo: {sorted(desconhecidas)}")

    validas = [r for r in registros if not _ausente(r[t])]
    n_removidas = len(registros) - len(validas)
    registros = validas
    n = len(registros)

    blocos, nomes_feat, descartadas = [], [], []
    n_orig = n_usados = n_aus = 0
    for j, (nome, tipo) in enumerate(zip(nomes, tipos)):
        if j == t:
            continue
        n_orig += 1
        if nome in ignorar:
            descartadas.append((nome, "ignorada por escolha (vazamento do alvo / identificador)"))
            continue
        brutos = [r[j] for r in registros]

        if tipo == "texto":                       # STRING: numérica disfarçada ou categórica?
            col = _tentar_numerico(brutos)
            if col is not None:
                tipo = "numerico"
        else:
            col = None

        if tipo == "numerico":
            if col is None:
                col = np.array([_para_float(v) for v in brutos], dtype=float)
            if np.isnan(col).all() or np.nanmax(col) == np.nanmin(col):
                descartadas.append((nome, "constante"))
                continue
            n_aus += int(np.isnan(col).sum())
            blocos.append(col[:, None]); nomes_feat.append(nome); n_usados += 1
        else:                                      # categórico -> one-hot
            valores = [None if _ausente(v) else _norm(v) for v in brutos]
            observados = {v for v in valores if v is not None}
            if len(observados) <= 1:
                descartadas.append((nome, "constante"))
                continue
            if isinstance(tipo, list):
                cats = [_norm(c) for c in tipo]
            else:
                if len(observados) > max_categorias:
                    descartadas.append(
                        (nome, f"texto livre / identificador (> {max_categorias} valores distintos)"))
                    continue
                cats = sorted(observados)
            if len(cats) == 2:
                # atributo binário -> UMA coluna 0/1 (duas colunas seriam espelhadas/redundantes)
                cats = [next((c for c in cats if c.lower() in _POSITIVOS), cats[-1])]
            mapa = {c: i for i, c in enumerate(cats)}
            oh = np.zeros((n, len(cats)))
            for i, v in enumerate(valores):
                if v is None:
                    n_aus += 1
                    continue
                k = mapa.get(v)
                if k is not None:
                    oh[i, k] = 1.0
            blocos.append(oh)
            nomes_feat.extend(f"{nome}={c}" for c in cats)
            n_usados += 1

    X = np.hstack(blocos)
    if tarefa == "classificacao":
        brutos = np.array([_norm(r[t]) for r in registros])
        classes, y = np.unique(brutos, return_inverse=True)
        # ordena classes numéricas por valor ("2" antes de "10")
        try:
            ordem = np.argsort([float(c) for c in classes])
            inv = np.empty_like(ordem); inv[ordem] = np.arange(len(ordem))
            classes, y = classes[ordem], inv[y]
        except ValueError:
            pass
    else:
        y = _tentar_numerico([r[t] for r in registros])
        if y is None or np.isnan(y).any():
            raise ValueError("Para regressão o alvo precisa ser numérico.")
        classes = None

    return Dataset(X, y, nomes_feat, nomes[t], tarefa, classes, n_orig, n_usados,
                   n_aus, n_removidas, descartadas)


# ----------------------------------------------------------------------------
# Pré-processamento (ajustado SOMENTE no treino de cada fold -> sem vazamento)
# ----------------------------------------------------------------------------
class Preprocessador:
    """Pré-processamento numérico, ajustado SOMENTE no treino de cada fold.

    1. Imputação de valores ausentes pela média do treino.
    2. Winsorização (opcional): corta cada coluna "contínua" (> 2 valores distintos)
       nos percentis (p_inf, p_sup) do treino -> reduz o efeito de outliers extremos.
    3. Padronização z-score: (x - media) / desvio.
    4. Remoção de atributos redundantes: colunas constantes e, de cada par com
       |correlação de Pearson| >= corr_max, a de índice maior (mantém a primeira).
       Evita colinearidade (instabilidade da regressão linear e das covariâncias).
    """

    def __init__(self, winsorizar=(1, 99), corr_max=0.99):
        self.winsorizar = winsorizar
        self.corr_max = corr_max

    def fit(self, X):
        validos = ~np.isnan(X)
        cont = np.maximum(validos.sum(axis=0), 1)
        self.media_ = np.where(validos, X, 0.0).sum(axis=0) / cont     # média p/ imputar
        Xi = np.where(validos, X, self.media_)
        d = X.shape[1]
        self.inf_ = np.full(d, -np.inf)
        self.sup_ = np.full(d, np.inf)
        if self.winsorizar is not None:
            p_inf, p_sup = self.winsorizar
            lo, hi = np.percentile(Xi, [p_inf, p_sup], axis=0)
            continua = np.array([len(np.unique(Xi[:, j])) > 2 for j in range(d)])
            self.inf_[continua], self.sup_[continua] = lo[continua], hi[continua]
            Xi = np.clip(Xi, self.inf_, self.sup_)
        self.media_pos_ = Xi.mean(axis=0)
        desvio = Xi.std(axis=0)
        constante = desvio < 1e-12
        self.desvio_ = np.where(constante, 1.0, desvio)

        manter = np.where(~constante)[0]
        if self.corr_max is not None and len(manter) > 1:
            Z = (Xi[:, manter] - self.media_pos_[manter]) / self.desvio_[manter]
            C = np.abs(Z.T @ Z) / len(Z)                    # |correlação de Pearson|
            mantidas = []
            for j in range(len(manter)):
                if not mantidas or C[j, mantidas].max() < self.corr_max:
                    mantidas.append(j)
            manter = manter[mantidas]
        self.manter_ = manter
        return self

    def transform(self, X):
        Xi = np.where(np.isnan(X), self.media_, X)
        Xi = np.clip(Xi, self.inf_, self.sup_)
        return ((Xi - self.media_pos_) / self.desvio_)[:, self.manter_]


# ============================================================================
# 3. DIVISÃO EM FOLDS – MÉTODO ÚNICO usado por todos os algoritmos
# ============================================================================
def criar_folds(n_amostras, k=5, semente=42, y=None):
    """Divide os índices 0..n-1 em k folds disjuntos.

    - y=None  : k-fold aleatório simples (usado na regressão).
    - y dado  : k-fold ESTRATIFICADO (usado na classificação), mantendo a
                proporção de cada classe em cada fold.
    Cada amostra aparece em exatamente um fold -> cada fold é teste 1 vez.
    """
    if not 2 <= k <= n_amostras:
        raise ValueError("k deve estar entre 2 e o número de amostras.")
    rng = np.random.RandomState(semente)

    if y is None:
        indices = rng.permutation(n_amostras)
        folds = [np.sort(f) for f in np.array_split(indices, k)]
    else:
        y = np.asarray(y)
        listas = [[] for _ in range(k)]
        deslocamento = 0  # continua o rodízio entre classes -> folds de tamanhos parecidos
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            rng.shuffle(idx)
            for pos, i in enumerate(idx):
                listas[(pos + deslocamento) % k].append(i)
            deslocamento += len(idx)
        folds = [np.sort(np.array(l, dtype=int)) for l in listas]

    validar_folds(folds, n_amostras)
    return folds


def validar_folds(folds, n_amostras):
    """Garante: folds disjuntos, cobrindo todas as amostras (cada fold testa 1x)."""
    todos = np.concatenate(folds)
    assert len(todos) == n_amostras, "Folds não cobrem todas as amostras."
    assert len(np.unique(todos)) == n_amostras, "Há amostras repetidas entre folds."
    assert all(len(f) > 0 for f in folds), "Há fold vazio."


def particoes(folds):
    """Gera (numero_do_fold, indices_treino, indices_teste)."""
    for i, teste in enumerate(folds):
        treino = np.concatenate([f for j, f in enumerate(folds) if j != i])
        yield i, treino, teste


# ============================================================================
# 4. MÉTRICAS (implementadas manualmente)
# ============================================================================
def matriz_confusao(y_true, y_pred, classes):
    pos = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[pos[t], pos[p]] += 1
    return cm


def acuracia(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def precisao_recall_f1(y_true, y_pred):
    """Média MACRO (todas as classes pesam igual)."""
    classes = np.unique(np.concatenate([y_true, y_pred]))
    cm = matriz_confusao(y_true, y_pred, classes)
    precs, recs, f1s = [], [], []
    for c in range(len(classes)):
        tp = cm[c, c]
        pred_c, real_c = cm[:, c].sum(), cm[c, :].sum()
        p = tp / pred_c if pred_c > 0 else 0.0
        r = tp / real_c if real_c > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precs.append(p); recs.append(r); f1s.append(f)
    return float(np.mean(precs)), float(np.mean(recs)), float(np.mean(f1s))


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def r2_ajustado(r2_valor, n, p):
    """1 - (1 - R²)(n - 1)/(n - p - 1); n = amostras de teste, p = nº de preditores."""
    if n - p - 1 <= 0:
        return float("nan")
    return float(1.0 - (1.0 - r2_valor) * (n - 1) / (n - p - 1))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


# ============================================================================
# 5. ALGORITMOS (kNN, Bayes univariado, Bayes multivariado, Regressão Linear)
# ============================================================================
class KNN:
    """k-vizinhos mais próximos para classificação (voto) ou regressão (média).

    Distâncias (força bruta, vetorizadas com numpy):
      - euclidiana: sqrt(sum (x_i - y_i)^2), calculada em float64 pela identidade
        ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x.y (usa BLAS, muito mais rápido);
      - manhattan : sum |x_i - y_i|, acumulada atributo a atributo em float32
        (mesmos vizinhos que float64 nos testes; ~3x mais rápido e usa menos memória).
    """

    def __init__(self, k=5, metrica="euclidiana", tarefa="classificacao",
                 max_elementos_lote=5_000_000):
        if metrica not in ("euclidiana", "manhattan"):
            raise ValueError("metrica deve ser 'euclidiana' ou 'manhattan'")
        self.k, self.metrica, self.tarefa = k, metrica, tarefa
        self.max_elementos_lote = max_elementos_lote

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=float)
        if self.metrica == "euclidiana":
            self.sq_ = np.einsum("ij,ij->i", self.X_, self.X_)          # ||y||^2
        else:
            self.XT32_ = np.ascontiguousarray(self.X_.T, dtype=np.float32)
        if self.tarefa == "classificacao":
            self.classes_, self.y_ = np.unique(y, return_inverse=True)
        else:
            self.y_ = np.asarray(y, dtype=float)
        return self

    def _indices_vizinhos(self, X, k):
        """Índices (m, k) dos k vizinhos mais próximos de cada linha de X."""
        n_tr = self.X_.shape[0]
        saida = []
        if self.metrica == "euclidiana":
            lote = max(1, self.max_elementos_lote // n_tr)
            for ini in range(0, len(X), lote):
                q = X[ini:ini + lote]
                D = np.einsum("ij,ij->i", q, q)[:, None] + self.sq_[None, :] - 2.0 * (q @ self.X_.T)
                np.maximum(D, 0.0, out=D)           # evita negativos por erro de arredondamento
                np.sqrt(D, out=D)
                saida.append(np.argpartition(D, k - 1, axis=1)[:, :k])
        else:
            X32 = X.astype(np.float32)
            lote = max(1, 100_000 // n_tr)           # bloco pequeno cabe no cache
            tmp = np.empty((lote, n_tr), dtype=np.float32)
            for ini in range(0, len(X32), lote):
                q = X32[ini:ini + lote]
                m = len(q)
                D = np.zeros((m, n_tr), dtype=np.float32)
                t = tmp[:m]
                for j in range(q.shape[1]):         # soma |x_j - y_j| atributo a atributo
                    np.subtract(q[:, j:j + 1], self.XT32_[j][None, :], out=t)
                    np.abs(t, out=t)
                    D += t
                saida.append(np.argpartition(D, k - 1, axis=1)[:, :k])
        return np.vstack(saida)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        k = min(self.k, self.X_.shape[0])
        idx = self._indices_vizinhos(X, k)
        viz = self.y_[idx]
        if self.tarefa == "classificacao":
            votos = np.zeros((len(viz), len(self.classes_)))
            for c in range(len(self.classes_)):
                votos[:, c] = (viz == c).sum(axis=1)
            return self.classes_[votos.argmax(axis=1)]     # empate: menor índice de classe
        return viz.mean(axis=1)


class BayesUnivariado:
    """Classificador bayesiano com densidade normal UNIVARIADA por atributo.

    p(x|w_c) = prod_j N(x_j; mu_cj, sigma_cj^2)   (atributos independentes)
    Decisão: argmax_c  log P(w_c) + sum_j log p(x_j|w_c)
    """

    def __init__(self, epsilon=0.1):
        # piso somado à variância (dados padronizados => variância global = 1).
        # Evita variância ~0 em atributos binários raros; valor escolhido em holdout interno.
        self.epsilon = epsilon

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_, yi = np.unique(y, return_inverse=True)
        self.priors_ = np.bincount(yi) / len(yi)
        self.medias_ = np.array([X[yi == c].mean(axis=0) for c in range(len(self.classes_))])
        self.vars_ = np.array([X[yi == c].var(axis=0) for c in range(len(self.classes_))]) + self.epsilon
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        log_post = np.empty((len(X), len(self.classes_)))
        for c in range(len(self.classes_)):
            mu, var = self.medias_[c], self.vars_[c]
            # log N(x;mu,s^2) = -1/2 [ log(2*pi*s^2) + ((x-mu)/s)^2 ]
            ll = -0.5 * np.sum(np.log(2 * np.pi * var) + (X - mu) ** 2 / var, axis=1)
            log_post[:, c] = np.log(self.priors_[c]) + ll
        return self.classes_[log_post.argmax(axis=1)]


class BayesMultivariado:
    """Classificador bayesiano com densidade normal MULTIVARIADA (cov. completa por classe).

    p(x|w_c) = 1/((2pi)^(d/2)|S_c|^(1/2)) exp(-1/2 (x-mu_c)^T S_c^-1 (x-mu_c))
    Usa log-verossimilhança e regularização S_c + reg*I (S_c costuma ser singular).
    """

    def __init__(self, reg=0.1):
        self.reg = reg   # lambda de S + lambda*I (dados padronizados); escolhido em holdout interno

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        d = X.shape[1]
        self.classes_, yi = np.unique(y, return_inverse=True)
        self.priors_ = np.bincount(yi) / len(yi)
        self.medias_, self.invs_, self.logdets_ = [], [], []
        for c in range(len(self.classes_)):
            Xc = X[yi == c]
            mu = Xc.mean(axis=0)
            S = np.cov(Xc, rowvar=False) if len(Xc) > 1 else np.zeros((d, d))
            S = np.atleast_2d(S) + self.reg * np.eye(d)
            _, logdet = np.linalg.slogdet(S)
            try:
                inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                inv = np.linalg.pinv(S)
            self.medias_.append(mu); self.invs_.append(inv); self.logdets_.append(logdet)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        d = X.shape[1]
        log_post = np.empty((len(X), len(self.classes_)))
        for c in range(len(self.classes_)):
            dif = X - self.medias_[c]
            maha = np.sum((dif @ self.invs_[c]) * dif, axis=1)   # (x-mu)^T S^-1 (x-mu)
            log_post[:, c] = np.log(self.priors_[c]) - 0.5 * (d * np.log(2 * np.pi) + self.logdets_[c] + maha)
        return self.classes_[log_post.argmax(axis=1)]


class RegressaoLinearMultipla:
    """Mínimos quadrados ordinários (OLS): minimiza sum (y - A beta)^2, A = [1, X].

    Solução: beta = A^+ y, com A^+ a pseudo-inversa de Moore-Penrose calculada por SVD
    (A = U S V^t  =>  A^+ = V S^-1 U^t, descartando valores singulares < rcond * s_max; 1e-6 estabilizou os 5 folds).
    Quando A^t A é inversível isto é idêntico à equação normal (A^t A)^-1 A^t y, mas é
    estável com atributos colineares (ex.: média = (máx + mín)/2), em que a equação
    normal eleva o número de condição ao quadrado e produz coeficientes absurdos.
    """

    def __init__(self, rcond=1e-6):
        self.rcond = rcond

    def fit(self, X, y):
        A = np.hstack([np.ones((len(X), 1)), np.asarray(X, dtype=float)])
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        s_inv = np.where(s > self.rcond * s[0], 1.0 / s, 0.0)
        self.beta_ = Vt.T @ (s_inv * (U.T @ np.asarray(y, dtype=float)))
        return self

    def predict(self, X):
        A = np.hstack([np.ones((len(X), 1)), np.asarray(X, dtype=float)])
        return A @ self.beta_


# ============================================================================
# 6. VALIDAÇÃO CRUZADA, TEMPOS E TABELAS
# ============================================================================
COLUNAS = {
    "classificacao": [("acuracia", "Acurácia"), ("precisao", "Precisão"), ("recall", "Recall"),
                      ("f1", "F1-Score"), ("tempo_treino", "Tempo Treino (s)"),
                      ("tempo_teste", "Tempo Teste (s)")],
    "regressao": [("r2", "R²"), ("r2_ajustado", "R² ajustado"), ("mae", "MAE"),
                  ("tempo_treino", "Tempo Treino (s)"), ("tempo_teste", "Tempo Teste (s)")],
}


def validacao_cruzada(fabrica_modelo, X, y, folds, tarefa, winsorizar=(1, 99)):
    """Roda todos os folds (cada um é teste exatamente 1 vez) e devolve métricas por fold."""
    chaves = [c for c, _ in COLUNAS[tarefa]]
    res = {c: [] for c in chaves}
    for i, tr, te in particoes(folds):
        prep = Preprocessador(winsorizar).fit(X[tr])          # ajusta SÓ no treino
        Xtr, Xte = prep.transform(X[tr]), prep.transform(X[te])
        ytr, yte = y[tr], y[te]

        modelo = fabrica_modelo()
        t0 = time.perf_counter(); modelo.fit(Xtr, ytr); t_treino = time.perf_counter() - t0
        t0 = time.perf_counter(); pred = modelo.predict(Xte); t_teste = time.perf_counter() - t0

        if tarefa == "classificacao":
            p, r, f = precisao_recall_f1(yte, pred)
            valores = {"acuracia": acuracia(yte, pred), "precisao": p, "recall": r, "f1": f}
        else:
            r2v = r2(yte, pred)
            valores = {"r2": r2v, "r2_ajustado": r2_ajustado(r2v, len(yte), X.shape[1]),
                       "mae": mae(yte, pred)}
        valores["tempo_treino"], valores["tempo_teste"] = t_treino, t_teste
        for c in chaves:
            res[c].append(valores[c])
    return {c: np.array(v) for c, v in res.items()}


def tabela_markdown(resultados, tarefa):
    rotulo = "Classificador" if tarefa == "classificacao" else "Regressor"
    cols = COLUNAS[tarefa]
    linhas = ["| " + " | ".join([rotulo] + [r for _, r in cols]) + " |",
              "|" + "---|" * (len(cols) + 1)]
    for nome, res in resultados.items():
        cel = [f"{res[c].mean():.4f} ± {res[c].std():.4f}" for c, _ in cols]
        linhas.append("| " + " | ".join([nome] + cel) + " |")
    return "\n".join(linhas)


def _fmt(chave, media, desvio):
    if chave == "mae":
        return f"{media:,.0f} ± {desvio:,.0f}"
    return f"{media:.2f} ± {desvio:.2f}"


def tabela_console(resultados, tarefa, numero):
    """Tabela no formato da Tabela 1 do PDF (média ± desvio padrão, tempos em segundos)."""
    rotulo = "Classificador" if tarefa == "classificacao" else "Regressor"
    cols = COLUNAS[tarefa]
    cab = [rotulo] + [r for _, r in cols]
    linhas = [[nome] + [_fmt(c, res[c].mean(), res[c].std()) for c, _ in cols]
              for nome, res in resultados.items()]
    larg = [max(len(l[i]) for l in [cab] + linhas) for i in range(len(cab))]
    fmt = lambda l: "  ".join(l[i].ljust(larg[i]) if i == 0 else l[i].rjust(larg[i]) for i in range(len(l)))
    barra = "-" * (sum(larg) + 2 * (len(larg) - 1))
    nome_tipo = "classificadores" if tarefa == "classificacao" else "regressores"
    saida = [f"Tabela com Médias, Desvio Padrão e Tempos (média ± desvio dos folds)", barra, fmt(cab), barra]
    saida += [fmt(l) for l in linhas] + [barra,
              f"Tabela {numero}: Análise comparativa do desempenho dos {nome_tipo}, com médias, desvios",
              "padrão (±) e tempos de treino e teste."]
    return "\n".join(saida)


# ============================================================================
# 7. ANÁLISE COMPARATIVA AUTOMÁTICA
# ============================================================================
def _m(res, nome, chave):
    return float(res[nome][chave].mean())


def gerar_analise(resultados, tarefa, tolerancia=0.01):
    nomes = list(resultados)
    if tarefa == "classificacao":
        quesitos = [("acuracia", "Acurácia", True), ("f1", "F1-Score", True)]
        principal, rot_p = "f1", "F1-Score"
    else:
        quesitos = [("r2", "R²", True), ("mae", "MAE", False)]
        principal, rot_p = "r2", "R²"

    L = []
    for chave, rot, maior in quesitos:
        f = max if maior else min
        melhor = f(nomes, key=lambda n: _m(resultados, n, chave))
        r = resultados[melhor][chave]
        L.append(f"- Melhor {rot}: **{melhor}** ({r.mean():.4f} ± {r.std():.4f}).")

    total = {n: _m(resultados, n, "tempo_treino") + _m(resultados, n, "tempo_teste") for n in nomes}
    mais_rap_tr = min(nomes, key=lambda n: _m(resultados, n, "tempo_treino"))
    mais_rap_te = min(nomes, key=lambda n: _m(resultados, n, "tempo_teste"))
    mais_rap = min(nomes, key=lambda n: total[n])
    L.append(f"- Mais rápido no treino: **{mais_rap_tr}** ({_m(resultados, mais_rap_tr, 'tempo_treino'):.4f} s).")
    L.append(f"- Mais rápido no teste: **{mais_rap_te}** ({_m(resultados, mais_rap_te, 'tempo_teste'):.4f} s).")
    L.append(f"- Mais rápido no total (treino + teste): **{mais_rap}** ({total[mais_rap]:.4f} s).")

    mais_preciso = max(nomes, key=lambda n: _m(resultados, n, principal))
    melhor_valor = _m(resultados, mais_preciso, principal)
    candidatos = [n for n in nomes if _m(resultados, n, principal) >= melhor_valor - tolerancia]
    equilibrio = min(candidatos, key=lambda n: total[n])
    L.append("")
    L.append(f"**Critério de equilíbrio:** entre os modelos cujo {rot_p} médio está a no máximo "
             f"{tolerancia:.2f} do melhor, escolhe-se o de menor tempo total (treino + teste).")
    L.append(f"**Melhor equilíbrio desempenho × tempo: {equilibrio}** "
             f"({rot_p} = {_m(resultados, equilibrio, principal):.4f}, tempo total = {total[equilibrio]:.4f} s).")
    if mais_preciso != equilibrio:
        razao = total[mais_preciso] / max(total[equilibrio], 1e-12)
        dif = melhor_valor - _m(resultados, equilibrio, principal)
        L.append(f"O mais preciso ({mais_preciso}) leva ~{razao:.1f}x o tempo do escolhido "
                 f"para ganhar {dif:.4f} em {rot_p}.")
    else:
        L.append("O modelo mais preciso também é o de melhor equilíbrio.")
    return "\n".join(L)


# ============================================================================
# 8. EXECUÇÃO
# ============================================================================
def achar_arquivo(candidatos):
    for c in candidatos:
        caminho = c if os.path.isabs(c) else os.path.join(PASTA, c)
        if os.path.isfile(caminho):
            return caminho
    raise FileNotFoundError(
        "Dataset não encontrado. Coloque o arquivo na pasta 'data/' com um destes nomes:\n  - "
        + "\n  - ".join(candidatos))


def executar(tarefa, numero_tabela):
    cfg = CONFIG[tarefa]
    k, n_folds, semente = CONFIG["k"], CONFIG["folds"], CONFIG["semente"]
    nome_tarefa = "CLASSIFICAÇÃO" if tarefa == "classificacao" else "REGRESSÃO"
    caminho = achar_arquivo(cfg["arquivos"])
    print(f"\n{'=' * 78}\n{nome_tarefa}  |  {os.path.basename(caminho)}\n{'=' * 78}")
    print("Lendo e tratando o dataset (leitor ARFF próprio, sem pandas)...", flush=True)
    ds = carregar_dataset(caminho, cfg["alvo"], tarefa, ignorar=cfg["ignorar"],
                          max_categorias=cfg["max_categorias"])
    resumo = ds.resumo()
    print(resumo.replace("**", "").replace("`", ""))
    if ds.n_atributos_originais <= 10 or len(ds.y) <= 1000:
        print("[ATENÇÃO] O PDF exige mais de 10 atributos e mais de 1000 instâncias.")

    # divisão ÚNICA em folds, compartilhada por todos os algoritmos
    y_estrat = ds.y if tarefa == "classificacao" else None
    folds = criar_folds(len(ds.y), n_folds, semente, y=y_estrat)
    tam = [len(f) for f in folds]
    print(f"\nValidação cruzada: {n_folds} folds ({'estratificado' if y_estrat is not None else 'aleatório'}), "
          f"cada fold é teste 1 vez. Tamanhos dos folds de teste: {tam}\n")

    if tarefa == "classificacao":
        modelos = {
            "K-vizinhos (Dist. Euclidiana)": lambda: KNN(k, "euclidiana", tarefa),
            "K-vizinhos (Dist. Manhattan)": lambda: KNN(k, "manhattan", tarefa),
            "Bayesiano (Univariado)": lambda: BayesUnivariado(),
            "Bayesiano (Multivariado)": lambda: BayesMultivariado(),
        }
    else:
        modelos = {
            "K-vizinhos (Dist. Euclidiana)": lambda: KNN(k, "euclidiana", tarefa),
            "K-vizinhos (Dist. Manhattan)": lambda: KNN(k, "manhattan", tarefa),
            "Regressão Linear Múltipla": lambda: RegressaoLinearMultipla(),
        }

    resultados = {}
    for nome, fab in modelos.items():
        print(f"-> treinando e testando: {nome} ...", flush=True)
        resultados[nome] = validacao_cruzada(fab, ds.X, ds.y, folds, tarefa, CONFIG["winsorizar"])

    analise = gerar_analise(resultados, tarefa)
    print("\n" + tabela_console(resultados, tarefa, numero_tabela))
    print("\nANÁLISE COMPARATIVA\n" + analise.replace("**", ""))
    return (f"# {nome_tarefa.capitalize()} ({os.path.basename(caminho)})\n\n## Dataset\n\n{resumo}\n\n"
            f"## Validação cruzada\n\n- {n_folds} folds, semente {semente}; tamanhos dos folds de teste: {tam}\n"
            f"- k do kNN: {k}\n\n## Tabela comparativa (média ± desvio padrão dos folds)\n\n"
            f"{tabela_markdown(resultados, tarefa)}\n\n## Análise\n\n{analise}\n")


if __name__ == "__main__":
    partes = [executar("classificacao", 1), executar("regressao", 2)]
    with open(os.path.join(PASTA, CONFIG["saida"]), "w", encoding="utf-8") as f:
        f.write("\n\n".join(partes))
    print(f"\n(Resultados salvos em '{CONFIG['saida']}')")


