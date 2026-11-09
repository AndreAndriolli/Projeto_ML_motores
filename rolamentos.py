"""
rolamentos.py
=============
Módulo com toda a lógica do pipeline de diagnóstico de rolamentos com base no
dataset de Paderborn (Lessmeier et al., PHME 2016).

Os notebooks só chamam as funções daqui. Assim a mesma lógica não fica copiada
em vários lugares e qualquer correção vale para o pipeline inteiro.

Seções:
    1. Constantes (sinais, condições, geometria do rolamento, rótulos)
    2. Leitura dos arquivos .mat
    3. Extração de features (tempo, frequência, tempo-frequência, envelope)
    4. Extração em lote, em paralelo
    5. Rótulos e metadados
    6. Seleção de features (transformadores scikit-learn)
    7. Modelos (incluindo ELM) e pipelines
    8. Divisões treino/teste por rolamento (Tabela 10 do artigo e GroupKFold)
    9. Avaliação e resumo dos resultados
"""
from __future__ import annotations

import re
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
from joblib import Parallel, delayed
from scipy import signal, stats
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectorMixin
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score)
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

try:  # scikit-learn >= 1.6
    from sklearn.utils.validation import validate_data
except ImportError:  # versões antigas: mesmo comportamento pelo método interno
    def validate_data(estimador, X="no_validation", y="no_validation", **kw):
        kw.pop("ensure_all_finite", None)
        return estimador._validate_data(X, y, force_all_finite=False, **kw)

try:  # PyWavelets é opcional: sem ele, usamos bandas de FFT equivalentes
    import pywt
    TEM_PYWT = True
except ImportError:  # pragma: no cover
    TEM_PYWT = False

# =============================================================================
# 1. CONSTANTES
# =============================================================================
FS = 64_000  # Hz. Taxa nominal das correntes e da vibração (artigo, seção 5)

# Nome do canal no .mat  ->  prefixo curto usado nas colunas de features
SINAIS = {"phase_current_1": "i1", "phase_current_2": "i2", "vibration_1": "vib"}

# Grupos de sensores usados na modelagem
GRUPOS_SENSORES = {"MCS": ("i1", "i2"), "Vibracao": ("vib",), "Fusao": ("i1", "i2", "vib")}

# Condições de operação (Tabela 6 do artigo): rotação [rpm], torque [Nm], força radial [N]
CONDICOES = {
    "N15_M07_F10": (1500, 0.7, 1000),  # setting 0, usada nos resultados do artigo
    "N09_M07_F10": (900, 0.7, 1000),
    "N15_M01_F10": (1500, 0.1, 1000),
    "N15_M07_F04": (1500, 0.7, 400),
}

# Geometria do rolamento 6203 (8 esferas, d = 6,75 mm, D = 28,55 mm, ângulo 0)
N_ESFERAS, D_ESFERA, D_PRIMITIVO = 8, 6.75, 28.55
_R = D_ESFERA / D_PRIMITIVO
FATORES_DEFEITO = {                     # multiplicam a frequência de rotação fr
    "bpfo": N_ESFERAS / 2 * (1 - _R),   # anel externo  ~3,05·fr
    "bpfi": N_ESFERAS / 2 * (1 + _R),   # anel interno  ~4,95·fr
    "bsf": 1 / (2 * _R) * (1 - _R**2),  # esfera        ~2,00·fr
}

# Rolamentos por origem
SAUDAVEIS = ["K001", "K002", "K003", "K004", "K005", "K006"]
ARTIFICIAIS = ["KA01", "KA03", "KA05", "KA06", "KA07", "KA08", "KA09",
               "KI01", "KI03", "KI05", "KI07", "KI08"]
MULTIPLOS = ["KB23", "KB24", "KB27"]

# Tabela 10 do artigo: 5 rolamentos por classe (cenário com danos reais)
TABELA_10 = {
    0: ["K001", "K002", "K003", "K004", "K005"],
    1: ["KA04", "KA15", "KA16", "KA22", "KA30"],
    2: ["KI04", "KI14", "KI16", "KI18", "KI21"],
}

NOMES_CLASSES = {
    "artigo": {0: "saudavel", 1: "anel_externo", 2: "anel_interno"},
    "4classes": {0: "saudavel", 1: "anel_externo", 2: "anel_interno", 3: "multiplo"},
}

_PADRAO_NOME = re.compile(r"^(N\d{2})_(M\d{2})_(F\d{2})_(K[A-Z]?\d{2,3})_(\d+)$")


# =============================================================================
# 2. LEITURA
# =============================================================================
def ler_mat(caminho):
    """Lê um .mat do Paderborn e devolve {nome_do_canal: array 1-D}.

    Usa os vetores brutos de `Y.Data`. Não interpola nem junta canais com taxas
    diferentes, o que corrige o erro de taxa de amostragem da versão anterior.
    """
    caminho = Path(caminho)
    mat = scipy.io.loadmat(caminho, squeeze_me=True, struct_as_record=False)
    # A variável interna tem o nome do arquivo. Por segurança, pega a única
    # variável que não começa com "__".
    chave = caminho.stem if caminho.stem in mat else next(k for k in mat if not k.startswith("__"))
    canais = np.atleast_1d(mat[chave].Y)
    return {str(c.Name): np.asarray(c.Data, dtype=np.float64).ravel() for c in canais}


def interpretar_nome(nome):
    """'N15_M07_F10_KA04_1' -> dicionário com condição, rolamento, medição etc."""
    m = _PADRAO_NOME.match(Path(nome).stem)
    if m is None:
        raise ValueError(f"Nome fora do padrão Paderborn: {nome}")
    n, mm, f, rolamento, medicao = m.groups()
    condicao = f"{n}_{mm}_{f}"
    rpm = int(n[1:]) * 100
    return {"arquivo": Path(nome).stem, "condicao": condicao, "rolamento": rolamento,
            "medicao": int(medicao), "rotacao_rpm": rpm,
            "torque_nm": int(mm[1:]) / 10, "forca_n": int(f[1:]) * 100}


def listar_arquivos(pasta_dados):
    """Todos os .mat abaixo de `pasta_dados` que seguem o padrão de nome."""
    arquivos = sorted(Path(pasta_dados).rglob("*.mat"))
    return [a for a in arquivos if _PADRAO_NOME.match(a.stem)]


# =============================================================================
# 3. FEATURES
# =============================================================================
def _amplitude_espectral(x, fs):
    """Espectro de amplitude unilateral com janela de Hann (resolução fs/N)."""
    janela = np.hanning(x.size)
    amp = 2.0 * np.abs(np.fft.rfft((x - x.mean()) * janela)) / janela.sum()
    return np.fft.rfftfreq(x.size, 1 / fs), amp


def _pico_perto(freqs, amp, f_alvo, tol_rel=0.02, tol_abs=1.0):
    """Maior amplitude numa janela ao redor de f_alvo. A tolerância cobre o
    escorregamento e a incerteza da geometria."""
    tol = max(tol_rel * f_alvo, tol_abs)
    i0, i1 = np.searchsorted(freqs, [f_alvo - tol, f_alvo + tol])
    return amp[i0:i1].max() if i1 > i0 else np.nan


def features_tempo(x):
    """Indicadores estatísticos no domínio do tempo (mesmas fórmulas de antes)."""
    abs_x = np.abs(x)
    rms = np.sqrt(np.mean(x**2))
    pico = abs_x.max()
    media_abs = abs_x.mean()
    hist, _ = np.histogram(x, bins=100)
    p = hist[hist > 0] / x.size
    return {
        "rms": rms,
        "desvio": x.std(),
        "pico": pico,
        "pico_a_pico": np.ptp(x),
        "fator_crista": pico / rms,
        "fator_folga": pico / np.mean(np.sqrt(abs_x)) ** 2,
        "fator_impulso": pico / media_abs,
        "fator_forma": rms / media_abs,
        "curtose": stats.kurtosis(x, fisher=False, bias=False),
        "assimetria": stats.skew(x, bias=False),
        "integral_linha": np.abs(np.diff(x)).sum(),
        "entropia": -np.sum(p * np.log2(p)),
    }


def features_frequencia(x, fs, freqs, amp):
    """FFT e PSD (Welch). Welch gera uma PSD menos ruidosa que o periodograma."""
    f_psd, psd = signal.welch(x, fs=fs, nperseg=8192, detrend="constant")
    potencia = np.trapezoid(psd, f_psd)
    i_pico = np.argmax(amp[1:]) + 1  # ignora a componente DC
    return {
        "fft_pico": amp[i_pico],
        "fft_freq_pico": freqs[i_pico],
        "fft_energia": np.sum(amp**2),
        "psd_integral": potencia,
        "centroide_espectral": np.trapezoid(f_psd * psd, f_psd) / potencia,
    }


def features_wpd(x, fs, nivel=3, wavelet="db4"):
    """Energia relativa dos 2**nivel nós da Wavelet Packet (artigo, seção 7.1).

    Sem PyWavelets, divide o espectro em 2**nivel bandas iguais. Essa é a
    partição ideal de frequência que a WPD aproxima.
    """
    n_bandas = 2**nivel
    if TEM_PYWT:
        wp = pywt.WaveletPacket(x, wavelet=wavelet, mode="periodization", maxlevel=nivel)
        energias = np.array([np.sum(no.data**2) for no in wp.get_level(nivel, order="freq")])
    else:
        potencia = np.abs(np.fft.rfft(x)) ** 2
        energias = np.array([b.sum() for b in np.array_split(potencia, n_bandas)])
    rel = energias / energias.sum()
    return {f"wpd_e{k}": rel[k] for k in range(n_bandas)}


def features_envelope(x, fs, fr, banda=(1_000, 15_000)):
    """Espectro de envelope (vibração). Amplitude nas frequências de defeito
    relativa ao nível de fundo, em dB. Isso torna o valor independente do
    ganho do sensor."""
    sos = signal.butter(4, banda, btype="bandpass", fs=fs, output="sos")
    env = np.abs(signal.hilbert(signal.sosfiltfilt(sos, x)))
    freqs, amp = _amplitude_espectral(env, fs)
    faixa = (freqs > 5) & (freqs < 10 * fr * FATORES_DEFEITO["bpfi"])
    fundo = np.median(amp[faixa])
    saida = {"env_fr": 20 * np.log10(_pico_perto(freqs, amp, fr) / fundo)}
    for defeito in ("bpfo", "bpfi"):
        f_def = fr * FATORES_DEFEITO[defeito]
        for h in (1, 2):
            saida[f"env_{defeito}_{h}x"] = 20 * np.log10(_pico_perto(freqs, amp, h * f_def) / fundo)
    return saida


def features_corrente(fs, fr, freqs, amp):
    """Corrente do motor (MCS): bandas laterais f1 ± f_defeito em relação à
    fundamental f1. f1 é estimada como o maior pico entre 10 e 400 Hz."""
    faixa = (freqs > 10) & (freqs < 400)
    i1 = np.flatnonzero(faixa)[np.argmax(amp[faixa])]
    f1, a1 = freqs[i1], amp[i1]
    saida = {"f1_hz": f1}
    for defeito in ("bpfo", "bpfi"):
        f_def = fr * FATORES_DEFEITO[defeito]
        laterais = [_pico_perto(freqs, amp, abs(f1 + s * f_def)) for s in (-1, 1)]
        saida[f"lat_{defeito}_db"] = 20 * np.log10(np.nanmean(laterais) / a1)
    harm = sum(_pico_perto(freqs, amp, k * f1) ** 2 for k in range(2, 8) if k * f1 < fs / 2)
    saida["thd"] = np.sqrt(harm) / a1
    return saida


def extrair_features_sinal(x, fs, fr, tipo):
    """Todas as features de um sinal. `tipo` é 'corrente' ou 'vibracao'."""
    freqs, amp = _amplitude_espectral(x, fs)  # calculada uma vez e reaproveitada
    feats = {**features_tempo(x), **features_frequencia(x, fs, freqs, amp), **features_wpd(x, fs)}
    if tipo == "vibracao":
        feats.update(features_envelope(x, fs, fr))
    else:
        feats.update(features_corrente(fs, fr, freqs, amp))
    return feats


# =============================================================================
# 4. EXTRAÇÃO EM LOTE
# =============================================================================
def processar_arquivo(caminho, n_segmentos=1, fs=FS):
    """Lê um arquivo e devolve uma lista de linhas (uma por segmento).

    Só as features voltam para o processo principal. O sinal bruto é descartado
    logo depois, o que mantém a memória baixa.
    """
    meta = interpretar_nome(caminho)
    fr = meta["rotacao_rpm"] / 60
    try:
        canais = ler_mat(caminho)
        linhas = [dict(meta, segmento=s) for s in range(n_segmentos)]
        for canal, prefixo in SINAIS.items():
            x = canais[canal]
            tipo = "vibracao" if prefixo == "vib" else "corrente"
            for s, parte in enumerate(np.array_split(x, n_segmentos)):
                for nome, valor in extrair_features_sinal(parte, fs, fr, tipo).items():
                    valor = float(valor)  # inf (ex.: log de 0) vira NaN e é imputado depois
                    linhas[s][f"{prefixo}__{nome}"] = valor if np.isfinite(valor) else np.nan
        return linhas
    except Exception as erro:  # um arquivo corrompido não derruba o lote
        return [dict(meta, segmento=-1, erro=repr(erro))]


def extrair_dataset(arquivos, n_segmentos=1, n_jobs=-1, progresso=True, verbose=0):
    """Extrai as features de todos os arquivos em paralelo (um processo por núcleo).

    Com progresso=True, mostra numa única linha quantos arquivos já foram
    processados, a porcentagem, o tempo decorrido e uma estimativa do tempo
    restante. Os resultados chegam conforme cada arquivo termina (gerador), sem
    esperar o lote inteiro.

    Devolve (df_features, df_erros).
    """
    total = len(arquivos)
    passo = max(1, total // 200)  # atualiza a linha a cada ~0,5 %
    t0 = time.perf_counter()
    tarefas = Parallel(n_jobs=n_jobs, verbose=verbose, return_as="generator_unordered")(
        delayed(processar_arquivo)(a, n_segmentos) for a in arquivos)
    resultados = []
    for k, linhas in enumerate(tarefas, start=1):
        resultados.append(linhas)
        if progresso and (k % passo == 0 or k == total):
            decorrido = time.perf_counter() - t0
            restante = decorrido / k * (total - k)
            print(f"\r{k}/{total} arquivos ({100 * k / total:5.1f} %) | "
                  f"{decorrido:6.0f} s decorridos | ~{restante:5.0f} s restantes",
                  end="", flush=True)
    if progresso:
        print()
    df = pd.DataFrame([linha for lista in resultados for linha in lista])
    # Os arquivos terminam fora de ordem; ordenar deixa o CSV reprodutível
    df = df.sort_values(["rolamento", "condicao", "medicao", "segmento"], kind="stable")
    if "erro" not in df:
        return df.reset_index(drop=True), pd.DataFrame(columns=["arquivo", "erro"])
    erros = df.loc[df["erro"].notna(), ["arquivo", "erro"]]
    df = df.loc[df["erro"].isna()].drop(columns="erro")
    return df.reset_index(drop=True), erros.reset_index(drop=True)


# =============================================================================
# 5. RÓTULOS
# =============================================================================
def rotular(rolamento, esquema="artigo"):
    """Classe de um rolamento.

    esquema='artigo'  : 0 saudável, 1 anel externo, 2 anel interno.
                        KB23 e KB24 -> interno, KB27 -> externo (dano dominante).
    esquema='4classes': igual, mas KB* vira a classe 3 (danos múltiplos).
    """
    if rolamento.startswith("K0"):
        return 0
    if rolamento.startswith("KA"):
        return 1
    if rolamento.startswith("KI"):
        return 2
    if rolamento.startswith("KB"):
        if esquema == "4classes":
            return 3
        return 1 if rolamento == "KB27" else 2
    raise ValueError(f"Rolamento desconhecido: {rolamento}")


def adicionar_rotulos(df, esquema="artigo"):
    """Acrescenta origem, classe, nome da classe e alvo binário."""
    df = df.copy()
    origem = np.select(
        [df["rolamento"].isin(SAUDAVEIS), df["rolamento"].isin(ARTIFICIAIS),
         df["rolamento"].isin(MULTIPLOS)],
        ["saudavel", "artificial", "real_multiplo"], default="real")
    df["origem"] = origem
    df["classe"] = df["rolamento"].map(lambda r: rotular(r, esquema)).astype(int)
    df["classe_nome"] = df["classe"].map(NOMES_CLASSES[esquema])
    df["danificado"] = (df["classe"] > 0).astype(int)
    return df


def colunas_features(df, sensores=("i1", "i2", "vib")):
    """Colunas de feature dos sensores escolhidos (pelo prefixo 'i1__' etc.)."""
    prefixos = tuple(f"{s}__" for s in sensores)
    return [c for c in df.columns if c.startswith(prefixos)]


# =============================================================================
# 6. SELEÇÃO DE FEATURES
# =============================================================================
class RemoverCorrelacionadas(SelectorMixin, BaseEstimator):
    """Remove features (quase) constantes e, de cada par com |r| >= limiar,
    descarta a que tem maior correlação média com as demais.

    Não usa o rótulo. Mesmo assim fica dentro do Pipeline, para que a decisão
    seja tomada só com os dados de treino de cada fold.
    """

    def __init__(self, limiar=0.95):
        self.limiar = limiar

    def fit(self, X, y=None):
        X = validate_data(self, X, ensure_all_finite="allow-nan")
        mascara = np.nanstd(X, axis=0) > 1e-12
        idx = np.flatnonzero(mascara)
        corr = np.abs(np.corrcoef(X[:, idx], rowvar=False))
        np.fill_diagonal(corr, 0.0)
        corr = np.nan_to_num(corr)
        vivos = np.ones(idx.size, dtype=bool)
        while True:
            sub = corr[np.ix_(vivos, vivos)]
            if sub.size == 0 or sub.max() < self.limiar:
                break
            i, j = np.unravel_index(np.argmax(sub), sub.shape)
            pos = np.flatnonzero(vivos)
            # descarta a feature mais redundante com o restante
            vivos[pos[i] if sub[i].mean() >= sub[j].mean() else pos[j]] = False
        mascara[idx[~vivos]] = False
        self.mascara_ = mascara
        return self

    def _get_support_mask(self):
        check_is_fitted(self)
        return self.mascara_


METODOS_SELECAO = ("artigo", "razao")


def normalizar_metodo(metodo):
    """Aceita variações de escrita ('razão', 'Razao', ' ARTIGO ') e devolve
    'artigo' ou 'razao'. Qualquer outro valor gera um erro que mostra o valor recebido."""
    import unicodedata
    texto = unicodedata.normalize("NFKD", str(metodo)).encode("ascii", "ignore").decode().strip().lower()
    if texto not in METODOS_SELECAO:
        raise ValueError(f"metodo_sel inválido: {metodo!r}. Use 'artigo' ou 'razao'.")
    return texto


class SelecaoSeparacao(SelectorMixin, BaseEstimator):
    """Seleção por distância de separação entre classes.

    metodo='artigo' : fórmula da seção 7.1. Normaliza para [0, 1], calcula
        d_j = 1/Nc² · Σ_k Σ_c mean_i (x_ijk − m_jc)², normaliza por max(d) e
        mantém d̄_j >= limiar (0,7 no artigo).
    metodo='razao'  : variante que divide a distância entre médias de classes
        pela variância dentro das classes (estilo Fisher). Na fórmula do
        artigo, uma variância interna grande também aumenta d_j, o que favorece
        features ruidosas. A razão corrige isso.

    `min_features` garante que pelo menos k features sobrevivam.
    """

    def __init__(self, limiar=0.7, metodo="artigo", min_features=3):
        self.limiar = limiar
        self.metodo = metodo
        self.min_features = min_features

    def fit(self, X, y):
        X, y = validate_data(self, X, y)
        mn, mx = X.min(axis=0), X.max(axis=0)
        Xn = (X - mn) / np.where(mx > mn, mx - mn, 1.0)
        classes = np.unique(y)
        medias = np.array([Xn[y == c].mean(axis=0) for c in classes])
        variancias = np.array([Xn[y == c].var(axis=0) for c in classes])
        nc = len(classes)
        # Σ_k Σ_c (m_k − m_c)²  (distância entre médias de classes)
        entre = ((medias[:, None, :] - medias[None, :, :]) ** 2).sum(axis=(0, 1)) / nc**2
        dentro = variancias.mean(axis=0)
        metodo = normalizar_metodo(self.metodo)
        if metodo == "artigo":
            # mean_i (x_ik − m_c)² = var_k + (m_k − m_c)²
            d = dentro + entre
        else:  # "razao"
            d = entre / (dentro + 1e-12)
        self.distancias_ = d / d.max() if d.max() > 0 else d
        mascara = self.distancias_ >= self.limiar
        if mascara.sum() < self.min_features:
            mascara[np.argsort(self.distancias_)[::-1][: self.min_features]] = True
        self.mascara_ = mascara
        return self

    def _get_support_mask(self):
        check_is_fitted(self)
        return self.mascara_


# =============================================================================
# 7. MODELOS
# =============================================================================
class ELMClassifier(ClassifierMixin, BaseEstimator):
    """Extreme Learning Machine: camada oculta aleatória e fixa, saída por
    mínimos quadrados regularizados (solução fechada, treino instantâneo)."""

    def __init__(self, n_ocultos=200, alpha=1e-2, random_state=None):
        self.n_ocultos = n_ocultos
        self.alpha = alpha
        self.random_state = random_state

    def _oculta(self, X):
        return np.tanh(X @ self.W_ + self.b_)

    def fit(self, X, y):
        X, y = validate_data(self, X, y)
        rng = np.random.default_rng(self.random_state)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        self.W_ = rng.normal(0, 1 / np.sqrt(X.shape[1]), (X.shape[1], self.n_ocultos))
        self.b_ = rng.normal(0, 1, self.n_ocultos)
        H = self._oculta(X)
        T = -np.ones((X.shape[0], self.classes_.size))
        T[np.arange(X.shape[0]), y_idx] = 1.0
        self.beta_ = np.linalg.solve(H.T @ H + self.alpha * np.eye(self.n_ocultos), H.T @ T)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return self.classes_[np.argmax(self._oculta(X) @ self.beta_, axis=1)]


def criar_pipeline(modelo, limiar_corr=0.95, limiar_sel=0.7, metodo_sel="artigo"):
    """imputação -> padronização -> remoção de correlacionadas -> seleção -> modelo."""
    metodo_sel = normalizar_metodo(metodo_sel)  # valida já aqui, antes de abrir os processos
    return Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("escala", StandardScaler()),
        ("corr", RemoverCorrelacionadas(limiar_corr)),
        ("sel", SelecaoSeparacao(limiar_sel, metodo_sel)),
        ("clf", modelo),
    ])


def criar_modelos(seed=42, **kw_pipeline):
    """Os 7 algoritmos do artigo, cada um dentro do mesmo pipeline.

    BT (boosted trees) -> HistGradientBoosting; SVM-PSO -> SVM com busca em
    grade (ver GRADES); NN -> MLP com uma camada oculta.
    """
    base = {
        "CART": DecisionTreeClassifier(random_state=seed),
        "RF": RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=seed),
        "BT": HistGradientBoostingClassifier(max_iter=200, random_state=seed),
        "NN": MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-3, max_iter=3000,
                            random_state=seed),
        "SVM": SVC(kernel="rbf", C=10, gamma="scale"),
        "ELM": ELMClassifier(n_ocultos=200, random_state=seed),
        "kNN": KNeighborsClassifier(n_neighbors=5),
    }
    return {nome: criar_pipeline(m, **kw_pipeline) for nome, m in base.items()}


# Hiperparâmetros buscados por validação interna agrupada por rolamento
# (substitui o PSO do artigo; aceita qualquer modelo do dicionário).
GRADES = {
    "SVM": {"clf__C": [1, 10, 100], "clf__gamma": ["scale", 0.01, 0.1]},
    "kNN": {"clf__n_neighbors": [3, 5, 9, 15]},
}


# =============================================================================
# 8. DIVISÕES TREINO/TESTE (sempre por rolamento)
# =============================================================================
def divisoes_artigo(df, cenario="B"):
    """Folds do artigo (seções 7.3 e 7.4).

    Cenário B: Tabela 10, 5 rolamentos por classe, 3 para treino e 2 para
        teste. As 10 combinações de 2 posições de teste se aplicam a todas as
        classes ao mesmo tempo.
    Cenário C: B mais KB23/KB24 (classe interno, um no treino e o outro no
        teste, alternando) e KB27 sempre no teste. Gera 20 folds.

    Devolve uma lista de (nome_fold, idx_treino, idx_teste) com índices de `df`.
    """
    rol = df["rolamento"].to_numpy()
    folds = []
    for pos_teste in combinations(range(5), 2):
        teste = [TABELA_10[c][p] for c in TABELA_10 for p in pos_teste]
        treino = [r for c in TABELA_10 for r in TABELA_10[c] if r not in teste]
        extras = [([], [])] if cenario == "B" else [(["KB23"], ["KB24", "KB27"]),
                                                    (["KB24"], ["KB23", "KB27"])]
        for tr_extra, te_extra in extras:
            tr, te = treino + tr_extra, teste + te_extra
            nome = "teste=" + ",".join(te)
            folds.append((nome, np.flatnonzero(np.isin(rol, tr)), np.flatnonzero(np.isin(rol, te))))
    return folds


def divisoes_grupo(df, n_splits=5, seed=42):
    """StratifiedGroupKFold: nenhum rolamento aparece no treino e no teste ao
    mesmo tempo, e cada fold tenta manter a proporção das classes."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(f"fold{k}", tr, te) for k, (tr, te) in
            enumerate(cv.split(df, df["classe"], groups=df["rolamento"]))]


# =============================================================================
# 9. AVALIAÇÃO
# =============================================================================
def _voto_majoritario(preds_por_modelo):
    """Votação dura. Em caso de empate, vence a classe votada primeiro na
    ordem dos modelos."""
    P = np.column_stack(list(preds_por_modelo.values()))
    saida = np.empty(P.shape[0], dtype=P.dtype)
    for i, linha in enumerate(P):
        valores, contagem = np.unique(linha, return_counts=True)
        empatados = set(valores[contagem == contagem.max()])
        saida[i] = next(v for v in linha if v in empatados)
    return saida


def _avaliar_fold(df, colunas, alvo, nome_fold, tr, te, modelos, grades, ensemble):
    X_tr, X_te = df.iloc[tr][colunas], df.iloc[te][colunas]
    y_tr = df.iloc[tr][alvo].to_numpy()
    g_tr = df.iloc[tr]["rolamento"].to_numpy()
    preds, selecionadas = {}, None
    for nome, modelo in modelos.items():
        estimador = clone(modelo)
        grade = (grades or {}).get(nome)
        if grade:
            n_grupos = min(3, len(np.unique(g_tr)))
            busca = GridSearchCV(estimador, grade, cv=GroupKFold(n_splits=n_grupos),
                                 scoring="balanced_accuracy", n_jobs=1)
            estimador = busca.fit(X_tr, y_tr, groups=g_tr).best_estimator_
        else:
            estimador.fit(X_tr, y_tr)
        preds[nome] = estimador.predict(X_te)
        if selecionadas is None and isinstance(estimador, Pipeline):
            selecionadas = list(estimador[:-1].get_feature_names_out())
    if ensemble and len(preds) > 1:
        preds["Ensemble"] = _voto_majoritario(preds)
    base = df.iloc[te][["arquivo", "rolamento", "condicao", alvo]].rename(columns={alvo: "y_true"})
    tabelas = [base.assign(modelo=n, y_pred=p, fold=nome_fold) for n, p in preds.items()]
    return pd.concat(tabelas, ignore_index=True), {"fold": nome_fold, "features": selecionadas}


def avaliar(df, colunas, divisoes, modelos, alvo="classe", grades=None, ensemble=True,
            n_jobs=-1, verbose=0):
    """Treina e testa todos os modelos em todos os folds (folds em paralelo).

    Devolve (predicoes, selecoes): uma linha por amostra de teste × modelo × fold
    e, para cada fold, a lista de features que passaram pela seleção.
    """
    saidas = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_avaliar_fold)(df, colunas, alvo, nome, tr, te, modelos, grades, ensemble)
        for nome, tr, te in divisoes)
    preds = pd.concat([p for p, _ in saidas], ignore_index=True)
    return preds, [s for _, s in saidas]


def resumo(preds):
    """Métricas por fold e depois média ± desvio entre folds, por modelo."""
    def metricas(g):
        return pd.Series({
            "acuracia": accuracy_score(g["y_true"], g["y_pred"]),
            "acuracia_bal": balanced_accuracy_score(g["y_true"], g["y_pred"]),
            "f1_macro": f1_score(g["y_true"], g["y_pred"], average="macro"),
        })
    por_fold = preds.groupby(["modelo", "fold"])[["y_true", "y_pred"]].apply(metricas)
    tabela = por_fold.groupby("modelo").agg(["mean", "std"])
    return tabela.sort_values(("acuracia", "mean"), ascending=False)


def matriz_confusao(preds, modelo, rotulos=None, normalizar=None):
    """Matriz de confusão somada sobre todos os folds (linhas = classe real)."""
    p = preds[preds["modelo"] == modelo]
    rotulos = sorted(p["y_true"].unique()) if rotulos is None else rotulos
    return confusion_matrix(p["y_true"], p["y_pred"], labels=rotulos, normalize=normalizar)


def acerto_por(preds, modelo, chave="rolamento"):
    """Taxa de acerto por rolamento (ou por condição). Mostra onde o modelo erra."""
    p = preds[preds["modelo"] == modelo]
    return (p.assign(acerto=p["y_true"] == p["y_pred"])
             .groupby(chave)["acerto"].agg(["mean", "size"])
             .rename(columns={"mean": "taxa_acerto", "size": "n"})
             .sort_values("taxa_acerto"))


def frequencia_selecao(selecoes):
    """Em quantos folds cada feature foi selecionada (estabilidade da seleção)."""
    todas = pd.Series([f for s in selecoes if s["features"] for f in s["features"]])
    return (todas.value_counts() / len(selecoes)).rename("fracao_folds")


# =============================================================================
# 10. TEXTO AUTOMÁTICO SOBRE OS ERROS
# =============================================================================
# Nível do dano (Tabela 5 do artigo) e amaciamento dos saudáveis (Tabela 7)
NIVEL_DANO = {"KA04": 1, "KA15": 1, "KA16": 2, "KA22": 1, "KA30": 1,
              "KB23": 2, "KB24": 3, "KB27": 1,
              "KI04": 1, "KI14": 1, "KI16": 3, "KI17": 1, "KI18": 2, "KI21": 1}
AMACIAMENTO = {"K001": ">50 h", "K002": "19 h", "K003": "1 h",
               "K004": "5 h", "K005": "10 h", "K006": "16 h"}
# Rolamentos com mais erros relatados no artigo (seção 7.3, 3 classes)
ERROS_ARTIGO = {"MCS": ["KA04", "KA22"], "Vibracao": ["KA04", "KI21"]}


_ROTULO_TEXTO = {"saudavel": "saudável", "Vibracao": "Vibração", "Fusao": "Fusão"}


def _txt(nome):
    """Nome interno -> texto para leitura ('anel_externo' -> 'anel externo', 'saudavel' -> 'saudável')."""
    return _ROTULO_TEXTO.get(nome, nome).replace("_", " ")


def _pct(v):
    return f"{100 * v:.0f} %"


def descrever_rolamento(rol, esquema="artigo"):
    """'KA15' -> 'KA15 (anel externo, nível 1)'; 'K002' -> 'K002 (saudável, 19 h de amaciamento)'."""
    if rol in AMACIAMENTO:
        return f"{rol} (saudável, {AMACIAMENTO[rol]} de amaciamento)"
    classe = _txt(NOMES_CLASSES[esquema][rotular(rol, esquema)])
    return f"{rol} ({classe}, nível {NIVEL_DANO.get(rol, '?')})"


def texto_erros(preds, sensor, esquema="artigo", nomes=None, modelo="Ensemble", limiar=0.6):
    """Parágrafo sobre onde o modelo erra num sensor: a confusão mais frequente,
    os rolamentos com acerto abaixo de `limiar` e, no esquema de 3 classes do
    artigo, a comparação com os rolamentos difíceis relatados no artigo.

    `nomes` = {rótulo: nome da classe}. O padrão é o do esquema. No binário use
    {0: "sem_defeito", 1: "com_defeito"}.
    """
    nomes = NOMES_CLASSES[esquema] if nomes is None else nomes
    rotulos = list(nomes)
    cm = matriz_confusao(preds, modelo, rotulos=rotulos)
    fora = cm.copy()
    np.fill_diagonal(fora, 0)
    i, j = np.unravel_index(fora.argmax(), fora.shape)
    real, prev = (_txt(nomes[rotulos[k]]) for k in (i, j))
    linhas = [f"**{_txt(sensor)}:** a confusão mais frequente é **{real} prevista como {prev}** "
              f"({fora[i, j]} amostras, {100 * fora[i, j] / cm[i].sum():.0f} % da classe {real})."]

    acerto = acerto_por(preds, modelo)
    dificeis = acerto[acerto["taxa_acerto"] < limiar]
    if dificeis.empty:
        linhas.append(f"Todos os rolamentos tiveram acerto de pelo menos {_pct(limiar)}.")
    else:
        lista = "; ".join(f"{descrever_rolamento(r, esquema)}: {_pct(v)}"
                          for r, v in dificeis["taxa_acerto"].items())
        n = len(dificeis)
        alvo = "1 rolamento com acerto" if n == 1 else f"{n} rolamentos com acerto"
        linhas.append(f"Os erros se concentram em {alvo} abaixo de {_pct(limiar)}: {lista}.")

    if esquema == "artigo" and len(nomes) == 3 and sensor in ERROS_ARTIGO:
        ref = ERROS_ARTIGO[sensor]
        comuns = sorted(set(ref) & set(dificeis.index))
        linhas.append(f"No artigo, os erros deste sensor se concentraram em {' e '.join(ref)}, "
                      "ambos de nível 1. " + (f"Em comum com este trabalho: {', '.join(comuns)}."
                      if comuns else "Nenhum deles aparece entre os mais difíceis aqui."))
    return " ".join(linhas)


def relatorio_erros(preds_por_sensor, esquema="artigo", nomes=None, modelo="Ensemble",
                    limiar=0.6, titulo="Onde o ensemble erra"):
    """Texto em Markdown para um experimento inteiro, um item por sensor.

    Uso no notebook:
        from IPython.display import Markdown, display
        display(Markdown(R.relatorio_erros(preds_B, esquema)))
    """
    nomes = NOMES_CLASSES[esquema] if nomes is None else nomes
    texto = (f"### {titulo}\n"
             "A matriz de confusão mostra **quais classes** se confundem. A taxa de acerto por "
             "rolamento mostra **quais rolamentos** causam os erros.\n\n"
             f"Esquema de rótulos: **{esquema}** ({len(nomes)} classes: "
             f"{', '.join(_txt(n) for n in nomes.values())}).")
    if esquema != "artigo":
        texto += (" O artigo usa apenas 3 classes, então não há referência direta "
                  "para comparar os erros por rolamento.")
    itens = [f"- {texto_erros(p, s, esquema, nomes, modelo, limiar)}"
             for s, p in preds_por_sensor.items()]
    return texto + "\n\n" + "\n\n".join(itens)


# =============================================================================
# Utilidade para os gráficos do notebook 02
# =============================================================================
def espectro_envelope(x, fs=FS, banda=(1_000, 15_000)):
    """Espectro de envelope completo (para plotar)."""
    sos = signal.butter(4, banda, btype="bandpass", fs=fs, output="sos")
    env = np.abs(signal.hilbert(signal.sosfiltfilt(sos, x)))
    return _amplitude_espectral(env, fs)

