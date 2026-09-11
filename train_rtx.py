"""
===============================================================================
 TRENING DEKODERA CW — PLIK SAMODZIELNY, do uruchomienia na maszynie z GPU
===============================================================================

Jeden plik, bez importów z resztą projektu. Kopiujesz go na maszynę z RTX 3050
i uruchamiasz. Nie wymaga żadnych plików obok siebie — dane potrafi wygenerować
sam.

ZALEŻNOŚCI: tylko tensorflow i numpy.
Front-end (mel, STFT, skala dB) jest policzony na samym numpy — nie ma tu
librosy, bo nie ma jej ani w środowisku treningowym z TF 2.10, ani na
docelowym KV260.

Sprawdzone na dwóch zestawach:
    TF 2.10.0 + keras 2.10.0 + numpy 1.21.6   (natywne GPU na Windows)
    TF 2.21.0 + keras 3.15   + numpy 2.5      (CPU)
Format zapisu modelu dobiera się sam: .h5 dla Kerasa 2, .keras dla 3.

    # 1. zbiór treningowy (raz; 200 tys. próbek to ok. 790 MB i kilka minut
    #    na wielu rdzeniach)
    python train_rtx.py generate --n 200000

    # 2. trening; można przerywać i wznawiać dowolną liczbę razy
    python train_rtx.py train --epochs 200 --batch 256 --mixed

    # 3. po przerwie: TO SAMO polecenie wznawia od ostatniej epoki
    python train_rtx.py train --epochs 200 --batch 256 --mixed

WZNAWIANIE — o to tu chodzi
---------------------------
Po każdej epoce zapisywane są cztery rzeczy:

    runs/<nazwa>/last.h5        model po ostatniej ukończonej epoce
    runs/<nazwa>/best.h5        model o najlepszej dokładności walidacji
    runs/<nazwa>/state.json     numer epoki, najlepszy wynik, historia
    runs/<nazwa>/log.csv        metryki wszystkich epok, dopisywane

Ponowne uruchomienie `train` wczytuje last.keras i state.json i kontynuuje od
następnej epoki. Wyłączenie komputera w środku epoki kosztuje najwyżej tę
jedną epokę. Nic nie trzeba podawać ręcznie — wznowienie jest domyślne.
Świeży start: --fresh.

ZGODNOŚĆ Z RESZTĄ PROJEKTU
--------------------------
Stałe front-endu poniżej są kopią dsp/config.py. Kopia jest tu świadomie, bo
plik ma być samodzielny — a przed rozjechaniem się chroni odcisk FINGERPRINT:
diag.py w projekcie porównuje go z config.py i zgłasza błąd przy różnicy.
Jeśli zmieniasz cokolwiek w sekcji KONFIGURACJA, zmień to w obu miejscach
i wygeneruj zbiór od nowa.
===============================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Poziom 1 = ukryj tylko INFO. NIE ustawiać 2 — poziom 2 ukrywa WARNING,
# a to właśnie na tym poziomie TensorFlow zgłasza brakujące biblioteki CUDA:
#   "Could not load dynamic library 'cudart64_110.dll'"
# Z poziomem 2 trening po cichu zjeżdża na CPU i nie wiadomo dlaczego.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")


# =============================================================================
# KONFIGURACJA — kopia dsp/config.py, pilnowana odciskiem FINGERPRINT
# =============================================================================

# --- sygnał ---
SR = 8000
CLIP_SECONDS = 4.0
CLIP_SAMPLES = int(SR * CLIP_SECONDS)

# --- front-end (kontrakt modelu; zmiana unieważnia wszystkie wagi) ---
N_FFT = 512
HOP_LENGTH = 160
N_MELS = 32
FMIN = 400.0
FMAX = 1200.0
IMG_FRAMES = 128
IMG_BINS = N_MELS
INPUT_SHAPE = (IMG_FRAMES, IMG_BINS, 1)

# Skala dB — WARTOŚCI ZMIERZONE narzędziem tools/calibrate.py, nie zgadnięte.
#   szum 0.015 RMS: mediana -26.7 dB, 95 pct -21.5 dB
#   sygnał amp 0.3: szczyt +17 dB, najwyższy zmierzony +22.6 dB
DB_REF = 1.0
DB_MIN = -30.0
DB_MAX = 24.0

# --- alfabet: indeks = klasa; 0 = "puste radio" ---
ALPHABET = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
N_CLASSES = len(ALPHABET)
CHAR_TO_ID = {c: i for i, c in enumerate(ALPHABET)}
ID_TO_CHAR = {i: c for i, c in enumerate(ALPHABET)}

MORSE_DICT = {
    "A": ".-",    "B": "-...",  "C": "-.-.",  "D": "-..",   "E": ".",
    "F": "..-.",  "G": "--.",   "H": "....",  "I": "..",    "J": ".---",
    "K": "-.-",   "L": ".-..",  "M": "--",    "N": "-.",    "O": "---",
    "P": ".--.",  "Q": "--.-",  "R": ".-.",   "S": "...",   "T": "-",
    "U": "..-",   "V": "...-",  "W": ".--",   "X": "-..-",  "Y": "-.--",
    "Z": "--..",
    "0": "-----", "1": ".----", "2": "..---", "3": "...--", "4": "....-",
    "5": ".....", "6": "-....", "7": "--...", "8": "---..", "9": "----.",
}

# --- timing CW ---
WPM = 20.0
# 7 WPM rozrzutu = 13-27 WPM. Przy WPM_JITTER = 0 zbiór miał dokładnie jedno
# tempo i model gubił znaki krótkie na pierwszym nagraniu z pasma, które
# szło 15 WPM (kropka 80 ms zamiast 60 ms). Dolna granica wynika z okna
# sieci: poniżej 11,7 WPM najdłuższy znak z przerwami się w nim nie mieści.
WPM_JITTER = 7.0
DASH_UNITS = 3
GAP_ELEMENT_UNITS = 1
GAP_CHAR_UNITS = 3
GAP_WORD_UNITS = 7
KEY_RAMP_MS = 5.0

# --- generator zbioru ---
SILENCE_FRACTION = 0.15
CHARS_PER_CLIP = 3
LABEL_INDEX = 1
NOISE_RMS = 0.015
SIGNAL_AMP_MIN = 0.03
SIGNAL_AMP_MAX = 0.50
TONE_CENTER = 750.0
TONE_SPREAD = 80.0
SEED = 12345
STORE_DTYPE = "uint8"

# --- model kanału radiowego ---
NOISE_TILT = (-1.0, 0.0)
NOISE_RMS_SPREAD = (0.5, 2.0)
QSB_DEPTH = (0.0, 0.6)
QRM_PROB = 0.35
QRM_AMP = (0.05, 0.35)
QRN_PROB = 0.30
QRN_MAX = 6
QRN_AMP = (0.1, 0.6)
# Timing ma DWA źródła o różnym charakterze:
#   FIST        szum na pojedynczym elemencie (drżenie ręki)
#   FIST_DRIFT  WOLNE błądzenie tempa przez nadanie — "jak łapa boli".
#               Proces SKORELOWANY; modelowanie go jako szumu niezależnego
#               dawało sygnał nerwowy o stałym tempie średnim, czego na
#               kluczu ręcznym nie ma.
#   GAP_JITTER  rozjazd PRZERW międzyznakowych, osobno od elementów —
#               przy manipulatorze elementy idą równo, a przerwy nie.
FIST = (0.0, 0.25)
FIST_DRIFT = (0.0, 0.15)
FIST_DRIFT_TAU_S = 2.0
GAP_JITTER = (0.0, 0.40)

# Zapadanie zasilania nadajnika: naduszenie klucza rozładowuje kondensator,
# napięcie spada, generator LC się przestraja. Ton spada W TRAKCIE elementu
# (750 Hz na początku kreski, 730 Hz na końcu) i wraca w przerwie; z tego
# SAMEGO mechanizmu spada amplituda. To opisuje trzecia litera RST (Tone).
# NIE to samo co dryf VFO, który jest wolny i niezależny od klucza.
CHIRP_HZ = (0.0, 25.0)
SAG_DB = (0.0, 3.0)
SAG_TAU_MS = (20.0, 120.0)

# Przydźwięk sieci na obwiedni nośnej — puste kondensatory w zasilaczu.
HUM_PROB = 0.20
HUM_DEPTH = (0.05, 0.35)
HUM_HZ = (50.0, 100.0)

DRIFT_HZ = 8.0                  # wolny dryf VFO, niezależny od klucza

# ARW odbiornika: trzecie źródło bujania amplitudy, o innej skali czasu
# niż zanik (sekundy) i zapadanie zasilacza (dziesiątki ms).
AGC_PROB = 0.5
AGC_TAU_MS = (50.0, 500.0)
AGC_DEPTH = (0.2, 0.8)

# Odcisk front-endu. Musi być identyczny z config.fingerprint_str().
# Kolejność pól: alfabetyczna po nazwie klucza.
FINGERPRINT = (
    f"ALPHABET={ALPHABET};CLIP_SECONDS={CLIP_SECONDS};DB_MAX={DB_MAX};"
    f"DB_MIN={DB_MIN};DB_REF={DB_REF};FMAX={FMAX};FMIN={FMIN};"
    f"HOP_LENGTH={HOP_LENGTH};IMG_BINS={IMG_BINS};IMG_FRAMES={IMG_FRAMES};"
    f"MEL_CONVENTION=librosa/slaney;N_FFT={N_FFT};N_MELS={N_MELS};SR={SR}"
)


# =============================================================================
# ZGODNOŚĆ ZE STARSZYM TENSORFLOW
#
# Ten plik ma działać zarówno na TF 2.10 + keras 2.10 (ostatnia wersja
# z natywnym GPU na Windows, i ta która współpracuje z Vitis AI), jak i na
# TF 2.16+ z Kerasem 3. Trzy rzeczy się między nimi różnią:
#
#   1. `import keras` w TF 2.10 daje keras 2.10, ale kanoniczną drogą jest
#      tf.keras. W TF 2.16+ tf.keras JEST Kerasem 3. Używamy tf.keras
#      wszędzie, bo działa w obu.
#   2. Format ".keras" istnieje dopiero w Kerasie 3 (i częściowo 2.13+).
#      Na keras 2.10 zapis pod taką nazwą albo zawiedzie, albo cicho
#      zapisze katalog SavedModel. Dlatego rozszerzenie dobiera się
#      do wersji: ".h5" dla Kerasa 2, ".keras" dla 3.
#   3. Starsze wersje nie przyjmują obiektów Path w niektórych miejscach —
#      wszędzie podajemy str().
# =============================================================================
def keras_api():
    """Moduł Keras właściwy dla zainstalowanego TensorFlow."""
    import tensorflow as tf
    return tf.keras


def keras_major() -> int:
    import tensorflow as tf
    return int(tf.keras.__version__.split(".")[0])


def model_ext() -> str:
    """Rozszerzenie pliku modelu obsługiwane przez zainstalowany Keras."""
    return ".keras" if keras_major() >= 3 else ".h5"


def find_model(run_dir: Path, stem: str):
    """Szuka zapisanego modelu w obu formatach — żeby katalog stanu
    zrobiony na jednej wersji Kerasa dał się wznowić na drugiej."""
    for ext in (model_ext(), ".keras", ".h5"):
        p = run_dir / (stem + ext)
        if p.exists():
            return p
    return None


# Biblioteki, których szuka wheel TensorFlow 2.10 na Windows. Gdy którejś
# nie ma na PATH, TF zgłasza to jako WARNING i po cichu zjeżdża na CPU.
_CUDA_DLL_TF210 = ("cudart64_110.dll", "cublas64_11.dll",
                   "cublasLt64_11.dll", "cufft64_10.dll",
                   "curand64_10.dll", "cusolver64_11.dll",
                   "cusparse64_11.dll", "cudnn64_8.dll")


def gpu_diagnosis() -> tuple[bool, str]:
    """(czy_widzi_GPU, wyjaśnienie). Odpowiada na pytanie DLACZEGO nie widzi.

    Samo "GPU: BRAK" jest bezużyteczne — trzeba wiedzieć, czy wheel wogóle
    jest zbudowany z CUDA, jakiej wersji oczekuje i która biblioteka się nie
    wczytała. Bez tego zostaje zgadywanie, a trening po cichu idzie na CPU.
    """
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        names = ", ".join(g.name.replace("/physical_device:", "")
                          for g in gpus)
        return True, f"GPU WIDZIANE: {names}"

    out = ["GPU NIE JEST WIDZIANE — trening pójdzie na CPU"]

    built = False
    try:
        built = bool(tf.test.is_built_with_cuda())
    except Exception:
        pass
    out.append(f"  wheel zbudowany z CUDA: {built}")

    info = {}
    try:
        info = dict(tf.sysconfig.get_build_info())
    except Exception:
        pass
    if info.get("cuda_version") or info.get("cudnn_version"):
        out.append(f"  ten wheel oczekuje CUDA {info.get('cuda_version')} "
                   f"i cuDNN {info.get('cudnn_version')}")

    major_minor = tuple(int(x) for x in tf.__version__.split(".")[:2])

    # --- Linux / WSL2 ---
    # Tu najczęstsza przyczyna jest inna niż na Windows: zwykły
    # `pip install tensorflow` NIE ciągnie bibliotek CUDA. Trzeba wariantu
    # [and-cuda], który dokłada koła nvidia-*. Sterownika linuksowego
    # w WSL nie instaluje się wcale — sterownik z Windows wystawia się
    # jako libcuda.so.
    if sys.platform.startswith("linux"):
        try:
            import importlib.util as _u
            has_wheels = _u.find_spec("nvidia") is not None
        except Exception:
            has_wheels = False
        in_wsl = "microsoft" in Path("/proc/version").read_text().lower() \
            if Path("/proc/version").exists() else False
        out.append(f"  system: Linux{' (WSL)' if in_wsl else ''}, "
                   f"koła CUDA z pipa: {has_wheels}")
        if not has_wheels:
            out.append("  -> BRAKUJE bibliotek CUDA z pipa. Zwykły "
                       "`pip install tensorflow` ich nie ciągnie:")
            out.append("       pip install 'tensorflow[and-cuda]'")
            out.append("     CUDA Toolkit z instalatora NVIDII nie jest do "
                       "tego potrzebny —\n     wystarczy sterownik na "
                       "Windows. Toolkit służy do KOMPILOWANIA.")
        else:
            out.append("  -> koła CUDA są, więc sprawdź sterownik na "
                       "Windows: nvidia-smi\n     wewnątrz WSL musi "
                       "pokazywać kartę. W WSL NIE instaluje się\n"
                       "     sterownika linuksowego.")
        return False, "\n".join(out)

    # Kolejność ma znaczenie: na Windows z TF >= 2.11 przyczyna jest
    # jednoznaczna i trzeba ją podać PRZED ogólnym "wheel bez CUDA",
    # bo te wheele są bez CUDA właśnie dlatego.
    if major_minor >= (2, 11) and sys.platform == "win32":
        out.append("  -> TensorFlow >= 2.11 NIE OBSŁUGUJE GPU na Windows "
                   "natywnie, nawet\n     z zainstalowaną CUDA. Potrzebny "
                   "WSL2 albo TF 2.10.")
    elif not built:
        out.append("  -> zainstalowany TensorFlow to wersja BEZ CUDA. "
                   "Sprawdź, czy nie masz\n     dwóch instalacji Pythona "
                   "i czy uruchamiasz tę właściwą:")
        out.append(f"     {sys.executable}")
    else:
        # Najczęstszy przypadek: wheel z CUDA, ale bibliotek nie ma na PATH.
        missing = [d for d in _CUDA_DLL_TF210 if not _find_dll(d)]
        if missing:
            out.append("  -> nie znaleziono na PATH bibliotek: "
                       + ", ".join(missing))
            out.append("     Dla TF 2.10 potrzebne są CUDA 11.2 i cuDNN 8.1, "
                       "a katalogi\n     bin/ obu muszą być w PATH.")
        else:
            out.append("  -> biblioteki CUDA są na PATH, więc problem jest "
                       "w sterowniku\n     albo w wersji cuDNN. Uruchom "
                       "z TF_CPP_MIN_LOG_LEVEL=0, żeby zobaczyć,\n     która "
                       "biblioteka odmawia wczytania.")
        out.append("     Sprawdź też: nvidia-smi (czy sterownik żyje)")
    return False, "\n".join(out)


def _find_dll(name: str) -> bool:
    """Czy biblioteka jest osiągalna przez PATH."""
    import shutil
    if shutil.which(name):
        return True
    for d in os.environ.get("PATH", "").split(os.pathsep):
        if d and Path(d, name).exists():
            return True
    return False


def env_report() -> tuple[bool, str]:
    """(czy_GPU, raport). Wypisywane na starcie treningu."""
    import tensorflow as tf
    has_gpu, gpu_txt = gpu_diagnosis()
    lines = [f"TensorFlow {tf.__version__}   Keras {tf.keras.__version__}   "
             f"numpy {np.__version__}   Python {sys.version.split()[0]}",
             f"interpreter: {sys.executable}",
             f"format zapisu modelu: {model_ext()}",
             gpu_txt]
    return has_gpu, "\n".join(lines)


def dot_seconds(wpm: float = WPM) -> float:
    """Kropka w sekundach; słowo wzorcowe PARIS = 50 jednostek."""
    return 1.2 / float(wpm)


def db_span() -> float:
    return DB_MAX - DB_MIN


def summary() -> str:
    return (f"SR={SR} Hz  klip={CLIP_SECONDS}s  "
            f"mel: n_fft={N_FFT} hop={HOP_LENGTH} n_mels={N_MELS} "
            f"pasmo={FMIN:.0f}-{FMAX:.0f} Hz\n"
            f"obraz={IMG_FRAMES}x{IMG_BINS} "
            f"({IMG_FRAMES*HOP_LENGTH/SR:.2f}s)  "
            f"dB: ref={DB_REF} zakres=[{DB_MIN}, {DB_MAX}]\n"
            f"morse: WPM={WPM} kropka={dot_seconds()*1000:.0f}ms  "
            f"klas={N_CLASSES}")


# =============================================================================
# MORSE — kodowanie i kluczowanie
# =============================================================================
def text_to_units(text: str, with_tags: bool = False):
    """Tekst -> [(czy_ton, długość_w_jednostkach), ...]. Bez przerwy końcowej.

    with_tags=True dokłada numer znaku, do którego należy element (przerwy
    międzyznakowe: -1). Po to, żeby dało się wskazać DOKŁADNE granice znaku
    z etykiety — bez tego nie da się sprawdzić wzrokiem, który z trzech
    nadanych znaków opisuje etykieta.
    """
    seq: list[tuple] = []
    words = [w for w in text.upper().split() if w]
    char_no = 0
    for wi, word in enumerate(words):
        chars = [ch for ch in word if ch in MORSE_DICT]
        for ci, ch in enumerate(chars):
            code = MORSE_DICT[ch]
            for si, sym in enumerate(code):
                seq.append((True, float(DASH_UNITS if sym == "-" else 1),
                            char_no))
                if si < len(code) - 1:
                    seq.append((False, float(GAP_ELEMENT_UNITS), char_no))
            char_no += 1
            if ci < len(chars) - 1:
                seq.append((False, float(GAP_CHAR_UNITS), -1))
        if wi < len(words) - 1:
            seq.append((False, float(GAP_WORD_UNITS), -1))
    return seq if with_tags else [(m, n) for m, n, _ in seq]


def _raised_cosine(n: int) -> np.ndarray:
    """Narastanie 0->1 na n próbkach, z zerową pochodną na obu końcach."""
    t = np.arange(1, n + 1, dtype=np.float64) / (n + 1)
    return (0.5 * (1.0 - np.cos(np.pi * t))).astype(np.float32)


def keying_envelope(text: str, wpm: float = WPM, sr: int = SR,
                    ramp_ms: float = KEY_RAMP_MS, fist: float = 0.0,
                    drift: float = 0.0, gap_jitter: float = 0.0,
                    rng: np.random.Generator | None = None,
                    return_spans: bool = False):
    """Obwiednia kluczowania w [0, 1].

    Granice elementów liczone są od zera (round(pozycja * dot)), więc błąd
    zaokrąglenia nie kumuluje się po długim nadaniu.

    fist — rozjazd ręcznego klucza, multiplikatywny na każdym elemencie
    osobno. To on odpowiada za większość trudności w dekodowaniu.

    return_spans=True zwraca (env, spans) z granicami każdego znaku
    w próbkach, PO uwzględnieniu rozjazdu.
    """
    seq = text_to_units(text, with_tags=True)
    if not seq:
        empty = np.zeros(0, dtype=np.float32)
        return (empty, []) if return_spans else empty

    if fist > 0.0 or drift > 0.0 or gap_jitter > 0.0:
        r = rng if rng is not None else np.random.default_rng()
        n_el = len(seq)

        # WOLNE błądzenie tempa: proces skorelowany, nie szum niezależny.
        # Kolejne elementy mają PODOBNE odchylenie — operator zwalnia
        # i przyspiesza w skali sekund.
        if drift > 0.0 and n_el > 1:
            el_s = dot_seconds(wpm) * 2.0
            alpha = float(np.exp(-el_s / max(0.2, FIST_DRIFT_TAU_S)))
            w = r.standard_normal(n_el)
            slow = np.empty(n_el)
            slow[0] = w[0]
            for i in range(1, n_el):
                slow[i] = alpha * slow[i - 1] + np.sqrt(1 - alpha ** 2) * w[i]
            slow *= drift / 2.0
        else:
            slow = np.zeros(n_el)

        out = []
        for i, (m, n, tag) in enumerate(seq):
            scale = 1.0 + slow[i]
            if fist > 0.0:
                scale *= 1.0 + r.uniform(-fist, fist)
            if gap_jitter > 0.0 and not m and tag < 0:
                scale *= 1.0 + r.uniform(-gap_jitter, gap_jitter)
            out.append((m, max(0.35, n * scale), tag))
        seq = out

    dot = dot_seconds(wpm) * sr
    total = sum(n for _, n, _ in seq)
    env = np.zeros(int(round(total * dot)), dtype=np.float32)
    ramp_len = int(round(ramp_ms * 1e-3 * sr))

    bounds: dict[int, list[int]] = {}
    pos = 0.0
    for is_mark, units, tag in seq:
        a, b = int(round(pos * dot)), int(round((pos + units) * dot))
        pos += units
        # Granice znaku z ELEMENTÓW, nie z przerw: znak zaczyna się
        # pierwszym tonem i kończy ostatnim.
        if is_mark and tag >= 0:
            if tag in bounds:
                bounds[tag][1] = b
            else:
                bounds[tag] = [a, b]
        if not is_mark or b <= a:
            continue
        n = b - a
        block = np.ones(n, dtype=np.float32)
        r_len = min(ramp_len, n // 2)
        if r_len > 0:
            ramp = _raised_cosine(r_len)
            block[:r_len] = ramp
            block[-r_len:] = ramp[::-1]
        env[a:b] = block

    if not return_spans:
        return env
    return env, [(t, ab[0], ab[1]) for t, ab in sorted(bounds.items())]


def psu_sag(env: np.ndarray, sr: int = SR,
            tau_ms: float = 60.0) -> np.ndarray:
    """Stan zapadania zasilania nadajnika, [0, 1].

    Podąża za obwiednią klucza z opóźnieniem (filtr pierwszego rzędu — takie
    jest rozładowanie i doładowanie pojemności). Dla DŁUGIEJ kreski dochodzi
    blisko jedynki, więc odchylenie jest największe na jej końcu. Dla serii
    kresek nie wraca do zera w przerwach i zapadanie się kumuluje.
    """
    from scipy.signal import lfilter
    a = float(np.exp(-1.0 / (max(1e-3, tau_ms) * 1e-3 * sr)))
    return lfilter([1.0 - a], [1.0, -a], np.asarray(env, dtype=np.float64))


def synth_cw(text: str, wpm: float = WPM, tone: float = TONE_CENTER,
             amp: float = 0.3, sr: int = SR, ramp_ms: float = KEY_RAMP_MS,
             phase: float = 0.0, fist: float = 0.0,
             fist_drift: float = 0.0, gap_jitter: float = 0.0,
             drift_hz: float = 0.0, chirp_hz: float = 0.0,
             sag_db: float = 0.0, sag_tau_ms: float = 60.0,
             hum_depth: float = 0.0, hum_hz: float = 100.0,
             rng: np.random.Generator | None = None,
             return_spans: bool = False):
    """Tekst -> nadanie CW. Bez szumu.

    DWA RÓŻNE ZJAWISKA CZĘSTOTLIWOŚCIOWE:
      drift_hz — WOLNY dryf VFO, liniowy, niezależny od klucza
      chirp_hz — SZYBKI spad tonu W TRAKCIE elementu, skorelowany z kluczem
                 (naduszenie 750 Hz, koniec kreski 730 Hz). Zapadanie
                 zasilacza przestraja LC. Trzecia litera RST (Tone).
    sag_db liczy się z TEGO SAMEGO stanu co chirp, bo to jedna przyczyna.
    """
    out = keying_envelope(text, wpm, sr, ramp_ms, fist=fist,
                          drift=fist_drift, gap_jitter=gap_jitter, rng=rng,
                          return_spans=return_spans)
    env, spans = out if return_spans else (out, [])
    if env.size == 0:
        return (env, spans) if return_spans else env

    t = np.arange(env.size, dtype=np.float64) / sr
    env = np.asarray(env, dtype=np.float64)
    sag = psu_sag(env, sr, sag_tau_ms) \
        if (chirp_hz > 0.0 or sag_db > 0.0) else None

    f_inst = np.full(env.size, float(tone))
    if abs(drift_hz) > 1e-9:
        span = t[-1] if t.size > 1 else 1.0
        f_inst = f_inst + float(drift_hz) * (t / span)
    if chirp_hz > 0.0:
        f_inst = f_inst - float(chirp_hz) * sag

    # Faza to CAŁKA częstotliwości chwilowej. Wstawienie f(t) wprost do
    # sin(2*pi*f(t)*t) dałoby przesunięcie dwukrotnie większe od zadanego,
    # a przy chirpie skorelowanym z kluczem — skoki fazy na elementach.
    ph = 2.0 * np.pi * np.cumsum(f_inst) / sr + phase

    a_env = env
    if sag_db > 0.0:
        a_env = a_env * (10.0 ** (-float(sag_db) / 20.0 * sag))
    if hum_depth > 0.0:
        a_env = a_env * (1.0 + float(hum_depth)
                         * np.sin(2.0 * np.pi * float(hum_hz) * t))

    wave = (amp * a_env * np.sin(ph)).astype(np.float32)
    return (wave, spans) if return_spans else wave


# =============================================================================
# KANAŁ RADIOWY — co pasmo robi z sygnałem
# =============================================================================
def band_noise(n: int, rng: np.random.Generator, rms: float = NOISE_RMS,
               tilt: float = 0.0) -> np.ndarray:
    """Szum o zadanym RMS i nachyleniu widma mocy f^tilt (0 biały, -1 różowy)."""
    if abs(tilt) < 1e-9:
        x = rng.standard_normal(n)
    else:
        spec = (rng.standard_normal(n // 2 + 1)
                + 1j * rng.standard_normal(n // 2 + 1))
        f = np.arange(spec.size, dtype=np.float64)
        f[0] = 1.0
        spec *= f ** (tilt / 2.0)
        spec[0] = 0.0
        x = np.fft.irfft(spec, n=n)
    r = float(np.sqrt(np.mean(x * x)))
    return (x * (rms / r)).astype(np.float32) if r > 1e-12 \
        else np.zeros(n, dtype=np.float32)


def apply_qsb(audio: np.ndarray, rng: np.random.Generator,
              depth: float, sr: int = SR) -> np.ndarray:
    """Zanik: suma dwóch wolnych modulacji, nie jeden sinus — pojedynczy
    sinus daje regularne pulsowanie, którego model uczy się jako wzoru."""
    if depth <= 0.0 or audio.size == 0:
        return audio
    t = np.arange(audio.size, dtype=np.float64) / sr
    f1, f2 = rng.uniform(0.15, 1.5), rng.uniform(0.15, 1.5) * 2.7
    p1, p2 = rng.uniform(0, 2 * np.pi, size=2)
    m = 0.65 * np.sin(2 * np.pi * f1 * t + p1) + \
        0.35 * np.sin(2 * np.pi * f2 * t + p2)
    return (audio * np.clip(1.0 - depth * 0.5 * (1.0 - m), 0.0, 1.0)
            ).astype(np.float32)


def apply_agc(audio: np.ndarray, tau_ms: float, depth: float,
              sr: int = SR) -> np.ndarray:
    """ARW odbiornika: ściska dynamikę z własną stałą czasową.

    Trzecie, niezależne źródło bujania amplitudy — obok zaniku (sekundy)
    i zapadania zasilacza nadajnika (dziesiątki ms). Skutek: pierwszy
    element po dłuższej przerwie jest przesterowany, potem wzmocnienie
    schodzi. Model uczony bez ARW widzi ten element jako inny kształt.
    """
    from scipy.signal import lfilter
    x = np.asarray(audio, dtype=np.float64)
    if depth <= 0.0 or x.size == 0:
        return audio
    a = float(np.exp(-1.0 / (max(1e-3, tau_ms) * 1e-3 * sr)))
    envd = lfilter([1.0 - a], [1.0, -a], np.abs(x))
    ref = float(np.percentile(envd, 95)) + 1e-9
    gain = (ref / np.maximum(envd, ref * 1e-3)) ** float(depth)
    return (x * np.clip(gain, 0.0, 20.0)).astype(np.float32)


def add_qrn(audio: np.ndarray, rng: np.random.Generator,
            n_crashes: int, amp: float, sr: int = SR) -> np.ndarray:
    """Trzaski: impulsy szerokopasmowe -> PIONOWA kreska w obrazie mel."""
    if n_crashes <= 0 or audio.size == 0:
        return audio
    out = audio.copy()
    for _ in range(n_crashes):
        pos = int(rng.integers(0, audio.size))
        length = int(rng.integers(int(0.0005 * sr), int(0.006 * sr) + 1))
        length = min(length, audio.size - pos)
        if length <= 1:
            continue
        decay = np.exp(-np.arange(length) / max(1.0, length / 3.0))
        out[pos:pos + length] += (rng.standard_normal(length) * decay
                                  * amp * rng.uniform(0.4, 1.0)
                                  ).astype(np.float32)
    return out


def add_qrm(audio: np.ndarray, rng: np.random.Generator,
            amp: float, sr: int = SR) -> np.ndarray:
    """Druga stacja: inny ton w CAŁYM pasmie analizy, inne tempo, inny tekst."""
    if amp <= 0.0 or audio.size == 0:
        return audio
    tone = float(rng.uniform(FMIN + 30.0, FMAX - 30.0))
    wpm = float(rng.uniform(15.0, 35.0))
    text = "".join(str(rng.choice(list(ALPHABET[1:]))) for _ in range(6))
    wave = synth_cw(text, wpm=wpm, tone=tone, amp=amp, sr=sr,
                    phase=float(rng.uniform(0, 2 * np.pi)),
                    fist=float(rng.uniform(0.0, 0.2)), rng=rng)
    if wave.size == 0:
        return audio
    out = audio.copy()
    # Start losowy, ale musi ZAHACZAĆ o okno widziane przez sieć: front-end
    # wycina środkowe IMG_FRAMES ramek, więc pierwsze i ostatnie 0,72 s klipu
    # nie trafiają do obrazu. Przy losowaniu z całego klipu ok. 12% nadań QRM
    # wypadało w odrzuconych brzegach — zbiór miał znacznik "obca stacja",
    # a na obrazie jej nie było.
    vis_a = window_crop_start(audio.size) * HOP_LENGTH
    vis_b = vis_a + IMG_FRAMES * HOP_LENGTH
    start = int(rng.integers(vis_a - wave.size + 1, vis_b))
    a0, a1 = max(0, start), min(audio.size, start + wave.size)
    if a1 > a0:
        out[a0:a1] += wave[a0 - start:a1 - start]
    return out


def receive(text: str, rng: np.random.Generator, realism: bool = True
            ) -> tuple[np.ndarray, dict]:
    """Nadanie przepuszczone przez kanał. Zwraca (audio, pełny opis)."""
    meta: dict = {}
    tilt = float(rng.uniform(*NOISE_TILT)) if realism else 0.0
    nr = NOISE_RMS * (float(rng.uniform(*NOISE_RMS_SPREAD)) if realism else 1.0)
    audio = band_noise(CLIP_SAMPLES, rng, rms=nr, tilt=tilt)
    meta.update(noise_rms=nr, noise_tilt=tilt)

    if text:
        tone = TONE_CENTER + float(rng.uniform(-TONE_SPREAD, TONE_SPREAD))
        amp = float(rng.uniform(SIGNAL_AMP_MIN, SIGNAL_AMP_MAX))
        wpm = float(np.clip(WPM + (rng.uniform(-WPM_JITTER, WPM_JITTER)
                                   if WPM_JITTER > 0 else 0.0), 5.0, 60.0))
        # --- operator i klucz ---
        fist = float(rng.uniform(*FIST)) if realism else 0.0
        fist_drift = float(rng.uniform(*FIST_DRIFT)) if realism else 0.0
        gap_jit = float(rng.uniform(*GAP_JITTER)) if realism else 0.0
        # --- nadajnik ---
        drift = float(rng.uniform(-DRIFT_HZ, DRIFT_HZ)) if realism else 0.0
        chirp = float(rng.uniform(*CHIRP_HZ)) if realism else 0.0
        sag = float(rng.uniform(*SAG_DB)) if realism else 0.0
        sag_tau = float(rng.uniform(*SAG_TAU_MS)) if realism else 60.0
        hum, hum_f = 0.0, 100.0
        if realism and rng.random() < HUM_PROB:
            hum = float(rng.uniform(*HUM_DEPTH))
            hum_f = float(rng.choice(HUM_HZ))

        wave, spans = synth_cw(text, wpm=wpm, tone=tone, amp=amp,
                               phase=float(rng.uniform(0, 2 * np.pi)),
                               fist=fist, fist_drift=fist_drift,
                               gap_jitter=gap_jit, drift_hz=drift,
                               chirp_hz=chirp, sag_db=sag,
                               sag_tau_ms=sag_tau, hum_depth=hum,
                               hum_hz=hum_f, rng=rng, return_spans=True)
        qsb = float(rng.uniform(*QSB_DEPTH)) if realism else 0.0
        wave = apply_qsb(wave, rng, depth=qsb)
        if wave.size < CLIP_SAMPLES:
            s = (CLIP_SAMPLES - wave.size) // 2
            audio[s:s + wave.size] += wave
            offset = s
        else:
            s = (wave.size - CLIP_SAMPLES) // 2
            audio += wave[s:s + CLIP_SAMPLES]
            offset = -s

        # Granice znaku Z ETYKIETY w próbkach klipu, z FAKTYCZNEGO timingu
        # po rozjeździe — nominalny wypadałby przy fist=0.25 nawet o pół
        # znaku obok.
        lab_a = lab_b = float("nan")
        for tag, a, b in spans:
            if tag == LABEL_INDEX:
                lab_a, lab_b = a + offset, b + offset
                break

        meta.update(tone=tone, amp=amp, wpm=wpm, fist=fist,
                    fist_drift=fist_drift, gap_jitter=gap_jit,
                    drift=drift, chirp=chirp, sag=sag, hum=hum,
                    qsb=qsb, text=text, lab_a=lab_a, lab_b=lab_b)
    else:
        meta.update(tone=float("nan"), amp=0.0, wpm=WPM, fist=0.0,
                    fist_drift=0.0, gap_jitter=0.0,
                    drift=0.0, chirp=0.0, sag=0.0, hum=0.0,
                    qsb=0.0, text="",
                    lab_a=float("nan"), lab_b=float("nan"))

    qrm = 0.0
    if realism and rng.random() < QRM_PROB:
        qrm = float(rng.uniform(*QRM_AMP))
        audio = add_qrm(audio, rng, amp=qrm)
    meta["qrm"] = qrm

    qrn = 0
    if realism and rng.random() < QRN_PROB:
        qrn = int(rng.integers(1, QRN_MAX + 1))
        audio = add_qrn(audio, rng, n_crashes=qrn,
                        amp=float(rng.uniform(*QRN_AMP)))
    meta["qrn"] = qrn

    # ARW odbiornika — ostatni etap przed ogranicznikiem karty, bo taka jest
    # kolejność w torze: antena -> mieszacz -> filtr -> ARW -> AF -> karta.
    agc_tau = 0.0
    if realism and rng.random() < AGC_PROB:
        agc_tau = float(rng.uniform(*AGC_TAU_MS))
        audio = apply_agc(audio, agc_tau, float(rng.uniform(*AGC_DEPTH)))
    meta["agc_tau"] = agc_tau

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    meta.update(peak=peak, clipped=peak > 1.0)
    np.clip(audio, -1.0, 1.0, out=audio)          # ogranicznik karty
    return audio, meta


# =============================================================================
# FRONT-END — audio -> obraz. JEDNA ścieżka.
# =============================================================================
# Front-end policzony NA SAMYM NUMPY, bez librosy. Trzy powody:
#   1. środowisko treningowe (TF 2.10) librosy nie ma;
#   2. docelowy KV260 też jej nie będzie miał — ten kod jest wzorcem
#      do przepisania na C;
#   3. jedna implementacja zamiast dwóch.
# Zgodność z librosą jest zmierzona w projekcie (diag.py TEST 14):
# krawędzie pasm 0,0 Hz różnicy, cały obraz 1,2e-07 = 6e-06 dB.
#
# TRZY KONWENCJE, KTÓRE MUSZĄ SIĘ ZGADZAĆ CO DO CYFRY
#   skala mel  SLANEY (liniowa poniżej 1000 Hz), nie HTK. Filtrbank
#              tf.signal.linear_to_mel_weight_matrix jest w HTK i NIE
#              jest wymienny.
#   norm       "slaney": trójkąty o jednakowym POLU, nie wysokości
#              (szczyt wychodzi 0,0395, nie 1,0).
#   dopełnienie STFT zerami, nie odbiciem. Sprawdzone: "constant" zgadza
#              się z librosą do 4e-16, "reflect" rozjeżdża o 7,6e-03.
#
# Nie ma tu parametru top_db. librosa.power_to_db ma go domyślnie na 80,
# co obcina wynik WZGLĘDEM MAKSIMUM klipu i po cichu przywraca skalę
# ruchomą nawet przy stałym ref. Tutaj nie da się go włączyć przez pomyłkę.

_F_SP = 200.0 / 3.0                      # herców na mel w części liniowej
_MIN_LOG_HZ = 1000.0                     # próg przejścia na logarytm
_MIN_LOG_MEL = _MIN_LOG_HZ / _F_SP       # = 15.0
_LOGSTEP = np.log(6.4) / 27.0


def hz_to_mel(f):
    """Herce -> mele, skala Slaneya (librosa htk=False)."""
    f = np.asarray(f, dtype=np.float64)
    mel = f / _F_SP
    hi = f >= _MIN_LOG_HZ
    if np.any(hi):
        mel = np.where(hi, _MIN_LOG_MEL + np.log(np.maximum(f, 1e-30)
                                                 / _MIN_LOG_HZ) / _LOGSTEP,
                       mel)
    return mel


def mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64)
    hz = m * _F_SP
    hi = m >= _MIN_LOG_MEL
    if np.any(hi):
        hz = np.where(hi, _MIN_LOG_HZ * np.exp(_LOGSTEP * (m - _MIN_LOG_MEL)),
                      hz)
    return hz


def mel_filterbank(sr: int = SR, n_fft: int = N_FFT, n_mels: int = N_MELS,
                   fmin: float = FMIN, fmax: float = FMAX) -> np.ndarray:
    """Macierz [n_mels, n_fft//2+1] — jak librosa.filters.mel(norm="slaney").
    Trójkąty liczone w HERCACH (nie w melach — to różnica wobec tf.signal),
    potem dzielone przez szerokość pasma, żeby miały równe pole."""
    n_bins = n_fft // 2 + 1
    fft_hz = np.linspace(0.0, sr / 2.0, n_bins, dtype=np.float64)
    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax),
                                  n_mels + 2))
    fb = np.zeros((n_mels, n_bins), dtype=np.float64)
    diff = np.diff(edges)
    ramps = edges[:, None] - fft_hz[None, :]
    for i in range(n_mels):
        fb[i] = np.maximum(0.0, np.minimum(-ramps[i] / diff[i],
                                           ramps[i + 2] / diff[i + 1]))
    fb *= (2.0 / (edges[2:n_mels + 2] - edges[:n_mels]))[:, None]
    return fb


# Filtrbank zależy tylko od stałych, więc liczymy go raz. W procesach
# roboczych generatora liczyłby się od nowa dla każdego kawałka.
_MEL_FB = mel_filterbank()


def hann_periodic(n: int) -> np.ndarray:
    """Okno Hanna OKRESOWE (fftbins=True) — takie bierze librosa do analizy.
    Wariant symetryczny (np.hanning) różni się jedną próbką."""
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


_HANN = hann_periodic(N_FFT)


def to_net_image(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    """audio -> [IMG_FRAMES, IMG_BINS] float32 w [0, 1]. Bez librosy."""
    x = np.asarray(audio, dtype=np.float64)

    # center=True: dopełnienie n_fft//2 z obu stron, ramka i wyśrodkowana
    # na próbce i*hop, liczba ramek = 1 + len//hop.
    pad = N_FFT // 2
    x = np.pad(x, pad, mode="constant")
    n_frames = 1 + (x.size - N_FFT) // HOP_LENGTH

    # Ramkowanie bez kopiowania: widok o zadanym kroku w pamięci.
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n_frames, N_FFT),
        strides=(x.strides[0] * HOP_LENGTH, x.strides[0]))

    spec = np.fft.rfft(frames * _HANN, n=N_FFT, axis=1)
    power = spec.real ** 2 + spec.imag ** 2

    # astype(float32) tutaj, a nie dopiero na końcu: dokładnie tak liczy
    # dsp/melref.py w projekcie, a różnica precyzji przed logarytmem daje
    # rozjazd 6e-08 w obrazie. Niegroźny, ale wtedy TEST 12 nie mógłby
    # wymagać zgodności CO DO BITU, a to najmocniejsze zabezpieczenie
    # przed rozjechaniem się tych dwóch plików.
    mel = (power @ _MEL_FB.T).astype(np.float32)

    # Ten sam porządek zaokrągleń co w melref: float32 po logarytmie,
    # PRZED normalizacją. Zamiana kolejności daje rozjazd 6e-08.
    db = (10.0 * np.log10(np.maximum(1e-10,
                                     np.asarray(mel, dtype=np.float64)))
          - 10.0 * np.log10(max(1e-10, DB_REF))).astype(np.float32)
    img = np.clip((db - DB_MIN) / db_span(), 0.0, 1.0).astype(np.float32)

    n = img.shape[0]
    if n == IMG_FRAMES:
        return img
    if n > IMG_FRAMES:
        s = (n - IMG_FRAMES) // 2
        return img[s:s + IMG_FRAMES]
    pad_f = IMG_FRAMES - n
    return np.pad(img, ((pad_f // 2, pad_f - pad_f // 2), (0, 0)),
                  mode="constant")


# =============================================================================
# GENERATOR ZBIORU
# =============================================================================
META_FIELDS = ("tone", "amp", "wpm", "fist", "drift", "qsb", "qrm", "qrn",
               "noise_rms", "noise_tilt", "peak",
               # Położenie znaku z etykiety w ramkach okna sieci — X-Ray
               # rysuje po tym biały wskaźnik pod kafelkiem.
               "lab_x0", "lab_x1")


def clip_frames(n_samples: int = CLIP_SAMPLES) -> int:
    """Ile ramek mel daje klip. librosa z center=True zwraca 1 + n//hop —
    ta jedna dodatkowa ramka jest źródłem pomyłek o jedną ramkę."""
    return 1 + n_samples // HOP_LENGTH


def window_crop_start(n_samples: int = CLIP_SAMPLES) -> int:
    return max(0, (clip_frames(n_samples) - IMG_FRAMES) // 2)


def sample_to_window_frame(sample: float,
                           n_samples: int = CLIP_SAMPLES) -> float:
    """Próbka audio w klipie -> numer ramki w oknie sieci."""
    return sample / HOP_LENGTH - window_crop_start(n_samples)


def make_clip(rng: np.random.Generator, realism: bool = True):
    """Jeden klip + etykieta + opis. Etykietą jest ŚRODKOWY znak."""
    if rng.random() > SILENCE_FRACTION:
        chars = [str(rng.choice(list(ALPHABET[1:])))
                 for _ in range(CHARS_PER_CLIP)]
        text = "".join(chars)
        target = CHAR_TO_ID[chars[LABEL_INDEX]]
    else:
        text, target = "", 0
    audio, meta = receive(text, rng, realism=realism)
    return audio, target, meta


def _gen_chunk(args):
    """Jeden kawałek zbioru — osobny proces. Ziarno z numeru kawałka, więc
    wynik nie zależy od liczby rdzeni ani kolejności zakończenia."""
    chunk_id, count, seed, realism = args
    rng = np.random.default_rng([seed, chunk_id])
    X = np.empty((count, IMG_FRAMES, IMG_BINS), dtype=np.uint8)
    y = np.empty(count, dtype=np.int16)
    cols = {k: np.empty(count, dtype=np.float32) for k in META_FIELDS}
    texts = []
    for i in range(count):
        audio, target, meta = make_clip(rng, realism=realism)
        X[i] = np.rint(to_net_image(audio) * 255.0).astype(np.uint8)
        y[i] = target
        for src, dst in (("lab_a", "lab_x0"), ("lab_b", "lab_x1")):
            v = meta.get(src, np.nan)
            meta[dst] = (np.nan if not np.isfinite(v)
                         else sample_to_window_frame(v))
        for k in META_FIELDS:
            cols[k][i] = meta.get(k, np.nan)
        texts.append(meta["text"])
    return X, y, cols, texts


def generate(n: int, out_path: Path, seed: int = SEED, realism: bool = True,
             workers: int | None = None) -> Path:
    """Zbiór treningowy -> .npz. Równolegle na wszystkich rdzeniach."""
    import concurrent.futures as cf

    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    per = 500
    chunks = [(i, min(per, n - i * per), seed, realism)
              for i in range((n + per - 1) // per)]

    print(f"generuję {n} próbek na {workers} procesach "
          f"({len(chunks)} kawałków po {per})")

    Xs, ys, colss, texts = [], [], [], []
    t0 = time.time()
    done = 0
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        for X, y, cols, tx in ex.map(_gen_chunk, chunks):
            Xs.append(X); ys.append(y); colss.append(cols); texts.extend(tx)
            done += len(y)
            el = time.time() - t0
            print(f"\r  {done}/{n}  {el:.0f}s  "
                  f"pozostało ~{el/done*(n-done):.0f}s   ", end="", flush=True)
    print()

    X = np.concatenate(Xs); y = np.concatenate(ys)
    cols = {k: np.concatenate([c[k] for c in colss]) for k in META_FIELDS}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, X=X, y=y, text=np.array(texts),
             fingerprint=FINGERPRINT, dtype_note=STORE_DTYPE,
             meta=f"n={n};seed={seed};wpm={WPM};realism={int(realism)}",
             **cols)

    counts = np.bincount(y, minlength=N_CLASSES)
    print(f"\nZapisano {out_path}  "
          f"({out_path.stat().st_size/1024/1024:.0f} MB, uint8)")
    print(f"klasa 0 (puste radio): {counts[0]} ({100*counts[0]/n:.1f}%)")
    print(f"znaki: min {counts[1:].min()}, max {counts[1:].max()} na klasę")
    if realism:
        print(f"QRM {100*np.mean(cols['qrm']>0):.0f}%   "
              f"QRN {100*np.mean(cols['qrn']>0):.0f}%   "
              f"QSB>0.3 {100*np.mean(cols['qsb']>0.3):.0f}%")
    return out_path


# =============================================================================
# MODEL
# =============================================================================
# Typy warstw, które przechodzą przez kompilator Vitis AI na DPUCZDX8G
# (Zynq UltraScale+, czyli KV260). Lista celowo wąska — cokolwiek poza nią
# albo nie skompiluje się, albo zostanie wydzielone do wykonania na ARM-ie,
# co przy strumieniu audio zabija przepustowość.
DPU_SAFE_LAYERS = {
    "InputLayer", "Conv2D", "DepthwiseConv2D", "SeparableConv2D",
    "Conv2DTranspose", "BatchNormalization", "Activation", "ReLU",
    "MaxPooling2D", "AveragePooling2D", "GlobalAveragePooling2D",
    "Add", "Concatenate", "Flatten", "Reshape", "Dense", "Dropout",
}


def build_model(dropout: float = 0.3, gru_units: int = 96,
                learning_rate: float = 1e-3, arch: str = "dpu"):
    """Model klasyfikatora znaku.

    arch="dpu"  — SAM SPLOT, bez warstw rekurencyjnych. Ta wersja idzie na
                  KV260. DPUCZDX8G jest akceleratorem CNN i nie obsługuje
                  GRU ani LSTM; RNN w Vitis AI ma osobne nakładki
                  (DPURADR16L na U25, DPURAHR16L na U50LV), których na Zynq
                  nie ma. Model z GRU albo się nie skompiluje, albo zostanie
                  rozcięty i część pójdzie na ARM.

    arch="gru"  — wersja z dwukierunkowym GRU. NIE DA SIĘ jej wdrożyć na
                  KV260; zostaje jako punkt odniesienia, żeby wiedzieć, ile
                  dokładności kosztuje rezygnacja z rekurencji.

    WSPÓLNA ZASADA W OBU: nie ma GlobalAveragePooling po osi czasu.
    W Morse'ie informacja jest w KOLEJNOŚCI — "..-" i "-.." mają ten sam
    zestaw elementów i tę samą średnią energię, więc uśrednienie po czasie
    zrównuje U z D.
    """
    keras = keras_api()
    layers = keras.layers

    inputs = keras.Input(shape=INPUT_SHAPE, name="obraz")
    x = inputs

    if arch == "gru":
        for filters, pool in ((32, (2, 2)), (64, (2, 2)), (128, (1, 2))):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            x = layers.MaxPooling2D(pool)(x)
        t, f, ch = x.shape[1], x.shape[2], x.shape[3]
        x = layers.Reshape((t, f * ch), name="czas_cechy")(x)
        # GRU w konfiguracji domyślnej (tanh/sigmoid, bez recurrent_dropout),
        # żeby na GPU wchodziło jądro cuDNN.
        x = layers.Bidirectional(layers.GRU(gru_units), name="gru")(x)
        name = "morse_crnn"

    elif arch == "dpu":
        # Sześć bloków splotowych, cztery redukcje 2x2. Zasięg widzenia
        # po osi czasu, liczony narastająco z wkładem KAŻDEJ warstwy —
        # także redukujących, o czym łatwo zapomnieć:
        #   rf = 1, skok = 1
        #   conv3   rf =   3    pool2  rf =   4, skok 2
        #   conv3   rf =   8    pool2  rf =  10, skok 4
        #   conv3   rf =  18    pool2  rf =  22, skok 8
        #   conv3   rf =  38    pool2  rf =  46, skok 16
        #   conv3   rf =  78
        #   conv3   rf = 110 ramek = 2,20 s
        # Najdłuższy znak ('0' = 19 jednostek) zajmuje 1,17 s, a z przerwami
        # międzyznakowymi 1,50 s. Bez pokrycia obu granic naraz neuron nie ma
        # z czego odczytać długości znaku. Ostatni splot jest właśnie po to:
        # bez niego zasięg to 78 ramek = 1,56 s, czyli 0,06 s zapasu.
        #
        # Rozszerzone sploty (dilation) dałyby ten sam zasięg taniej, ale
        # na DPUCZDX8G mają ograniczenia zależne od konfiguracji rdzenia,
        # więc zasięg budujemy zwykłym stosem.
        for filters, pool in ((32, (2, 2)), (48, (2, 2)), (64, (2, 2)),
                              (96, (2, 2)), (128, None), (128, None)):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            if pool is not None:
                x = layers.MaxPooling2D(pool)(x)

        # Flatten po 8 zgrubnych krokach czasu (8 x 2 x 128 = 2048).
        # Głowa MUSI być zależna od pozycji: etykietą jest ŚRODKOWY z trzech
        # nadanych znaków, więc "znajdź jakikolwiek znak" to zła odpowiedź.
        # Maksimum po czasie (position-invariant) zwracałoby najmocniejszy
        # znak w oknie, czyli często sąsiada. Dense po 8 krokach jest
        # zależny od pozycji, ale zgrubnie — a środek znaku z etykiety
        # błądzi +/-24 ramki, czyli +/-1,5 kroku na tym poziomie.
        x = layers.Flatten()(x)
        x = layers.Dense(128, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        name = "morse_cnn_dpu"

    else:
        raise ValueError(f"nieznana architektura: {arch!r} "
                         f"(dostępne: 'dpu', 'gru')")

    x = layers.Dropout(dropout)(x)
    # dtype="float32" jawnie: przy mixed_float16 softmax musi liczyć się
    # w pełnej precyzji, inaczej strata potrafi wyjść NaN.
    # Na DPUCZDX8G softmax ma własny blok sprzętowy, a i tak jest tani.
    outputs = layers.Dense(N_CLASSES, activation="softmax", name="znak",
                           dtype="float32")(x)

    model = keras.Model(inputs, outputs, name=name)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def check_dpu_compatible(model, verbose: bool = True) -> list[str]:
    """Zwraca listę warstw, których DPUCZDX8G nie obsługuje.

    Kontrola statyczna, na typach warstw — nie zastępuje kompilatora Vitis
    AI, ale wychwytuje najkosztowniejszy błąd: wytrenowanie modelu, którego
    nie da się wdrożyć. Kompilator sprawdzi jeszcze rozmiary jąder, liczbę
    kanałów i kolejność warstw.
    """
    bad = [f"{l.name} ({type(l).__name__})" for l in model.layers
           if type(l).__name__ not in DPU_SAFE_LAYERS]
    if verbose:
        if bad:
            print("UWAGA: warstwy nieobsługiwane przez DPUCZDX8G (KV260):")
            for b in bad:
                print(f"    {b}")
            print("  Model wytrenuje się na GPU, ale na KV260 albo się nie "
                  "skompiluje,\n  albo zostanie rozcięty i część pójdzie "
                  "na ARM.")
        else:
            print("kontrola DPUCZDX8G: wszystkie typy warstw obsługiwane")
    return bad


# =============================================================================
# DANE DO TRENINGU
# =============================================================================
def _pliki_zbioru(spec: str) -> list:
    """Wzorzec -> lista plikow. Jeden plik dziala jak wczesniej, wzorzec
    z gwiazdka skleja kilka czesci w jeden zbior.

    PO CO CZESCI. Zmierzone: generowanie idzie 238 probek/s (11 procesow),
    a karta konsumuje 8000/s. Generowanie w locie zaglodziloby GPU 34-krotnie
    -- epoka trwalaby 13 minut zamiast 23 sekund. Wiec dane robi sie
    ZAWCZASU, a jedyny sposob na wiecej danych bez czekania to trzymac je
    w kilku plikach i skleic przy wczytaniu.
    """
    from glob import glob
    p = Path(spec)
    if p.is_file():
        return [p]
    trafienia = sorted(Path(x) for x in glob(spec))
    if not trafienia:
        raise SystemExit(
            f"nie ma zbioru: {spec}\n"
            f"Najpierw: python {Path(__file__).name} generate --n 200000\n"
            f"albo w czesciach: python {Path(__file__).name} generate "
            f"--n 200000 --shards 5 --out czesci/morse.npz")
    return trafienia


def _sprawdz_odcisk(data, gdzie: Path):
    fp = str(data["fingerprint"]) if "fingerprint" in data else ""
    if fp and fp != FINGERPRINT:
        a = dict(kv.split("=", 1) for kv in fp.split(";") if "=" in kv)
        b = dict(kv.split("=", 1) for kv in FINGERPRINT.split(";") if "=" in kv)
        diffs = [f"  {k}: zbiór={a.get(k,'-')}  skrypt={b.get(k,'-')}"
                 for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
        raise SystemExit(f"{gdzie} powstal na innych parametrach front-endu:\n"
                         + "\n".join(diffs)
                         + "\n\nWygeneruj zbiór od nowa albo przywróć te "
                           "wartości w sekcji KONFIGURACJA.")


def _wolna_pamiec_mb() -> float:
    """MemAvailable z /proc/meminfo. Zwraca -1, gdy nie da sie odczytac."""
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return -1.0


def load_dataset(spec):
    pliki = _pliki_zbioru(str(spec))

    # Sprawdzenie pamieci PRZED wczytaniem. Sklejenie dziesieciu czesci po
    # 800 MB to 8 GB -- na maszynie z 15 GB zmiesci sie, na mniejszej nie,
    # a objawem bylby zabity proces bez zadnego komunikatu (OOM killer).
    lacznie_mb = sum(p.stat().st_size for p in pliki) / 1024 / 1024
    wolne_mb = _wolna_pamiec_mb()
    if len(pliki) > 1:
        print(f"zbiór w {len(pliki)} częściach, razem {lacznie_mb:.0f} MB")

    # SZCZYT, NIE ROZMIAR PLIKU. Pierwsza wersja tego strażnika porównywała
    # z wolną pamięcią sam rozmiar danych i przepuściła 1 mln próbek na
    # maszynie z 15 GB — po czym trening padł po minucie.
    #
    # Skąd mnożnik. Na drodze do model.fit() te same obrazy istnieją
    # w kilku kopiach naraz:
    #   1x  sklejone X po np.concatenate
    #   1x  X[idx] w make_pipeline — indeksowanie tablicą ROBI KOPIĘ
    #   1x  to samo przeniesione do tf.data
    # czyli około trzykrotności, zanim cokolwiek policzy się na karcie.
    SZCZYT = 3.0
    potrzeba_mb = lacznie_mb * SZCZYT
    if wolne_mb > 0 and potrzeba_mb > 0.8 * wolne_mb:
        raise SystemExit(
            f"PRZERWANO: zbiór ma {lacznie_mb:.0f} MB, ale w drodze do "
            f"model.fit() istnieje w ~{SZCZYT:.0f} kopiach, czyli potrzeba "
            f"~{potrzeba_mb:.0f} MB.\n"
            f"Wolnej pamięci jest {wolne_mb:.0f} MB.\n"
            f"Bez tego sprawdzenia proces zostałby zabity przez OOM killer "
            f"BEZ KOMUNIKATU, po kilkudziesięciu sekundach treningu.\n"
            f"Weź mniej części: przy {wolne_mb:.0f} MB bezpieczne jest "
            f"około {wolne_mb * 0.8 / SZCZYT / 820:.0f} x 200 tys. próbek.")

    Xs, ys, metki = [], [], []
    for p in pliki:
        data = np.load(p, allow_pickle=False)
        _sprawdz_odcisk(data, p)
        Xs.append(np.asarray(data["X"]))
        ys.append(np.asarray(data["y"]).astype(np.int32))
        metki.append(str(data["meta"]) if "meta" in data else "")

    if len(pliki) == 1:
        return Xs[0], ys[0], metki[0]

    X = np.concatenate(Xs)
    del Xs                      # 8 GB nie moze lezec w dwoch kopiach
    y = np.concatenate(ys)

    # Opis zbioru idzie do stanu treningu, zeby wznowienie wykrylo podmiane
    # danych. Musi byc DETERMINISTYCZNY dla tego samego zestawu czesci --
    # inaczej kazde uruchomienie wygladaloby jak zmiana zbioru i kasowaloby
    # najlepszy wynik. Odcisk front-endu tego nie wykrywa: zmiana
    # WPM_JITTER czy chirpu daje inne dane przy tym samym front-endzie.
    ziarna = sorted({m.split("seed=")[1].split(";")[0]
                     for m in metki if "seed=" in m})
    meta = (f"czesci={len(pliki)};n={len(y)};seeds={','.join(ziarna)};"
            f"wpm={WPM};realism=1")
    return X, y, meta


def stratified_split(y: np.ndarray, val_fraction: float, seed: int):
    """Podział warstwowy: każda klasa oddaje ten sam UDZIAŁ na walidację.

    Przy losowym podziale rzadsza klasa może dostać kilka próbek i
    val_accuracy przestaje cokolwiek znaczyć dla tej klasy.
    """
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_fraction)))
        va.append(idx[:n_val]); tr.append(idx[n_val:])
    tr, va = np.concatenate(tr), np.concatenate(va)
    rng.shuffle(tr); rng.shuffle(va)
    return tr, va


def class_weights(y: np.ndarray) -> np.ndarray:
    """Wagi wyrównujące udział klas.

    Klasa 0 ma 15% zbioru, każdy znak po ~2,4%. Najtańsze minimum straty przy
    takim rozkładzie to "zawsze odpowiadaj 0" — daje 15% dokładności bez
    uczenia się czegokolwiek, i model dokładnie w to wpada.
    """
    counts = np.bincount(y, minlength=N_CLASSES).astype(np.float64)
    present = counts > 0
    w = np.zeros(N_CLASSES, dtype=np.float32)
    w[present] = len(y) / (present.sum() * counts[present])
    return w


def make_pipeline(X, y, idx, batch: int, training: bool,
                  weights: np.ndarray | None = None):
    """tf.data; konwersja uint8 -> float32 dopiero w partii.

    Obrazy leżą jako uint8 (200 tys. próbek = 790 MB). W float32 byłoby
    3,1 GB, plus tyle samo na kopię przy podziale — stąd brały się braki
    pamięci przy dużych zbiorach.
    """
    import tensorflow as tf
    parts = [X[idx], y[idx]]
    if weights is not None:
        parts.append(weights[y[idx]])
    ds = tf.data.Dataset.from_tensor_slices(tuple(parts))

    def prep(img, label, *rest):
        img = tf.expand_dims(tf.cast(img, tf.float32) / 255.0, -1)
        return (img, label, rest[0]) if rest else (img, label)

    if training:
        ds = ds.shuffle(min(len(idx), 50000), reshuffle_each_iteration=True)
    return (ds.map(prep, num_parallel_calls=tf.data.AUTOTUNE)
              .batch(batch).prefetch(tf.data.AUTOTUNE))


# =============================================================================
# WZNAWIALNY TRENING
# =============================================================================
class StateSaver:
    """Zapis stanu po każdej epoce — sedno wznawiania.

    Kolejność zapisu ma znaczenie: najpierw model, potem state.json.
    Gdyby state.json trafił na dysk pierwszy, a prąd padł przed zapisem
    modelu, wznowienie ruszyłoby z niezgodnym numerem epoki.
    """

    def __init__(self, run_dir: Path, keras_mod):
        self.dir = run_dir
        self.keras = keras_mod
        self.ext = model_ext()
        self.state = {"epoch": 0, "best_val_acc": -1.0, "history": {},
                      "dataset": ""}

    # --- odczyt ---
    def load(self, dataset_meta: str = ""):
        """Wczytuje stan i SPRAWDZA, czy zbiór jest ten sam.

        PO CO. Zdarzyło się dokładnie tak: 200 epok na zbiorze o jednym
        tempie (20 WPM, bez chirpu) dało val_accuracy 98,68% i accuracy
        treningową 100,00%. Potem zbiór wygenerowano od nowa na szerszym
        configu (13-27 WPM, chirp, zapadanie zasilacza, przydźwięk, ARW)
        i wznowiono trening. W epoce 201 strata TRENINGOWA skoczyła
        z 0,0001 na 4,33 — model spotkał inne dane.

        Skutek uboczny był gorszy niż sam skok: `best_val_acc` zostało
        na 98,68% ZMIERZONYM NA INNYM ROZKŁADZIE, więc nowe zadanie nie
        miało jak go pobić i best.keras nigdy się nie zaktualizował.
        Po 100 epokach na trudnym zbiorze best.keras nadal był starym
        modelem z łatwego, a jedynym użytecznym plikiem było last.keras.

        Dlatego stan pamięta teraz opis zbioru. Przy zmianie licznik
        najlepszego wyniku jest zerowany, bo porównywanie dokładności
        z dwóch różnych rozkładów nie ma sensu.
        """
        p = self.dir / "state.json"
        if p.exists():
            self.state = json.loads(p.read_text(encoding="utf-8"))
        stary = self.state.get("dataset", "")
        if stary and dataset_meta and stary != dataset_meta:
            print("=" * 72)
            print("UWAGA: ZBIÓR TRENINGOWY SIĘ ZMIENIŁ")
            print(f"  poprzednio: {stary}")
            print(f"  teraz:      {dataset_meta}")
            print(f"  Zeruję najlepszy wynik ({self.state.get('best_val_acc', -1)*100:.2f}%)"
                  f" — był mierzony na INNYM rozkładzie.")
            print("  Model jest dalej wczytywany z last, czyli trening jest")
            print("  ciepłym startem. Jeśli nowy zbiór różni się mocno,")
            print("  rozważ --fresh: wagi wyuczone na jednym tempie to zły")
            print("  punkt wyjścia dla zbioru o wielu, a learning rate zdążył")
            print("  zejść do minimum, więc odbudowa się czołga.")
            print("=" * 72)
            self.state["best_val_acc"] = -1.0
        self.state["dataset"] = dataset_meta or stary
        return self.state

    def resume_model(self):
        p = find_model(self.dir, "last")
        if p is not None and self.state.get("epoch", 0) > 0:
            return self.keras.models.load_model(str(p))
        return None

    # --- zapis ---
    def callback(self):
        keras = self.keras
        saver = self

        class _Cb(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                saver.model.save(str(saver.dir / ("last" + saver.ext)))

                acc = float(logs.get("val_accuracy", -1.0))
                if acc > saver.state["best_val_acc"]:
                    saver.state["best_val_acc"] = acc
                    saver.model.save(str(saver.dir / ("best" + saver.ext)))

                saver.state["epoch"] = int(epoch) + 1
                hist = saver.state.setdefault("history", {})
                for k, v in logs.items():
                    hist.setdefault(k, []).append(float(v))
                (saver.dir / "state.json").write_text(
                    json.dumps(saver.state, indent=1), encoding="utf-8")

                print(f"    [stan zapisany: epoka {saver.state['epoch']}, "
                      f"najlepsza walidacja {saver.state['best_val_acc']*100:.2f}%]",
                      flush=True)

        return _Cb()


def confusion_report(model, ds, y_true: np.ndarray) -> None:
    """Raport pomyłek. Macierz 37x37 w terminalu jest nieczytelna, a
    interesuje nas i tak tylko to, co się z czym myli."""
    probs = model.predict(ds, verbose=0)
    pred = probs.argmax(axis=1)
    print(f"\nDokładność na walidacji: {np.mean(pred == y_true)*100:.2f}%")

    m0 = y_true == 0
    if m0.any():
        print(f"  klasa 0 (puste radio): {np.mean(pred[m0]==0)*100:.1f}% "
              f"poprawnie ({m0.sum()} próbek)")
    if (~m0).any():
        print(f"  znaki:                 "
              f"{np.mean(pred[~m0]==y_true[~m0])*100:.1f}% poprawnie "
              f"({(~m0).sum()} próbek)")
        print(f"  znak wzięty za ciszę:  "
              f"{np.mean(pred[~m0]==0)*100:.1f}%")

    pairs = {}
    for t, p in zip(y_true, pred):
        if t != p:
            pairs[(int(t), int(p))] = pairs.get((int(t), int(p)), 0) + 1
    if pairs:
        print("\nNajczęstsze pomyłki (prawda -> predykcja):")
        for (t, p), cnt in sorted(pairs.items(), key=lambda kv: -kv[1])[:20]:
            ct = "PUSTE" if t == 0 else ID_TO_CHAR[t]
            cp = "PUSTE" if p == 0 else ID_TO_CHAR[p]
            print(f"  {ct:>5} {MORSE_DICT.get(ct,''):<6} -> "
                  f"{cp:>5} {MORSE_DICT.get(cp,''):<6}  {cnt}x")
        print("\nPomyłki między znakami różniącymi się jednym elementem "
              "(U '..-' / V '...-', A '.-' / R '.-.') znaczą, że model gubi "
              "TIMING, a nie że brakuje mu pojemności.")


def train(args):
    import tensorflow as tf
    keras = keras_api()

    run_dir = Path(args.run)
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- GPU ---
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    if args.mixed and gpus:
        # W keras 2.10 to tf.keras.mixed_precision.set_global_policy — ta
        # sama nazwa co w Kerasie 3, więc jedno wywołanie wystarcza.
        keras.mixed_precision.set_global_policy("mixed_float16")
        print("mixed_float16 włączone (Ampere i nowsze liczą tak ~2x szybciej)")
    elif args.mixed:
        print("--mixed pominięte: brak GPU")

    # --- dane ---
    X, y, ds_meta = load_dataset(args.dataset)
    print(f"opis zbioru: {ds_meta or '(brak)'}")
    print(f"zbiór: {X.shape} {X.dtype} ({X.nbytes/1024/1024:.0f} MB)")
    tr, va = stratified_split(y, args.val, args.seed)
    print(f"podział warstwowy: trening={len(tr)}  walidacja={len(va)}")

    w = None if args.no_class_weights else class_weights(y[tr])
    if w is not None:
        print(f"wagi klas: klasa 0 -> {w[0]:.3f}, "
              f"znaki -> {w[1:][w[1:]>0].mean():.3f}")

    ds_tr = make_pipeline(X, y, tr, args.batch, True, weights=w)
    ds_va = make_pipeline(X, y, va, args.batch, False)

    # --- model: wznowienie albo świeży ---
    saver = StateSaver(run_dir, keras)
    state = ({"epoch": 0, "best_val_acc": -1.0, "dataset": ds_meta}
             if args.fresh else saver.load(ds_meta))
    if args.fresh:
        saver.state = state
    model = None if args.fresh else saver.resume_model()

    if model is None:
        model = build_model(dropout=args.dropout, gru_units=args.gru,
                            learning_rate=args.lr, arch=args.arch)
        model.summary()
        check_dpu_compatible(model)
        print("\nstart od zera")
    else:
        print(f"\nWZNOWIENIE: model z epoki {state['epoch']}, "
              f"najlepsza walidacja {state['best_val_acc']*100:.2f}%")
        # Learning rate PO wznowieniu. Bez tego nie widać najczęstszej
        # przyczyny "trening stoi": ReduceLROnPlateau przez dziesiątki
        # epok płaskiego val_loss zsuwa go do min_lr, a po podmianie
        # zbioru model musi się uczyć od nowa przy 1e-6 i czołga się.
        try:
            lr = float(keras.backend.convert_to_numpy(
                model.optimizer.learning_rate))
        except Exception:
            try:
                lr = float(model.optimizer.learning_rate.numpy())
            except Exception:
                lr = float("nan")
        print(f"            learning rate: {lr:.2e}")
        if lr < 1e-5:
            print("            To jest praktycznie minimum. Jeśli zbiór się "
                  "zmienił,\n            trening nie odbuduje się w rozsądnym "
                  "czasie — użyj --fresh\n            albo --lr 1e-3.")

    saver.model = model
    initial_epoch = int(state.get("epoch", 0))
    if initial_epoch >= args.epochs:
        print(f"Zadana liczba epok ({args.epochs}) już osiągnięta "
              f"({initial_epoch}). Podnieś --epochs, żeby uczyć dalej.")
        confusion_report(model, ds_va, y[va])
        return

    cbs = [
        saver.callback(),
        keras.callbacks.CSVLogger(str(run_dir / "log.csv"), append=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=4, min_lr=1e-6, verbose=1),
    ]
    if args.patience > 0:
        cbs.append(keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=args.patience,
            restore_best_weights=True, verbose=1))

    print(f"\ntrening: epoki {initial_epoch+1}..{args.epochs}, "
          f"batch {args.batch}")
    print("Przerwanie w dowolnym momencie jest bezpieczne — po każdej epoce "
          "stan jest na dysku.\n")

    model.fit(ds_tr, validation_data=ds_va, epochs=args.epochs,
              initial_epoch=initial_epoch, callbacks=cbs, verbose=1)

    best = find_model(run_dir, "best")
    if best is not None:
        print(f"\nnajlepszy model: {best}")
        model = keras.models.load_model(str(best))
    confusion_report(model, ds_va, y[va])


# =============================================================================
# CLI
# =============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Trening dekodera CW — plik samodzielny")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="wygeneruj zbiór treningowy")
    g.add_argument("--n", type=int, default=200000)
    g.add_argument("--out", type=Path, default=Path("morse_dataset.npz"))
    g.add_argument("--seed", type=int, default=SEED)
    g.add_argument("--shards", type=int, default=1,
                   help="ile osobnych czesci po --n probek kazda. Kazda "
                        "dostaje inne ziarno, wiec dane sa naprawde rozne. "
                        "Trening sklei je przez wzorzec, np. --dataset "
                        "'czesci/morse_*.npz'")
    g.add_argument("--workers", type=int, default=None)
    g.add_argument("--no-realism", action="store_true",
                   help="bez modelu kanału (QSB/QRM/QRN/dryf/fist)")

    t = sub.add_parser("train", help="trenuj (domyślnie wznawia)")
    t.add_argument("--dataset", default="morse_dataset.npz",
                   help="plik zbioru albo WZORZEC, np. 'czesci/morse_*.npz'")
    t.add_argument("--run", type=Path, default=Path("runs/cw1"),
                   help="katalog stanu: last.keras, best.keras, state.json")
    t.add_argument("--arch", choices=("dpu", "gru"), default="dpu",
                   help="dpu = sam splot, wdrażalny na KV260 (domyślnie); "
                        "gru = z rekurencją, NIE do wdrożenia na KV260, "
                        "tylko jako punkt odniesienia")
    t.add_argument("--epochs", type=int, default=200)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--val", type=float, default=0.08)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--gru", type=int, default=96)
    t.add_argument("--dropout", type=float, default=0.3)
    t.add_argument("--patience", type=int, default=0,
                   help="0 = bez early stopping (przy treningu z dnia na "
                        "dzień lepiej sterować liczbą epok ręcznie)")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--fresh", action="store_true",
                   help="zignoruj zapisany stan i zacznij od zera")
    t.add_argument("--mixed", action="store_true",
                   help="mixed_float16 — na RTX 3050 ok. 2x szybciej")
    t.add_argument("--require-gpu", action="store_true",
                   help="przerwij, jeśli karta nie jest widziana — do "
                        "treningu zostawianego na noc")
    t.add_argument("--no-class-weights", action="store_true")

    args = ap.parse_args(argv)

    print("=" * 72)
    print("DEKODER CW — " + ("GENEROWANIE ZBIORU" if args.cmd == "generate"
                             else "TRENING"))
    print("=" * 72)
    print(summary())
    print("-" * 72)
    if args.cmd == "train":
        has_gpu, report = env_report()
        print(report)
        print("-" * 72)
        if args.require_gpu and not has_gpu:
            # Świadomie przerywamy. Zostawienie treningu na noc i odkrycie
            # o 8:00, że policzył się na CPU, kosztuje całą noc.
            print("\nPRZERWANO: podano --require-gpu, a karta nie jest "
                   "widziana.\nUsuń --require-gpu, jeśli chcesz mimo to "
                   "trenować na CPU.")
            return 2

    if args.cmd == "generate":
        if args.shards <= 1:
            generate(args.n, args.out, seed=args.seed,
                     realism=not args.no_realism, workers=args.workers)
        else:
            # Kazda czesc z INNYM ziarnem -- inaczej powstalyby identyczne
            # pliki i "wiecej danych" byloby zludzeniem.
            baza = args.out.with_suffix("")
            for k in range(args.shards):
                cel = Path(f"{baza}_{k:02d}.npz")
                if cel.exists():
                    print(f"[{k+1}/{args.shards}] {cel} juz jest — pomijam")
                    continue
                print(f"\n[{k+1}/{args.shards}] {cel}")
                generate(args.n, cel, seed=args.seed + 1000 * k,
                         realism=not args.no_realism, workers=args.workers)
            print(f"\nGotowe. Trening: --dataset '{baza}_*.npz'")
    else:
        train(args)


if __name__ == "__main__":
    sys.exit(main())
