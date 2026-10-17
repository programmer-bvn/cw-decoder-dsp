"""SIEĆ ZBOCZY — pierwszy etap dwuetapowego dekodera (17.10).

POMYSŁ OPERATORA. „Podział procesu na odzyskiwanie zboczy i rozpoznawanie
znaków — dwuetapowa detekcja." Klasyczne dekodery biorą obwiednię jednego
przekroju („nożem przez kopiec") i progują komparatorem; zbocze odebrane
nie leży jednak tam, gdzie nadane — tor nadawczy, filtr p.cz., ARW, pogłos
i szum przesuwają je i rozmywają („kość" na końcach znaków). Sieć, jak przy
kolorowaniu czarno-białych zdjęć, uczy się, jak wygląda zbocze PO torze,
i wskazuje, gdzie było zbocze ORYGINALNE. Generator zna je co do próbki.

    pierwsza sieć (ta):  widmo [czas 5 ms x pasma]  ->  P(klucz wciśnięty) co 5 ms
    dalej:               grzebień na jej wyjściu -> tempo -> normalizacja ->
                         druga sieć albo tabela Morse'a

WEJŚCIE. Widmo w CZASIE RZECZYWISTYM (tempo liczy się dopiero z wyjścia tej
sieci), krok 5 ms, okno FFT 32 ms (N_FFT_ZBOCZA = 256): rozdzielczość
31 Hz < szerokość pasma mel (~48 Hz), a zbocze dwa razy ostrzejsze w czasie
niż przy oknie 64 ms, które jest dłuższe niż kropka przy 22 WPM (55 ms) —
rozmowa z operatorem 17.10 (RBW / filtr wideo w analizatorze widma).
Okno sieci: RAMKI_ZBOCZA = 512 ramek = 2,56 s, jak w głównej sieci.
"""

from __future__ import annotations

import numpy as np

from . import config as C
from . import frontend, melref, radio

HOP_ZBOCZA = 40                    # 5 ms
N_FFT_ZBOCZA = 256                 # 32 ms
RAMKI_ZBOCZA = 512                 # 2,56 s
ZNAKOW_W_KLIPIE = 4                # więcej zboczy na klip niż 3

# Zakresy szersze niż w głównym generatorze (17.10, tester_zbocza na
# nagraniach): runs/zbocza_1 (SN_DB -3..30, tło od -12 dB, narastanie 5 ms)
# gubił się na obrazach CZYSTSZYCH niż w treningu — mikrofon (tło -30...-20
# dB, zmierzone 12.10) i SN0FMT po wyrównaniu poziomu: czarne tło. Z szumem
# dosypanym do 25 dB mic1-3 czytane prawie bezbłędnie (mic3 6/6 słów, próg
# klasyczny 1/6). Narastanie: IC-7300 ma do wyboru 2/4/6/8 ms.
ZAKRESY_ZBOCZY = {"SN_DB": (-3.0, 45.0), "TLO_DB": (-32.0, 7.0),
                  "KEY_RAMP_MS": (2.0, 8.0)}


_FB = None


def _fb():
    global _FB
    if _FB is None:
        _FB = melref.mel_filterbank(C.SR, N_FFT_ZBOCZA, C.N_MELS, C.FMIN, C.FMAX)
    return _FB


def obraz_zboczy(audio: np.ndarray) -> np.ndarray:
    """audio -> [ramki co 5 ms, N_MELS] w [0,1], skala dB jak w głównej sieci."""
    p = melref.stft_power(np.asarray(audio, dtype=np.float32), N_FFT_ZBOCZA,
                          HOP_ZBOCZA)
    db = melref.power_to_db((p @ _fb().T).astype(np.float64))
    return frontend.normalize_db(db).astype(np.float32)


def klucz_ramek(klucz: np.ndarray, n_ramek: int) -> np.ndarray:
    """Przebieg klucza (próbki) -> wartość w każdej ramce (średnia z +/-2,5 ms
    wokół środka ramki; ramka i ma środek w próbce i*HOP, jak stft center)."""
    k = np.asarray(klucz, dtype=np.float64)
    h = HOP_ZBOCZA // 2
    kp = np.pad(k, (h, h + HOP_ZBOCZA))
    sk = np.cumsum(np.r_[0.0, kp])
    i = np.arange(n_ramek) * HOP_ZBOCZA
    return ((sk[i + 2 * h] - sk[i]) / (2 * h)).astype(np.float32)


def wytnij(x: np.ndarray, n: int = RAMKI_ZBOCZA) -> tuple[np.ndarray, int]:
    """Środkowe n ramek (dopełnione zerami, gdy krócej). Zwraca (wycinek, start)."""
    s0 = max(0, (x.shape[0] - n) // 2)
    w = x[s0:s0 + n]
    if w.shape[0] < n:
        w = np.concatenate([w, np.zeros((n - w.shape[0],) + w.shape[1:],
                                        dtype=w.dtype)])
    return w, s0


def klip_zboczy(rng: np.random.Generator, realism: bool = True):
    """(obraz [512, N_MELS], klucz [512] w 0..1, opis). Tempo PRAWDZIWE z
    TEMPO_ZAKRES — sieć zboczy pracuje przed normalizacją tempa."""
    if rng.random() > C.SILENCE_FRACTION:
        tekst = "".join(rng.choice(list(C.ALPHABET[1:]), ZNAKOW_W_KLIPIE))
    else:
        tekst = ""
    wpm = float(rng.uniform(*C.TEMPO_ZAKRES))
    audio, meta = radio.receive(tekst, rng, n_samples=C.CLIP_SAMPLES,
                                wpm=wpm, realism=realism, tempo_staly=True,
                                z_kluczem=bool(tekst), zakresy=ZAKRESY_ZBOCZY)
    obraz, s0 = wytnij(obraz_zboczy(audio))
    k = meta.get("klucz")
    if k is None:
        kl = np.zeros(RAMKI_ZBOCZA, dtype=np.float32)
    else:
        kl_all = klucz_ramek(k, 1 + audio.size // HOP_ZBOCZA)
        kl, _ = wytnij(kl_all[:, None])
        kl = kl[:, 0]
    meta["wpm"] = wpm
    return obraz, kl, meta


# --------------------------------------------------------------------------
# Zbocza z przebiegu i miara błędu ich położenia
# --------------------------------------------------------------------------
def zbocza(p: np.ndarray, prog: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Przebieg P(klucz) [ramki] -> (zbocza narastające, opadające) w ramkach,
    z interpolacją liniową przejścia przez próg (ułamek ramki)."""
    p = np.asarray(p, dtype=np.float64)
    nad = p >= prog
    i = np.flatnonzero(np.diff(nad.astype(np.int8)))
    t = []
    for k in i:
        a, b = p[k], p[k + 1]
        t.append(k + (prog - a) / (b - a) if b != a else k + 0.5)
    t = np.array(t)
    gora = nad[i + 1] if i.size else np.zeros(0, bool)
    return t[gora], t[~gora]


def blad_zboczy(p_pred: np.ndarray, p_true: np.ndarray, tol_ramek: float
                ) -> dict:
    """Dopasowanie zboczy przewidzianych do prawdziwych (każde prawdziwe do
    najbliższego przewidzianego tego samego rodzaju w odległości < tol).
    Zwraca błędy [ramki] trafionych, liczbę pominiętych i fałszywych."""
    out = {"bledy": [], "pominiete": 0, "falszywe": 0, "prawdziwych": 0}
    for tp, pp in zip(zbocza(p_true), zbocza(p_pred)):
        out["prawdziwych"] += tp.size
        uzyte = np.zeros(pp.size, bool)
        for x in tp:
            if pp.size:
                d = np.abs(pp - x)
                d[uzyte] = np.inf
                j = int(np.argmin(d))
                if d[j] < tol_ramek:
                    uzyte[j] = True
                    out["bledy"].append(float(pp[j] - x))
                    continue
            out["pominiete"] += 1
        out["falszywe"] += int((~uzyte).sum())
    return out


# --------------------------------------------------------------------------
# Sieć na długim nagraniu
# --------------------------------------------------------------------------
def przebieg_sieci(audio: np.ndarray, model, krok: int = RAMKI_ZBOCZA // 2
                   ) -> np.ndarray:
    """P(klucz) co 5 ms na całym nagraniu: okna 512 ramek na zakładkę
    (krok 256), z każdego brana środkowa połowa — brzegi okna sieć widzi
    tylko z jednej strony."""
    import tensorflow as tf
    obraz = obraz_zboczy(audio)
    n = obraz.shape[0]
    pad = (RAMKI_ZBOCZA - krok) // 2
    x = np.concatenate([np.zeros((pad, obraz.shape[1]), np.float32), obraz,
                        np.zeros((RAMKI_ZBOCZA, obraz.shape[1]), np.float32)])
    starty = list(range(0, n, krok))
    okna = np.stack([x[s:s + RAMKI_ZBOCZA] for s in starty])[..., None]
    logit = model.predict(okna, verbose=0, batch_size=32)
    p = tf.sigmoid(logit).numpy()[:, pad:pad + krok].reshape(-1)
    return p[:n]


# Histereza (17.10, pomysł operatora).
# mic1, 27-31 s: wyjście sieci po kropce N wisiało na 0,5-0,9 (z ogona
# powstał doklejony element: N -> D), w kresce Q spadło do 0,35 (kreska
# rozcięta: Q -> ?). Jeden próg 0,5 robił z każdego zachwiania zbocze.
PROG_WL = 0.6           # element zaczyna się powyżej
PROG_WYL = 0.3          # ... i kończy dopiero poniżej
# Odrzucanie krótkich impulsów (sklejanie odcinków < 0,5 u) sprawdzone
# 17.10 w dwóch wersjach i USUNIĘTE: zysku brak, a niszczy informację,
# którą grzebień naprawia sam (HISTORIA, „Histereza i odrzucanie...").


def odcinki_z_przebiegu(p: np.ndarray, prog: float | None = None,
                        histereza: tuple[float, float] | None = None
                        ) -> list[tuple[bool, float]]:
    """P(klucz) co 5 ms -> [(czy_znak, czas_s), ...] jak grzebien.odcinki_przebiegu.

    prog: jeden próg (domyślnie 0,5, gdy brak histerezy).
    histereza: (próg włączenia, próg wyłączenia)."""
    p = np.asarray(p, dtype=np.float64)
    if p.size < 20:
        return []
    if histereza is not None:
        wl, wyl = histereza
        on = np.zeros(p.size, bool)
        stan = bool(p[0] >= wl)
        for i, v in enumerate(p):
            if stan and v < wyl:
                stan = False
            elif not stan and v >= wl:
                stan = True
            on[i] = stan
    else:
        on = p >= (0.5 if prog is None else prog)
    zm = np.flatnonzero(np.diff(on.astype(np.int8))) + 1
    gr = np.r_[0, zm, on.size]
    d = HOP_ZBOCZA / C.SR
    return [(bool(on[a]), (b - a) * d) for a, b in zip(gr[:-1], gr[1:])]

