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
# POGLOS (17.10, zboczy_3): ogon po elemencie zmierzony na mikrofonie
# (tools/ogon.py: -26...-29 dB przy 10 ms, ~10 dB na 50 ms, garb 15-30 ms).
ZAKRESY_ZBOCZY = {"SN_DB": (-3.0, 45.0), "TLO_DB": (-32.0, 7.0),
                  "KEY_RAMP_MS": (2.0, 8.0),
                  "POGLOS": {"prob": 0.4, "L_db": (-35.0, -18.0),
                             "tau_ms": (20.0, 80.0), "pre_ms": (0.0, 15.0)},
                  # QRM (19.10, zboczy_4): test QRM — obca stacja 60 Hz obok,
                  # nawet 6 dB słabsza, zabierała wszystkie słowa (65/257).
                  "QRM": {"prob": 0.6, "blisko": 0.5, "df_hz": (30.0, 200.0),
                          "poziom_db": (-10.0, 12.0)}}


# ZBOCZY_5 (20.10): drugie wejście — obraz DROBNY wokół naszej stacji.
# Pasmo 40 m w tłoku: kłąb stacji 680-740 Hz (20-80 Hz obok) — pasma mel
# ~48 Hz przy 750 Hz zlewały je z naszą (P > 0,5 przez 70% czasu, 86 błędów
# na 105 znaków). Okno 128 ms: 7,8 Hz na prążek, 32 prążki = 625-867 Hz;
# po tune.retune nasza stacja jest zawsze przy 750 Hz. Rozmyte w czasie —
# zbocza daje dalej obraz mel (32 ms), częstotliwość ten.
N_FFT_DROBNY = 1024
DROBNY_PRAZKI = 32
DROBNY_PIERWSZY = int(round((C.TONE_CENTER - 125.0) / (C.SR / N_FFT_DROBNY)))   # prążek 80 = 625 Hz
# Ten sam ton — ten sam odcień w obu obrazach. Zmierzone 20.10 (ton 740-760 Hz,
# amplituda 0,01-0,5): mel - drobny = -26,4...-26,5 dB (okno 4x dłuższe
# i wagi filtrów mel). Szum biały: -20,7 dB — tło w drobnym 5,8 dB ciemniejsze,
# bo prążek jest węższy: to jest właśnie zysk z tego obrazu.
DROBNY_KOREKTA_DB = -26.4
ZAKRESY_ZBOCZY_5 = {**{k: v for k, v in ZAKRESY_ZBOCZY.items() if k != "QRM"},
                    "TON_SPREAD_HZ": 10.0,
                    "QRM": {"prob": 0.35, "blisko": 0.6, "df_hz": (15.0, 200.0),
                            "poziom_db": (-10.0, 6.0)},
                    "TLOK": {"prob": 0.3, "liczba": (2, 4), "df_hz": (15.0, 200.0),
                             "poziom_db": (-8.0, 4.0)}}

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


def obraz_drobny(audio: np.ndarray) -> np.ndarray:
    """audio -> [ramki co 5 ms, DROBNY_PRAZKI] w [0,1]: 625-867 Hz co 7,8 Hz.
    Te same ramki co obraz_zboczy (ten sam krok, ramka wyśrodkowana)."""
    p = melref.stft_power(np.asarray(audio, dtype=np.float32), N_FFT_DROBNY, HOP_ZBOCZA)
    p = p[:, DROBNY_PIERWSZY:DROBNY_PIERWSZY + DROBNY_PRAZKI]
    return frontend.normalize_db(melref.power_to_db(p) + DROBNY_KOREKTA_DB).astype(np.float32)


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


def klip_zboczy_5(rng: np.random.Generator, realism: bool = True):
    """Jak klip_zboczy, z zakresami ZAKRESY_ZBOCZY_5 i obrazem drobnym:
    (obraz [512, N_MELS], drobny [512, DROBNY_PRAZKI], klucz [512], opis)."""
    if rng.random() > C.SILENCE_FRACTION:
        tekst = "".join(rng.choice(list(C.ALPHABET[1:]), ZNAKOW_W_KLIPIE))
    else:
        tekst = ""
    wpm = float(rng.uniform(*C.TEMPO_ZAKRES))
    audio, meta = radio.receive(tekst, rng, n_samples=C.CLIP_SAMPLES,
                                wpm=wpm, realism=realism, tempo_staly=True,
                                z_kluczem=bool(tekst), zakresy=ZAKRESY_ZBOCZY_5)
    obraz, s0 = wytnij(obraz_zboczy(audio))
    drobny, _ = wytnij(obraz_drobny(audio))
    k = meta.get("klucz")
    if k is None:
        kl = np.zeros(RAMKI_ZBOCZA, dtype=np.float32)
    else:
        kl_all = klucz_ramek(k, 1 + audio.size // HOP_ZBOCZA)
        kl, _ = wytnij(kl_all[:, None])
        kl = kl[:, 0]
    meta["wpm"] = wpm
    return obraz, drobny, kl, meta


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
    tylko z jednej strony. Sieć z dwoma wejściami (zboczy_5) dostaje też
    obraz drobny."""
    drobne = [obraz_drobny(audio)] if len(model.inputs) == 2 else None
    return przebieg_z_obrazow([obraz_zboczy(audio)], model, krok, drobne)[0]


def _okna(obrazy: list, krok: int):
    pad = (RAMKI_ZBOCZA - krok) // 2
    okna, gdzie = [], []
    for obraz in obrazy:
        n = obraz.shape[0]
        x = np.concatenate([np.zeros((pad, obraz.shape[1]), np.float32),
                            np.asarray(obraz, np.float32),
                            np.zeros((RAMKI_ZBOCZA, obraz.shape[1]), np.float32)])
        st = list(range(0, n, krok))
        gdzie.append((len(okna), len(st), n))
        okna.extend(x[s:s + RAMKI_ZBOCZA] for s in st)
    return np.stack(okna)[..., None], gdzie


_FUNKCJE: dict = {}


def _logit(model, wejscie, wsad: int = 64) -> np.ndarray:
    """Sieć na oknach, wsadami po `wsad`, przez JEDNĄ skompilowaną funkcję.

    model.predict kompilował sieć od nowa dla każdego nowego rozmiaru
    ostatniego, niepełnego wsadu — odczyt na żywo (21.10): bufor rośnie
    do 20 s, liczba okien co krok inna, 15-27 s na krok przez pierwsze
    kroki i na końcu nagrania. Tu wymiar wsadu jest dowolny (None).
    Wsad 64: przy 256 przydzielacz karty zgłaszał brak pamięci (19.10).
    """
    import tensorflow as tf
    f = _FUNKCJE.get(id(model))
    if f is None:
        spec = [tf.TensorSpec((None,) + tuple(i.shape[1:]), tf.float32) for i in model.inputs]
        f = tf.function(lambda *x: model(list(x) if len(x) > 1 else x[0], training=False),
                        input_signature=spec)
        _FUNKCJE[id(model)] = f
    wej = wejscie if isinstance(wejscie, list) else [wejscie]
    n = wej[0].shape[0]
    return np.concatenate([f(*[tf.constant(w[s:s + wsad], tf.float32) for w in wej]).numpy()
                           for s in range(0, n, wsad)])


def przebieg_z_obrazow(obrazy: list, model, krok: int = RAMKI_ZBOCZA // 2,
                       drobne: list | None = None) -> list:
    """Jak przebieg_sieci, ale dla wielu obrazów naraz — jedno wywołanie
    sieci na wszystkie okna (zbiór drugiego etapu: setki tysięcy klipów;
    pojedyncze wywołania kosztowałyby ~30 ms narzutu każde)."""
    import tensorflow as tf
    pad = (RAMKI_ZBOCZA - krok) // 2
    okna, gdzie = _okna(obrazy, krok)
    wejscie = [okna, _okna(drobne, krok)[0]] if drobne is not None else okna
    logit = _logit(model, wejscie)
    p = tf.sigmoid(logit).numpy()[:, pad:pad + krok]
    return [p[i0:i0 + k].reshape(-1)[:n] for i0, k, n in gdzie]



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

