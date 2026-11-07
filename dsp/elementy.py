"""ETAP A klasyfikatora elementów (06.11, gałąź `elementy`): przegródka grzebienia
-> ton / dziura / śmieć.

POMYSŁ OPERATORA (06.11, rysunek „grzebień przesuwa się i szuka"). Grzebień
z zębami co jednostkę u jedzie po wodospadzie; sieć ocenia, co leży między
zębami. Dwa etapy: A — każda przegródka osobno („wzorce rozwalonych kropek
i dziur"), B — grupowanie (3 tony obok siebie = kreska, długość dziur =
rodzaj przerwy) z kotwicą i zatrzaśnięciem zegara. Ten moduł to A.

Dlaczego wodospad, a nie P z sieci zboczy: sieć zboczy widzi 34 ms i oddaje
jedną liczbę co 5 ms — sylaba mowy przy 750 Hz wygląda w niej jak wciśnięty
klucz, a P mowy i P litery R prawie się nie różnią (halucynacje zboczy_5/6,
HISTORIA 06.11). Na wodospadzie ton CW to pozioma grań na jednej wysokości,
mowa — pływające prążki; tę różnicę widać tylko w 2D (operator: „kolor jako
wysokość — przestrzennie traktować widmo").

WYCINEK. Czas pocięty grzebieniem: K kolumn na u, PRZEGRODEK przegródek
(środkowa oceniana, sąsiedzi jako kontekst). Częstotliwość: ton +/- PASMO_HZ
w WIERSZE rzędach (stała skala w Hz — odstęp harmonicznych mowy ma znaczenie).
Okno FFT dobrane do u (u/2, w granicach OKNO_S), żeby przy każdym tempie
kropka zajmowała tyle samo kolumn. Poziom: dB nad medianą całego wodospadu
klipu (mediana = szum, bo ton zajmuje jeden-dwa rzędy z WIERSZE).

ETYKIETA środkowej przegródki: TON, gdy klucz wciśnięty przez > połowę
przegródki (prawdziwy przebieg klucza z generatora); ŚMIEĆ, gdy nie, a w paśmie
wycinka jest negatyw (mowa, śpiew, nagranie z pokoju...) co najmniej na
poziomie szumu; inaczej DZIURA. Ułamek wciśnięcia zapisywany obok (miękka
etykieta, do sprawdzania przegródek na granicy elementu).

W treningu jednostka i ton „zmierzone" z błędem (BLAD_U, BLAD_TONU_HZ) —
w odczycie dają je cepstrum i track_tone, nie generator.
"""

from __future__ import annotations

import numpy as np

from . import config as C
from . import radio
from . import zbocza as Z

K = 4                         # kolumn na jednostkę
PRZEGRODEK = 3                # przegródek w wycinku (środkowa oceniana)
KOLUMNY = K * PRZEGRODEK      # 12
PASMO_HZ = 300.0              # ton +/- [Hz]
WIERSZE = 24                  # rzędy częstotliwości (25 Hz)
N_FFT = 1024
OKNO_S = (0.010, 0.040)       # granice długości okna FFT [s]
DB_ZAKRES = 40.0              # dB nad medianą -> [0, 1]
BLAD_U = 0.10
BLAD_TONU_HZ = 15.0
PROG_SMIECIA = 2.0             # 3 dB ponad tło

DZIURA, TON, SMIEC = 0, 1, 2
NAZWY = ("dziura", "ton", "śmieć")

ZAKRESY_ELEMENTOW = {**Z.ZAKRESY_ZBOCZY_6,
                     "NEGATYW": {**Z.ZAKRESY_ZBOCZY_6["NEGATYW"], "zwroc_audio": True}}
ZNAKOW = 6                    # znaków w klipie: przy 10 WPM 4 s mieści ~3-4


def jednostka(wpm: float) -> float:
    return 1.2 / wpm


def kolumny_czasu(n_probek: int, u: float, faza: float = 0.0, sr: int = C.SR) -> np.ndarray:
    """Środki kolumn [s] co u/K od fazy, tak żeby całe okno FFT mieściło się w klipie."""
    hop = u / K
    pol = okno(u) / 2
    t0 = faza % hop
    t = np.arange(t0, n_probek / sr, hop)
    return t[(t >= pol) & (t <= n_probek / sr - pol)]


def okno(u: float) -> float:
    return float(np.clip(u / 2, *OKNO_S))


def wodospad(audio: np.ndarray, ton_hz: float, u: float, t: np.ndarray,
             sr: int = C.SR) -> np.ndarray:
    """[WIERSZE, len(t)] moc w dB (bez normalizacji): kolumna = widmo okna
    Hanna długości okno(u) wyśrodkowanego w t, rzędy ton +/- PASMO_HZ."""
    L = int(round(okno(u) * sr))
    w = np.hanning(L).astype(np.float32)
    a = np.asarray(audio, np.float32)
    s = np.round(t * sr).astype(int) - L // 2
    s = np.clip(s, 0, a.size - L)
    ramki = a[s[:, None] + np.arange(L)[None, :]] * w
    P = np.abs(np.fft.rfft(ramki, N_FFT, axis=1)) ** 2 / (w ** 2).sum()
    f = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    fr = ton_hz + np.linspace(-PASMO_HZ, PASMO_HZ, WIERSZE)
    # Dla krótkich okien prążki FFT (7,8 Hz) są gęstsze niż rozdzielczość okna —
    # interpolacja wystarcza; przy długich (40 ms = 25 Hz) i tak rzędy co 25 Hz.
    out = np.stack([np.interp(fr, f, p) for p in P], axis=1)
    return 10 * np.log10(out + 1e-12)


def normalizuj(db: np.ndarray, mediana: float | None = None) -> np.ndarray:
    m = float(np.median(db)) if mediana is None else mediana
    return np.clip((db - m) / DB_ZAKRES, -0.25, 1.0).astype(np.float32)


def ulamek_klucza(klucz: np.ndarray, t: np.ndarray, u: float, sr: int = C.SR) -> np.ndarray:
    """Dla każdej kolumny j: średni klucz w przegródce [t_j - hop/2, t_j - hop/2 + u)
    — czyli w przegródce zaczynającej się na kolumnie j."""
    k = np.r_[0.0, np.cumsum(np.asarray(klucz, np.float64))]
    a = np.clip(np.round((t - u / K / 2) * sr).astype(int), 0, klucz.size)
    b = np.clip(np.round((t - u / K / 2 + u) * sr).astype(int), 0, klucz.size)
    return ((k[b] - k[a]) / np.maximum(b - a, 1)).astype(np.float32)


def smiec_przegrodek(x_neg: np.ndarray | None, neg_db: float, ton_hz: float,
                     t: np.ndarray, u: float, sr: int = C.SR) -> np.ndarray:
    """Dla każdej kolumny j: czy negatyw w przegródce od j, w paśmie wycinka,
    wystaje ponad tło (szum + stała część negatywu) o PROG_SMIECIA.
    Poziom negatywu był ustawiony jako neg_db względem szumu w 400-1200 Hz.
    Przybliżenie: szum traktowany jako płaski w paśmie."""
    if x_neg is None:
        return np.zeros(t.size, bool)
    db = wodospad(x_neg, ton_hz, u, t, sr)                 # [W, T]
    moc = (10 ** (db / 10)).mean(axis=0)                   # średnia po rzędach
    w = np.convolve(moc, np.ones(K) / K)[K - 1:K - 1 + t.size]   # przegródka od j
    # Odniesienie: średnia gęstość negatywu w środku pasma sieci (800 +/- 300 Hz,
    # cały klip) — tym poziomem negatyw = szum + neg_db. Względna gęstość
    # w przegródce + neg_db = przegródka względem szumu.
    db_ref = wodospad(x_neg, (C.FMIN + C.FMAX) / 2, u, t, sr)
    ref = float((10 ** (db_ref / 10)).mean())
    r = w / max(ref, 1e-30) * 10 ** (neg_db / 10)          # przegródka / szum
    # Tło = szum + to, co negatyw daje stale (mediana): szum pokoju na +5 dB
    # po normalizacji wodospadu wygląda jak głośniejszy szum, nie jak obiekt
    # (podgląd 06.11: bank +5 dB cały „śmieć", choć nic nie widać).
    return r > PROG_SMIECIA * (1.0 + float(np.median(r)))


def klip(rng: np.random.Generator, realism: bool = True):
    """Jeden klip -> (wodospad dB [W, T], etykiety [T] (klasa przegródki od
    kolumny j), ułamki [T], opis). Siatka z jednostki i tonu „zmierzonych"."""
    zak = ZAKRESY_ELEMENTOW
    tekst = ("".join(rng.choice(list(C.ALPHABET[1:]), ZNAKOW))
             if rng.random() > zak["PUSTE"] else "")
    wpm = float(rng.uniform(*C.TEMPO_ZAKRES))
    audio, meta = radio.receive(tekst, rng, n_samples=C.CLIP_SAMPLES, wpm=wpm,
                                realism=realism, tempo_staly=True,
                                z_kluczem=bool(tekst), zakresy=zak)
    u_pr = jednostka(wpm)
    u = u_pr * (1 + float(rng.uniform(-BLAD_U, BLAD_U)))
    ton_pr = meta["tone"] if tekst else C.TONE_CENTER
    ton = ton_pr + float(rng.uniform(-BLAD_TONU_HZ, BLAD_TONU_HZ))
    t = kolumny_czasu(audio.size, u, faza=float(rng.uniform(0, u)))
    db = wodospad(audio, ton, u, t)
    kl = meta.get("klucz")
    ul = (ulamek_klucza(kl, t, u) if kl is not None else np.zeros(t.size, np.float32))
    sm = smiec_przegrodek(meta.get("negatyw_audio"), meta.get("negatyw_db", 0.0), ton, t, u)
    et = np.where(ul > 0.5, TON, np.where(sm, SMIEC, DZIURA)).astype(np.uint8)
    meta.pop("negatyw_audio", None)
    meta.pop("klucz", None)
    meta.update(wpm=wpm, u=u, u_prawdziwe=u_pr, ton_siatki=ton, tekst=tekst)
    return db, et, ul, meta


def wycinki(db: np.ndarray, et: np.ndarray, ul: np.ndarray, krok: int = 1):
    """Wodospad klipu -> (X [n, W, KOLUMNY], y [n], ułamek [n]): przegródka
    oceniana zaczyna się w kolumnie K (po jednej przegródce kontekstu z lewej)."""
    x = normalizuj(db)
    T = x.shape[1]
    j = np.arange(0, T - KOLUMNY + 1, krok)
    X = np.stack([x[:, i:i + KOLUMNY] for i in j])
    return X, et[j + K], ul[j + K]


# --------------------------------------------------------------------------
# Łączenie z siecią zboczy (07.11, pomysł operatora: „jak centrum badawcze —
# dwie metody, kaskada filtrów"). Pomiar elementy_2 (HISTORIA): zbocza_6
# trafia zbocze do 0,2 ms, ale na pokoju daje 63 elementy/min; przegródki
# 1,0 ms i 21/min. Dwa sposoby, oba do pomiaru:
#   iloczyn — niezależne świadectwa: suma log-szans (hipoteza: błędy obu sieci
#             są skorelowane, bo to ten sam dźwięk — stąd wagi do strojenia);
#   kaskada — zbocza wyzwalają, przegródki potwierdzają: odcinek z P_zboczy
#             > 0,5 zostaje, gdy średnie P(ton) przegródek na nim >= PROG;
#             inaczej P_zboczy tłumione do min(P_zboczy, P_przegródek).
# --------------------------------------------------------------------------
EPS_P = 1e-3


def _logit(p):
    p = np.clip(np.asarray(p, np.float64), EPS_P, 1 - EPS_P)
    return np.log(p / (1 - p))


def polacz_iloczyn(pz, pe, wz=1.0, we=1.0):
    return (1 / (1 + np.exp(-(wz * _logit(pz) + we * _logit(pe))))).astype(np.float32)


def polacz_kaskada(pz, pe, prog=0.5):
    pz = np.asarray(pz, np.float32)
    out = pz.copy()
    w = np.r_[False, pz > 0.5, False]
    d = np.diff(w.astype(np.int8))
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if float(np.mean(pe[a:b])) < prog:
            out[a:b] = np.minimum(pz[a:b], pe[a:b])
    return out
