"""BLOKADA SZUMÓW (23.10) — pomysł operatora: „suwakiem ustawić minimalny poziom".

Objaw: w odczycie na żywo, w długiej ciszy między nadaniami, szum pasma
czytany jako kropki — „E EI EE I ESIE" (sesje 23.10 211147 i 214326).
Viterbi jest bez bramki (HISTORIA „Grzebień z cepstrum + Viterbi": bramka
S/N zabijała słabe sygnały, Viterbi czytał przy 0 dB), ale wzmocnienie
z okna samego szumu podnosi szum do poziomu, w którym sieć widzi elementy.

Blokada jak squelch w radiu: poziom tonu naszej (po przestrojeniu na
TONE_CENTER) względem podłogi szumu; gdzie przez TRZYMANIE_S w obie strony
nic nie wystaje ponad próg — P zastąpione ciszą. Próg 0 = blokada wyłączona.
"""

from __future__ import annotations

import numpy as np

from . import config as C
from . import melref
from . import zbocza as Z

N_FFT = 1024                    # 128 ms, prążek 7,8 Hz — jak obraz drobny sieci
PRAZKI = 1                      # ± prążków wokół TONE_CENTER (±8 Hz)
PODLOGA_PCT = 20                # podłoga szumu: percentyl poziomu w oknie
PODLOGA_S = 10.0
TRZYMANIE_S = 0.6               # otwarta tyle przed i po ostatnim elemencie ponad progiem
P_CISZA = 0.02


def poziom_tonu(y: np.ndarray) -> np.ndarray:
    """dB co ramkę sieci (krok HOP_ZBOCZA, ramka wyśrodkowana) w ±PRAZKI wokół TONE_CENTER."""
    p = melref.stft_power(np.asarray(y, np.float32), N_FFT, Z.HOP_ZBOCZA)
    k = int(round(C.TONE_CENTER / (C.SR / N_FFT)))
    return (10 * np.log10(p[:, k - PRAZKI:k + PRAZKI + 1].sum(1) + 1e-12)).astype(np.float32)


def podloga(poz: np.ndarray, okno_s: float = PODLOGA_S) -> np.ndarray:
    """Podłoga szumu co ramkę: percentyl w oknie okno_s (bieżąco, co 1 s)."""
    n, krok = int(okno_s / (Z.HOP_ZBOCZA / C.SR)), int(1.0 / (Z.HOP_ZBOCZA / C.SR))
    if poz.size <= n:
        return np.full(poz.size, np.percentile(poz, PODLOGA_PCT) if poz.size else 0.0, np.float32)
    srodki, wart = [], []
    for s in range(0, poz.size - n + 1, krok):
        srodki.append(s + n // 2)
        wart.append(np.percentile(poz[s:s + n], PODLOGA_PCT))
    return np.interp(np.arange(poz.size), srodki, wart).astype(np.float32)


def otwarta(poz: np.ndarray, prog_db: float, pod: np.ndarray | None = None) -> np.ndarray:
    """Ramki, w których blokada przepuszcza (bool)."""
    if prog_db <= 0 or not poz.size:
        return np.ones(poz.size, bool)
    pod = podloga(poz) if pod is None else pod
    nad = (poz - pod) >= prog_db
    h = int(TRZYMANIE_S / (Z.HOP_ZBOCZA / C.SR))
    # rozszerzenie o h w obie strony: suma bieżąca zamiast pętli
    c = np.r_[0, np.cumsum(nad)]
    i = np.arange(poz.size)
    return (c[np.minimum(i + h + 1, poz.size)] - c[np.maximum(i - h, 0)]) > 0


def zastosuj(p: np.ndarray, otw: np.ndarray) -> np.ndarray:
    return np.where(otw, p, P_CISZA).astype(np.float32)
