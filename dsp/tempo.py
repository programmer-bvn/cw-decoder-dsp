"""POMIAR TEMPA z bufora elementów — pod normalizację tempa (16.10).

PO CO. Okno sieci ma stałą długość w SEKUNDACH (2,56 s), a znak — w
JEDNOSTKACH. Przy 12 WPM w oknie mieszczą się ~2 znaki i ~40% kroków ze
znakiem to znaki ucięte krawędzią; przy 30 WPM kropka jest krótsza niż
krok wyjścia fcn (HISTORIA „Wolne tempo", „Maska uciętych"). Operator
(16.10): „do bufora nabierać ileś elementów, z tego szacować tempo
i normalizować do tempa standardowego, tak jak się normalizuje amplitudę".
Ten moduł to pierwszy krok: sam pomiar, bez zmiany treningu.

JAK. Obwiednia w wąskim paśmie przy tonie (po przestrojeniu na 750 Hz),
bloki 5 ms, próg Otsu w dB -> odcinki znak / przerwa. Szukana jest jedna
jednostka u i „waga" b, dla których

    znak    = k * u + b,  k in {1, 3}
    przerwa = k * u - b,  k in {1, 3, 7}

Waga b to wydłużenie znaku kosztem przerwy przez pogłos, filtr p.cz.
i ARW (mic3 z pamięci keyera: kropka 75 ms, przerwa w znaku 45 ms przy
u = 60 ms). Bez niej znaki dawałyby u za duże, a przerwy — za małe.
Błąd liczony WZGLĘDNIE (do k * u), żeby kreska nie ważyła 9 razy więcej
niż kropka. Siatka u: 30-140 ms (8,6-40 WPM).
"""

from __future__ import annotations

import numpy as np

from . import config as C

BLOK_S = 0.005
U_SIATKA = np.arange(0.030, 0.1405, 0.001)       # jednostka [s]
# Waga / u. Do 16.10 wieczór +/-0,4 — mikrofon (pogłos pokoju) dobijał do
# +0,40, czyli granicy siatki.
B_UDZIAL = np.linspace(-0.7, 0.7, 57)
MIN_ELEMENTOW = 8


def _prog_otsu(v: np.ndarray) -> float:
    h, e = np.histogram(v, bins=64)
    c = 0.5 * (e[:-1] + e[1:])
    w = np.cumsum(h)
    s = np.cumsum(h * c)
    w1 = w[-1] - w
    ok = (w > 0) & (w1 > 0)
    m0 = np.where(ok, s / np.maximum(w, 1), 0.0)
    m1 = np.where(ok, (s[-1] - s) / np.maximum(w1, 1), 0.0)
    return float(c[int(np.argmax(np.where(ok, w * w1 * (m0 - m1) ** 2, -1)))])


def odcinki(audio: np.ndarray, ton_hz: float = C.TONE_CENTER,
            sr: int = C.SR, wygladzenie_s: float = 0.0
            ) -> tuple[np.ndarray, np.ndarray]:
    """(długości znaków [s], długości przerw [s]) — bez skrajnych odcinków,
    bo bufor tnie je w przypadkowym miejscu. wygladzenie_s: średnia krocząca
    mocy przed progiem (opis w zmierz_tempo)."""
    from . import tune
    x = tune.bandpass(np.asarray(audio, dtype=np.float32), ton_hz, 50.0, sr=sr)
    h = int(round(BLOK_S * sr))
    n = x.size // h
    if n < 20:
        return np.zeros(0), np.zeros(0)
    p = (x[:n * h].astype(np.float64) ** 2).reshape(n, h).mean(axis=1)
    k = int(round(wygladzenie_s / BLOK_S))
    if k > 1:
        p = np.convolve(p, np.ones(k) / k, mode="same")
    db = 10.0 * np.log10(p + 1e-12)
    on = db > _prog_otsu(db)
    zm = np.flatnonzero(np.diff(on.astype(np.int8))) + 1
    gr = np.r_[0, zm, n]
    run = [(bool(on[a]), (b - a) * BLOK_S) for a, b in zip(gr[:-1], gr[1:])]
    run = run[1:-1]
    znaki = np.array([d for m, d in run if m])
    przerwy = np.array([d for m, d in run if not m])
    # Przerwy dłuższe niż słowo przy 9 WPM to brak nadawania, nie przerwa
    # w tekście — nie mówią nic o tempie.
    przerwy = przerwy[przerwy < 7 * U_SIATKA[-1] * 1.3]
    return znaki, przerwy


def dopasuj(znaki: np.ndarray, przerwy: np.ndarray) -> dict:
    """Jednostka i waga najlepiej tłumaczące odcinki (siatka u x b)."""
    wynik = {"wpm": float("nan"), "u_s": float("nan"), "waga": float("nan"),
             "blad": float("nan"), "n": int(znaki.size + przerwy.size)}
    if znaki.size + przerwy.size < MIN_ELEMENTOW or znaki.size < 3:
        return wynik
    km = np.array([1.0, 3.0])
    kg = np.array([1.0, 3.0, 7.0])
    najl = (np.inf, None, None)
    for u in U_SIATKA:
        b = B_UDZIAL * u                                    # [B]
        # [element, k, b] — cała siatka wagi naraz
        em = ((znaki[:, None, None] - (km[None, :, None] * u + b))
              / (km[None, :, None] * u)) ** 2
        eg = ((przerwy[:, None, None] - (kg[None, :, None] * u - b))
              / (kg[None, :, None] * u)) ** 2
        e = (em.min(axis=1).sum(axis=0) + eg.min(axis=1).sum(axis=0)) / (
            znaki.size + przerwy.size)                       # [B]
        i = int(np.argmin(e))
        if e[i] < najl[0]:
            najl = (float(e[i]), float(u), float(b[i]))
    e, u, b = najl
    wynik.update(wpm=float(1.2 / u), u_s=float(u), waga=float(b / u),
                 blad=float(np.sqrt(e)))
    return wynik


def zmierz_tempo(audio: np.ndarray, ton_hz: float = C.TONE_CENTER,
                 sr: int = C.SR) -> dict:
    """Tempo [WPM] z całego bufora audio z jedną stacją przy ton_hz.

    DWA PRZEBIEGI. Przy S/N 5 dB szum dzieli długi znak na krótkie kawałki,
    które wyglądają jak szybkie elementy: jeden przebieg zawyżał tempo przy
    10-15 WPM o 40-60% (walidacja 16.10). Pierwszy przebieg daje
    przybliżoną jednostkę u1, drugi mierzy na mocy wygładzonej oknem u1/2
    — krótszym niż najkrótszy element, więc nie zlewa kropek, a dłuższym
    niż migotanie szumu.
    """
    # Pierwszy przebieg przy kilku wygładzeniach: na nagraniach z radia
    # (trzaski, ARW, migotanie szumu) 10 ms nie wystarczało — pomiar dobijał
    # do u = 30 ms (40 WPM) w większości okien. Wynik na granicy siatki
    # odpada; z pozostałych wygrywa najmniejszy błąd dopasowania.
    kandydaci = []
    for wyg in (0.010, 0.020, 0.030, 0.045):
        r1 = dopasuj(*odcinki(audio, ton_hz, sr, wygladzenie_s=wyg))
        if not np.isfinite(r1["u_s"]):
            continue
        r2 = dopasuj(*odcinki(audio, ton_hz, sr, wygladzenie_s=0.5 * r1["u_s"]))
        r = r2 if np.isfinite(r2["u_s"]) else r1
        if U_SIATKA[0] + 1e-4 < r["u_s"] < U_SIATKA[-1] - 1e-4:
            kandydaci.append(r)
    if not kandydaci:
        return dopasuj(np.zeros(0), np.zeros(0))
    return min(kandydaci, key=lambda r: r["blad"])


def tempo_w_czasie(audio: np.ndarray, okno_s: float = 8.0,
                   krok_s: float = 2.0, ton_hz: float = C.TONE_CENTER,
                   sr: int = C.SR) -> list[tuple[float, dict]]:
    """Pomiar w przesuwanym buforze: [(środek okna [s], wynik), ...]."""
    n, k = int(okno_s * sr), int(krok_s * sr)
    out = []
    for s in range(0, max(1, audio.size - n + 1), k):
        out.append(((s + n / 2) / sr, zmierz_tempo(audio[s:s + n], ton_hz, sr)))
    return out
