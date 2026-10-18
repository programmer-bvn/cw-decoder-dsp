"""GRZEBIEŃ Z CEPSTRUM + VITERBI (19.10) — odczyt bez bramki, z budżetu prawdopodobieństw.

Pomysł operatora: jednostka z cepstrum („fizyka", HISTORIA „Cepstrum
przebiegu kluczowania"), do przebiegu przykładany grzebień, w każdej komórce
prawdopodobieństwo zepsutego elementu, a odczytem jest NAJBARDZIEJ
PRAWDOPODOBNY ciąg zgodny z gramatyką Morse'a („najkorzystniejsza tabela
prawdopodobieństw"; jak maska przesuwana po zapisie F2F w kasowniku).

Wejście: P(klucz wciśnięty) co 5 ms z sieci zboczy (dsp/zbocza.py).
Model: ciąg odcinków (semi-Markow):

    KROPKA ~1 u, KRESKA ~3 u                      (elementy)
    PRZERWA_Z ~1 u   — w znaku
    PRZERWA_L ~3 u   — między literami (ręka: 2,5-10 u, HISTORIA 18.10)
    PRZERWA_S ~7 u   — między słowami
    CISZA     >= 10 u — między nadaniami (sama się powtarza)

Wynik odcinka = WAGA_EMISJI * suma log-prawdopodobieństw ramek (filtr
dopasowany do długości: przy 17 WPM 14 ramek na jednostkę) + kara za
długość (log-normalna wokół wzorca x u). Siatka „sprężysta": każdy odcinek
może mieć własną długość, więc tempo ręki może pływać. Viterbi wybiera
najlepszy podział całego przebiegu — bez bramki S/N: szum daje po prostu
ciszę, bo żaden podział na elementy nie wygrywa z nią budżetem.
"""

from __future__ import annotations

import numpy as np

from . import config as C
from . import zbocza as Z
from .grzebien import _ODWROTNY, NIEZNANY

DT = Z.HOP_ZBOCZA / C.SR                 # 5 ms
EPS = 1e-3                               # P obcięte do [EPS, 1-EPS]
WAGA_EMISJI = 0.3                        # ramki sąsiednie są skorelowane
KARA_DLUGOSCI = 4.0                      # waga kary log-normalnej

# (nazwa, czy_znak, wzorzec [u], sigma log, min [u], max [u])
ODCINKI = [
    ("K1", True, 1.0, 0.30, 0.4, 2.0),
    ("K3", True, 3.0, 0.25, 2.0, 5.0),
    ("PZ", False, 1.0, 0.35, 0.4, 2.0),
    ("PL", False, 3.5, 0.35, 2.0, 6.0),
    ("PS", False, 7.0, 0.30, 5.0, 12.0),
    ("CI", False, 15.0, 0.80, 10.0, 40.0),
]
NAZWY = [o[0] for o in ODCINKI]
# kto może być przed kim (gramatyka)
POPRZEDNICY = {
    "K1": ["PZ", "PL", "PS", "CI"], "K3": ["PZ", "PL", "PS", "CI"],
    "PZ": ["K1", "K3"], "PL": ["K1", "K3"], "PS": ["K1", "K3"],
    "CI": ["K1", "K3", "CI"],
}


# --------------------------------------------------------------------------
# Jednostka z cepstrum
# --------------------------------------------------------------------------
def jednostka_cepstrum(p: np.ndarray, okno_s: float = 8.0, krok_s: float = 4.0
                       ) -> tuple[float, np.ndarray]:
    """u [s] z cepstrum przebiegu: minimum sumy c(u) + c(3u) + 0,5 c(7u)
    (ujemne szpilki przy długościach odcinków), mediana po oknach.
    Zwraca (u, u w kolejnych oknach)."""
    n, k = int(okno_s / DT), int(krok_s / DT)
    q = np.arange(n) * DT
    uu = np.arange(0.025, 0.200, 0.001)
    wyn = []
    for s in range(0, max(1, p.size - n + 1), k):
        x = np.asarray(p[s:s + n], np.float64)
        if x.size < n // 2 or x.std() < 0.2:
            continue
        x = (x - x.mean()) * np.hanning(x.size)
        S = np.abs(np.fft.rfft(x, 4 * x.size)) ** 2
        c = np.fft.irfft(np.log(S + 1e-9 * S.max()))[:x.size]
        qx = q[:x.size]
        suma = np.interp(uu, qx, c) + np.interp(3 * uu, qx, c) + 0.5 * np.interp(7 * uu, qx, c)
        wyn.append(uu[np.argmin(suma)])
    wyn = np.array(wyn)
    return (float(np.median(wyn)) if wyn.size else float("nan")), wyn


# --------------------------------------------------------------------------
# Viterbi po odcinkach
# --------------------------------------------------------------------------
def dekoduj_najlepiej(p: np.ndarray, kandydaci: list[float]) -> tuple[str, float, float]:
    """Jednostka też jest parametrem: odczyt dla każdej kandydatki (cepstrum,
    grzebień, ...), wygrywa najlepszy wynik Viterbiego na ramkę.
    19.10: cepstrum przy S/N 20 dB dało raz 110 ms zamiast 67 ms (qso2, qso5
    z zestawu zaszumionego: 0 słów) — zgodne pomiary niczego nie zmieniają,
    niezgodne rozstrzyga dopasowanie do całego przebiegu. -> (tekst, u, wynik)."""
    naj = ("", float("nan"), -np.inf)
    for u in {round(float(x), 4) for x in kandydaci if np.isfinite(x) and x > 0}:
        tekst, _, wynik = dekoduj(p, u, z_wynikiem=True)
        if wynik > naj[2]:
            naj = (tekst, u, wynik)
    return naj


def dekoduj(p: np.ndarray, u: float, z_wynikiem: bool = False):
    """P co 5 ms + jednostka -> (tekst ze spacjami, odcinki [(nazwa, od, do)])
    [, wynik na ramkę]."""
    p = np.clip(np.asarray(p, np.float64), EPS, 1 - EPS)
    T = p.size
    cm = np.r_[0.0, np.cumsum(np.log(p))] * WAGA_EMISJI
    cg = np.r_[0.0, np.cumsum(np.log(1 - p))] * WAGA_EMISJI
    uf = u / DT                                   # ramek na jednostkę
    par = []
    for nazwa, znak, wz, sig, mn, mx in ODCINKI:
        d = np.arange(max(1, int(round(mn * uf))), max(2, int(round(mx * uf))) + 1)
        kara = -KARA_DLUGOSCI * (np.log(d / (wz * uf))) ** 2 / (2 * sig ** 2)
        par.append((znak, d, kara))
    S = len(ODCINKI)
    delta = np.full((T + 1, S), -np.inf)
    skad_d = np.zeros((T + 1, S), np.int32)
    skad_s = np.full((T + 1, S), -1, np.int8)
    idx = {n: i for i, n in enumerate(NAZWY)}
    poprz = [[idx[x] for x in POPRZEDNICY[n]] for n in NAZWY]
    start = np.zeros(S); start[[idx["K1"], idx["K3"]]] = -2.0   # można zacząć od czegokolwiek
    for t in range(1, T + 1):
        for s, (znak, d, kara) in enumerate(par):
            dd = d[d <= t]
            if not dd.size:
                continue
            a = t - dd
            em = (cm[t] - cm[a]) if znak else (cg[t] - cg[a])
            kd = kara[:dd.size]
            prev = delta[a][:, poprz[s]]                      # [len(dd), |poprz|]
            jp = np.argmax(prev, axis=1)
            best_prev = prev[np.arange(dd.size), jp]
            best_prev = np.where(a == 0, np.maximum(best_prev, start[s]), best_prev)
            wyn = best_prev + em + kd
            j = int(np.argmax(wyn))
            delta[t, s] = wyn[j]
            skad_d[t, s] = dd[j]
            skad_s[t, s] = -1 if (a[j] == 0 and best_prev[j] == start[s]) else poprz[s][jp[j]]
    # powrót
    s = int(np.argmax(delta[T])); t = T
    odc = []
    while t > 0 and s >= 0:
        d = int(skad_d[t, s]); odc.append((NAZWY[s], t - d, t))
        s_prev = int(skad_s[t, s]); t -= d; s = s_prev
    odc.reverse()
    # odcinki -> tekst
    tekst, kod = [], ""
    for nazwa, a, b in odc:
        if nazwa in ("K1", "K3"):
            kod += "." if nazwa == "K1" else "-"
        elif nazwa in ("PL", "PS", "CI"):
            if kod:
                tekst.append(_ODWROTNY.get(kod, NIEZNANY)); kod = ""
            if nazwa in ("PS", "CI") and tekst and tekst[-1] != " ":
                tekst.append(" ")
    if kod:
        tekst.append(_ODWROTNY.get(kod, NIEZNANY))
    if z_wynikiem:
        return "".join(tekst).strip(), odc, float(np.max(delta[T])) / max(1, T)
    return "".join(tekst).strip(), odc
