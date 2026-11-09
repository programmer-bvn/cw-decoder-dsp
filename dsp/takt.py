"""TAKT Z CEPSTRUM — „okienka" na cepstrum przebiegu P (08.11, pomysł operatora).

Skąd: na stałej nośnej z IC-7300 dekoder pokazywał 24–33 WPM i wypisywał
„E E E…”. Operator: „grzebień powinien na nośnej pokazać tempo nieskończenie
małe — nic się nie zmienia, to co mierzyć”; poszarpana nośna ma dawać
w cepstrum szum, a skoro taktu nie da się fizycznie wyznaczyć, to prędkości
nie znamy. Próg na poziomie tonu (kontrast) odpada: słaba stacja też ma mały
kontrast, a rytm ma ten sam.

Fizyka: telegrafia to odcinki długości u (kropka, odstęp w znaku), 3u
(kreska, odstęp między literami), 7u (między słowami). Prostokąty takich
długości dają w cepstrum przebiegu P ujemne szpilki przy u, 3u, 7u
(dsp.viterbi.jednostka_cepstrum bierze minimum sumy tych trzech miejsc — ale
ZAWSZE jakieś minimum jest, także w samym szumie).

Tu: dla każdej kandydatki u — wąskie okienko wokół k·u (±TOL) i tło tuż obok
(od ±TOL do ±TLO). Dołek w jednostkach rozrzutu tła (z, mediana/MAD) osobno
dla k = 1, 3, 7. Takt w oknie = u z największym min(z1, z3): dołek musi być
w OBU okienkach naraz. To odsiewa i szum, i pomyłki o czynnik 3 (u' = 3u
potrzebowałoby dołka przy 9u, u' = u/3 — przy u/3). 7u tylko informacyjnie:
w 8 s nie zawsze wypada odstęp między słowami.

Dwie grupy (09.11, zgłoszenia SQ2MMX): ręka i bug nie trzymają siatki k·u —
QSO EA4BB z DM6 (FTDX10, 14 MHz): kropki 20–60 ms, kreski 100–160, kreska
~2,5 kropki. Dołków przy u i 3u (±6 %) nie ma, okna z ≥ 4 — 5 %, bramka
zostawiała 47 słów z 354. Szersze okienko przy 3u (±20–30 %) odzyskiwało
mało (79–94), a na negatywach puszczało AM 13–16 zn./min. Druga miara:
długości znaków i osobno przerw (do 0,6 s) w oknie dzielą się na dwie grupy
(log, najlepszy podział, eta = wariancja międzygrupowa / całkowita), grupy
w stosunku 2–4,5. Okno ma takt, gdy z ≥ próg ALBO eta ≥ GRUPY_ETA. Pomiar
(zbocza_5, podtrzymanie 12 s): 1458 -> 1459 słów z 1675; zn./min nośna 0,4,
pokój 1,3, AM 4,4 -> 4,7, SSB 2,0; raport 2: 47 -> 202 słowa, raport 1:
143 -> 251 (bez bramki 354 i 329, ale z 32-38 % pojedynczych E/T).
Same dwie grupy (bez cepstrum): negatywy 0,0–0,2, ale 1419 słów — gubią
część klucza elektronicznego; stąd LUB.
"""

from __future__ import annotations

import numpy as np

from dsp.viterbi import DT

OKNO_S, KROK_S = 8.0, 4.0
U_OD, U_DO = 0.025, 0.200          # jak jednostka_cepstrum: 6–48 WPM
TOL = 0.06                         # okienko dołka ±6 % (trening sieci: u ±10 %)
TLO = 0.35                         # tło od ±TOL do ±TLO wokół k·u
MIN_STD = 0.02                     # płaskie P (cisza) — nie ma czego mierzyć
GRUPY_ETA = 0.8                    # 0,7: negatywy AM 8,3 zn./min (09.11); 0,8: 4,7
GRUPY_STOSUNEK = (2.0, 4.5)        # kreska/kropka, odstęp liter/odstęp w znaku
GRUPY_MIN = 8                      # mniej odcinków w oknie — nie ma czego dzielić
GRUPY_PRZERWA_MAX_S = 0.6          # dłuższe to słowa i cisza, nie u/3u


def cepstrum(x: np.ndarray) -> np.ndarray:
    """Rzeczywiste cepstrum okna P (jak w jednostka_cepstrum): c[i] przy q = i·DT."""
    x = np.asarray(x, np.float64)
    x = (x - x.mean()) * np.hanning(x.size)
    S = np.abs(np.fft.rfft(x, 4 * x.size)) ** 2
    return np.fft.irfft(np.log(S + 1e-9 * S.max()))[:x.size]


def _z(c: np.ndarray, q0: float) -> float:
    """Dołek przy q0 względem tła obok, w jednostkach rozrzutu tła."""
    i = np.arange(c.size) * DT
    tol = max(TOL * q0, 2 * DT)
    w = np.abs(i - q0) <= tol
    t = (np.abs(i - q0) > tol) & (np.abs(i - q0) <= TLO * q0)
    if not w.any() or t.sum() < 6:
        return float("nan")
    tlo = c[t]
    med = np.median(tlo)
    rozrzut = 1.4826 * np.median(np.abs(tlo - med)) + 1e-9
    return float((med - c[w].min()) / rozrzut)


def _eta(d: np.ndarray) -> float:
    """Najlepszy podział log(długości) na dwie grupy -> eta; 0, gdy za mało
    odcinków albo stosunek grup poza GRUPY_STOSUNEK."""
    if d.size < GRUPY_MIN:
        return 0.0
    l = np.sort(np.log(d))
    if l.var() <= 0:
        return 0.0
    eta, st = 0.0, 0.0
    for i in range(2, l.size - 1):
        a, b = l[:i], l[i:]
        e = (a.size * b.size / l.size ** 2) * (b.mean() - a.mean()) ** 2 / l.var()
        if e > eta:
            eta, st = e, float(np.exp(b.mean() - a.mean()))
    return float(eta) if GRUPY_STOSUNEK[0] <= st <= GRUPY_STOSUNEK[1] else 0.0


def dwie_grupy(p: np.ndarray) -> float:
    """eta okna P: mniejsza z (znaki, przerwy). Odcinki jak w odczycie (histereza)."""
    from dsp import zbocza as Z
    oo = Z.odcinki_z_przebiegu(np.asarray(p, np.float32), histereza=(Z.PROG_WL, Z.PROG_WYL))
    zn = np.array([d for j, d in oo if j and d >= 0.01])
    pr = np.array([d for j, d in oo if not j and 0.01 <= d <= GRUPY_PRZERWA_MAX_S])
    return min(_eta(zn), _eta(pr))


def ma_takt(o: dict, prog_z: float) -> bool:
    """Okno z wyraźnym taktem: dołki cepstrum (klucz elektroniczny) albo dwie grupy (ręka)."""
    return (np.isfinite(o["z"]) and o["z"] >= prog_z) or o.get("eta", 0.0) >= GRUPY_ETA


def okno(p: np.ndarray, uu: np.ndarray | None = None) -> dict:
    """Jedno okno P -> {'u', 'z' (min z1,z3 przy u), 'z1', 'z3', 'z7', 'eta'};
    u = nan, gdy P płaskie albo okno za krótkie na 3u z tłem."""
    nic = dict(u=float("nan"), z=float("nan"), z1=float("nan"), z3=float("nan"), z7=float("nan"), eta=0.0)
    p = np.asarray(p, np.float64)
    if p.std() < MIN_STD:
        return nic
    nic["eta"] = dwie_grupy(p)
    c = cepstrum(p)
    if uu is None:
        uu = np.arange(U_OD, U_DO, 0.001)
    uu = uu[3 * uu * (1 + TLO) < c.size * DT]
    if not uu.size:
        return nic
    z1 = np.array([_z(c, u) for u in uu])
    z3 = np.array([_z(c, 3 * u) for u in uu])
    s = np.minimum(z1, z3)
    if not np.isfinite(s).any():
        return nic
    i = int(np.nanargmax(s))
    z7 = _z(c, 7 * uu[i]) if 7 * uu[i] * (1 + TLO) < c.size * DT else float("nan")
    return dict(u=float(uu[i]), z=float(s[i]), z1=float(z1[i]), z3=float(z3[i]), z7=z7, eta=nic["eta"])


def okna(p: np.ndarray, okno_s: float = OKNO_S, krok_s: float = KROK_S) -> list[dict]:
    """Okna jak w jednostka_cepstrum (8 s co 4 s); 't' — początek okna [s]."""
    n, k = int(okno_s / DT), int(krok_s / DT)
    wyn = []
    for s in range(0, max(1, p.size - n + 1), k):
        x = p[s:s + n]
        if x.size < n // 2:
            continue
        wyn.append({"t": s * DT, **okno(x)})
    return wyn
