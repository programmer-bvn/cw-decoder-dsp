"""PRZECIWSYGNAŁ (19.10): odtworzenie obcej stacji i odjęcie jej od nagrania.

Pomysł operatora: „generowanie przeciwsygnału". Test QRM (tools/test_qrm,
sufit bez obcej stacji 145/257 — nagrania ucięte do 40 s), cała droga:
bez niczego 75, notch ±10 / ±20 / ±30 Hz na obcej 115 / 131 / 138,
przeciwsygnał z rzutem ±40 Hz 138, ±80 Hz z wyciętą naszą stacją **142**
(częstotliwości z opisu nagrania); automatycznie (obce stacje z widma,
nasza z pętli tonu) 104 — straty tylko tam, gdzie pętla wybrała obcą
stację (7/16), w pozostałych 9 tyle co z opisu. Eksperyment: scratchpad
qrm_ic.py, HISTORIA 19.10.

Dlaczego nie sam notch: sieć zboczy myli się od obcej stacji w DOWOLNYM
miejscu wejścia (120-200 Hz obok, w paśmie ±50 Hz naszej SIR 15-45 dB,
a słowa i tak giną: 120 Hz +12 dB 4/27 przy suficie 11). Notch wycina tylko
środek obcej, jej zbocza (wstęgi boczne) zostają; szerszy notch zjada naszą.
Przeciwsygnał odtwarza obcą Z WSTĘGAMI (±80 Hz), a pasmo naszej stacji
z estymaty wycina.

Odrzucone w eksperymencie: kształt z samej maski kluczowania (111 — błąd
zboczy), wyuczony wzorzec zbocza obcej (kliknięcia; bez zysku, w paśmie
naszej gorzej).
"""

from __future__ import annotations

import numpy as np

from . import config as C

ODSTEP_HZ = 25.0          # obca stacja: kluczowany szczyt dalej niż tyle od naszej
B_KLUCZ_HZ = 25.0         # obwiednia do wyznaczenia kluczowania obcej
B_RZUT_HZ = 80.0          # pasmo odtwarzanej obcej (z wstęgami zboczy): 40 -> 137, 80 -> 142, 150 gorzej w SIR
B_NASZA_HZ = 20.0         # pasmo naszej stacji wycinane z estymaty obcej
OKNO_S = 0.4              # uśrednianie amplitudy i fazy obcej; nasza (df >= 25 Hz) wiruje i znika
# Kandydat kluczowany RAZEM z naszą (korelacja obwiedni ±15 Hz) to nie obca
# stacja, tylko nasz produkt: ręka 19.10, 420 i 1100 Hz przy tonie 760: +0,68
# (tttt 0,31-0,40); prawdziwe obce w teście QRM: -0,02...+0,33.
KORELACJA_MAX = 0.5
# Spójność nośnej: |Σ z·e^{-j2πδt}| / Σ|z| w chwilach wciśnięcia, najlepsze δ
# w ±3 Hz. Pierwsza wersja bez tego (regresja 19.10): QRM 75 -> 104, ale
# reszta -14 słów — kasowała stałe garby banku szumów (521, 607, 689 Hz:
# spójność 0,05-0,11) i szum pasma 20 m (do 0,37). Prawdziwe obce w teście
# QRM: 0,58-0,96.
SPOJNOSC_MIN = 0.5
# Spójność i odejmowanie w OKNACH (zakładka 50%, okno Hanna), nie z całości.
# Pasmo 40 m 20.10 (qso6, kluczownik na tle pasma): stacja 684 Hz, 76 Hz
# od naszej, 6-10 dB słabsza — spójność z 147 s 0,18 (dryf, przerwy), w oknach
# 10 s 0,14-0,86. Z całości odrzucona: 32 błędy / 105 znaków; wymuszone
# odjęcie całości 28; w oknach 10 s z progiem w oknie 20.
OKNO_KASOWANIA_S = 10.0
# W krótkim oknie szum łatwiej przechodzi próg spójności (pierwsza wersja
# okien: F5OHS, „Nagrywanie (6)", 1/1 -> 0/1 — skasowane szczyty szumu
# 470-1004 Hz). Drugi próg w oknie: poziom obcej (P90 obwiedni ±25 Hz)
# względem naszej. Zmierzone 20.10 na oknach ze spójnością >= 0,5:
# prawdziwe obce (QRM, 40 m 684 Hz) -6,5...+12 dB; szum mediana -22 dB,
# P90 -15,8; od -12 dB w górę tylko produkty obcej +12 dB w QRM.
POZIOM_MIN_DB = -12.0


def _lp(z: np.ndarray, b_hz: float, sr: int, taps: int = 801) -> np.ndarray:
    """Dolnoprzepustowy FIR o fazie zerowej = scipy filtfilt(b, [1], z), przez FFT.

    21.10 (odczyt na żywo): filtfilt liczył splot w dziedzinie czasu — 38
    wywołań na bufor 20 s, 1,24 s z 1,6 s przeciwsygnału. Tu to samo:
    odbicie nieparzyste na brzegach (padlen = 3·taps) i stan początkowy
    lfilter_zi (sygnał stały przed początkiem), splot oaconvolve.
    Zmierzone: różnica wobec filtfilt <= 3e-15, 3-4x szybciej.
    """
    from scipy.signal import firwin, oaconvolve
    b = firwin(taps, b_hz / (sr / 2))
    x = np.asarray(z)
    if x.size <= 3 * taps:
        from scipy.signal import filtfilt
        return filtfilt(b, [1.0], x.real) + 1j * filtfilt(b, [1.0], x.imag)
    pl = 3 * taps
    xp = np.concatenate([2 * x[0] - x[pl:0:-1], x, 2 * x[-1] - x[-2:-pl - 2:-1]])

    def raz(v):
        return oaconvolve(np.concatenate([np.full(taps - 1, v[0]), v]), b, mode="valid")

    return raz(raz(xp)[::-1])[::-1][pl:-pl]


def _srednia(x: np.ndarray, n: int) -> np.ndarray:
    n = max(1, int(n))
    return np.convolve(x, np.ones(n) / n, mode="same")


def obce_stacje(x: np.ndarray, f_nasza: float, sr: int = C.SR) -> list[tuple[float, float]]:
    """Kluczowane szczyty widma poza ±ODSTEP_HZ od naszej: [(Hz, poziom dB)], od najgłośniejszej.

    Kryterium jak tune.track_tone (kontrast P90-P10 >= 10 dB, poziom >= 8 dB nad
    medianą), ale z całego nagrania — obca stacja bez dryfu (do sprawdzenia na paśmie).
    """
    nb, ns, nfft = int(0.2 * sr), int(0.05 * sr), 4096
    f = np.fft.rfftfreq(nfft, 1 / sr)
    band = (f >= C.FMIN) & (f <= C.FMAX)
    fb = f[band]
    w = np.hanning(nb)
    starty = range(0, max(1, x.size - nb), ns)
    S = np.array([(np.abs(np.fft.rfft(x[s:s + nb] * w, nfft)) ** 2)[band] for s in starty])
    k = int(round(5.0 / (f[1] - f[0])))
    S = np.apply_along_axis(lambda r: np.convolve(r, np.ones(2 * k + 1) / (2 * k + 1), "same"), 1, S)
    D = 10 * np.log10(np.maximum(S, 1e-30) / max(float(np.median(S)), 1e-30))
    g, d = np.percentile(D, 90, 0), np.percentile(D, 10, 0)
    ok = (g >= 8.0) & (g - d >= 10.0)
    poz = np.where(ok, g, -np.inf)
    lm = np.flatnonzero((poz[1:-1] >= poz[:-2]) & (poz[1:-1] > poz[2:]) & ok[1:-1]) + 1
    kand = sorted(((float(fb[i]), float(g[i])) for i in lm if abs(fb[i] - f_nasza) >= ODSTEP_HZ),
                  key=lambda t: -t[1])
    out: list[tuple[float, float]] = []
    for fi, gi in kand:                                   # jeden szczyt na stację
        if all(abs(fi - o) >= ODSTEP_HZ for o, _ in out):
            out.append((fi, gi))
    return out


def odejmij(x: np.ndarray, f: float, f_nasza: float, sr: int = C.SR) -> np.ndarray:
    """Odtwarza stację na f i odejmuje ją od x.

    1. z: x sprowadzone do zera na f (sygnał analityczny);
    2. kluczowanie k(t): obwiednia ±B_KLUCZ_HZ, histereza w dB;
    3. dokładna częstotliwość: koherentna suma z w chwilach wciśnięcia (±3 Hz);
    4. c(t): amplituda i faza obcej — średnia z·k w OKNO_S / średnia k;
    5. kształt: rzut z (±B_RZUT_HZ, bez pasma naszej) na fazę c, w [0; 1,3];
    6. x - Re(c·kształt·e^{jφ}).
    """
    from scipy.signal import hilbert
    x = np.asarray(x, dtype=np.float64)
    n = np.arange(x.size)
    ph = 2 * np.pi * f * n / sr
    z = hilbert(x) * np.exp(-1j * ph)
    e = 20 * np.log10(np.abs(_lp(z, B_KLUCZ_HZ, sr)) + 1e-9)
    dol, gora = np.percentile(e, 10), np.percentile(e, 95)
    prog, h = 0.5 * (dol + gora), 0.1 * (gora - dol)
    krok = sr // 1000
    k = np.zeros(x.size, bool)
    st = False
    for i in range(0, x.size, krok):
        v = e[i]
        if st and v < prog - h:
            st = False
        elif not st and v > prog + h:
            st = True
        k[i:i + krok] = st
    if not k.any():
        return x
    zl = _lp(z, 100.0, sr)
    dz = 20
    zd, kd = zl[::dz], k[::dz]
    siatka = np.arange(-3.0, 3.001, 0.02)
    v = _suma_siatki(np.where(kd, zd, 0), sr / dz, 0.02)
    rot = np.exp(-2j * np.pi * siatka[int(np.argmax(v))] * n / sr)
    kk = k.astype(float)
    m_ok = int(OKNO_S * sr)
    c = _srednia(zl * rot * kk, m_ok) / np.maximum(_srednia(kk, m_ok), 1e-3)
    zr = _lp(z * rot, B_RZUT_HZ, sr)
    dlt = np.exp(-2j * np.pi * (f_nasza - f) * n / sr)
    zr = zr - _lp(zr * dlt, B_NASZA_HZ, sr) / dlt
    m = np.clip(np.real(zr * np.conj(c)) / np.maximum(np.abs(c) ** 2, 1e-12), 0, 1.3)
    return x - np.real(c * m * np.exp(1j * ph) / rot)


def _obwiednia(an: np.ndarray, f: float, sr: int) -> np.ndarray:
    z = an * np.exp(-2j * np.pi * f * np.arange(an.size) / sr)
    return np.abs(_lp(z, 15.0, sr))[::40]


def _spojnosc(an: np.ndarray, f: float, sr: int) -> float:
    zl = _lp(an * np.exp(-2j * np.pi * f * np.arange(an.size) / sr), 25.0, sr)[::20]
    e = np.abs(zl)
    k = e > 0.5 * np.percentile(e, 95)
    if not k.any():
        return 0.0
    s = float(np.sum(e[k]))
    return float(np.max(_suma_siatki(np.where(k, zl, 0), sr / 20, 0.05))) / max(s, 1e-12)


def _suma_siatki(x: np.ndarray, fs: float, krok_hz: float, zakres_hz: float = 3.0) -> np.ndarray:
    """|Σ x[n]·e^{-j2π d n/fs}| dla d = -zakres...+zakres co krok_hz, rosnąco.

    To jest FFT z dopełnieniem zerami — prążki dokładnie w punktach siatki.
    21.10 (odczyt na żywo): pętle po siatce (121 i 301 punktów) kosztowały
    ~1 s na krok 2 s nawet bez niczego do skasowania.
    """
    n0 = int(round(fs / krok_hz))
    m = -(-x.size // n0)                     # N >= długości: bez zawijania
    X = np.fft.fft(x, n0 * m)
    k = np.arange(-int(round(zakres_hz / krok_hz)), int(round(zakres_hz / krok_hz)) + 1) * m
    return np.abs(X[k % (n0 * m)])


def _poziom_db(an: np.ndarray, f: float, sr: int) -> float:
    z = an * np.exp(-2j * np.pi * f * np.arange(an.size) / sr)
    return 20 * np.log10(np.percentile(np.abs(_lp(z, 25.0, sr)), 90) + 1e-12)


def odejmij_oknami(x: np.ndarray, f: float, f_nasza: float, sr: int = C.SR,
                   tor: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[np.ndarray, int]:
    """`odejmij` w oknach OKNO_KASOWANIA_S tam, gdzie obca przechodzi progi w oknie.

    Nasza w oknie = mediana toru pętli tonu (`tor` = times, freqs z
    tune.track_tone), nie mediana z całości: F5OHS („Nagrywanie (6)") nadawał
    na 712 Hz przez pierwsze 10 s, potem szum i pętla na 757 — mediana 757
    i kasowaliśmy NASZĄ stację (1/1 -> 0/1). Okno, w którym kandydat leży
    bliżej niż ODSTEP_HZ od toru, zostaje nietknięte.
    Zwraca (sygnał, liczba okien z kasowaniem). Nagranie krótsze od okna — jedno okno.
    """
    from scipy.signal import hilbert
    n = min(int(OKNO_KASOWANIA_S * sr), x.size)
    krok = max(1, n // 2)
    starty = list(range(0, max(1, x.size - n + 1), krok))
    if starty[-1] + n < x.size:
        starty.append(x.size - n)
    w = np.hanning(n) if len(starty) > 1 else np.ones(n)
    est = np.zeros_like(x)
    wag = np.zeros_like(x)
    uzyte = 0
    for s in starty:
        seg = x[s:s + n]
        wag[s:s + n] += w
        fn = f_nasza
        if tor is not None:
            w_t = (tor[0] >= s / sr) & (tor[0] < (s + n) / sr) & np.isfinite(tor[1])
            if w_t.any():
                fn = float(np.median(tor[1][w_t]))
        if abs(f - fn) < ODSTEP_HZ:
            continue
        an = hilbert(seg)
        if _poziom_db(an, f, sr) - _poziom_db(an, fn, sr) >= POZIOM_MIN_DB \
                and _spojnosc(an, f, sr) >= SPOJNOSC_MIN:
            est[s:s + n] += (seg - odejmij(seg, f, fn, sr)) * w
            uzyte += 1
    return x - est / np.maximum(wag, 1e-9), uzyte


def kasuj_obce(x: np.ndarray, f_nasza: float, sr: int = C.SR,
               tor: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[np.ndarray, list]:
    """Odejmuje kolejno obce kluczowane stacje (od najgłośniejszej): [(Hz, dB, okien)].

    `tor` (times, freqs z tune.track_tone): nasza stacja okno po oknie — patrz
    odejmij_oknami. Bez toru: f_nasza w każdym oknie.
    """
    from scipy.signal import hilbert
    if not np.isfinite(f_nasza):
        return np.asarray(x), []
    y = np.asarray(x, dtype=np.float64)
    obce = obce_stacje(y, f_nasza, sr)
    if obce:
        an = hilbert(y)
        e0 = _obwiednia(an, f_nasza, sr)
        obce = [(fo, g) for fo, g in obce
                if np.corrcoef(e0, _obwiednia(an, fo, sr))[0, 1] <= KORELACJA_MAX]
    wyn = []
    for fo, g in obce:
        y, uzyte = odejmij_oknami(y, fo, f_nasza, sr, tor)
        if uzyte:
            wyn.append((fo, g, uzyte))
    return y.astype(np.float32), wyn
