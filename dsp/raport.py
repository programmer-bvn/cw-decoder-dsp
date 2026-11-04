"""Sugerowany raport RST z odbioru na żywo: R z pewności dekodera, T z tętnienia.

Skala RST (W2BSR, QST 1934) jest opisana słownie; liczbowo zdefiniowany jest
tylko S (IARU R1: S9 = −73 dBm, 6 dB na stopień — tools/radio/smetr.py). R i T
to nasze przełożenie, ustalone z operatorem 03-04.11.2026:

R — „procent dopasowania do idealnego sygnału, uśredniony w długim czasie”.
    Pewność sieci wzdłuż ścieżki Viterbiego (ton: p, przerwa: 1 − p), średnia na
    znak, potem po ostatnich OKNO_ZNAKOW znakach („nie na podstawie 2-3, tylko
    20-30”); tylko znaki ponad szumem (NAD_SZUMEM_DB). Pomiar 04.11, dekoder na
    żywo na nagraniach z probki/: manipulator i kluczownik 0,985; SN0FMT 0,96;
    OK1FPS 0,93-0,95 (odczyt z pojedynczymi błędami); LA4CIR w czasie: 0,965 przy
    czystym tekście („AGE ES CAN SEE UR”), 0,918 z błędami („RI6 7V M5 B3”),
    0,87-0,76 przy sieczce; Nagrywanie (8) i śpiew operatora 0,82 (sieczka).
    Progi R_PROGI ułożone na tych liczbach — do sprawdzenia z oceną słuchową.

T — definicja mówi tylko o tętnieniu zasilania, nie o częstotliwości (operator
    04.11), więc T liczymy z modulacji amplitudy 50 i 100 Hz w czasie trwania
    elementów: głębokość tętnienia [%] -> T3-T9 progami operatora („2-5 % czysty,
    10 to 8, 15 to 7, rzędu 20 % słyszalne tętnienie, od 0 do maksimum
    niefiltrowany”). T1-T2: zasilanie prądem zmiennym bez prostowania — nadajnik
    przez pół okresu nie nadaje, w elemencie są przerwy. Zmierzone 03.11: szum
    i linia 110 kV dają pozorną modulację 50/100 Hz (u słabych stacji 2-4 %), więc
    od tętnienia odejmowana jest ta sama miara na częstotliwościach obok (H_TLA);
    gdy samo tło przekracza MAKS_TLO, T jest nieoznaczone („?”). Wolne zmiany
    (QSB, ARW odbiornika, siadające zasilanie) usuwa odjęcie prostej z obwiedni
    elementu — 04.11 ARW dawało spadek 10-20 % w kresce u prawie każdej stacji.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy import signal

OKNO_ZNAKOW = 25
NAD_SZUMEM_DB = 8.0                 # znak stacji, nie szumu: elementy tyle ponad podłogę
OKNO_ELEMENTOW = 75                 # ~25 znaków po ~3 elementy
MIN_ZNAKOW, MIN_ELEMENTOW = 10, 20
# (największy brak pewności 1 − R, stopień)
R_PROGI = [(0.05, 5), (0.10, 4), (0.15, 3), (0.20, 2)]
# (największa głębokość tętnienia [%], stopień) — progi operatora 04.11
T_PROGI = [(5, 9), (10, 8), (15, 7), (25, 6), (40, 5), (60, 4)]
H_TETNIENIA = (50.0, 100.0)
H_TLA = (65.0, 80.0, 120.0)         # nie harmoniczne sieci, poniżej filtra 130 Hz
# [%] pozorne tętnienie z szumu, powyżej którego T nieoznaczone. Test 04.11 (syntetyczny,
# 40 kresek): SNR 12 dB w 250 Hz — tło 10,3 %, a 10 % tętnienia zmierzone jako 8,9 %;
# SNR 6 dB — tło 15,9 %.
MAKS_TLO = 15.0
BRZEG_S = 0.008                     # obcięte z brzegów elementu (zbocza, niedokładne granice)
MIN_ELEMENT_S = 0.020               # 50 Hz: co najmniej jeden okres
PRZERWY_UDZIAL = 0.15               # udział próbek < 20 % mediany: element przerywany (T1-T2)
# Głębokość > 100 % bywa tylko przy przerwach w nadawaniu: prąd zmienny bez
# prostowania (półokresy) to ~170 % (50 Hz: 0,5/(1/π), 100 Hz: 0,21·π), prostowanie
# dwupołówkowe bez filtra 67 % (T3-T4). Próg między nimi:
PRZERWY_PROC = 120.0
F_FILTRA = 130.0


def stopien_r(pewnosc: float) -> int:
    brak = 1.0 - pewnosc
    for prog, r in R_PROGI:
        if brak <= prog:
            return r
    return 1


def stopien_t(tetnienie_proc: float, przerwy: bool, dominuje_50: bool) -> int:
    """Przerwy: półokresowe (dominuje 50 Hz, jednopołówkowo) T1, inaczej T2 — hipoteza."""
    if przerwy:
        return 1 if dominuje_50 else 2
    for prog, t in T_PROGI:
        if tetnienie_proc <= prog:
            return t
    return 3


def pomiar_elementu(x: np.ndarray, sr: int, f0: float, margines: int) -> dict | None:
    """Rzeczywisty dźwięk elementu z `margines` próbek z obu stron (do filtra)
    -> kwadraty głębokości modulacji na H_TETNIENIA + H_TLA i udział przerw."""
    z = x.astype(np.float64) * np.exp(-2j * np.pi * f0 * np.arange(x.size) / sr)
    sos = signal.butter(4, F_FILTRA, fs=sr, output="sos")
    z = signal.sosfiltfilt(sos, z)
    brzeg = margines + int(BRZEG_S * sr)
    a = np.abs(z[brzeg:z.size - brzeg])
    if a.size < MIN_ELEMENT_S * sr or not np.any(a > 0):
        return None
    sr_a = float(np.mean(a))
    t = np.arange(a.size) / sr
    e = a / sr_a - 1.0
    e = e - np.polyval(np.polyfit(t, e, 1), t)          # QSB, ARW, siadanie — nie tętnienie
    # okno Hanna: bez niego 100 Hz przeciekało do częstotliwości tła (test 04.11:
    # przy 100 % tętnienia „tło” 6 %); |H|² — filtr w obie strony podcina 100 Hz o ~11 %
    w = np.hanning(a.size)
    _, hf = signal.sosfreqz(sos, worN=np.array(H_TETNIENIA + H_TLA), fs=sr)
    m2 = {h: (2 * abs(np.sum(e * w * np.exp(-2j * np.pi * h * t))) / np.sum(w) / abs(g) ** 2) ** 2
          for h, g in zip(H_TETNIENIA + H_TLA, hf)}
    return dict(m2=m2, przerwy=float(np.mean(a < 0.2 * np.median(a))))


class Raport:
    def __init__(self) -> None:
        self.znaki: deque[float] = deque(maxlen=OKNO_ZNAKOW)
        self.elementy: deque[dict] = deque(maxlen=OKNO_ELEMENTOW)

    def znak(self, pewnosc: float) -> None:
        if np.isfinite(pewnosc):
            self.znaki.append(float(pewnosc))

    def element(self, pomiar: dict | None) -> None:
        if pomiar is not None:
            self.elementy.append(pomiar)

    def wynik(self) -> dict:
        """-> {r, pewnosc, t, tetnienie_proc, tlo_proc, przerwy, n_znakow, n_elementow};
        r / t = None, gdy za mało danych, t = "?" przy zbyt głośnym tle."""
        w: dict = dict(r=None, t=None, n_znakow=len(self.znaki), n_elementow=len(self.elementy))
        if len(self.znaki) >= MIN_ZNAKOW:
            p = float(np.mean(self.znaki))
            w.update(r=stopien_r(p), pewnosc=round(p, 3))
        if len(self.elementy) >= MIN_ELEMENTOW:
            sr_m2 = {h: float(np.mean([e["m2"][h] for e in self.elementy])) for h in H_TETNIENIA + H_TLA}
            tlo2 = float(np.mean([sr_m2[h] for h in H_TLA]))
            tet2 = sum(max(0.0, sr_m2[h] - tlo2) for h in H_TETNIENIA)
            tet, tlo = 100 * np.sqrt(tet2), 100 * np.sqrt(tlo2)
            przerwy = tet > PRZERWY_PROC or float(np.median([e["przerwy"] for e in self.elementy])) > PRZERWY_UDZIAL
            w.update(tetnienie_proc=round(float(tet), 1), tlo_proc=round(float(tlo), 1), przerwy=przerwy)
            if tlo > MAKS_TLO and not przerwy:
                w["t"] = "?"
            else:
                w["t"] = stopien_t(tet, przerwy, sr_m2[50.0] >= sr_m2[100.0])
        return w
