"""GRZEBIEŃ: odzyskanie taktu morsa, „nuty", regenerator i klasyczny odczyt.

POMYSŁ OPERATORA (17.10). Jak odzyskiwanie zegara w transmisji cyfrowej,
tylko że mors nie idzie jednym taktem: przerwa między znakami to 3
jednostki „albo więcej, i to więcej może być bardzo dużo". Wewnątrz znaku
wszystkie krawędzie leżą na siatce jednostki u. Po sprowadzeniu przerw
między znakami do 3 jednostek przykłada się do sygnału „wirtualny
przebieg prostokątny" — jak grzebień do gwintu — i dopasowanie na dużej
grupie kropek i kresek przestaje zależeć od pojedynczej zaszumionej
krawędzi. Wynik to „nuty": elementy w jednostkach taktu, z których da się
odtworzyć czysty, bezszumny sygnał (regenerator / „idealny filtr CW"),
podać go sieci (tester) albo przełożyć wprost na znaki (klasyczny odczyt).

DOPASOWANIE. Odcinki znak/przerwa z dsp.tempo.odcinki (z wygładzeniem).
Przerwa dłuższa niż PROG_ZNAKU jednostki zamyka znak — grzebień zaczyna
od nowa (to jest ta normalizacja przerw między znakami do 3 jednostek).
Wewnątrz znaku krawędź narastająca leży w k·u − b/2, opadająca w k·u + b/2
(b = waga: pogłos, filtr p.cz. i ARW wydłużają znaki kosztem przerw).
Szukamy u, b minimalizujących sumę ODPORNEJ funkcji błędu położeń krawędzi
względem siatki — krawędź dalej niż pół jednostki od siatki (trzask,
zanik) liczy się stałą karą, nie kwadratem.

UWAGA (17.10, rozmowa z operatorem). Regeneracja to TWARDA decyzja:
każdy fragment staje się znakiem albo przerwą. Przy słabej stacji
regenerator odtworzy błąd na czysto. Do słuchania i jako tester — nie
jako zamiennik sieci przy słabym sygnale.
"""

from __future__ import annotations

import numpy as np

from . import config as C
from . import morse, tempo

U_SIATKA = tempo.U_SIATKA
B_UDZIAL = np.linspace(-0.6, 0.6, 49)
PROG_ZNAKU = 2.0          # przerwa > 2 u = koniec znaku (1 u w znaku, 3 u między)
KARA = 0.25               # (błąd/u)^2 obcięty do (0,5)^2 — krawędź spoza siatki


def _znaki_z_odcinkow(on_off: list[tuple[bool, float]], u: float):
    """[(znaki: lista (start, koniec) w s od początku znaku), ...] i czasy
    początku znaków. Przerwa > PROG_ZNAKU·u zamyka znak."""
    znaki, t, biezacy, start = [], 0.0, [], None
    for jest, d in on_off:
        if jest:
            if start is None:
                start = t
            biezacy.append((t - start, t - start + d))
        elif d > PROG_ZNAKU * u and biezacy:
            znaki.append((start, biezacy))
            biezacy, start = [], None
        t += d
    if biezacy:
        znaki.append((start, biezacy))
    return znaki


def _blad(znaki, u: float, b: float) -> tuple[float, int]:
    """Odporny błąd położeń krawędzi względem siatki (średnio na krawędź)."""
    e, n = 0.0, 0
    for _, el in znaki:
        for a, z in el:
            ra = (a + b / 2) / u          # narastająca w k·u − b/2
            rz = (z - b / 2) / u          # opadająca w k·u + b/2
            for r in (ra, rz):
                e += min((r - round(r)) ** 2, KARA)
                n += 1
    return (e / n if n else np.inf), n


def odcinki_przebiegu(audio: np.ndarray, ton_hz: float = C.TONE_CENTER,
                      sr: int = C.SR, wygladzenie_s: float = 0.015
                      ) -> list[tuple[bool, float]]:
    """Cały przebieg znak/przerwa po kolei [(czy_znak, czas_s), ...]."""
    from . import tune
    x = tune.bandpass(np.asarray(audio, dtype=np.float32), ton_hz, 50.0, sr=sr)
    h = int(round(tempo.BLOK_S * sr))
    n = x.size // h
    if n < 20:
        return []
    p = (x[:n * h].astype(np.float64) ** 2).reshape(n, h).mean(axis=1)
    k = int(round(wygladzenie_s / tempo.BLOK_S))
    if k > 1:
        p = np.convolve(p, np.ones(k) / k, mode="same")
    db = 10.0 * np.log10(p + 1e-12)
    on = db > tempo._prog_otsu(db)
    zm = np.flatnonzero(np.diff(on.astype(np.int8))) + 1
    gr = np.r_[0, zm, n]
    return [(bool(on[a]), (b - a) * tempo.BLOK_S) for a, b in zip(gr[:-1], gr[1:])]


def dopasuj_grzebien(on_off: list[tuple[bool, float]],
                     u_start: float | None = None) -> dict:
    """Podziałka u i waga b grzebienia; pewność = 1 − średni błąd / KARA."""
    wynik = {"wpm": float("nan"), "u_s": float("nan"), "waga": float("nan"),
             "blad": float("nan"), "pewnosc": float("nan"), "krawedzi": 0}
    if sum(1 for j, _ in on_off if j) < 4:
        return wynik
    siatka = U_SIATKA if u_start is None else np.linspace(
        0.85 * u_start, 1.15 * u_start, 61)
    najl = (np.inf, None, None, 0)
    for u in siatka:
        znaki = _znaki_z_odcinkow(on_off, u)
        for bu in B_UDZIAL:
            # Błąd WZGLĘDNY (w jednostkach u): drobniejsza siatka u/2 też
            # trafia w każdą krawędź, ale ten sam rozrzut w sekundach to przy
            # niej 2x więcej jednostek — przegrywa sama, bez kary.
            e, n = _blad(znaki, u, bu * u)
            if n >= 8 and e < najl[0]:
                najl = (e, u, bu * u, n)
    e, u, b, n = najl
    if u is None:
        return wynik
    e0, _ = _blad(_znaki_z_odcinkow(on_off, u), u, b)
    wynik.update(wpm=float(1.2 / u), u_s=float(u), waga=float(b / u),
                 blad=float(e0), pewnosc=float(1.0 - e0 / KARA), krawedzi=n)
    return wynik


def nuty(on_off: list[tuple[bool, float]], u: float, b: float) -> list:
    """Znaki jako elementy w jednostkach: [(czas_startu_znaku_s,
    [(start_u, dlugosc_u), ...]), ...] — krawędzie dociągnięte do siatki."""
    out = []
    for start, el in _znaki_z_odcinkow(on_off, u):
        e = []
        for a, z in el:
            ka = int(round((a + b / 2) / u))
            kz = int(round((z - b / 2) / u))
            if kz > ka:
                e.append((ka, kz - ka))
        if e:
            out.append((start, e))
    return out


_ODWROTNY = {v: k for k, v in C.MORSE_DICT.items()}


def odczyt_klasyczny(nuty_: list) -> str:
    """Nuty -> tekst według tabeli Morse'a (bez sieci): element 1 u =
    kropka, ≥ 2 u = kreska. Nieznany kod -> '?'."""
    tekst = []
    for start, el in nuty_:
        kod = "".join("." if d < 2 else "-" for _, d in el)
        tekst.append(_ODWROTNY.get(kod, "?"))
    return "".join(tekst)


def regeneruj(nuty_: list, u: float, sr: int = C.SR,
              ton_hz: float = C.TONE_CENTER, amp: float = 0.3,
              dlugosc_s: float | None = None) -> np.ndarray:
    """Czysty sygnał z nut: elementy w zmierzonym takcie, początki znaków
    w ich prawdziwych miejscach (przerwy między znakami zostają, jakie były)."""
    koniec = max((s + (e[-1][0] + e[-1][1]) * u for s, e in nuty_), default=0.0)
    n = int(round((dlugosc_s if dlugosc_s else koniec + 0.5) * sr))
    env = np.zeros(n, dtype=np.float32)
    rampa = max(1, int(C.KEY_RAMP_MS * 1e-3 * sr))
    for s, el in nuty_:
        for k, d in el:
            a = int(round((s + k * u) * sr))
            z = int(round((s + (k + d) * u) * sr))
            if a >= n:
                continue
            z = min(z, n)
            blok = np.ones(z - a, dtype=np.float32)
            r = min(rampa, (z - a) // 2)
            if r > 0:
                w = 0.5 - 0.5 * np.cos(np.pi * np.arange(r) / r)
                blok[:r] = w
                blok[-r:] = w[::-1]
            env[a:z] = np.maximum(env[a:z], blok)
    t = np.arange(n) / sr
    return (amp * env * np.sin(2 * np.pi * ton_hz * t)).astype(np.float32)


# Próg słowa: przerwa między znakami >= tyle jednostek to przerwa między
# słowami (połowa drogi między 3 u a 7 u).
PROG_SLOWA_U = 5.0


def regeneruj_wzorcowo(nuty_u: list, sr: int = C.SR,
                       ton_hz: float = C.TONE_CENTER, amp: float = 0.3
                       ) -> np.ndarray:
    """Regenerat w PEŁNI znormalizowany (17.10, operator: „źle się słucha"
    regeneratu, który miał znaki na siatce, ale przerwy między nimi jak
    w nagraniu i jedno tempo na całość). Tu: każdy znak w jednostce SWOJEGO
    okna, przerwa między literami 3 u, między słowami 7 u (próg
    PROG_SLOWA_U) — jak z dobrego keyera. Wejście: [(start_s, el, u_s), ...]."""
    odc, t = [], 0.0
    prev_koniec = None
    for ts, el, u in nuty_u:
        if prev_koniec is not None:
            przerwa_u = (ts - prev_koniec) / u
            t += (7 if przerwa_u >= PROG_SLOWA_U else 3) * u
        for k, d in el:
            odc.append((t + k * u, d * u))
        dl = (el[-1][0] + el[-1][1]) * u
        t += dl
        prev_koniec = ts + dl
    n = int(round((t + 0.5) * sr))
    env = np.zeros(n, dtype=np.float32)
    rampa = max(1, int(C.KEY_RAMP_MS * 1e-3 * sr))
    for a_s, d_s in odc:
        a = int(round(a_s * sr))
        z = min(n, int(round((a_s + d_s) * sr)))
        blok = np.ones(z - a, dtype=np.float32)
        r = min(rampa, (z - a) // 2)
        if r > 0:
            w = 0.5 - 0.5 * np.cos(np.pi * np.arange(r) / r)
            blok[:r] = w
            blok[-r:] = w[::-1]
        env[a:z] = np.maximum(env[a:z], blok)
    tt = np.arange(n) / sr
    return (amp * env * np.sin(2 * np.pi * ton_hz * tt)).astype(np.float32)


def zmierz(audio: np.ndarray, ton_hz: float = C.TONE_CENTER,
           sr: int = C.SR) -> dict:
    """Grzebień na buforze: tempo z dsp.tempo jako start, dopasowanie
    grzebienia w ±15% wokół niego (najpierw zgrubnie, potem dokładnie)."""
    r0 = tempo.zmierz_tempo(audio, ton_hz, sr)
    u0 = r0["u_s"] if np.isfinite(r0["u_s"]) else None
    wyg = 0.5 * u0 if u0 else 0.015
    oo = odcinki_przebiegu(audio, ton_hz, sr, wygladzenie_s=wyg)
    r = dopasuj_grzebien(oo, u_start=u0)
    r["on_off"] = oo
    r["wpm_dlugosci"] = r0["wpm"]
    return r
