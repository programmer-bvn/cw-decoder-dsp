"""NEGATYWY — dźwięk, który NIE jest telegrafią, do treningu sieci zboczy (06.11).

Pomiar 05.11 (tools/halucynacje, runs/zbocza_5, odczyt całości): mikrofon
w pokoju 21,3 fałszywych znaków/min, radiofonia AM z KiwiSDR 35,7, SSB 29,9.
Generator znał jako „brak tonu” tylko szum (puste radio, probki/szum, QRN) —
mowy, muzyki i pływającego tonu sieć nie widziała, więc każde wybrzuszenie
obwiedni przy 750 Hz bierze za kropkę. Operator 05.11: „sieć sama z siebie
nie powinna rozpoznawać zboczy tam, gdzie ich nie ma”.

Źródła:
  - bank prawdziwy: CZĘŚĆ TRENINGOWA banku W:\\nagrania_cw\\negatywy
    (pokoj_mikrofon, minuty 0000-0299 nocy 04/05.11), sklejona do jednego
    pliku int16 8 kHz (zbuduj_bank). Minuty 0300+ i całe KiwiSDR zostają
    do pomiaru — sieć ich nie widzi (tools/halucynacje pomija TRENING.txt);
  - syntetyczne: mowa (ciąg impulsów krtaniowych przez formanty, sylaby
    3-6/s), śpiew (nuty z vibrato 4-7 Hz — operator śpiewał do dekodera
    04.11: wahania FM ~35 Hz), gwizdy (przestrajany ton), muzyka (akordy
    z harmonicznymi i perkusją).

Śpiew i gwizd mają ZAWSZE ruch częstotliwości (vibrato >= 12 Hz, przestrajanie,
zmiana nuty o >= półton). Nuta stała w tonie, włączana i wyłączana, to
w torze radiowym kluczowany ton — z etykietą 0 uczyłaby sieć sprzeczności.

Retune: odczyt (tune.retune) sprowadza najsilniejszy ton do 750 Hz, także
na mowie i muzyce. Część negatywów przesuwamy tak samo: najsilniejszy prążek
na ton naszej stacji ± 10 Hz.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from . import config as C
from . import sn

SR = C.SR
# Bank treningowy: jeden plik int16 8 kHz (np.load mmap — procesy generatora
# dzielą go przez pamięć podręczną systemu, zamiast trzymać po 300 MB każdy).
BANK_KAT = Path(os.environ.get("CW_NEGATYWY",
                               r"W:\nagrania_cw\negatywy" if os.name == "nt"
                               else "/mnt/w/nagrania_cw/negatywy"))
BANK_PLIK = "trening_8k.npy"
TRENING_POKOJ = 300                 # minuty 0000-0299 z pokoj_mikrofon

_BANK = None


# --------------------------------------------------------------------------
# bank prawdziwy
# --------------------------------------------------------------------------
def pliki_treningowe(kat: Path = BANK_KAT) -> list[Path]:
    return sorted((kat / "pokoj_mikrofon").glob("*.wav"))[:TRENING_POKOJ]


def zbuduj_bank(kat: Path = BANK_KAT) -> Path:
    """Część treningową sklei do kat/trening_8k.npy i zapisze kat/TRENING.txt
    (nazwy plików, których pomiar halucynacji nie liczy)."""
    import soundfile as sf
    from scipy.signal import resample_poly
    pliki = pliki_treningowe(kat)
    if not pliki:
        raise SystemExit(f"brak nagrań w {kat / 'pokoj_mikrofon'}")
    czesci = []
    for p in pliki:
        x, sr = sf.read(p, dtype="float32")
        if x.ndim > 1:
            x = x[:, 0]
        if sr != SR:
            x = resample_poly(x, SR, sr).astype(np.float32)
        czesci.append(np.round(np.clip(x, -1, 1) * 32767).astype(np.int16))
    wyj = kat / BANK_PLIK
    np.save(wyj, np.concatenate(czesci))
    (kat / "TRENING.txt").write_text(
        "# pliki w banku treningowym negatywów (dsp/negatywy.py) — pomiar ich nie liczy\n"
        + "".join(f"{p.parent.name}/{p.name}\n" for p in pliki), encoding="utf-8")
    return wyj


def bank():
    """int16 memmap albo None (inna maszyna — wtedy same syntetyczne)."""
    global _BANK
    if _BANK is None:
        p = BANK_KAT / BANK_PLIK
        _BANK = np.load(p, mmap_mode="r") if p.exists() else False
    return _BANK if _BANK is not False else None


def z_banku(n: int, rng: np.random.Generator) -> np.ndarray:
    b = bank()
    s = int(rng.integers(0, b.shape[0] - n))
    x = np.asarray(b[s:s + n], dtype=np.float32) / 32768.0
    return x[::-1].copy() if rng.random() < 0.5 else x


# --------------------------------------------------------------------------
# syntetyczne
# --------------------------------------------------------------------------
def _gladka(n: int, rng, tempo_hz: float, lo: float, hi: float) -> np.ndarray:
    """Wolno zmienny przebieg w [lo, hi] (szum dolnoprzepustowy przez interpolację)."""
    k = max(2, int(n / SR * tempo_hz) + 2)
    return np.interp(np.arange(n), np.linspace(0, n, k), rng.uniform(lo, hi, k))


def _faza(f: np.ndarray) -> np.ndarray:
    return 2 * np.pi * np.cumsum(f) / SR


def _formanty(x: np.ndarray, rng, n_syl: int) -> np.ndarray:
    """Pasmowe filtry formantów zmieniane co sylabę (sumowane z wagami)."""
    from scipy.signal import butter, sosfilt
    n = x.size
    out = np.zeros(n, np.float32)
    granice = np.linspace(0, n, n_syl + 1).astype(int)
    for a, b in zip(granice[:-1], granice[1:]):
        y = np.zeros(b - a, np.float32)
        for lo, hi in ((250, 900), (800, 2400), (2000, 3200)):
            f0 = rng.uniform(lo, hi)
            bw = rng.uniform(60, 200)
            sos = butter(2, [max(50, f0 - bw), min(SR / 2 - 50, f0 + bw)], "band", fs=SR, output="sos")
            y += rng.uniform(0.3, 1.0) * sosfilt(sos, x[a:b]).astype(np.float32)
        out[a:b] = y
    return out


def mowa(n: int, rng) -> np.ndarray:
    """Impulsy krtaniowe f0 80-260 Hz z intonacją, formanty co sylabę,
    obwiednia sylab 3-6/s, szelesty (spółgłoski), pauzy."""
    f0 = _gladka(n, rng, 2.0, *sorted(rng.uniform(80, 260, 2)))
    ph = np.cumsum(f0 / SR)
    imp = (np.diff(np.floor(ph), prepend=0) > 0).astype(np.float32)
    imp = np.convolve(imp, np.hanning(9), mode="same")
    syl = rng.uniform(3, 6)
    n_syl = max(1, int(n / SR * syl))
    x = _formanty(imp, rng, n_syl)
    x += _formanty(rng.normal(0, 0.05, n).astype(np.float32), rng, n_syl) * (rng.random() < 0.6)
    t = np.arange(n) / SR
    env = np.clip(np.sin(np.pi * syl * t + rng.uniform(0, np.pi)) ** 2
                  * _gladka(n, rng, 1.5, 0.0, 1.6), 0, 1)
    return (x * env).astype(np.float32)


def spiew(n: int, rng) -> np.ndarray:
    """Nuty 0,2-1 s, legato albo z przerwą, vibrato 4-7 Hz i 12-60 Hz,
    harmoniczne — jak operator śpiewający do dekodera 04.11."""
    f = np.zeros(n)
    env = np.zeros(n)
    i, nuta = 0, rng.uniform(350, 1100)
    while i < n:
        d = int(rng.uniform(0.2, 1.0) * SR)
        f[i:i + d] = nuta
        a = np.ones(min(d, n - i))
        r = min(len(a) // 3, int(rng.uniform(0.02, 0.1) * SR))
        if r > 0:
            a[:r] = np.linspace(0, 1, r)
            a[-r:] = np.linspace(1, 0, r) if rng.random() < 0.5 else a[-r:]
        env[i:i + d] = a
        i += d
        krok = rng.choice([-1, 1]) * rng.uniform(1, 5)          # >= półton
        nuta = float(np.clip(nuta * 2 ** (krok / 12), 350, 1100))
    f = np.convolve(f, np.ones(200) / 200, mode="same")          # portamento ~25 ms
    t = np.arange(n) / SR
    vib = rng.uniform(12, 60) * np.sin(2 * np.pi * rng.uniform(4, 7) * t + rng.uniform(0, 6.3))
    ph = _faza(f + vib)
    x = sum(rng.uniform(0.1, 1.0) / k * np.sin(k * ph) for k in range(1, 5))
    return (x * env * _gladka(n, rng, 1.0, 0.5, 1.0)).astype(np.float32)


def gwizd(n: int, rng) -> np.ndarray:
    """Ton przestrajany: przemiatanie, falowanie albo skoki z poślizgiem
    (heterodyna, gwizd na paśmie, kręcenie gałką strojenia)."""
    rodzaj = rng.integers(0, 3)
    t = np.arange(n) / SR
    if rodzaj == 0:
        f = np.linspace(*rng.uniform(300, 1300, 2), n)
    elif rodzaj == 1:
        f = rng.uniform(450, 1050) + rng.uniform(20, 250) * np.sin(
            2 * np.pi * rng.uniform(0.3, 6) * t + rng.uniform(0, 6.3))
    else:
        f = _gladka(n, rng, rng.uniform(1, 6), 350, 1200)
    x = np.sin(_faza(f))
    if rng.random() < 0.4:
        x = x + rng.uniform(0.1, 0.5) * np.sin(2 * _faza(f))
    return (x * _gladka(n, rng, 2.0, 0.3, 1.0)).astype(np.float32)


def muzyka(n: int, rng) -> np.ndarray:
    """Akordy 2-4 nut z harmonicznymi, zmiana co 0,2-0,8 s, perkusja."""
    x = np.zeros(n, np.float32)
    i = 0
    t = np.arange(n) / SR
    while i < n:
        d = min(n - i, int(rng.uniform(0.2, 0.8) * SR))
        tt = t[:d]
        a = np.exp(-tt * rng.uniform(0.5, 6))
        for _ in range(int(rng.integers(2, 5))):
            f0 = 110 * 2 ** (rng.integers(0, 36) / 12)
            for k in range(1, 6):
                if k * f0 < SR / 2 - 200:
                    x[i:i + d] += (rng.uniform(0.2, 1.0) / k * a
                                   * np.sin(2 * np.pi * k * f0 * tt + rng.uniform(0, 6.3)))
        i += d
    if rng.random() < 0.7:                                       # bęben: szum z szybkim zanikiem
        okres = int(rng.uniform(0.25, 0.6) * SR)
        uderz = np.exp(-np.arange(int(0.15 * SR)) / (0.03 * SR)) * rng.normal(0, 1, int(0.15 * SR))
        for s in range(int(rng.integers(0, okres)), n - uderz.size, okres):
            x[s:s + uderz.size] += rng.uniform(0.5, 2.0) * uderz
    return x


RODZAJE = {"bank": None, "mowa": mowa, "spiew": spiew, "gwizd": gwizd, "muzyka": muzyka}


def _przesun(x: np.ndarray, df: float) -> np.ndarray:
    from scipy.signal import hilbert
    t = np.arange(x.size) / SR
    return np.real(hilbert(x) * np.exp(2j * np.pi * df * t)).astype(np.float32)


def _najsilniejszy(x: np.ndarray) -> float:
    from scipy.signal import welch
    f, p = welch(x, SR, nperseg=1024)
    m = (f >= C.FMIN) & (f <= C.FMAX)
    return float(f[m][np.argmax(p[m])])


def negatyw(n: int, rng: np.random.Generator, szum: np.ndarray, ton_hz: float,
            par: dict) -> tuple[np.ndarray, dict]:
    """Negatyw o długości n, poziom par["poziom_db"] względem szumu
    (moc w paśmie 400-1200 Hz).
    Zwraca (audio, opis)."""
    nazwy = [k for k in RODZAJE if k != "bank" or bank() is not None]
    wagi = np.array([par["wagi"][k] for k in nazwy], float)
    rodzaj = str(rng.choice(nazwy, p=wagi / wagi.sum()))
    x = z_banku(n, rng) if rodzaj == "bank" else RODZAJE[rodzaj](n, rng)
    x = x - x.mean()
    if rng.random() < par["retune"]:
        x = _przesun(x, ton_hz + float(rng.uniform(-10, 10)) - _najsilniejszy(x))
    elif rodzaj == "bank":
        x = _przesun(x, float(rng.uniform(-300, 300)))
    # Poziom w CAŁYM paśmie sieci (400-1200 Hz), nie przy tonie: gwizd przy
    # 1100 Hz liczony przy 750 dostałby wzmocnienie o kilkadziesiąt dB.
    sr_, szer = (C.FMIN + C.FMAX) / 2, C.FMAX - C.FMIN
    moc = sn.moc_w_pasmie(x, sr_, szer)
    poz = float(rng.uniform(*par["poziom_db"]))
    cel = sn.moc_w_pasmie(szum, sr_, szer) * 10 ** (poz / 10)
    x = x * np.sqrt(cel / max(moc, 1e-30))
    return x.astype(np.float32), {"negatyw": rodzaj, "negatyw_db": poz}
