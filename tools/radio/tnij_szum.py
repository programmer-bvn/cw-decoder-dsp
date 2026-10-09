# Kopia robocza z sesji 09.10 (scratchpad) -- nagrywanie z IC-7300MK2 przez USB.
"""Tnie nagranie szumu na odcinki o stałej szerokości pasa (zmierzonej
z widma co 1 s) i zapisuje je osobno, z nazwą wg zmierzonej szerokości."""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sur = Path(sys.argv[1])            # oryginał 48 kHz
kat = Path(sys.argv[2])            # probki/szum
baza = sys.argv[3]                 # np. szum_ic7300_usb_3566kHz
sys.path.insert(0, str(kat.parent.parent))

a, sr = sf.read(sur)
m0 = a[:, 0]
n = 4096
w = np.hanning(n)
f = np.fft.rfftfreq(n, 1 / sr)
pasmo = (f > 150) & (f < 3500)


def szerokosc(seg):
    P = np.zeros(n // 2 + 1)
    for k in range(0, len(seg) - n, n // 2):
        P += np.abs(np.fft.rfft(seg[k:k + n] * w)) ** 2
    Pd = 10 * np.log10(P / P[pasmo].max() + 1e-15)
    p = f[pasmo][Pd[pasmo] > -6]
    return p.max() - p.min(), p.min(), p.max()


sek = int(len(m0) / sr)
szer = [szerokosc(m0[t * sr:(t + 1) * sr]) for t in range(sek)]
# 200 i 250 Hz w jednej klasie: pomiar szerokości skacze między nimi
# przy tym samym filtrze (40 m, 09.10) i tnie nagranie na drobne kawałki.
klasa = [min((225, 500, 1200, 2400), key=lambda c: abs(c - s[0]))
         for s in szer]

odcinki, start = [], 0
for t in range(1, sek + 1):
    if t == sek or klasa[t] != klasa[start]:
        odcinki.append((start, t, klasa[start]))
        start = t

for t0, t1, kl in odcinki:
    a0, a1 = t0 + 1, t1 - 1                      # margines 1 s na przełączenie
    if a1 - a0 < 10:
        print(f"  pomijam {t0}-{t1} s ({kl} Hz): krócej niż 10 s")
        continue
    sz = np.median([szer[t][0] for t in range(a0, a1)])
    lo = np.median([szer[t][1] for t in range(a0, a1)])
    hi = np.median([szer[t][2] for t in range(a0, a1)])
    nazwa = f"{baza}_{int(round(sz, -1))}Hz_{a0:03d}-{a1:03d}s"
    kaw = a[a0 * sr:a1 * sr]
    (kat / "surowe").mkdir(parents=True, exist_ok=True)
    sf.write(kat / "surowe" / f"{nazwa}_surowe.wav", kaw, sr, subtype="PCM_24")
    import librosa
    m8 = librosa.resample(kaw[:, 0].astype(np.float32), orig_sr=sr, target_sr=8000)
    sf.write(kat / f"{nazwa}.wav", m8, 8000, subtype="PCM_16")
    (kat / f"{nazwa}.txt").write_text(
        "# SZUM PASMA - bez sygnalu; do banku tla generatora.\n"
        f"zrodlo:        {sur.name}, {a0}-{a1} s\n"
        f"pas -6 dB:     {lo:.0f}-{hi:.0f} Hz, szerokosc {sz:.0f} Hz "
        f"(zmierzona z widma co 1 s)\n"
        "tor:           USB audio z IC-7300MK2, AF Output Level 50%, "
        "wejscie Windows 50%\n", encoding="utf-8")
    print(f"  {a0:3d}-{a1:3d} s  pas {lo:.0f}-{hi:.0f} Hz ({sz:.0f} Hz)  -> {nazwa}")
