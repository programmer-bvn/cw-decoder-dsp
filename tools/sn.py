"""S/N nagrań w jednej mierze z generatorem (dsp/sn.py).

    python -m tools.sn                       # wszystkie probki/*.wav
    python -m tools.sn probki/szum           # katalog
    python -m tools.sn plik.wav --okno 5

PO CO. 12.10 arkusz X-Ray (out/xray_sn.png) pokazał, że nie wiemy, jakie
S/N mają nasze nagrania, więc nie wiemy, czy trening je pokrywa. Nagranie
jest najpierw przestrajane na 750 Hz (jak przy odczycie), potem mierzone
w oknach — S/N stacji zmienia się z zanikami, a stacja bywa przerywana.
Okno bez ważnego pomiaru to „brak stacji albo S/N < 0 dB" — tego pomiar
nie rozróżnia (walidacja w dsp/sn.py).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsp import config as C
from dsp import frontend, sn, tune


def sn_nagrania(sciezka: Path, okno_s: float = 5.0, retune: bool = True):
    """(lista S/N z ważnych okien, liczba okien) dla jednego pliku."""
    audio = frontend.load_audio(sciezka)
    if retune:
        t, f = tune.track_tone(audio)
        audio = tune.retune(audio, t, f)
    n = int(okno_s * C.SR)
    if audio.size < n:
        n = audio.size
    wyniki, okien = [], 0
    for s in range(0, max(1, audio.size - n + 1), max(1, n // 2)):
        okien += 1
        r = sn.zmierz_sn(audio[s:s + n], C.TONE_CENTER)
        if np.isfinite(r["sn_db"]):
            wyniki.append(r["sn_db"])
    return wyniki, okien


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="S/N nagrań w paśmie "
                                 f"{C.SN_REF_BW_HZ:.0f} Hz wokół tonu")
    ap.add_argument("sciezki", nargs="*", default=["probki"])
    ap.add_argument("--okno", type=float, default=5.0, help="sekundy")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    pliki = []
    for p in map(Path, a.sciezki):
        pliki += sorted(p.glob("*.wav")) if p.is_dir() else [p]

    w = [f"S/N w paśmie {C.SN_REF_BW_HZ:.0f} Hz, okna {a.okno:.0f} s "
         f"(zakład 50%); ważne = okna z tonem stacji",
         f"{'nagranie':52s} {'ważne':>9s} {'10%':>6s} {'mediana':>7s} "
         f"{'90%':>6s}"]
    for p in pliki:
        v, okien = sn_nagrania(p, a.okno)
        if v:
            q = np.percentile(v, [10, 50, 90])
            w.append(f"{p.name[:52]:52s} {len(v):4d}/{okien:<4d} "
                     f"{q[0]:6.1f} {q[1]:7.1f} {q[2]:6.1f}")
        else:
            w.append(f"{p.name[:52]:52s} {0:4d}/{okien:<4d}   brak stacji "
                     f"albo S/N < 0 dB")
    tekst = "\n".join(w)
    print(tekst)
    if a.out:
        Path(a.out).write_text(tekst + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
