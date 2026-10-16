"""TRAFNOŚĆ fcn rozbita na to, co liczy się przy odczycie, i to, co nie.

PO CO (16.10). val_accuracy liczy WSZYSTKIE kroki ze znakiem, także znaki
ucięte krawędzią okna, na których z zasady nie da się trafić (pół kreski).
Przy 12-18 WPM to 39-44% kroków ze znakiem i model zgaduje na nich w ~46%
(HISTORIA „Wolne tempo"). Przy odczycie nagrań liczą się środki okien
(dsp.sekwencja.kroki_nagrania), więc miarą, która mówi o pracy na paśmie,
jest trafność na znakach CAŁYCH i w ŚRODKU okna. Model z maską uciętych
(--maska-uciete) może wypaść gorzej na val_accuracy, a lepiej tutaj.

Syntetyk, pełny realizm, te same klipy dla każdego modelu (stałe ziarno),
więc porównanie modeli jest uczciwe — i ma dużo mniejszy rozrzut niż
26 słów z nagrań.

    python -m tools.trafnosc runs/fcn_A/best.keras runs/fcn_B/best.keras
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsp import config as C
from dsp import frontend, radio

PRZEDZIALY = ((12, 18), (18, 25), (25, 32))
# Środek okna = DOKŁADNIE to, co odczyt nagrań bierze z każdego okna:
# dsp.sekwencja.kroki_nagrania(krok_okna=32) zszywa środkowe 32 ramki,
# czyli kroki 24-39 z 64. Do 16.10 13:50 było 16-47 (połowa okna) —
# zawierało kroki, których odczyt nie używa, i krzywdziło model z maską.
SRODEK = (24, 40)


def zestaw(wpm_od: float, wpm_do: float, n: int, ziarno: int):
    """Klipy z tempem w [wpm_od, wpm_do), etykiety kroków i maska uciętych."""
    stare = C.WPM, C.WPM_JITTER
    C.WPM, C.WPM_JITTER = 0.5 * (wpm_od + wpm_do), 0.5 * (wpm_do - wpm_od)
    try:
        rng = np.random.default_rng(ziarno)
        X, Y, U = [], [], []
        while len(X) < n:
            t = "".join(rng.choice(list(C.ALPHABET[1:]), C.CHARS_PER_CLIP))
            a, m = radio.receive(t, rng, realism=True)
            X.append(frontend.to_net_image(a))
            Y.append(frontend.frame_labels(m["spans"], m["offset"], t, a.size))
            u = np.zeros(C.IMG_FRAMES, bool)
            for _, s0, s1 in m["spans"]:
                fa = frontend.sample_to_window_frame(s0 + m["offset"], a.size)
                fb = frontend.sample_to_window_frame(s1 + m["offset"], a.size)
                if fa < 0 or fb > C.IMG_FRAMES:
                    u[max(0, int(np.ceil(fa))):
                      min(C.IMG_FRAMES, int(np.floor(fb)) + 1)] = True
            U.append(u)
    finally:
        C.WPM, C.WPM_JITTER = stare
    Y, U = np.stack(Y), np.stack(U)
    yk = np.where(Y[:, 0::2] > 0, Y[:, 0::2], Y[:, 1::2])   # jak _do_float
    uk = U[:, 0::2] | U[:, 1::2]
    return np.stack(X)[..., None], yk, uk


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="trafność fcn: całe / ucięte / środek")
    ap.add_argument("modele", nargs="+")
    ap.add_argument("--n", type=int, default=400, help="klipów na przedział")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    from dsp.model import load_model
    dane = [(f"{lo}-{hi} WPM", zestaw(lo, hi, a.n, 1610 + lo))
            for lo, hi in PRZEDZIALY]
    w = [f"TRAFNOŚĆ (syntetyk, pełny realizm, {a.n} klipów na przedział, "
         f"trafne kroki ze znakiem [%])", ""]
    for sciezka in a.modele:
        model = load_model(Path(sciezka))
        w.append(f"  {sciezka}")
        w.append(f"    {'tempo':12s} {'wszystkie':>10s} {'całe':>7s} "
                 f"{'ucięte':>7s} {'środek':>7s}  udział uciętych")
        for nazwa, (X, yk, uk) in dane:
            pr = model.predict(X, verbose=0).argmax(-1)
            zn = yk > 0
            ok = pr == yk
            sr = np.zeros_like(zn)
            sr[:, SRODEK[0]:SRODEK[1]] = True

            def p(m):
                return 100.0 * ok[m].mean() if m.any() else float("nan")
            w.append(f"    {nazwa:12s} {p(zn):9.1f} {p(zn & ~uk):7.1f} "
                     f"{p(zn & uk):7.1f} {p(zn & sr):7.1f}  "
                     f"{100.0 * (zn & uk).sum() / max(1, zn.sum()):5.1f}%")
        w.append("")
    tekst = "\n".join(w)
    print(tekst)
    if a.out:
        Path(a.out).write_text(tekst + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
