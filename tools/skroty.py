"""SKRÓTY: czy model czyta stację, czy jakąś przypadkową cechę generatora.

PO CO. 14.10 model z walidacją 92% nie widział silnej, CZYSTEJ stacji —
odpowiadał „cisza" przez całe okno. Przyczyna: druga stacja (QRM, klasa 0)
była generowana bez chirpu, a stacja docelowa prawie zawsze z chirpem.
Sieć nauczyła się skrótu „bez chirpu = obca stacja". Walidacja tego nie
pokazała, bo walidacja ma ten sam skrót co trening. Wyszło dopiero w teście
„po jednym" (HISTORIA „Noc 14.10").

Test: klipy ze stacją przy S/N 15-30 dB, wszystkie zjawiska kanału
i nadajnika wyłączone, potem włączane pojedynczo, na koniec pełny realizm.
Model, który czyta STACJĘ, nie może tracić na czystym sygnale — czysty
sygnał jest najłatwiejszy. Miara: odsetek kroków 40 ms ze znakiem, w których
model podał dobrą klasę (jak etykiety fcn).

    python -m tools.skroty --model runs/fcn_4000000_czesci_sn/best.keras
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

# Stan „wszystko wyłączone". S/N z górnej połowy — sygnał ma być łatwy.
CZYSTO = dict(ZAPADANIE_PROB=0.0, QSB_DEPTH=(0.0, 0.0), AGC_PROB=0.0,
              REAL_NOISE_PROB=0.0, QRM_PROB=0.0, QRN_PROB=0.0, HUM_PROB=0.0,
              GAP_JITTER=(0.0, 0.0), DRIFT_HZ=0.0, SN_DB=(15.0, 30.0))
POJEDYNCZO = [
    ("zapadanie zasilacza (chirp+sag) ZAWSZE", {"ZAPADANIE_PROB": 1.0}),
    ("zanik (QSB)", {"QSB_DEPTH": None}),
    ("ARW", {"AGC_PROB": 1.0}),
    ("tło z banku", {"REAL_NOISE_PROB": 1.0}),
    ("druga stacja (QRM)", {"QRM_PROB": 1.0}),
    ("trzaski (QRN)", {"QRN_PROB": 1.0}),
    ("dłuższe przerwy", {"GAP_JITTER": None}),
]
# Próg ostrzeżenia: o tyle punktów czysty sygnał może być gorszy od pełnego
# realizmu, zanim to uznamy za skrót. 14.10 było 42% wobec 87%.
PROG_PP = 10.0


def _zestaw(ustaw: dict, n: int, ziarno: int):
    oryg = {k: getattr(C, k) for k in CZYSTO}
    try:
        for k, v in CZYSTO.items():
            setattr(C, k, v)
        for k, v in ustaw.items():
            setattr(C, k, oryg[k] if v is None else v)
        rng = np.random.default_rng(ziarno)
        X, Y = [], []
        while len(X) < n:
            t = "".join(rng.choice(list(C.ALPHABET[1:]), C.CHARS_PER_CLIP))
            a, m = radio.receive(t, rng, realism=True)
            X.append(frontend.to_net_image(a))
            Y.append(frontend.frame_labels(m["spans"], m["offset"], t, a.size))
        return np.stack(X)[..., None], np.stack(Y)
    finally:
        for k, v in oryg.items():
            setattr(C, k, v)


def _trafnosc(model, X, Y) -> float:
    pr = model.predict(X, verbose=0).argmax(-1)
    # Etykiety ramek -> kroki 40 ms tak jak w treningu (_do_float).
    a, b = Y[:, 0::2], Y[:, 1::2]
    yk = np.where(a > 0, a, b)
    mk = yk > 0
    return 100.0 * float(np.mean(pr[mk] == yk[mk]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="test skrótów modelu fcn")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    from dsp.model import load_model
    model = load_model(Path(a.model))
    if len(model.output_shape) != 3:
        print("skroty: tylko model fcn (znak na krok)")
        return 1

    wiersze = [("wszystko wyłączone (czysta stacja)", {})] + POJEDYNCZO + [
        ("pełny realizm", {k: None for k in CZYSTO if k != "SN_DB"})]
    w = [f"SKRÓTY: {a.model}",
         f"  {a.n} klipów na wiersz, S/N 15-30 dB; trafne kroki ze znakiem",
         ""]
    wyniki = {}
    for nazwa, ustaw in wiersze:
        X, Y = _zestaw(ustaw, a.n, 14)
        wyniki[nazwa] = _trafnosc(model, X, Y)
        w.append(f"  {nazwa:40s} {wyniki[nazwa]:5.1f}%")
    czysto = wyniki[wiersze[0][0]]
    pelny = wyniki["pełny realizm"]
    w.append("")
    if czysto < pelny - PROG_PP:
        w.append(f"  UWAGA: czysta stacja {czysto:.1f}% < pełny realizm "
                 f"{pelny:.1f}% — model korzysta z jakiejś cechy generatora")
        w.append("  zamiast ze stacji. Wiersz, który najbardziej podnosi "
                 "wynik, wskazuje tę cechę.")
    else:
        w.append(f"  OK: czysta stacja {czysto:.1f}%, pełny realizm "
                 f"{pelny:.1f}% — skrótu nie widać.")
    tekst = "\n".join(w)
    print(tekst)
    if a.out:
        Path(a.out).write_text(tekst + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
