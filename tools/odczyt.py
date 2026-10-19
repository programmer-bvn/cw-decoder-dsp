"""ODCZYT: plik audio -> tekst, całą drogą dwuetapową (19.10).

    python -m tools.odczyt nagranie.wav
    python -m tools.odczyt nagranie.wav --odcinki      # też podział na odcinki

Droga (każdy krok zmierzony, HISTORIA 17-19.10):
  1. tune.track_tone (ton kluczowany); przeciwsygnał — obce kluczowane
     stacje odtworzone i odjęte (dsp/przeciwsygnal); retune na 750 Hz
  2. sieć zboczy (runs/zbocza_3): P(klucz wciśnięty) co 5 ms
  3. jednostka: cepstrum przebiegu (u + 3u + 7u) i grzebień (mediana okien)
  4. viterbi.dekoduj_najlepiej: odcinki zgodne z gramatyką Morse'a, bez
     bramki; jednostkę wybiera wynik Viterbiego.
Na nagraniach z treścią (19.10): 821/828 słów, tabela z bramką 685.
Z przeciwsygnałem (19.10): wszystkie zestawy 1350/1541 (bez: 1321), QRM
75 -> 104, poza QRM żadne nagranie bez zmian.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsp import config as C, frontend, grzebien, przeciwsygnal as PS, tune, viterbi as V, zbocza as Z

MODEL_ZBOCZY = Path("runs/zbocza_3/best.keras")


def przygotuj(a: np.ndarray, kasuj: bool = True) -> tuple[np.ndarray, float]:
    from tools import wav2net
    t, f = tune.track_tone(a)
    ton = float(np.nanmedian(f)) if np.isfinite(f).any() else float("nan")
    if kasuj:
        a, _ = PS.kasuj_obce(a, ton)
    a = tune.retune(a, t, f)
    g = wav2net.auto_gain_db(a)
    return np.clip(a * 10 ** (g / 20), -1, 1).astype(np.float32), ton


def jednostka_grzebienia(pz: np.ndarray) -> float:
    """Mediana jednostki grzebienia z okien 8 s (P z histerezą)."""
    n8, k4 = int(8 / V.DT), int(4 / V.DT)
    us = []
    for s in range(0, max(1, pz.size - n8 + 1), k4):
        oo = Z.odcinki_z_przebiegu(pz[s:s + n8], histereza=(Z.PROG_WL, Z.PROG_WYL))
        r = grzebien.dopasuj_grzebien(oo)
        if np.isfinite(r["u_s"]) and r["pewnosc"] >= 0.85:
            us.append(r["u_s"])
    return float(np.median(us)) if us else float("nan")


def odczytaj(sciezka: Path, model=None) -> dict:
    from tensorflow import keras
    model = model or keras.models.load_model(MODEL_ZBOCZY)
    a, ton = przygotuj(frontend.load_audio(sciezka))
    pz = Z.przebieg_sieci(a, model)
    u_c, _ = V.jednostka_cepstrum(pz)
    u_g = jednostka_grzebienia(pz)
    tekst, u, wynik = V.dekoduj_najlepiej(pz, [u_c, u_g])
    return dict(tekst=tekst, u=u, wpm=1.2 / u if np.isfinite(u) else float("nan"),
                ton=ton, u_cepstrum=u_c, u_grzebien=u_g, wynik=wynik, p=pz)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pliki", type=Path, nargs="+")
    ap.add_argument("--odcinki", action="store_true")
    a = ap.parse_args()
    from tensorflow import keras
    model = keras.models.load_model(MODEL_ZBOCZY)
    for p in a.pliki:
        r = odczytaj(p, model)
        print(f"{p.name}: ton {r['ton']:.0f} Hz, tempo {r['wpm']:.1f} WPM "
              f"(u: cepstrum {r['u_cepstrum'] * 1000:.0f} ms, grzebień {r['u_grzebien'] * 1000:.0f} ms, "
              f"wybrane {r['u'] * 1000:.0f} ms), wynik {r['wynik']:.3f}")
        print(f"  {r['tekst']}")
        if a.odcinki:
            _, odc = V.dekoduj(r["p"], r["u"])
            print("  " + " ".join(f"{n}:{(b - a_) * V.DT * 1000:.0f}" for n, a_, b in odc[:80]))


if __name__ == "__main__":
    main()
