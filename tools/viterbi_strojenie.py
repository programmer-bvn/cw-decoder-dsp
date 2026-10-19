"""STROJENIE VITERBIEGO (19.10): przebiegi sieci zboczy raz, siatka ustawień na procesorze.

    python -m tools.viterbi_strojenie licz  out/viterbi_p.npz [--model runs/zbocza_3/best.keras]
    python -m tools.viterbi_strojenie siatka out/viterbi_p.npz

`licz`: dla każdego nagrania z treścią (katalogi jak tester_viterbi) P co 5 ms,
jednostki z cepstrum i z grzebienia, oczekiwane słowa — do jednego pliku.
`siatka`: WAGA_EMISJI x KARA_DLUGOSCI (dsp/viterbi), słowa razem i osobno
dla zestawów. Uwaga na dopasowanie do zestawu testowego: ustawienie wybierać
po WSZYSTKICH zestawach naraz, nie po jednym.
"""

from __future__ import annotations

import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsp import viterbi as V

KATALOGI = ["probki", "probki/kluczownik", "probki/reka", "probki/zaszumione",
            "probki/zaszumione_real", "probki/qrm"]


def licz(wyj: Path, model_sc: str) -> None:
    from tensorflow import keras
    from dsp import frontend, zbocza as Z
    from tools import nagrania
    from tools.odczyt import jednostka_grzebienia, przygotuj
    mz = keras.models.load_model(model_sc)
    dane = []
    for kat in KATALOGI:
        for p in sorted(Path(kat).glob("*.wav")):
            chce = nagrania.oczekiwane(p.with_suffix(".txt"))
            if not chce:
                continue
            a, _ = przygotuj(frontend.load_audio(p))
            pz = Z.przebieg_sieci(a, mz).astype(np.float32)
            dane.append(dict(kat=kat, plik=p.name, chce=chce, p=pz,
                             u_c=V.jednostka_cepstrum(pz)[0], u_g=jednostka_grzebienia(pz)))
            print(f"\r  {len(dane)} nagrań", end="", flush=True)
    print()
    with open(wyj, "wb") as f:
        pickle.dump(dict(model=model_sc, dane=dane), f)
    print(f"zapisano {wyj}")


def siatka(plik: Path) -> None:
    from tools import nagrania
    from tools import tester_zbocza as T
    d = pickle.load(open(plik, "rb"))["dane"]
    print(f"{'WAGA':>5s} {'KARA':>5s}  " + "  ".join(f"{k.split('/')[-1][:10]:>10s}" for k in KATALOGI) + "   RAZEM")
    for waga in (0.15, 0.3, 0.6):
        for kara in (2.0, 4.0, 8.0):
            V.WAGA_EMISJI, V.KARA_DLUGOSCI = waga, kara
            sk = Counter(); n = Counter()
            for x in d:
                tekst = V.dekoduj_najlepiej(x["p"], [x["u_c"], x["u_g"]])[0]
                sk[x["kat"]] += T.slowa(tekst.replace(" ", ""), x["chce"])[0]
                n[x["kat"]] += len(x["chce"])
            print(f"{waga:5.2f} {kara:5.1f}  " + "  ".join(f"{sk[k]:4d}/{n[k]:<5d}" for k in KATALOGI)
                  + f"   {sum(sk.values())}/{sum(n.values())}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "licz":
        model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "runs/zbocza_3/best.keras"
        licz(Path(sys.argv[2]), model)
    else:
        siatka(Path(sys.argv[2]))
