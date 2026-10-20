"""TESTER: grzebień z cepstrum + Viterbi (dsp/viterbi) obok tabeli z bramką (19.10).

    python -m tools.tester_viterbi [katalog ...]

Te same nagrania, ten sam przebieg sieci zboczy_3 i ta sama pętla tonu.
Tabela = tools.tester_zbocza.droga (okna 8 s, bramka S/N >= 3 dB i pewność
grzebienia >= 0,85, histereza); Viterbi = bez bramki, jednostka: kandydatki
z cepstrum i z grzebienia, wygrywa lepszy wynik Viterbiego.
Słowa jak w testerach: każde słowo z pola „nadane:" szukane w tekście bez spacji.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsp import config as C, frontend, tune, viterbi as V, zbocza as Z

KATALOGI = ["probki", "probki/kluczownik", "probki/reka", "probki/zaszumione",
            "probki/zaszumione_real", "probki/qrm"]


def main() -> None:
    from tensorflow import keras
    from tools import nagrania, wav2net
    from tools import tester_zbocza as T
    argv0 = sys.argv[1:]
    model = "runs/zbocza_3/best.keras"
    if "--model" in argv0:
        i = argv0.index("--model"); model = argv0[i + 1]; del sys.argv[1 + i:1 + i + 2]
    mz = keras.models.load_model(model)
    print(f"model zboczy: {model}" + ("" if "--bez-kasowania" in sys.argv else "  + przeciwsygnał"))
    # --filtr HZ: po przestrojeniu filtr ±HZ wokół tonu, zanim sieć zboczy
    # (19.10, test QRM: obca stacja 60 Hz obok mieszała się w przebieg)
    argv = sys.argv[1:]
    filtr = None
    if "--filtr" in argv:
        i = argv.index("--filtr"); filtr = float(argv[i + 1]); del argv[i:i + 2]
    # przeciwsygnał — obce kluczowane stacje odjęte przed przestrojeniem, jak
    # w tools.odczyt (dsp/przeciwsygnal, 19.10: 1321 -> 1350/1541);
    # --bez-kasowania: droga sprzed 19.10 (odniesienie out/viterbi_3_zb3.txt)
    kasuj = "--bez-kasowania" not in argv
    if not kasuj:
        argv.remove("--bez-kasowania")
    from dsp import przeciwsygnal as PS
    # ton CW radia z opisu nagrania (pole „ton CW:", CI-V; 20.10) — środek
    # filtra pętli tonu; --bez-tonu-cw: pętla sama szuka środka (jak przed 20.10)
    z_tonem = "--bez-tonu-cw" not in argv
    if not z_tonem:
        argv.remove("--bez-tonu-cw")
    from tools.radio import civ
    katalogi = argv or KATALOGI
    n8, k4 = 8 * C.SR, 4 * C.SR
    suma = Counter()
    print(f"{'nagranie':58s} {'tabela':>8s} {'viterbi':>8s}   u wybrane")
    for kat in katalogi:
        sk = Counter()
        for p in sorted(Path(kat).glob("*.wav")):
            chce = nagrania.oczekiwane(p.with_suffix(".txt"))
            if not chce:
                continue
            a = frontend.load_audio(p)
            t, f = tune.track_tone(a, ton_cw=civ.ton_z_opisu(p.with_suffix(".txt")) if z_tonem else None)
            if kasuj and np.isfinite(f).any():
                a, _ = PS.kasuj_obce(a, float(np.nanmedian(f)), tor=(t, f))
            a = tune.retune(a, t, f)
            if filtr:
                a = tune.bandpass(a, C.TONE_CENTER, filtr)
            g = wav2net.auto_gain_db(a); a = np.clip(a * 10 ** (g / 20), -1, 1).astype(np.float32)
            pz = Z.przebieg_sieci(a, mz)
            starty = list(range(0, max(1, a.size - n8 + 1), k4))
            if starty[-1] < a.size - n8:
                starty.append(a.size - n8)
            r5 = n8 // Z.HOP_ZBOCZA
            oo = [Z.odcinki_z_przebiegu(pz[s // Z.HOP_ZBOCZA:s // Z.HOP_ZBOCZA + r5],
                                        histereza=(Z.PROG_WL, Z.PROG_WYL)) for s in starty]
            tab, nuty_, _, _ = T.droga(a, oo, starty, n8)
            u_c, _ = V.jednostka_cepstrum(pz)
            u_g = float(np.median([x[2] for x in nuty_])) if nuty_ else float("nan")
            vit, u, _ = V.dekoduj_najlepiej(pz, [u_c, u_g])
            s_t, s_v = T.slowa(tab, chce)[0], T.slowa(vit.replace(" ", ""), chce)[0]
            sk["tab"] += s_t; sk["vit"] += s_v; sk["n"] += len(chce)
            print(f"{str(p)[:58]:58s} {s_t:3d}/{len(chce):<3d} {s_v:4d}/{len(chce):<3d}  {u * 1000:5.0f} ms", flush=True)
            if s_v != s_t:
                print(f"      tabela : {tab[:100]}\n      viterbi: {vit[:110]}", flush=True)
        print(f"  RAZEM {kat}: tabela {sk['tab']}/{sk['n']}, viterbi {sk['vit']}/{sk['n']}", flush=True)
        suma.update(sk)
    print(f"RAZEM: tabela {suma['tab']}/{suma['n']}, viterbi {suma['vit']}/{suma['n']}")


if __name__ == "__main__":
    main()
