"""CZAS ODCZYTU NA KV260 — sieć na ARM (numpy) kontra DPU, z podziałem na etapy (07.11).

Pierwszy pomiar (bez DPU, HISTORIA „KV260: odczyt na samym ARM") dał
0,76× czasu nagrania, połowa w przygotowaniu dźwięku. Tu te same nagrania,
etapy osobno: wczytanie, przygotuj (pętla tonu, kasowanie obcych), sieć,
dekoder (Viterbi). Na płytce:
    sudo xmutil loadapp kv260-benchmark-b4096
    ~/venv/bin/python -m tools.pomiar_kv260 PLIKI.wav [--xmodel ...] [--npz ...]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pliki", type=Path, nargs="+")
    ap.add_argument("--npz", type=Path, default=Path("runs/zbocza_5/siec.npz"))
    ap.add_argument("--xmodel", type=Path, default=Path("out/dpu/xmodel/zbocza_5.xmodel"))
    a = ap.parse_args()
    import soundfile as sf
    from dsp import frontend, viterbi as V, zbocza as Z
    from dsp.siec_dpu import SiecDPU
    from dsp.siec_numpy import SiecNumpy
    from tools import nagrania, odczyt, tester_zbocza as T
    from tools.radio import civ
    sieci = {"numpy": SiecNumpy(a.npz), "DPU": SiecDPU(a.xmodel)}
    suma = {}
    print(f"{'nagranie':34s} {'dł.':>6s} {'wczyt':>6s} {'przyg':>6s} | sieć: numpy / DPU | dekoder | słowa numpy / DPU")
    for p in a.pliki:
        d = sf.info(p).duration
        chce = nagrania.oczekiwane(p.with_suffix(".txt"))
        t = time.perf_counter(); x = frontend.load_audio(p); t_w = time.perf_counter() - t
        t = time.perf_counter(); x, _ = odczyt.przygotuj(x, ton_cw=civ.ton_z_opisu(p.with_suffix(".txt")))
        t_p = time.perf_counter() - t
        wyn = {}
        for nazwa, m in sieci.items():
            t = time.perf_counter(); pz = Z.przebieg_sieci(x, m); t_s = time.perf_counter() - t
            t = time.perf_counter()
            tekst = V.dekoduj_najlepiej(pz, [V.jednostka_cepstrum(pz)[0], odczyt.jednostka_grzebienia(pz)])[0]
            t_d = time.perf_counter() - t
            sl = T.slowa(tekst.replace(" ", ""), chce)[0] if chce else -1
            wyn[nazwa] = (t_s, t_d, sl, tekst)
        for k, v in (("dł", d), ("wczyt", t_w), ("przyg", t_p), ("numpy", wyn["numpy"][0]), ("DPU", wyn["DPU"][0]),
                     ("dek", wyn["DPU"][1])):
            suma[k] = suma.get(k, 0) + v
        print(f"{p.name[:34]:34s} {d:5.1f}s {t_w:5.2f}s {t_p:5.2f}s | {wyn['numpy'][0]:6.2f}s / {wyn['DPU'][0]:5.2f}s |"
              f" {wyn['DPU'][1]:5.2f}s | {wyn['numpy'][2]:3d} / {wyn['DPU'][2]:3d} / {len(chce)}"
              f"{'' if wyn['numpy'][3] == wyn['DPU'][3] else '  TEKST RÓŻNY'}", flush=True)
    D = suma["dł"]
    print(f"RAZEM {D:.0f} s nagrań: wczytanie {suma['wczyt'] / D:.3f}×, przygotuj {suma['przyg'] / D:.3f}×,"
          f" sieć numpy {suma['numpy'] / D:.3f}× / DPU {suma['DPU'] / D:.3f}×, dekoder {suma['dek'] / D:.3f}×")
    cn = (suma["wczyt"] + suma["przyg"] + suma["numpy"] + suma["dek"]) / D
    cd = (suma["wczyt"] + suma["przyg"] + suma["DPU"] + suma["dek"]) / D
    print(f"  cały odczyt: z siecią na ARM {cn:.3f}× czasu nagrania, z DPU {cd:.3f}×")


if __name__ == "__main__":
    main()
