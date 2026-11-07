"""INT8 SIECI ZBOCZY — czy 8 bitów jak w DPU psuje odczyt (07.11, KV260).

Kalibracja na klipach z generatora (nie na nagraniach testowych — pozycje
nie widzą testu), potem ten sam test słów co tools.tester_viterbi (te same
katalogi, ta sama obróbka: pętla tonu, kasowanie obcych, wzmocnienie)
i halucynacje na pokoju: sieć float (dsp.siec_numpy) obok int8
(dsp.siec_int8), na tych samych P różnice logitów.

    python -m tools.int8_ocena [--wagi runs/zbocza_5/siec.npz] [--klipow 60] [--out PLIK]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsp import config as C, frontend, przeciwsygnal as PS, radio, tune, viterbi as V, zbocza as Z
from dsp.siec_int8 import SiecInt8
from dsp.siec_numpy import SiecNumpy


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--wagi", type=Path, default=Path("runs/zbocza_5/siec.npz"))
    ap.add_argument("--klipow", type=int, default=60)
    ap.add_argument("--bank", type=Path, default=Path("/mnt/w/nagrania_cw/negatywy"))
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    from tools import nagrania, odczyt, tester_zbocza as T, wav2net
    from tools.radio import civ
    from tools.tester_viterbi import KATALOGI
    linie = []

    def pisz(s):
        print(s, flush=True); linie.append(s)

    mf, mq = SiecNumpy(a.wagi), SiecInt8(a.wagi)
    rng = np.random.default_rng(2026_11_07)
    mq.zacznij_kalibracje()
    for _ in range(a.klipow):
        tekst = "".join(rng.choice(list(C.ALPHABET[1:]), 8))
        au, _ = radio.receive(tekst, rng, n_samples=C.CLIP_SAMPLES, wpm=float(rng.uniform(*C.TEMPO_ZAKRES)),
                              realism=True, zakresy=Z.ZAKRESY_ZBOCZY_6)
        Z.przebieg_sieci(au.astype(np.float32), mq)
    p = mq.koniec_kalibracji()
    cel = a.wagi.with_name(a.wagi.stem + "_int8_pozycje.npz")
    mq.zapisz(cel)
    pisz(f"INT8 — wagi {a.wagi}, kalibracja na {a.klipow} klipach generatora -> {cel}")
    pisz("  pozycje aktywacji (skala 2^-p): " + ", ".join(f"{k}:{v}" for k, v in p.items()))

    def czytaj(pz):
        u_c, _ = V.jednostka_cepstrum(pz)
        return V.dekoduj_najlepiej(pz, [u_c, odczyt.jednostka_grzebienia(pz)])[0]

    sl, roz, zgodne, ramek = Counter(), [], 0, 0
    for kat in KATALOGI:
        for pth in sorted(Path(kat).glob("*.wav")):
            chce = nagrania.oczekiwane(pth.with_suffix(".txt"))
            if not chce:
                continue
            x = frontend.load_audio(pth)
            t, f = tune.track_tone(x, ton_cw=civ.ton_z_opisu(pth.with_suffix(".txt")))
            if np.isfinite(f).any():
                x, _ = PS.kasuj_obce(x, float(np.nanmedian(f)), tor=(t, f))
            x = tune.retune(x, t, f)
            g = wav2net.auto_gain_db(x); x = np.clip(x * 10 ** (g / 20), -1, 1).astype(np.float32)
            pf, pq = Z.przebieg_sieci(x, mf), Z.przebieg_sieci(x, mq)
            roz.append(np.abs(pf - pq)); zgodne += int(np.sum((pf > 0.5) == (pq > 0.5))); ramek += pf.size
            sf_, sq_ = (T.slowa(czytaj(pp).replace(" ", ""), chce)[0] for pp in (pf, pq))
            sl["f" + kat] += sf_; sl["q" + kat] += sq_; sl["n" + kat] += len(chce)
        pisz(f"  {kat:24s} słowa float {sl['f' + kat]:4d}  int8 {sl['q' + kat]:4d}  / {sl['n' + kat]}")
    r = np.concatenate(roz)
    pisz(f"RAZEM słowa: float {sum(sl['f' + k] for k in KATALOGI)}, int8 {sum(sl['q' + k] for k in KATALOGI)}"
         f" / {sum(sl['n' + k] for k in KATALOGI)}")
    pisz(f"  |P_float - P_int8|: średnio {r.mean():.4f}, 99. percentyl {np.percentile(r, 99):.3f}, maks {r.max():.3f};"
         f" ramki zgodne (P>0,5) {zgodne / ramek:.4%}")

    tr = a.bank / "TRENING.txt"
    trening = set(tr.read_text(encoding="utf-8").split()) if tr.exists() else set()
    import soundfile as sf
    for kat in ("pokoj_mikrofon", "kiwi_am", "kiwi_ssb"):
        if not (a.bank / kat).is_dir():
            continue
        zn, mi = Counter(), 0.0
        for pth in sorted((a.bank / kat).glob("*.wav")):
            if f"{kat}/{pth.name}" in trening:
                continue
            x, _ = odczyt.przygotuj(frontend.load_audio(pth))
            for k, m in (("f", mf), ("q", mq)):
                zn[k] += sum(1 for z in czytaj(Z.przebieg_sieci(x, m)) if z != " ")
            mi += sf.info(pth).duration / 60
        pisz(f"  halucynacje {kat:15s} ({mi:4.0f} min): float {zn['f'] / mi:5.1f}, int8 {zn['q'] / mi:5.1f} znaków/min")
    if a.out:
        a.out.write_text("\n".join(linie) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
