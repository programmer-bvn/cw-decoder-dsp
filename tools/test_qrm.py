"""TEST QRM (19.10): druga stacja obok — ile słów stacji docelowej zostaje.

    python -m tools.test_qrm generuj        # probki/qrm/*.wav + .txt
    python -m tools.tester_viterbi probki/qrm

Stacja docelowa: nagranie kluczownika IC-7300MK2 (bez anteny, tekst znany),
przesunięta na środek filtra szumu (tak dostraja operator). Druga stacja:
INNE nagranie kluczownika (inny tekst), przesunięte o df od docelowej,
poziom względem docelowej. Szum naturalny z banku, S/N docelowej 10 dB.
Siatka: df 30 / 60 / 120 / 200 Hz x poziom -6 / 0 / +6 / +12 dB.
Pomiar całą drogą (tester_viterbi): pętla tonu, sieć zboczy, Viterbi.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import soundfile as sf

from dsp import config as C, frontend, sn
from tools.test_tonu import amplituda, pasmo, przesun, szumy, ton_nagrania

KAT = Path("probki/qrm")


def generuj(seed: int) -> None:
    rng = np.random.default_rng(seed)
    pl = sorted(Path("probki/kluczownik").glob("*_14[01][0-9].wav"))      # bez anteny
    stacje = [(frontend.load_audio(p).astype(np.float64), p) for p in pl]
    bank = [b for b in szumy() if (pasmo(b)[1] - pasmo(b)[0]) > 400]     # filtr >= ~500 Hz
    KAT.mkdir(parents=True, exist_ok=True)
    n = 0
    for df in (30.0, 60.0, 120.0, 200.0):
        for poz in (-6.0, 0.0, 6.0, 12.0):
            i = int(rng.integers(len(stacje)))
            j = (i + 1 + int(rng.integers(len(stacje) - 1))) % len(stacje)
            st, p_st = stacje[i]; s2, _ = stacje[j]
            st = st[:int(40 * C.SR)]
            sz = bank[int(rng.integers(len(bank)))]
            lo, hi = pasmo(sz)
            f_cel = 0.5 * (lo + hi) + float(rng.uniform(-20, 20))
            f_2 = f_cel + float(rng.choice([-1, 1])) * df
            st = przesun(st, f_cel - ton_nagrania(st))
            s2 = przesun(s2[:st.size], f_2 - ton_nagrania(s2[:st.size]))
            while sz.size < st.size:
                sz = np.concatenate([sz, sz[::-1]])
            sz = sz[:st.size] / (np.std(sz[:st.size]) + 1e-12) * 0.05
            amp = sn.amp_dla_sn(10.0, sz, f_cel) / max(amplituda(st, f_cel), 1e-9)
            a_cel = amp * amplituda(st, f_cel)
            s2 = s2 * a_cel * 10 ** (poz / 20) / max(amplituda(s2, f_2), 1e-9)
            x = sz + amp * st
            x[:s2.size] += s2[:x.size]
            x = (x * (0.5 / np.max(np.abs(x)))).astype(np.float32)
            nazwa = f"qrm_df{df:03.0f}_poz{poz:+03.0f}_{n:02d}"
            sf.write(KAT / f"{nazwa}.wav", x, C.SR)
            opis = p_st.with_suffix(".txt").read_text(encoding="utf-8")
            nad = re.search(r"^nadane:\s*(.*)$", opis, re.M).group(1)
            (KAT / f"{nazwa}.txt").write_text(
                "# TEST QRM — tools/test_qrm.py (19.10)\n\n"
                f"nadane:        {nad}\n"
                f"stacja:        {p_st.name}, ton {f_cel:.0f} Hz (środek filtra szumu {0.5 * (lo + hi):.0f} Hz), S/N 10 dB\n"
                f"druga stacja:  {stacje[j][1].name}, ton {f_2:.0f} Hz (df {df:.0f} Hz), poziom {poz:+.0f} dB wzgl. docelowej\n",
                encoding="utf-8")
            n += 1
    print(f"zapisano {n} przypadków w {KAT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["generuj"]); ap.add_argument("--seed", type=int, default=3)
    generuj(ap.parse_args().seed)
