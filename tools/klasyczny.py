"""DEKODER KLASYCZNY (19.10) — punkt odniesienia: detektor obwiedni, komparator,
pomiar długości, drzewko. Tak działa typowy dekoder na mikrokontrolerze.

    python -m tools.klasyczny                     # porównanie z Viterbim (CER)

1. filtr pasmowy ±100 Hz wokół tonu (po tune.track_tone + retune na 750 Hz),
   prostowanie, wygładzenie 10 ms;
2. komparator: próg = połowa między śledzonym szczytem (szybki narost, wolny
   spadek) a śledzonym tłem (odwrotnie), histereza ±10% rozpiętości;
3. długości impulsów i przerw;
4. jednostka adaptacyjna: średnia krocząca impulsów uznanych za kropki;
   kropka < 2 u <= kreska; przerwa < 2 u w znaku, < 5 u między literami,
   dalej między słowami;
5. drzewko (tablica kodów ITU, jak dsp.grzebien) — nieznany kod: „*".
"""

from __future__ import annotations

import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsp import config as C, frontend, tune, viterbi as V
from dsp.grzebien import _ODWROTNY, NIEZNANY


def obwiednia(a: np.ndarray) -> np.ndarray:
    x = tune.bandpass(a, C.TONE_CENTER, 100.0, taps=255)
    k = int(0.010 * C.SR)
    return np.convolve(np.abs(x), np.ones(k) / k, mode="same")


def komparator(e: np.ndarray, sr: int = C.SR) -> np.ndarray:
    """Próg adaptacyjny z histerezą (przetwarzanie próbka po próbce, co 1 ms)."""
    krok = sr // 1000
    e = e[::krok]
    szczyt, tlo = float(e.max()) * 0.5, float(np.median(e))
    a_sz, a_tl = 1 - np.exp(-1 / 2000.0), 1 - np.exp(-1 / 2000.0)    # ~2 s
    stan = False
    out = np.zeros(e.size, bool)
    for i, v in enumerate(e):
        szczyt = v if v > szczyt else szczyt + a_sz * (v - szczyt)
        tlo = v if v < tlo else tlo + a_tl * (v - tlo)
        prog = 0.5 * (szczyt + tlo); h = 0.1 * (szczyt - tlo)
        if stan and v < prog - h:
            stan = False
        elif not stan and v > prog + h:
            stan = True
        out[i] = stan
    return out                                                        # co 1 ms


def dekoduj(a: np.ndarray) -> str:
    on = komparator(obwiednia(a))
    zm = np.flatnonzero(np.diff(on.astype(np.int8))) + 1
    gr = np.r_[0, zm, on.size]
    odc = [(bool(on[s]), (e - s) / 1000.0) for s, e in zip(gr[:-1], gr[1:])]
    znaki = [d for z, d in odc if z]
    if not znaki:
        return ""
    u = float(np.percentile(znaki, 25))                               # start: krótkie impulsy
    tekst, kod = [], ""
    for z, d in odc:
        if z:
            if d < 0.3 * u:
                continue                                              # trzask
            if d < 2 * u:
                kod += "."; u = 0.9 * u + 0.1 * d
            else:
                kod += "-"; u = 0.9 * u + 0.1 * d / 3
        else:
            if d >= 2 * u and kod:
                tekst.append(_ODWROTNY.get(kod, NIEZNANY)); kod = ""
                if d >= 5 * u:
                    tekst.append(" ")
    if kod:
        tekst.append(_ODWROTNY.get(kod, NIEZNANY))
    return " ".join("".join(tekst).split())


def cer(a: str, b: str) -> int:
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (ca != cb))
    return d[len(b)]


def main() -> None:
    from tools.odczyt import przygotuj
    dane = pickle.load(open("out/viterbi_p.npz", "rb"))["dane"]
    gr = defaultdict(lambda: [0, 0, 0, 0])
    for x in dane:
        if x["kat"] not in ("probki/kluczownik", "probki/zaszumione", "probki/zaszumione_real") or "reka" in x["plik"]:
            continue
        p = Path(x["kat"]) / x["plik"]
        nad = re.search(r"^nadane:\s*(.*)$", p.with_suffix(".txt").read_text(encoding="utf-8"), re.M).group(1).replace(" ", "")
        a, _ = przygotuj(frontend.load_audio(p))
        kl = dekoduj(a).replace(" ", "")
        vt = V.dekoduj_najlepiej(x["p"], [x["u_c"], x["u_g"]])[0].replace(" ", "")
        m = re.search(r"sn([+-]\d+)", x["plik"]); sn = m.group(1) if m else "czyste"
        rodz = {"probki/kluczownik": "kluczownik", "probki/zaszumione": "syntetyk + szum",
                "probki/zaszumione_real": "kluczownik + szum"}[x["kat"]]
        g = gr[(rodz, sn)]; g[0] += cer(kl, nad); g[1] += cer(vt, nad); g[2] += len(nad); g[3] += 1
        if sn in ("czyste", "+6") and g[3] == 1:
            print(f"  przykład {p.name[:30]}\n    klasyczny: {kl[:90]}\n    nasz     : {vt[:90]}\n    nadane   : {nad[:90]}")
    print(f"{'nagrania':20s} {'S/N':>7s}   klasyczny   nasz (sieć zboczy + Viterbi)")
    for (r, sn), g in sorted(gr.items(), key=lambda kv: (kv[0][0], -999 if kv[0][1] == "czyste" else -int(kv[0][1]))):
        print(f"{r:20s} {sn:>7s}   {100 * (1 - g[0] / g[2]):6.1f}%     {100 * (1 - g[1] / g[2]):6.1f}%   ({g[2]} znaków)")


if __name__ == "__main__":
    main()
