"""TEST PĘTLI TONU (18.10): zestaw z PRAWDZIWYM tonem stacji i miara pętli.

    python -m tools.test_tonu generuj            # out/test_tonu/*.wav + prawda.json
    python -m tools.test_tonu ocen               # obecna dsp.tune.track_tone

Po co: pętla fazowa (do 18.10) zaczepiała się na szumie, gdy nagranie zaczyna się od
samego szumu (HISTORIA 18.10, „Pętla tonu ... OTWARTE"; na 20 m kosztowało
9 z 94 słów). Pięć poprawek odrzuconych, bo nie było czym rozstrzygnąć, który
ton jest dobry. Tu prawda jest znana z budowy (decyzja operatora: zamiast
sumatora rezystorowego — dodawanie nagrań programowo):

  stacja    nagranie kluczownika IC-7300MK2 (probki/kluczownik, ton zmierzony),
            przesunięte na ton docelowy 600-900 Hz
  szum      naturalny: bank probki/szum + szum własny odbiornika (antena
            odłączona), S/N stacji zadane (dsp/sn)
  przed     0-5 s samego szumu przed nadaniem
  obok      nic / druga stacja (inne nagranie, inny tekst) / stała nośna (sinus),
            60-200 Hz od stacji, poziom -10 / 0 / +6 dB względem stacji

Miara: udział czasu NADAWANIA, gdy pętla jest w ±10 Hz od tonu stacji;
czas do pierwszego takiego zaczepu.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import soundfile as sf

from dsp import config as C, frontend, sn, tune

KAT = Path("out/test_tonu")
TOL_HZ = 10.0


def ton_nagrania(a: np.ndarray) -> float:
    X = np.abs(np.fft.rfft(a * np.hanning(a.size)))
    fr = np.fft.rfftfreq(a.size, 1 / C.SR)
    p = (fr > C.FMIN) & (fr < C.FMAX)
    return float(fr[p][np.argmax(X[p])])


def przesun(a: np.ndarray, df: float) -> np.ndarray:
    """Stałe przesunięcie widma o df [Hz] (sygnał analityczny, jak tune.retune)."""
    t = np.array([0.0, a.size / C.SR])
    f = np.array([C.TONE_CENTER - df, C.TONE_CENTER - df])
    return tune.retune(a, t, f)                     # shift = target - f = df


def amplituda(a: np.ndarray, ton: float) -> float:
    from scipy.signal import hilbert
    x = tune.bandpass(a, ton, 100.0, taps=255)
    return float(np.percentile(np.abs(hilbert(x.astype(np.float64))), 95))


def wasko_prazek_db(x: np.ndarray) -> float:
    """Najwyższy wąski prążek ponad sąsiedztwo (±5-30 Hz) w paśmie CW [dB]."""
    n = 8192
    seg = x[:x.size // n * n].reshape(-1, n) * np.hanning(n)
    S = (np.abs(np.fft.rfft(seg, axis=1)) ** 2).mean(axis=0)
    fr = np.fft.rfftfreq(n, 1 / C.SR)
    b = (fr > C.FMIN) & (fr < C.FMAX)
    k = np.flatnonzero(b)[np.argmax(S[b])]
    sas = (np.abs(fr - fr[k]) > 5) & (np.abs(fr - fr[k]) < 30)
    return float(10 * np.log10(S[k] / np.median(S[sas])))


def szumy() -> list[np.ndarray]:
    # Bez plików z wąskim prążkiem (> 12 dB nad sąsiedztwem): to nie szum, tylko
    # nieopisana nośna/stacja (7025 kHz FIL 230 Hz: 18,5 dB przy 770 Hz).
    out = [x for x in (frontend.load_audio(p).astype(np.float64)
                       for p in sorted(Path("probki/szum").glob("*.wav")))
           if wasko_prazek_db(x) <= 12.0]
    # szum własny odbiornika: początek nieprzyciętych nagrań bez anteny
    for p in sorted(Path("probki/kluczownik/surowe").glob("*_14[01][0-9]_surowe.wav")):
        x = frontend.load_audio(p).astype(np.float64)
        out.append(x[:int(3.0 * C.SR)])
    return out


def pasmo(sz: np.ndarray) -> tuple[float, float]:
    """Pas -6 dB szumu (filtr odbiornika) z uśrednionego widma."""
    n = 2048
    seg = sz[:sz.size // n * n].reshape(-1, n) * np.hanning(n)
    S = (np.abs(np.fft.rfft(seg, axis=1)) ** 2).mean(axis=0)
    S = np.convolve(S, np.ones(9) / 9, mode="same")
    fr = np.fft.rfftfreq(n, 1 / C.SR)
    w = (fr > 150) & (fr < 2500)
    pr = S[w].max() / 4.0
    f = fr[w][S[w] >= pr]
    return float(f.min()), float(f.max())


def generuj(n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    stacje = [frontend.load_audio(p).astype(np.float64)
              for p in sorted(Path("probki/kluczownik").glob("*.wav"))]
    tony = [ton_nagrania(s) for s in stacje]
    bank = szumy()
    pasma = [pasmo(b) for b in bank]
    KAT.mkdir(parents=True, exist_ok=True)
    prawda = []
    for i in range(n):
        k = int(rng.integers(len(stacje)))
        st = stacje[k][:int(rng.uniform(15, 30) * C.SR)]
        ib = int(rng.integers(len(bank)))
        lo, hi = pasma[ib]
        lo, hi = max(lo, C.FMIN + 30), min(hi, C.FMAX - 30)
        # stacja dostrojona do środka filtra (±30 Hz) — tak dostraja operator;
        # obce sygnały wpadają obok (pole „obok")
        f_cel = float(0.5 * (lo + hi) + rng.uniform(-30.0, 30.0))
        st = przesun(st, f_cel - tony[k]).astype(np.float64)
        przed = float(rng.choice([0.0, 1.0, 2.0, 3.0, 5.0]))
        n_all = int(przed * C.SR) + st.size + C.SR
        sz = bank[ib]
        while sz.size < n_all:
            sz = np.concatenate([sz, sz[::-1]])
        s0 = int(rng.integers(0, sz.size - n_all + 1))
        sz = sz[s0:s0 + n_all]
        sn_db = float(rng.choice([6.0, 10.0, 20.0]))
        x = sz / (np.std(sz) + 1e-12) * 0.05
        amp = sn.amp_dla_sn(sn_db, x, f_cel) / max(amplituda(st, f_cel), 1e-9)
        x[int(przed * C.SR):int(przed * C.SR) + st.size] += amp * st
        rodzaj = str(rng.choice(["nic", "stacja", "nosna"]))
        mozliwe = [z * d for d in (60.0, 120.0, 200.0) for z in (-1, 1) if lo + 20 < f_cel + z * d < hi - 20]
        df = float(rng.choice(mozliwe)) if mozliwe else 0.0
        poz = float(rng.choice([-10.0, 0.0, 6.0]))
        f_obok = f_cel + df
        if rodzaj != "nic" and mozliwe:
            a_cel = amp * amplituda(st, f_cel)
            if rodzaj == "nosna":
                t = np.arange(n_all) / C.SR
                x += a_cel * 10 ** (poz / 20) * np.sin(2 * np.pi * f_obok * t + rng.uniform(0, 6.3))
            else:
                k2 = (k + 1 + int(rng.integers(len(stacje) - 1))) % len(stacje)
                s2 = stacje[k2][:x.size]
                s2 = przesun(s2, f_obok - tony[k2]).astype(np.float64)
                s2 *= a_cel * 10 ** (poz / 20) / max(amplituda(s2, f_obok), 1e-9)
                o2 = int(rng.integers(0, max(1, x.size - s2.size + 1)))
                x[o2:o2 + s2.size] += s2[:x.size - o2]
        else:
            rodzaj = "nic"
        x = (x * (0.5 / np.max(np.abs(x)))).astype(np.float32)
        nazwa = f"t{i:03d}.wav"
        sf.write(KAT / nazwa, x, C.SR)
        prawda.append(dict(plik=nazwa, ton=f_cel, pas=[lo, hi], przed_s=przed, nadanie_s=st.size / C.SR,
                           sn_db=sn_db, obok=rodzaj, df=df if rodzaj != "nic" else None,
                           poziom_db=poz if rodzaj != "nic" else None))
    (KAT / "prawda.json").write_text(json.dumps(prawda, indent=1), encoding="utf-8")
    print(f"zapisano {n} przypadków w {KAT}")


def ocen_petle(track, nazwa: str = "") -> dict:
    prawda = json.loads((KAT / "prawda.json").read_text(encoding="utf-8"))
    wyn = []
    for d in prawda:
        a = frontend.load_audio(KAT / d["plik"])
        t, f = track(a)
        w = (t >= d["przed_s"]) & (t <= d["przed_s"] + d["nadanie_s"])
        dobrze = np.abs(f - d["ton"]) <= TOL_HZ
        traf = float(np.mean(dobrze[w])) if w.any() else 0.0
        pierwszy = t[np.argmax(dobrze & (t >= d["przed_s"]))] - d["przed_s"] if (dobrze & (t >= d["przed_s"])).any() else np.inf
        wyn.append(dict(d, traf=traf, pierwszy=float(pierwszy)))
    tr = np.array([w["traf"] for w in wyn])
    print(f"{nazwa}: na tonie (±{TOL_HZ:.0f} Hz) przez nadanie: średnio {100 * tr.mean():.1f}%; "
          f"przypadków >= 90%: {np.mean(tr >= 0.9) * 100:.0f}%; całkiem obok (< 10%): {np.mean(tr < 0.1) * 100:.0f}%")
    for klucz, wart in (("przed_s", [0.0, 1.0, 2.0, 3.0, 5.0]), ("obok", ["nic", "stacja", "nosna"]),
                        ("sn_db", [6.0, 10.0, 20.0])):
        print(f"   wg {klucz:7s}: " + "  ".join(
            f"{v}: {100 * np.mean([w['traf'] for w in wyn if w[klucz] == v]):5.1f}%"
            for v in wart if any(w[klucz] == v for w in wyn)))
    return {"srednio": float(tr.mean()), "wyniki": wyn}


# Funkcja z dsp.tune do oceny (19.10: track_tone = ton kluczowany, 90,3%;
# poprzednia pętla fazowa 31,7% — HISTORIA).
METODY = {"track_tone": "track_tone"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generuj"); g.add_argument("--n", type=int, default=150); g.add_argument("--seed", type=int, default=1)
    o = sub.add_parser("ocen"); o.add_argument("--metoda", default="track_tone", choices=list(METODY))
    a = ap.parse_args()
    if a.cmd == "generuj":
        generuj(a.n, a.seed)
    else:
        ocen_petle(getattr(tune, METODY[a.metoda]), f"tune.{METODY[a.metoda]}")


if __name__ == "__main__":
    main()
