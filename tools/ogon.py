"""OGON PO ELEMENCIE (17.10): obwiednia przy tonie po zboczu opadającym,
nagrania kontra czysty syntetyk tym samym torem (filtr ±100 Hz, 255 odczepów,
filtfilt; obwiednia Hilberta, 2 ms). Mediana po zboczach z przerwą >= 150 ms.
NAR = to samo lustrzanie przed zboczem narastającym. Wykres: out/zbocza/ogon.png

    python -m tools.ogon

Powód: sieć zboczy na mic1 (27,6 s) długo nie schodziła z P po kresce —
przerwa w N skurczona do 0,44 u (HISTORIA, „Histereza i odrzucanie...").
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.signal import hilbert
from dsp import config as C, frontend, tune, morse, tempo
from tools import wav2net

TAU = np.arange(-30, 301, 5)  # ms po zboczu
def obwiednia_db(a):
    x = tune.bandpass(a, C.TONE_CENTER, 100.0, taps=255)
    e = np.abs(hilbert(x.astype(np.float64)))
    k = int(0.002 * C.SR); e = np.convolve(e, np.ones(k) / k, mode="same")
    return 20 * np.log10(e + 1e-9)

def krzywe(a):
    d = obwiednia_db(a)
    znak = np.percentile(d, 95)
    on = d > znak - 12
    zm = np.flatnonzero(np.diff(on.astype(np.int8))) + 1
    gr = np.r_[0, zm, on.size]
    odc = [(bool(on[s]), s, e) for s, e in zip(gr[:-1], gr[1:])]
    opad, nar, pod = [], [], []
    for i in range(1, len(odc) - 1):
        z, s, e = odc[i]
        if z:
            continue
        dl = (e - s) / C.SR
        if dl >= 0.15 and (odc[i - 1][2] - odc[i - 1][1]) / C.SR >= 0.03:
            opad.append(s)
        if dl >= 0.15 and (odc[i + 1][2] - odc[i + 1][1]) / C.SR >= 0.03:
            nar.append(e)
        if dl >= 0.6:
            pod.append(np.median(d[s + int(0.25 * C.SR):e - int(0.25 * C.SR)]))
    def wez(pkt, znak_):
        w = []
        for p in pkt:
            idx = p + (znak_ * TAU * C.SR / 1000).astype(int)
            if idx.min() >= 0 and idx.max() < d.size:
                w.append(d[idx] - znak)
        return np.median(np.array(w), axis=0) if w else np.full(TAU.size, np.nan), len(w)
    o, no = wez(opad, 1); n, nn = wez(nar, -1)
    return o, n, (np.median(pod) - znak) if pod else np.nan, no, nn

def wczytaj(n):
    a = frontend.load_audio(Path("probki") / n)
    t, f = tune.track_tone(a); a = tune.retune(a, t, f)
    g = wav2net.auto_gain_db(a); return np.clip(a * 10 ** (g / 20), -1, 1).astype(np.float32)

pliki = ["mic1_manipulator_12wpm.wav", "mic2_manipulator_20wpm.wav", "mic3_pamiec_15wpm.wav",
         "ic7300_usb_SN0FMT_3565kHz_20261009.wav", "ic7300_usb_SN0FMT_3565kHz_20261009_b.wav",
         "ic7300_usb_LA3WAA_7007kHz_FIL2_20261009_2146.wav", "ic7300_usb_DP75HSC_3526kHz_FIL3_20261010_1811.wav"]
pokaz = [10, 20, 30, 40, 60, 80, 120, 160, 200, 300]
ix = [int(np.flatnonzero(TAU == t)[0]) for t in pokaz]
print("dB względem znaku, mediana po zboczach.  OPAD = po zboczu opadającym (ogon), NAR = przed narastającym (lustrzanie)")
print(f"{'':34s} {'zboczy':>6s} " + " ".join(f"{t:>5d}" for t in pokaz) + "   podłoga")
fig, ax = plt.subplots(1, 1, figsize=(11, 6))
for ramp in (2.0, 5.0, 8.0):
    rng = np.random.default_rng(0)
    w = morse.synth_cw("PSE K CQ CQ DE SQ2BVN K", wpm=15, tone=C.TONE_CENTER, amp=0.3, ramp_ms=ramp, rng=rng)
    w = np.r_[np.zeros(4000), w, np.zeros(4000)].astype(np.float32) + 1e-5 * np.random.default_rng(1).standard_normal(w.size + 8000).astype(np.float32)
    o, n, pod, no, nn = krzywe(w)
    print(f"{'syntetyk, narastanie ' + str(ramp) + ' ms':34s} {no:6d} " + " ".join(f"{o[i]:5.0f}" for i in ix) + f"   {pod:6.0f}")
    ax.plot(TAU, o, "k--" if ramp == 5.0 else "k:", lw=1, label=f"syntetyk {ramp:.0f} ms")
for n in pliki:
    o, nr, pod, no, nn = krzywe(wczytaj(n))
    print(f"{n[:34]:34s} {no:6d} " + " ".join(f"{o[i]:5.0f}" for i in ix) + f"   {pod:6.0f}")
    print(f"{'   NAR (przed narastającym)':34s} {nn:6d} " + " ".join(f"{nr[i]:5.0f}" for i in ix))
    ax.plot(TAU, o, lw=1, label=n[:30])
ax.set_xlabel("ms po zboczu opadającym"); ax.set_ylabel("dB względem znaku"); ax.set_ylim(-70, 3); ax.grid(alpha=.3); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig("out/zbocza/ogon.png", dpi=90)
