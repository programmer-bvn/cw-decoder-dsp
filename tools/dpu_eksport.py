"""SIEĆ ZBOCZY NA DPU — kwantyzacja i kompilacja Vitis AI 3.5 (07.11, KV260).

best.keras jest z Keras 3 (TF 2.21); kontener Vitis AI 3.5 ma TF 2.12
z Keras 2 i takiego pliku nie wczyta. Dlatego sieć buduje się tu od nowa
z runs/zbocza_5/siec.npz (normalizacja już wliczona w sploty — DPU i tak
ją wlicza), w czystym tf.keras.

Dwa kroki, dwa środowiska:
1. w środowisku projektu (~/venv_gpu): okna kalibracyjne i testowe
   z generatora i z nagrań + logity numpy jako wzorzec:
       python -m tools.dpu_eksport okna
2. w kontenerze xilinx/vitis-ai-tensorflow2-cpu (tools/dpu_kompiluj.sh):
       python tools/dpu_eksport.py kontener
   budowa, zgodność z numpy, vai_q (kalibracja), porównanie float/int8,
   zapis quantized.h5; kompilację vai_c_tensorflow2 robi skrypt .sh.
Wyniki w out/dpu/ (npz i h5 poza gitem).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

KORZEN = Path(__file__).resolve().parent.parent
WAGI = KORZEN / "runs/zbocza_5/siec.npz"
OUT = KORZEN / "out/dpu"


def okna() -> None:
    sys.path.insert(0, str(KORZEN))
    from dsp import config as C, frontend, radio, zbocza as Z
    from dsp.siec_numpy import SiecNumpy
    from tools import odczyt
    m = SiecNumpy(WAGI)
    zebrane = []

    class Lapacz(SiecNumpy):
        def przod(self, wej):
            zebrane.append([np.asarray(w, np.float32) for w in wej])
            return m.przod(wej)

    lap = Lapacz(WAGI)
    # kalibracja: te same klipy co tools.int8_ocena (seed 2026_11_07)
    rng = np.random.default_rng(2026_11_07)
    for _ in range(60):
        tekst = "".join(rng.choice(list(C.ALPHABET[1:]), 8))
        au, _ = radio.receive(tekst, rng, n_samples=C.CLIP_SAMPLES, wpm=float(rng.uniform(*C.TEMPO_ZAKRES)),
                              realism=True, zakresy=Z.ZAKRESY_ZBOCZY_6)
        Z.przebieg_sieci(au.astype(np.float32), lap)
    kal = [np.concatenate([z[i] for z in zebrane]) for i in range(2)]
    # test: prawdziwe nagranie (kluczownik qso1) — inne niż kalibracja
    zebrane.clear()
    x, _ = odczyt.przygotuj(frontend.load_audio(KORZEN / "probki/kluczownik/qso1_cq_kluczownik17wpm_20261018_1224.wav"))
    Z.przebieg_sieci(x, lap)
    tst = [np.concatenate([z[i] for z in zebrane]) for i in range(2)]
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "okna.npz", kal_a=kal[0], kal_b=kal[1], tst_a=tst[0], tst_b=tst[1],
             tst_logit=m.przod(tst), kal_logit=m.przod([k[:64] for k in kal]))
    print(f"okna: kalibracja {kal[0].shape[0]}, test {tst[0].shape[0]} -> {OUT / 'okna.npz'}")


def zbuduj(w):
    import tensorflow as tf
    L = tf.keras.layers

    def splot(x, n, d=1, relu=True):
        W, b = w[n + "_W"], w[n + "_b"]
        c = L.Conv2D(W.shape[3], W.shape[:2], padding="same", dilation_rate=(d, 1), name=n)
        y = c(x)
        c.set_weights([W, b])
        return L.ReLU(name=n + "_relu")(y) if relu else y

    wa = L.Input((512, 32, 1), name="widmo_5ms")
    wb = L.Input((512, 32, 1), name="drobny_5ms")
    gal = []
    for wej, nazwy in ((wa, [f"conv2d{'' if i == 0 else f'_{i}'}" for i in range(5)]),
                       (wb, [f"conv2d_{i}" for i in range(5, 10)])):
        x = wej
        for n in nazwy:
            x = L.MaxPooling2D((1, 2), name=n + "_pool")(splot(x, n))
        gal.append(x)
    x = splot(L.Concatenate(name="zlacz")(gal), "conv2d_10")
    for i in range(11, 17):
        n = f"conv2d_{i}"
        x = L.Add(name=n + "_suma")([x, splot(x, n, 2 ** (i - 11))])
    x = splot(x, "logit", relu=False)
    return tf.keras.Model([wa, wb], x, name="zbocza_5")


def kontener() -> None:
    import tensorflow as tf
    from tensorflow_model_optimization.quantization.keras import vitis_quantize
    w = dict(np.load(WAGI))
    o = dict(np.load(OUT / "okna.npz"))
    m = zbuduj(w)
    tst = [o["tst_a"], o["tst_b"]]
    pf = m.predict(tst, batch_size=32, verbose=0)[..., 0, 0]
    print(f"TF {tf.__version__}: zgodność z numpy — maks |dlogit| {np.abs(pf - o['tst_logit']).max():.2e}"
          f" (zakres {np.abs(o['tst_logit']).max():.1f})")
    q = vitis_quantize.VitisQuantizer(m).quantize_model(
        calib_dataset=[o["kal_a"], o["kal_b"]], calib_batch_size=16, include_fast_ft=False)
    q.save(OUT / "zbocza_5_int8.h5")
    pq = q.predict(tst, batch_size=32, verbose=0)[..., 0, 0]
    sf, sq = 1 / (1 + np.exp(-pf)), 1 / (1 + np.exp(-pq))
    d = np.abs(sf - sq)
    print(f"vai_q int8: |P_float - P_int8| średnio {d.mean():.4f}, 99. percentyl {np.percentile(d, 99):.3f},"
          f" maks {d.max():.3f}; ramki zgodne (P>0,5) {np.mean((sf > 0.5) == (sq > 0.5)):.4%}")
    np.save(OUT / "tst_logit_vaiq.npy", pq)


if __name__ == "__main__":
    {"okna": okna, "kontener": kontener}[sys.argv[1]]()
