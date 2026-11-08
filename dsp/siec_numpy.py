"""SIEĆ ZBOCZY W NUMPY — bez TensorFlow (24.10, aplikacja dla SP2MSA).

TensorFlow to 1,5 GB na dysku i kilka sekund importu; exe z nim byłoby
ogromne. Sieć zboczy_5 (tools/zbocza_trening.model_zboczy_5) to same sploty,
normalizacja, ReLU, max-pooling i dodawanie — wagi raz z .keras do .npz
(normalizacja wliczona w sploty), przebieg w numpy. Zgodność z TensorFlow
sprawdzana przy budowie (`python -m dsp.siec_numpy best.keras wyj.npz`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

GALAZ_A = [f"conv2d{'' if i == 0 else f'_{i}'}" for i in range(5)]
GALAZ_B = [f"conv2d_{i}" for i in range(5, 10)]
ZLACZ = "conv2d_10"
ROZSZERZONE = [(f"conv2d_{i}", 2 ** (i - 11)) for i in range(11, 17)]


def z_keras(sciezka: Path) -> dict:
    """best.keras -> {nazwa: (W [kh,kw,cin,cout], b [cout])} z normalizacją w środku."""
    from tensorflow import keras
    m = keras.models.load_model(sciezka)
    warstwy = {l.name: l for l in m.layers}
    out = {}
    nr_bn = {f"conv2d{'' if i == 0 else f'_{i}'}": f"batch_normalization{'' if i == 0 else f'_{i}'}"
             for i in range(17)}
    for nazwa, bn in nr_bn.items():
        W = warstwy[nazwa].get_weights()[0].astype(np.float64)
        g, be, mu, var = (x.astype(np.float64) for x in warstwy[bn].get_weights())
        s = g / np.sqrt(var + warstwy[bn].epsilon)
        out[nazwa + "_W"] = (W * s).astype(np.float32)
        out[nazwa + "_b"] = (be - mu * s).astype(np.float32)
    W, b = warstwy["logit"].get_weights()
    out["logit_W"], out["logit_b"] = W.astype(np.float32), b.astype(np.float32)
    return out


def _splot(x: np.ndarray, W: np.ndarray, b: np.ndarray, d: int = 1) -> np.ndarray:
    """Conv2D „same”, krok 1, rozszerzenie d po czasie. x [B, T, F, C]."""
    kh, kw, cin, cout = W.shape
    B, T, F, _ = x.shape
    pt, pf = d * (kh - 1) // 2, (kw - 1) // 2
    xp = np.pad(x, ((0, 0), (pt, pt), (pf, pf), (0, 0)))
    kol = np.concatenate([xp[:, dy * d:dy * d + T, dx:dx + F, :] for dy in range(kh) for dx in range(kw)], axis=-1)
    return (kol.reshape(-1, kh * kw * cin) @ W.reshape(-1, cout)).reshape(B, T, F, cout) + b


class SiecNumpy:
    """Zamiennik modelu Keras dla dsp.zbocza: `inputs` (dwa wejścia) i `przod`."""

    inputs = ("widmo_5ms", "drobny_5ms")

    def __init__(self, sciezka: Path):
        z = np.load(sciezka)
        self.w = {k: z[k] for k in z.files}

    def _galaz(self, x: np.ndarray, nazwy: list[str]) -> np.ndarray:
        for n in nazwy:
            x = np.maximum(_splot(x, self.w[n + "_W"], self.w[n + "_b"]), 0)
            B, T, F, C = x.shape
            x = x.reshape(B, T, F // 2, 2, C).max(3)
        return x

    def przod(self, wejscie: list[np.ndarray]) -> np.ndarray:
        """[okna mel [B,512,32,1], okna drobne [B,512,32,1]] -> logit [B, 512]."""
        a = self._galaz(np.asarray(wejscie[0], np.float32), GALAZ_A)
        b = self._galaz(np.asarray(wejscie[1], np.float32), GALAZ_B)
        x = np.maximum(_splot(np.concatenate([a, b], -1), self.w[ZLACZ + "_W"], self.w[ZLACZ + "_b"]), 0)
        for n, d in ROZSZERZONE:
            x = x + np.maximum(_splot(x, self.w[n + "_W"], self.w[n + "_b"], d), 0)
        x = _splot(x, self.w["logit_W"], self.w["logit_b"])
        return x[..., 0, 0]


# --------------------------------------------------------------------------
# Sieć przegródek (dsp/elementy, etap A, elementy_2) — 08.11, operator:
# „przełącz program na kaskadę". Kaskada zbocza_6 + elementy_2 na nośnej
# z IC-7300: 6,3 zn./min zamiast 18,5 (zbocza_5) — a program liczy bez
# TensorFlow, więc i ta sieć w numpy. Budowa (tools/elementy_trening):
# 2x splot, pooling 2x1, 2x splot, pooling 2x2, splot, średnia, 64, 3.
# --------------------------------------------------------------------------
SPLOTY_E = [f"conv2d{'' if i == 0 else f'_{i}'}" for i in range(5)]
POOL_E = {1: (2, 1), 3: (2, 2)}          # po którym splocie jaki pooling


def z_keras_elementow(sciezka: Path) -> dict:
    from tensorflow import keras
    m = keras.models.load_model(sciezka)
    w = {l.name: l for l in m.layers}
    out = {}
    for i, n in enumerate(SPLOTY_E):
        bn = w[f"batch_normalization{'' if i == 0 else f'_{i}'}"]
        ww = w[n].get_weights()                   # splot bez biasu (przed normalizacją)
        W, b = ww[0].astype(np.float64), (ww[1].astype(np.float64) if len(ww) > 1 else 0.0)
        g, be, mu, var = (x.astype(np.float64) for x in bn.get_weights())
        sk = g / np.sqrt(var + bn.epsilon)
        out[n + "_W"] = (W * sk).astype(np.float32)
        out[n + "_b"] = ((b - mu) * sk + be).astype(np.float32)
    for n in ("dense", "dense_1"):
        W, b = w[n].get_weights()
        out[n + "_W"], out[n + "_b"] = W.astype(np.float32), b.astype(np.float32)
    return out


class SiecElementowNumpy:
    """Zamiennik modelu Keras sieci przegródek: przewiduj(X) -> softmax [n, 3]."""

    def __init__(self, sciezka: Path):
        z = np.load(sciezka)
        self.w = {k: z[k] for k in z.files}

    def przewiduj(self, X: np.ndarray, wsad: int = 512) -> np.ndarray:
        """X [n, WIERSZE, KOLUMNY] (float po normalizuj albo uint8 jak w treningu)."""
        out = []
        for i in range(0, len(X), wsad):
            x = X[i:i + wsad]
            x = (x.astype(np.float32) / 255.0 * 1.25 - 0.25 if x.dtype == np.uint8
                 else x.astype(np.float32))[..., None]
            for j, n in enumerate(SPLOTY_E):
                x = np.maximum(_splot(x, self.w[n + "_W"], self.w[n + "_b"]), 0)
                if j in POOL_E:
                    ph, pw = POOL_E[j]
                    B, H, W_, C_ = x.shape
                    x = x[:, :H // ph * ph, :W_ // pw * pw].reshape(B, H // ph, ph, W_ // pw, pw, C_).max((2, 4))
            x = x.mean((1, 2))
            x = np.maximum(x @ self.w["dense_W"] + self.w["dense_b"], 0)
            x = x @ self.w["dense_1_W"] + self.w["dense_1_b"]
            x = np.exp(x - x.max(1, keepdims=True))
            out.append(x / x.sum(1, keepdims=True))
        return np.concatenate(out) if out else np.zeros((0, 3), np.float32)


def main() -> None:
    """best.keras -> .npz i sprawdzenie zgodności z TensorFlow na losowych oknach.
    Sieć przegródek rozpoznana po nazwie modelu (elementy_a)."""
    zrodlo, cel = Path(sys.argv[1]), Path(sys.argv[2])
    from tensorflow import keras
    m = keras.models.load_model(zrodlo)
    rng = np.random.default_rng(0)
    if m.name == "elementy_a":
        np.savez(cel, **z_keras_elementow(zrodlo))
        import tensorflow as tf
        x = rng.uniform(-0.25, 1.0, (64, 24, 12)).astype(np.float32)
        a = tf.nn.softmax(m(x[..., None], training=False)).numpy()
        b = SiecElementowNumpy(cel).przewiduj(x)
        print(f"{cel}: maks. różnica P numpy - TensorFlow {np.abs(a - b).max():.2e}")
        return
    np.savez(cel, **z_keras(zrodlo))
    s = SiecNumpy(cel)
    x = [rng.random((4, 512, 32, 1), np.float32) for _ in range(2)]
    a, b = m.predict(x, verbose=0), s.przod(x)
    print(f"{cel}: maks. różnica logitów numpy - TensorFlow {np.abs(a - b).max():.2e} "
          f"(zakres {np.abs(a).max():.1f})")


if __name__ == "__main__":
    main()
