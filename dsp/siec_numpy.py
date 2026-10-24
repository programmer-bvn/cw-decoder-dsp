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


def main() -> None:
    """best.keras -> .npz i sprawdzenie zgodności z TensorFlow na losowych oknach."""
    zrodlo, cel = Path(sys.argv[1]), Path(sys.argv[2])
    np.savez(cel, **z_keras(zrodlo))
    from tensorflow import keras
    m = keras.models.load_model(zrodlo)
    s = SiecNumpy(cel)
    rng = np.random.default_rng(0)
    x = [rng.random((4, 512, 32, 1), np.float32) for _ in range(2)]
    a, b = m.predict(x, verbose=0), s.przod(x)
    print(f"{cel}: maks. różnica logitów numpy - TensorFlow {np.abs(a - b).max():.2e} "
          f"(zakres {np.abs(a).max():.1f})")


if __name__ == "__main__":
    main()
