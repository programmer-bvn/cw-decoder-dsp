"""SIEĆ ZBOCZY W INT8 — symulacja kwantyzacji jak w DPU (07.11, KV260).

Zanim narzędzia Vitis AI (docker, kilka GB przez Wi-Fi): czy 8 bitów w ogóle
wystarcza tej sieci? DPU (DPUCZDX8G) liczy na liczbach całkowitych int8,
a skala każdego tensora to potęga dwójki („fix position"): wartość = q * 2^-p,
q w [-128, 127]. Tu ta sama arytmetyka na numpy, na wagach dsp.siec_numpy:
- wagi i progi (bias) — int8, pozycja z maksimum |w| (jak vai_q dla wag);
- aktywacje — int8 po każdej warstwie (wejście, ReLU, złączenie, suma
  rezydualna, logit), pozycja z kalibracji: najmniejszy błąd kwadratowy
  na nagraniach (metoda „min MSE" kwantyzatora Vitis AI);
- złączenie gałęzi i suma rezydualna we wspólnej skali (DPU tego wymaga).
Pośrednie sumy splotu dokładne (DPU akumuluje szeroko). Hipoteza do pomiaru:
symulacja oddaje DPU z dokładnością do zaokrągleń — sprawdzi to dopiero
porównanie z płytką.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .siec_numpy import GALAZ_A, GALAZ_B, ROZSZERZONE, ZLACZ, SiecNumpy, _splot


def _pozycja_wag(w: np.ndarray) -> int:
    m = float(np.abs(w).max())
    return int(np.floor(np.log2(127 / m))) if m > 0 else 7


def kwant(x: np.ndarray, p: int) -> np.ndarray:
    s = 2.0 ** p
    return (np.clip(np.round(x * s), -128, 127) / s).astype(np.float32)


def pozycja_mse(x: np.ndarray, zakres=range(-6, 16)) -> int:
    """Pozycja (potęga dwójki) o najmniejszym błędzie kwadratowym na próbce."""
    x = np.asarray(x, np.float32).ravel()
    if x.size > 2_000_000:
        x = x[np.random.default_rng(0).choice(x.size, 2_000_000, replace=False)]
    return min(zakres, key=lambda p: float(np.mean((kwant(x, p) - x) ** 2)))


class SiecInt8(SiecNumpy):
    """SiecNumpy z kwantyzacją int8; pozycje aktywacji z `kalibruj` albo z pliku."""

    def __init__(self, sciezka: Path, pozycje: dict | Path | None = None):
        super().__init__(sciezka)
        self.wq = {}
        for k, w in self.w.items():
            self.wq[k] = kwant(w, _pozycja_wag(w))
        if isinstance(pozycje, (str, Path)):
            z = np.load(pozycje)
            pozycje = {k: int(z[k]) for k in z.files}
        self.p = pozycje            # None = przebieg zbierający (kalibracja)
        self._zbior: dict | None = None

    # punkt kwantyzacji aktywacji: w kalibracji zbiera, potem zaokrągla
    def _q(self, nazwa: str, x: np.ndarray) -> np.ndarray:
        if self._zbior is not None:
            self._zbior.setdefault(nazwa, []).append(x.ravel()[::7].copy())
            return x
        return kwant(x, self.p[nazwa]) if self.p is not None else x

    def _w(self, n):
        return (self.wq if self._zbior is None else self.w)[n + "_W"], \
               (self.wq if self._zbior is None else self.w)[n + "_b"]

    def _galaz_q(self, x, nazwy):
        for n in nazwy:
            W, b = self._w(n)
            x = self._q(n, np.maximum(_splot(x, W, b), 0))
            B, T, F, C = x.shape
            x = x.reshape(B, T, F // 2, 2, C).max(3)
        return x

    def przod(self, wejscie):
        a = self._galaz_q(self._q("wej_a", np.asarray(wejscie[0], np.float32)), GALAZ_A)
        b = self._galaz_q(self._q("wej_b", np.asarray(wejscie[1], np.float32)), GALAZ_B)
        x = self._q("zlacz_wej", np.concatenate([a, b], -1))
        W, bb = self._w(ZLACZ)
        x = self._q(ZLACZ, np.maximum(_splot(x, W, bb), 0))
        for n, d in ROZSZERZONE:
            W, bb = self._w(n)
            r = self._q(n, np.maximum(_splot(x, W, bb, d), 0))
            x = self._q(n + "_suma", x + r)
        W, bb = self._w("logit")
        x = self._q("logit", _splot(x, W, bb))
        return x[..., 0, 0]

    def zacznij_kalibracje(self) -> None:
        """Od teraz przod() liczy w float i zbiera aktywacje (np. przez
        zbocza.przebieg_sieci na kilku nagraniach); potem koniec_kalibracji()."""
        self._zbior = {}

    def koniec_kalibracji(self) -> dict:
        zb, self._zbior = self._zbior, None
        self.p = {k: pozycja_mse(np.concatenate(v)) for k, v in zb.items()}
        return self.p

    def zapisz(self, cel: Path) -> None:
        np.savez(cel, **{k: np.int8(v) for k, v in self.p.items()})
