"""DANE WIDOKU SYGNAŁU NA ŻYWO — liczone w procesie dekodera (08.11).

tools/na_zywo._praca_okna z cfg["widok"] po każdym kroku wysyła
("widok", NadawcaWidoku.krok(d)): nowe kolumny 5 ms (wejście, po obróbce,
oczami sieci), ogon P (przeliczany przez dekoder do PRZELICZ_S wstecz —
okno nadpisuje), a co ~1 s grzebień, zbocza, regenerat i odchyłki. Bez Qt:
proces dekodera startuje od zera (spawn) i nie potrzebuje okienek.

Siatka ta sama co w pliku (tools/gui/analiza): kolumna j = ramka sieci j,
okno wyśrodkowane na próbce j·HOP_ZBOCZA od początku nasłuchu.
"""

from __future__ import annotations

import numpy as np

from dsp import config as C, frontend, melref, zbocza as Z
from tools.gui import analiza as A

ZAKLADKA_S = 12.0          # ogon P do grzebienia (jak VITERBI_S dekodera)
PEWNE_OD_S = 4.0           # z ogona brane tylko to, co dalej niż tyle od jego początku
NAKLADKI_CO = 4            # co tyle kroków (0,25 s) grzebień i regenerat


def _u8(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def _p_dekodera(d) -> np.ndarray:
    """P, które widzi Viterbi: po kaskadzie z przegródkami (08.11), gdy jest."""
    hk = getattr(d, "hist_k", None)
    return hk if hk is not None and hk.size == d.hist.size else d.hist


class NadawcaWidoku:
    def __init__(self):
        self.j = None              # pierwsza jeszcze niewysłana kolumna
        self.kroki = 0
        self.a, self.b, _ = A._prazki()

    def _widmo(self, x: np.ndarray, x0: int, j0: int, j1: int) -> np.ndarray | None:
        """Kolumny j0..j1 widma 512 z sygnału x (x[0] = próbka x0), okna
        wyśrodkowane na j·hop — jak center=True w pliku."""
        hop, pol = Z.HOP_ZBOCZA, A.N_FFT_WEJ // 2
        s, e = j0 * hop - pol - x0, j1 * hop + pol - x0
        if s < 0 or e > x.size or j1 <= j0:
            return None
        p = melref.stft_power(np.asarray(x[s:e], np.float32), A.N_FFT_WEJ, hop, center=False)[:j1 - j0]
        return _u8(frontend.normalize_db(melref.power_to_db(p[:, self.a:self.b]) + A.KOREKTA_WEJ_DB))

    def krok(self, d) -> dict:
        hop = Z.HOP_ZBOCZA
        j1 = (d.n - A.N_FFT_WEJ // 2) // hop                  # pełne okno 512 dostępne
        if self.j is None:
            self.j = -(-(d.raw0 + A.N_FFT_WEJ // 2) // hop)
        j0 = max(self.j, -(-(d.raw0 + A.N_FFT_WEJ // 2) // hop))
        out: dict = {"j0": j0}
        if j1 > j0:
            out["wej"] = self._widmo(d.raw, d.raw0, j0, j1)
            y, y0 = d.ost_y, d.ost_y0
            out["dos"] = self._widmo(y, y0, j0, j1) if y.size else None
            if y.size:
                ob = Z.obraz_zboczy(y)                         # ramka i = próbka y0 + i·hop
                i0, i1 = j0 - y0 // hop, j1 - y0 // hop
                if i0 >= 0 and i1 <= ob.shape[0]:
                    out["siec"] = _u8(ob[i0:i1])
            self.j = j1
        if d.hist.size:
            k = min(d.hist.size, int(7.0 / Z.HOP_ZBOCZA * C.SR))   # PRZELICZ_S dekodera
            out["p"] = (d.h0 + d.hist.size - k, _p_dekodera(d)[-k:].copy())
        self.kroki += 1
        if self.kroki % NAKLADKI_CO == 0 and d.hist.size and np.isfinite(d.u):
            out["nakladki"] = self._nakladki(d)
        return out

    def _nakladki(self, d) -> dict:
        dt = Z.HOP_ZBOCZA / C.SR
        k = min(d.hist.size, int(ZAKLADKA_S / dt))
        h = d.h0 + d.hist.size - k
        pz = _p_dekodera(d)[-k:]
        nk = A.nakladki(pz, d.u)
        t0 = h * dt
        od = t0 + (PEWNE_OD_S if k * dt > 2 * PEWNE_OD_S else 0.0)
        regen = A.regenerat(nk["zbocza"], pz.size)
        o = A.odchylki(pz, nk["zbocza"], d.u)
        wynik = {"t_od": od, "zeby": nk["zeby"] + t0, "zbocza": nk["zbocza"] + t0,
                 "regen": (h, regen), "pary": o["pary"] + t0}
        if "rozrzut_ms" in o:
            wynik["opis"] = (f"— zbocza P od regeneratu (ostatnie {k * dt:.0f} s): rozrzut "
                             f"{o['rozrzut_ms']:.1f} ms = {o['rozrzut_u']:.2f} u, waga {o['waga_ms']:+.0f} ms")
        return wynik
