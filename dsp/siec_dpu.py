"""SIEĆ ZBOCZY NA DPU KV260 — zamiennik SiecNumpy dla dsp.zbocza (07.11).

Ten sam interfejs co dsp.siec_numpy.SiecNumpy (`inputs`, `przod`), więc
zbocza.przebieg_sieci i tools.odczyt.odczytaj działają bez zmian.
Model: out/dpu/xmodel/zbocza_5.xmodel z tools/dpu_kompiluj.sh. Wejścia
int8 ze skalą 2^fix_point z modelu, wyjście (logit) int8 -> float.
Zgodność z vai_q na PC: ramki 100%, logity identyczne 98,8%
(out/dpu/dpu_proba.txt). Na płytce przed użyciem:
`sudo xmutil loadapp kv260-benchmark-b4096` i stos VART
(srodowisko/kv260/buduj_vart.sh).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class SiecDPU:
    inputs = ("widmo_5ms", "drobny_5ms")

    def __init__(self, xmodel: Path):
        import vart
        import xir
        # graf trzymany w obiekcie: runner wskazuje na jego podgraf, a lokalny
        # graf zwolniony po __init__ dawał segfault w execute_async (07.11)
        self.g = g = xir.Graph.deserialize(str(xmodel))
        sg = [s for s in g.get_root_subgraph().toposort_child_subgraph()
              if s.has_attr("device") and s.get_attr("device") == "DPU"]
        if len(sg) != 1:
            raise RuntimeError(f"{xmodel}: podgrafów DPU {len(sg)}, oczekiwany 1 (część sieci poszłaby na CPU)")
        self.r = vart.Runner.create_runner(sg[0], "run")
        wej = self.r.get_input_tensors()
        # kolejność wejść w modelu nie jest gwarantowana — dopasowanie po nazwie
        self.kolej = [0 if "widmo" in t.name else 1 for t in wej]
        self.ksztalt = [tuple(t.dims) for t in wej]
        self.skala = [2.0 ** t.get_attr("fix_point") for t in wej]
        wy = self.r.get_output_tensors()[0]
        self.ksztalt_wy = tuple(wy.dims)
        self.skala_wy = 2.0 ** -wy.get_attr("fix_point")
        self.wsad = self.ksztalt[0][0]

    def przod(self, wejscie: list[np.ndarray]) -> np.ndarray:
        """[okna mel [B,512,32,1], okna drobne [B,512,32,1]] -> logit [B, 512]."""
        n = wejscie[0].shape[0]
        out = np.empty((n, self.ksztalt_wy[1]), np.float32)
        for s in range(0, n, self.wsad):
            bufy = []
            for k, ks, sk in zip(self.kolej, self.ksztalt, self.skala):
                x = np.zeros(ks, np.float32)
                kaw = np.asarray(wejscie[k][s:s + self.wsad], np.float32)
                x[:kaw.shape[0]] = kaw
                bufy.append(np.clip(np.round(x * sk), -128, 127).astype(np.int8))
            y = np.empty(self.ksztalt_wy, np.int8)
            self.r.wait(self.r.execute_async(bufy, [y]))
            m = min(self.wsad, n - s)
            out[s:s + m] = y[:m].reshape(m, -1).astype(np.float32) * self.skala_wy
        return out
