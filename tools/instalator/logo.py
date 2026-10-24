"""Logo Dekodera CW (24.10): u góry „CW” kropkami i kreskami (-.-. .--),
pod spodem kluczowany ton tego samego „CW” — sygnał i to, co z niego
odczytano. Rysowane w numpy (bez Pillow), 4x nadpróbkowanie dla gładkich
krawędzi; zapis PNG i ICO (PNG w środku, 16-256 px).

    python -m tools.instalator.logo      # -> tools/instalator/dekoder_cw.png, .ico
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

KAT = Path(__file__).resolve().parent
CW = [3, 1, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 3]       # -.-. (przerwa 3) .-- : długości na przemian znak/przerwa
TLO_GORA, TLO_DOL = np.array([12, 30, 52]), np.array([20, 58, 96])
BURSZTYN = np.array([255, 178, 0])
TURKUS = np.array([64, 220, 210])


def rysuj(n: int = 256, nadprobka: int = 4) -> np.ndarray:
    N = n * nadprobka
    y, x = np.mgrid[0:N, 0:N].astype(np.float32) / N
    rgba = np.zeros((N, N, 4), np.float32)
    # tło: zaokrąglony kwadrat z pionowym gradientem
    r = 0.18
    dx = np.maximum(np.abs(x - 0.5) - (0.5 - r), 0)
    dy = np.maximum(np.abs(y - 0.5) - (0.5 - r), 0)
    maska = (dx ** 2 + dy ** 2) <= r ** 2
    tlo = TLO_GORA[None, None] * (1 - y[..., None]) + TLO_DOL[None, None] * y[..., None]
    rgba[..., :3] = tlo
    rgba[..., 3] = maska * 255
    # „CW” kreskami i kropkami
    u = 0.84 / sum(CW)
    x0, yc, h = 0.08, 0.36, 0.84 / sum(CW) * 1.25
    pos = x0
    for i, d in enumerate(CW):
        if i % 2 == 0:
            a, b = pos, pos + d * u
            pr = h / 2
            cx = np.clip(x, a + pr, b - pr)
            el = ((x - cx) ** 2 + ((y - yc) / 1.0) ** 2) <= pr ** 2
            rgba[el, :3] = BURSZTYN
        pos += d * u
    # kluczowany ton pod spodem: sinus w czasie elementów, z łagodnym narastaniem
    # i opadaniem; rysowany jako łamana kolumna po kolumnie (stromy sinus inaczej
    # dawał przerwy albo pionowe kreski przez całe logo)
    ys, amp, f, gr = 0.66, 0.10, 34.0, 0.010
    t = x[0]
    w = np.zeros_like(t)
    pos = x0
    for i, d in enumerate(CW):
        if i % 2 == 0:
            a, b = pos, pos + d * u
            obw = np.clip(np.minimum(t - a, b - t) / (0.35 * u), 0, 1)
            w += np.where((t >= a) & (t <= b), amp * obw * np.sin(2 * np.pi * f * (t - a)), 0.0)
        pos += d * u
    yv = ys - w
    lo = np.minimum(yv, np.roll(yv, 1)) - gr
    hi = np.maximum(yv, np.roll(yv, 1)) + gr
    linia = (y >= lo[None]) & (y <= hi[None]) & (x >= 0.06) & (x <= 0.94)
    rgba[linia, :3] = TURKUS
    rgba[..., 3] *= maska
    # nadpróbkowanie -> n
    m = rgba.reshape(n, nadprobka, n, nadprobka, 4)
    al = m[..., 3:4]
    kol = (m[..., :3] * al).sum((1, 3)) / np.maximum(al.sum((1, 3)), 1e-6)
    a = al.mean((1, 3))
    return np.concatenate([kol, a], -1).round().clip(0, 255).astype(np.uint8)


def png(rgba: np.ndarray) -> bytes:
    h, w, _ = rgba.shape
    surowe = b"".join(b"\x00" + rgba[i].tobytes() for i in range(h))

    def kaw(typ: bytes, dane: bytes) -> bytes:
        return struct.pack(">I", len(dane)) + typ + dane + struct.pack(">I", zlib.crc32(typ + dane) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + kaw(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + kaw(b"IDAT", zlib.compress(surowe, 9)) + kaw(b"IEND", b""))


def ico(obrazy: list[np.ndarray]) -> bytes:
    dane = [png(o) for o in obrazy]
    nag = struct.pack("<HHH", 0, 1, len(dane))
    pos = 6 + 16 * len(dane)
    wpisy = b""
    for o, d in zip(obrazy, dane):
        s = o.shape[0]
        wpisy += struct.pack("<BBBBHHII", s % 256, s % 256, 0, 0, 1, 32, len(d), pos)
        pos += len(d)
    return nag + wpisy + b"".join(dane)


def main() -> None:
    (KAT / "dekoder_cw.png").write_bytes(png(rysuj(512)))
    (KAT / "dekoder_cw.ico").write_bytes(ico([rysuj(s, 8 if s < 64 else 4) for s in (16, 24, 32, 48, 64, 128, 256)]))
    # Inno Setup: obrazek kreatora (BMP 24-bit) — 164x314 i małe 55x55
    for nazwa, (w, h, s) in {"kreator.bmp": (164, 314, 150), "kreator_maly.bmp": (55, 55, 55)}.items():
        o = rysuj(s)
        tlo = np.full((h, w, 3), 255, np.uint8)
        y0, x0 = (h - s) // 2, (w - s) // 2
        a = o[..., 3:4] / 255.0
        tlo[y0:y0 + s, x0:x0 + s] = (o[..., :3] * a + tlo[y0:y0 + s, x0:x0 + s] * (1 - a)).astype(np.uint8)
        wiersz = (w * 3 + 3) // 4 * 4
        bity = b"".join(tlo[i, :, ::-1].tobytes().ljust(wiersz, b"\x00") for i in range(h - 1, -1, -1))
        (KAT / nazwa).write_bytes(b"BM" + struct.pack("<IHHI", 54 + len(bity), 0, 0, 54)
                                  + struct.pack("<IiiHHIIiiII", 40, w, h, 1, 24, 0, len(bity), 2835, 2835, 0, 0) + bity)
    print("logo:", *sorted(p.name for p in KAT.glob("*") if p.suffix in (".png", ".ico", ".bmp")))


if __name__ == "__main__":
    main()
