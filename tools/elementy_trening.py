"""Etap A klasyfikatora elementów: zbiór, trening, ocena (06.11, gałąź `elementy`).

    python -m tools.elementy_trening generuj out/elementy/trening.npz 60000 --seed 1
    python -m tools.elementy_trening generuj out/elementy/walidacja.npz 3000 --seed 99
    python -m tools.elementy_trening trenuj out/elementy runs/elementy_1 --epoki 20
    python -m tools.elementy_trening ocen runs/elementy_1/best.keras out/elementy/walidacja.npz
    python -m tools.elementy_trening negatywy runs/elementy_1/best.keras --bank /mnt/w/nagrania_cw/negatywy

Wycinek [WIERSZE x KOLUMNY] (dsp/elementy.py) -> dziura / ton / śmieć dla
przegródki środkowej. Walidacja to OSOBNE klipy (inne ziarno): wycinki z jednego
klipu nakładają się, losowy podział wycinków zawyżyłby wynik.

Architektura DPU-bezpieczna: Conv2D/BN/ReLU/MaxPooling/GlobalAveragePooling/Dense.
"""
import argparse
import os
import sys
import time
from pathlib import Path

if len(sys.argv) > 1 and sys.argv[1] == "generuj":
    for _zm in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_zm, "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsp import config as C, elementy as E

NAZWY_NEG = ("", "bank", "mowa", "spiew", "gwizd", "muzyka")
KROK = 2                      # co druga kolumna (pół przegródki) — sąsiednie wycinki prawie te same
PRZEDZIALY_SN = ((-3, 3), (3, 10), (10, 20), (20, 46))


def do_uint8(x):
    return np.round((np.clip(x, -0.25, 1.0) + 0.25) / 1.25 * 255).astype(np.uint8)


def z_uint8(x):
    return x.astype(np.float32) / 255.0 * 1.25 - 0.25


# --------------------------------------------------------------------------
# generowanie
# --------------------------------------------------------------------------
def _kawalek(arg):
    i, n, seed = arg
    rng = np.random.default_rng([seed, i])
    Xs, ys, us, sn, wpm, neg = [], [], [], [], [], []
    for _ in range(n):
        db, et, ul, m = E.klip(rng)
        X, y, u = E.wycinki(db, et, ul, krok=KROK)
        Xs.append(do_uint8(X)); ys.append(y); us.append(np.round(u * 255).astype(np.uint8))
        k = len(y)
        sn.append(np.full(k, m.get("snr_db", np.nan) if m["tekst"] else np.nan, np.float32))
        wpm.append(np.full(k, m["wpm"], np.float32))
        neg.append(np.full(k, NAZWY_NEG.index(m.get("negatyw", "")), np.uint8))
    return tuple(np.concatenate(a) for a in (Xs, ys, us, sn, wpm, neg))


def generuj(wyj: Path, n: int, seed: int, procesy: int | None) -> None:
    import concurrent.futures as cf
    procesy = procesy or max(1, (os.cpu_count() or 4) - 1)
    per = 100
    kaw = [(i, min(per, n - i * per), seed) for i in range((n + per - 1) // per)]
    czesci = []
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=procesy) as ex:
        for g, cz in enumerate(ex.map(_kawalek, kaw), 1):
            czesci.append(cz)
            el = time.time() - t0
            if g % 20 == 0 or g == len(kaw):
                print(f"  {min(g * per, n)}/{n} klipów  {el:.0f}s  pozostało ~{el / g * (len(kaw) - g):.0f}s",
                      flush=True)
    X, y, ul, sn, wpm, neg = (np.concatenate([c[i] for c in czesci]) for i in range(6))
    wyj.parent.mkdir(parents=True, exist_ok=True)
    tym = wyj.with_name(wyj.name + ".part")
    with open(tym, "wb") as f:
        np.savez(f, X=X, y=y, ul=ul, sn=sn, wpm=wpm, neg=neg,
                 meta=f"n={n};seed={seed};K={E.K};przegrodek={E.PRZEGRODEK};"
                      f"wiersze={E.WIERSZE};pasmo={E.PASMO_HZ};krok={KROK}")
        f.flush(); os.fsync(f.fileno())
    os.replace(tym, wyj)
    print(f"zapisano {wyj}: {len(y)} wycinków, dziura/ton/śmieć {np.bincount(y, minlength=3)}, "
          f"{wyj.stat().st_size / 2**20:.0f} MB, {time.time() - t0:.0f} s")


# --------------------------------------------------------------------------
# model i trening
# --------------------------------------------------------------------------
def model_elementow():
    from tensorflow import keras
    L = keras.layers
    wej = keras.Input((E.WIERSZE, E.KOLUMNY, 1))
    x = wej
    for f, pula in ((32, None), (32, (2, 1)), (64, None), (64, (2, 2)), (96, None)):
        x = L.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = L.BatchNormalization()(x)
        x = L.ReLU()(x)
        if pula:
            x = L.MaxPooling2D(pula)(x)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dense(64, activation="relu")(x)
    return keras.Model(wej, L.Dense(3)(x), name="elementy_a")


def trenuj(katalog_zbioru: Path, katalog: Path, epoki: int, batch: int) -> None:
    import tensorflow as tf
    from tensorflow import keras
    tr = np.load(katalog_zbioru / "trening.npz")
    va = np.load(katalog_zbioru / "walidacja.npz")
    Xt, yt, Xv, yv = tr["X"], tr["y"], va["X"], va["y"]
    print(f"trening {len(yt)} wycinków {np.bincount(yt, minlength=3)}, "
          f"walidacja {len(yv)} {np.bincount(yv, minlength=3)}")

    def ds(X, y, mieszaj):
        # CPU: inaczej from_tensor_slices ładuje zbiór do pamięci KARTY (CLAUDE.md).
        with tf.device("/cpu:0"):
            d = tf.data.Dataset.from_tensor_slices((X, y))
        if mieszaj:
            d = d.shuffle(200_000, reshuffle_each_iteration=True)
        f = lambda x, t: (tf.cast(x, tf.float32)[..., None] / 255.0 * 1.25 - 0.25, t)
        return d.batch(batch).map(f, num_parallel_calls=4).prefetch(4)

    m = model_elementow()
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
    print(f"etap A: {m.count_params()} parametrów")
    katalog.mkdir(parents=True, exist_ok=True)
    m.fit(ds(Xt, yt, True), validation_data=ds(Xv, yv, False), epochs=epoki, verbose=2,
          callbacks=[
              keras.callbacks.ModelCheckpoint(str(katalog / "best.keras"), save_best_only=True),
              keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2),
              keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
              keras.callbacks.CSVLogger(str(katalog / "log.csv"))])


def _przewiduj(model, X, wsad=4096):
    import tensorflow as tf
    out = []
    for i in range(0, len(X), wsad):
        x = z_uint8(X[i:i + wsad])[..., None] if X.dtype == np.uint8 else X[i:i + wsad][..., None]
        out.append(tf.nn.softmax(model(x, training=False)).numpy())
    return np.concatenate(out) if out else np.zeros((0, 3), np.float32)


# --------------------------------------------------------------------------
# ocena na syntetyku (osobne klipy)
# --------------------------------------------------------------------------
def ocen(sciezka: Path, zbior: Path) -> None:
    from tensorflow import keras
    m = keras.models.load_model(sciezka)
    z = np.load(zbior)
    y, sn, neg, ul = z["y"], z["sn"], z["neg"], z["ul"] / 255.0
    p = _przewiduj(m, z["X"])
    q = p.argmax(1)
    print(f"ETAP A — {sciezka}, {zbior}: {len(y)} wycinków")
    print("  macierz pomyłek (wiersz = prawda, kolumna = sieć):")
    print("            " + "".join(f"{n:>9}" for n in E.NAZWY))
    for i, n in enumerate(E.NAZWY):
        r = np.bincount(q[y == i], minlength=3)
        print(f"  {n:>9} " + "".join(f"{v:9d}" for v in r) + f"   trafnie {r[i] / max(r.sum(), 1):.1%}")
    ton = q == E.TON
    print(f"  ton: precyzja {np.mean(y[ton] == E.TON):.1%}, czułość {np.mean(ton[y == E.TON]):.1%}")
    pewne = (ul < 0.2) | (ul > 0.8)
    print(f"  bez przegródek na granicy elementu (ułamek <0,2 albo >0,8): "
          f"trafnie {np.mean(q[pewne] == y[pewne]):.1%} ({pewne.mean():.0%} wycinków); "
          f"wszystkie {np.mean(q == y):.1%}")
    for a, b in PRZEDZIALY_SN:
        w = (sn >= a) & (sn < b) & pewne
        if w.any():
            print(f"  S/N {a:>3}..{b:<3} dB: trafnie {np.mean(q[w] == y[w]):.1%}, "
                  f"ton czułość {np.mean(q[w & (y == E.TON)] == E.TON):.1%}  ({w.sum()})")
    bez_cw = np.isnan(sn)
    for i, n in enumerate(NAZWY_NEG):
        w = bez_cw & (neg == i)
        if w.any():
            print(f"  klipy bez CW, negatyw {n or 'brak':>7}: sieć „ton” w {np.mean(q[w] == E.TON):.2%} "
                  f"przegródek ({w.sum()})")


# --------------------------------------------------------------------------
# negatywy spoza treningu: ile przegródek „ton” na minutę
# --------------------------------------------------------------------------
def tony_na_minute(model, audio, sr, wpm_siatki=(15.0, 25.0, 35.0), ton_hz=C.TONE_CENTER):
    """Audio bez CW -> przegródek uznanych za ton na minutę (pewność > 0,5),
    średnio po kilku tempach siatki. Kawałki po 4 s jak w treningu (mediana
    wodospadu z kawałka). Przegródka = kolumny co u (bez nakładania)."""
    from scipy.signal import resample_poly
    if sr != C.SR:
        from math import gcd
        g = gcd(int(sr), C.SR)
        audio = resample_poly(audio, C.SR // g, int(sr) // g)
    n4 = C.CLIP_SAMPLES
    wyn = []
    for wpm in wpm_siatki:
        u = E.jednostka(wpm)
        tony = 0
        for s in range(0, len(audio) - n4 + 1, n4):
            t = E.kolumny_czasu(n4, u)
            db = E.wodospad(audio[s:s + n4], ton_hz, u, t)
            X, _, _ = E.wycinki(db, np.zeros(t.size, np.uint8), np.zeros(t.size, np.float32), krok=E.K)
            if len(X):
                tony += int((_przewiduj(model, X)[:, E.TON] > 0.5).sum())
        wyn.append(tony / (len(audio) / C.SR / 60))
    return float(np.mean(wyn))


def negatywy(sciezka: Path, bank: Path, out: Path | None) -> None:
    import soundfile as sf
    from tensorflow import keras
    m = keras.models.load_model(sciezka)
    trening = set()
    spis = bank / "TRENING.txt"
    if spis.exists():
        trening = {l.strip().replace("\\", "/") for l in spis.read_text(encoding="utf-8").splitlines() if l.strip()}
    linie = [f"ETAP A NA NEGATYWACH — {sciezka} (przegródki „ton” na minutę, siatki 15/25/35 WPM, ton 750 Hz)"]
    for kat in sorted(p for p in bank.iterdir() if p.is_dir() and p.name not in ("skrypty", "surowe_kiwi")):
        wyn = []
        for wav in sorted(kat.glob("*.wav")):
            if f"{kat.name}/{wav.name}" in trening:
                continue
            x, sr = sf.read(wav, dtype="float32")
            if x.ndim > 1:
                x = x.mean(1)
            wyn.append((tony_na_minute(m, x, sr), wav.name))
        if not wyn:
            continue
        v = np.array([w for w, _ in wyn])
        linie.append(f"  {kat.name}: {v.mean():.1f} /min (mediana {np.median(v):.1f}, p90 {np.percentile(v, 90):.1f}), "
                     f"{len(v)} kawałków")
        for w, n in sorted(wyn, reverse=True)[:3]:
            linie.append(f"      {w:.0f}/min {n}")
    tekst = "\n".join(linie)
    print(tekst)
    if out:
        out.write_text(tekst + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("generuj")
    g.add_argument("wyj", type=Path); g.add_argument("n", type=int)
    g.add_argument("--seed", type=int, default=1); g.add_argument("--procesy", type=int, default=None)
    t = sp.add_parser("trenuj")
    t.add_argument("zbior", type=Path); t.add_argument("katalog", type=Path)
    t.add_argument("--epoki", type=int, default=20); t.add_argument("--batch", type=int, default=512)
    o = sp.add_parser("ocen")
    o.add_argument("model", type=Path); o.add_argument("zbior", type=Path)
    n = sp.add_parser("negatywy")
    n.add_argument("model", type=Path)
    n.add_argument("--bank", type=Path, default=Path(os.environ.get("CW_NEGATYWY", "/mnt/w/nagrania_cw/negatywy")))
    n.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    if a.cmd == "generuj":
        generuj(a.wyj, a.n, a.seed, a.procesy)
    elif a.cmd == "trenuj":
        trenuj(a.zbior, a.katalog, a.epoki, a.batch)
    elif a.cmd == "ocen":
        ocen(a.model, a.zbior)
    else:
        negatywy(a.model, a.bank, a.out)


if __name__ == "__main__":
    main()
