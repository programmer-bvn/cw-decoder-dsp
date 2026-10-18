"""DRUGI ETAP: zbiór -> trening -> ocena na nagraniach (18.10).

    python -m tools.etap2_trening generuj out/etap2/czesc_0.npz 150000 --seed 1
    python -m tools.etap2_trening trenuj out/etap2 runs/etap2_1 --epoki 15
    python -m tools.etap2_trening ocen runs/etap2_1/best.keras

Zbiór: klipy z generatora sieci zboczy (dsp/etap2.klip, procesy), przebieg
P(klucz) liczony PRAWDZIWĄ siecią zboczy (karta, wiele klipów na raz), z niego
po 2 okna na klip (dsp/etap2.probka: inna „zmierzona" jednostka, inne okno).
Ocena: nagrania z treścią (probki, probki/kluczownik, probki/reka); jednostka
z grzebienia (mediana okien po bramce, jak tester_zbocza), odczyt drugim
etapem obok odczytu tabelą z tych samych nut.
"""

from __future__ import annotations

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

from dsp import config as C, etap2 as E, zbocza as Z

MODEL_ZBOCZY = Path("runs/zbocza_3/best.keras")
NA_KLIP = 2


def _klipy(arg):
    i, n, seed = arg
    rng = np.random.default_rng([seed, i])
    return [E.klip(rng) for _ in range(n)]


def generuj(wyj: Path, n: int, seed: int, procesy: int | None) -> None:
    import concurrent.futures as cf
    from tensorflow import keras
    mz = keras.models.load_model(MODEL_ZBOCZY)
    procesy = procesy or max(1, (os.cpu_count() or 4) - 1)
    n_klipow = (n + NA_KLIP - 1) // NA_KLIP
    per = 100
    kaw = [(i, min(per, n_klipow - i * per), seed) for i in range((n_klipow + per - 1) // per)]
    X = np.zeros((n_klipow * NA_KLIP, E.OKNO), np.uint8)
    Y = np.zeros((n_klipow * NA_KLIP, E.KROKI), np.int16)
    rng = np.random.default_rng([seed, 10 ** 6])
    j, t0 = 0, time.time()
    with cf.ProcessPoolExecutor(max_workers=procesy) as ex:
        bufor = []
        for wynik in ex.map(_klipy, kaw):
            bufor.extend(wynik)
            if len(bufor) >= 2000 or j + len(bufor) * NA_KLIP >= X.shape[0]:
                pz = Z.przebieg_z_obrazow([o.astype(np.float32) / 255.0 for o, *_ in bufor], mz)
                for (o, znaki, tekst, u), p in zip(bufor, pz):
                    for _ in range(NA_KLIP):
                        if j < X.shape[0]:
                            X[j], Y[j] = E.probka(p, znaki, tekst, u, rng); j += 1
                bufor = []
                el = time.time() - t0
                print(f"\r  {j}/{X.shape[0]}  {el:.0f}s  pozostało ~{el / max(1, j) * (X.shape[0] - j):.0f}s ",
                      end="", flush=True)
    print()
    wyj.parent.mkdir(parents=True, exist_ok=True)
    tymczasowy = wyj.with_name(wyj.name + ".part")
    with open(tymczasowy, "wb") as f:
        np.savez(f, X=X[:j], Y=Y[:j], meta=f"n={j};seed={seed};K={E.K};okno_u={E.OKNO_U};zbocza={MODEL_ZBOCZY}")
        f.flush(); os.fsync(f.fileno())
    os.replace(tymczasowy, wyj)
    print(f"zapisano {wyj} ({wyj.stat().st_size / 2**20:.0f} MB, {time.time() - t0:.0f} s)")


def trenuj(zbior: Path, katalog: Path, epoki: int, batch: int) -> None:
    import tensorflow as tf
    from tensorflow import keras
    pliki = sorted(zbior.glob("*.npz")) if zbior.is_dir() else [zbior]
    d = [np.load(p) for p in pliki]
    X = np.concatenate([x["X"] for x in d]); Y = np.concatenate([x["Y"] for x in d]).astype(np.int32)
    n = len(X); idx = np.random.default_rng(0).permutation(n); nv = max(min(2000, n // 10), n // 20)
    tr, va = idx[nv:], idx[:nv]
    print(f"zbiór {n}: trening {len(tr)}, walidacja {nv}; klasa 0 w krokach: {100 * (Y == 0).mean():.0f}%")

    def ds(ii, mieszaj):
        with tf.device("/cpu:0"):              # inaczej zbiór ląduje w pamięci karty
            dd = tf.data.Dataset.from_tensor_slices((X[ii], Y[ii]))
        if mieszaj:
            dd = dd.shuffle(50000, reshuffle_each_iteration=True)
        dd = dd.batch(batch).map(lambda x, y: (tf.cast(x, tf.float32)[:, :, None, None] / 255.0, y),
                                 num_parallel_calls=tf.data.AUTOTUNE)
        return dd.prefetch(tf.data.AUTOTUNE)

    m = E.model_etap2()
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["sparse_categorical_accuracy"])
    print(f"sieć drugiego etapu: {m.count_params()} parametrów")
    katalog.mkdir(parents=True, exist_ok=True)
    m.fit(ds(tr, True), validation_data=ds(va, False), epochs=epoki, verbose=2,
          callbacks=[keras.callbacks.ModelCheckpoint(str(katalog / "best.keras"), save_best_only=True),
                     keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2),
                     keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                     keras.callbacks.CSVLogger(str(katalog / "log.csv"))])


def ocen(sciezka: Path) -> None:
    from collections import Counter
    from tensorflow import keras
    from dsp import frontend, grzebien, tune
    from tools import nagrania, wav2net
    from tools import tester_zbocza as T
    m2 = keras.models.load_model(sciezka)
    mz = keras.models.load_model(MODEL_ZBOCZY)
    n8, k4 = 8 * C.SR, 4 * C.SR
    suma = Counter(); razem = 0
    print(f"drugi etap {sciezka} vs tabela (te same nuty sieci zboczy, histereza); słowa w całości")
    for kat in (Path("probki"), Path("probki/kluczownik"), Path("probki/reka")):
        for p in sorted(kat.glob("*.wav")):
            chce = nagrania.oczekiwane(p.with_suffix(".txt"))
            if not chce:
                continue
            a = frontend.load_audio(p)
            t, f = tune.track_tone(a); a = tune.retune(a, t, f)
            g = wav2net.auto_gain_db(a); a = np.clip(a * 10 ** (g / 20), -1, 1).astype(np.float32)
            starty = list(range(0, max(1, a.size - n8 + 1), k4))
            if starty[-1] < a.size - n8:
                starty.append(a.size - n8)
            pz = Z.przebieg_sieci(a, mz)
            r5 = n8 // Z.HOP_ZBOCZA
            oo = [Z.odcinki_z_przebiegu(pz[s // Z.HOP_ZBOCZA:s // Z.HOP_ZBOCZA + r5],
                                        histereza=(Z.PROG_WL, Z.PROG_WYL)) for s in starty]
            tab, nuty_, _, _ = T.droga(a, oo, starty, n8)
            u = float(np.median([x[2] for x in nuty_])) if nuty_ else C.dot_seconds(C.WPM)
            e2 = E.odczyt(pz, u, m2)
            s_t, s_2 = T.slowa(tab, chce)[0], T.slowa(e2, chce)[0]
            suma["tab"] += s_t; suma["e2"] += s_2; razem += len(chce)
            print(f"  {str(p)[:52]:52s} tabela {s_t:3d}/{len(chce)}  etap2 {s_2:3d}/{len(chce)}")
            if s_2 < s_t:
                print(f"      tabela: {tab[:90]}\n      etap2 : {e2[:90]}")
    print(f"RAZEM: tabela {suma['tab']}/{razem}, drugi etap {suma['e2']}/{razem}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generuj"); g.add_argument("wyj", type=Path); g.add_argument("n", type=int)
    g.add_argument("--seed", type=int, default=1); g.add_argument("--procesy", type=int, default=None)
    t = sub.add_parser("trenuj"); t.add_argument("zbior", type=Path); t.add_argument("katalog", type=Path)
    t.add_argument("--epoki", type=int, default=15); t.add_argument("--batch", type=int, default=256)
    o = sub.add_parser("ocen"); o.add_argument("model", type=Path)
    a = ap.parse_args()
    if a.cmd == "generuj":
        generuj(a.wyj, a.n, a.seed, a.procesy)
    elif a.cmd == "trenuj":
        trenuj(a.zbior, a.katalog, a.epoki, a.batch)
    else:
        ocen(a.model)


if __name__ == "__main__":
    main()
