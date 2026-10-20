"""SIEĆ ZBOCZY: zbiór -> trening -> porównanie z metodą klasyczną (17.10).

    python -m tools.zbocza_trening generuj  out/zbocza/czesc_0.npz 100000 --seed 1
    python -m tools.zbocza_trening trenuj   out/zbocza runs/zbocza_1 --epoki 20
    python -m tools.zbocza_trening ocen     runs/zbocza_1/best.keras
    python -m tools.zbocza_trening generuj  out/zbocza/z5/czesc_0.npz 100000 --seed 1 --wersja 5
    (zboczy_5, 20.10: drugie wejście — obraz drobny 7,8 Hz wokół 750 Hz, tłok)

Pierwszy etap dwuetapowego dekodera (dsp/zbocza.py): widmo co 5 ms ->
P(klucz wciśnięty) co 5 ms. Etykieta to PRAWDZIWY przebieg klucza z generatora
(radio.receive z_kluczem), zanim tor go zniekształcił.

ODNIESIENIE (17.10, 400 klipów na przedział): próg Otsu na przekroju przy
tonie — tak, jak robi to grzebień — poszerza znak SYMETRYCZNIE: narastające
zbocze wcześniej, opadające później, o 2 ms przy S/N 0 dB i 10 ms przy 25 dB
(okno 32 ms rozlewa energię, silniejszy sygnał wcześniej przekracza próg).
To „kość" operatora w liczbach. Sieć widzi poziom, więc może to oddać.

Architektura DPU-bezpieczna: same Conv2D/BN/ReLU/MaxPooling. Pasma zwijane
pulowaniem (1,2), potem sploty po czasie z dylatacją (zasięg ~1,3 s).
Wyjście to LOGIT — sigmoid dla 512 liczb liczy już procesor ARM.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

if len(sys.argv) > 1 and sys.argv[1] == "generuj":
    # jak w train_rtx generate: jeden wątek BLAS na proces (16.10: 355 -> 65 s)
    for _zm in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_zm, "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsp import config as C, sn, tempo, zbocza as Z

POLA = ("snr_db", "wpm", "tone", "waga", "ramp_ms", "tlo_db")
PRZEDZIALY_SN = ((-3, 3), (3, 10), (10, 20), (20, 31), (31, 46))


# --------------------------------------------------------------------------
# generowanie
# --------------------------------------------------------------------------
def _kawalek(arg):
    i, n, seed, wersja = arg
    rng = np.random.default_rng([seed, i])
    X = np.zeros((n, Z.RAMKI_ZBOCZA, C.N_MELS), np.uint8)
    X2 = np.zeros((n, Z.RAMKI_ZBOCZA, Z.DROBNY_PRAZKI), np.uint8) if wersja == 5 else None
    K = np.zeros((n, Z.RAMKI_ZBOCZA), np.uint8)
    M = {k: np.full(n, np.nan, np.float32) for k in POLA}
    for j in range(n):
        if wersja == 5:
            img, drob, kl, m = Z.klip_zboczy_5(rng)
            X2[j] = np.round(np.clip(drob, 0, 1) * 255)
        else:
            img, kl, m = Z.klip_zboczy(rng)
        X[j] = np.round(np.clip(img, 0, 1) * 255)
        K[j] = np.round(np.clip(kl, 0, 1) * 255)
        for k in POLA:
            if m.get(k) is not None and m.get("text", "") != "":
                M[k][j] = m[k]
    return X, X2, K, M


def generuj(wyj: Path, n: int, seed: int, procesy: int | None, wersja: int = 3) -> None:
    import concurrent.futures as cf
    procesy = procesy or max(1, (os.cpu_count() or 4) - 1)
    per = 250
    kaw = [(i, min(per, n - i * per), seed, wersja) for i in range((n + per - 1) // per)]
    Xs, X2s, Ks, Ms = [], [], [], []
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=procesy) as ex:
        for X, X2, K, M in ex.map(_kawalek, kaw):
            Xs.append(X); X2s.append(X2); Ks.append(K); Ms.append(M)
            g = sum(len(x) for x in Xs)
            el = time.time() - t0
            print(f"\r  {g}/{n}  {el:.0f}s  pozostało ~{el / g * (n - g):.0f}s ",
                  end="", flush=True)
    print()
    wyj.parent.mkdir(parents=True, exist_ok=True)
    tymczasowy = wyj.with_name(wyj.name + ".part")
    with open(tymczasowy, "wb") as f:
        dodatkowe = {"X2": np.concatenate(X2s)} if wersja == 5 else {}
        np.savez(f, X=np.concatenate(Xs), K=np.concatenate(Ks), **dodatkowe,
                 **{k: np.concatenate([m[k] for m in Ms]) for k in POLA},
                 meta=f"n={n};seed={seed};hop={Z.HOP_ZBOCZA};nfft={Z.N_FFT_ZBOCZA};wersja={wersja}")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tymczasowy, wyj)
    print(f"zapisano {wyj} ({wyj.stat().st_size / 2**20:.0f} MB, {time.time() - t0:.0f} s)")


def wczytaj(sciezka: Path):
    pliki = sorted(sciezka.glob("*.npz")) if sciezka.is_dir() else [sciezka]
    if not pliki:
        raise SystemExit(f"brak części zbioru zboczy w {sciezka}")
    d = [np.load(p) for p in pliki]
    X = np.concatenate([x["X"] for x in d]); K = np.concatenate([x["K"] for x in d])
    X2 = np.concatenate([x["X2"] for x in d]) if "X2" in d[0].files else None
    M = {k: np.concatenate([x[k] for x in d]) for k in POLA}
    return X, X2, K, M


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
def model_zboczy():
    from tensorflow import keras
    L = keras.layers
    wej = keras.Input((Z.RAMKI_ZBOCZA, C.N_MELS, 1), name="widmo_5ms")
    x = wej
    # pasma: 32 -> 16 -> 8 -> 4 -> 2 -> 1, czas bez zmian (co 5 ms)
    for f in (32, 48, 64, 96, 128):
        x = L.Conv2D(f, (5, 3), padding="same", use_bias=False)(x)
        x = L.BatchNormalization()(x)
        x = L.ReLU()(x)
        x = L.MaxPooling2D((1, 2))(x)
    # czas: dylatacje 1..32 -> zasięg ~ +/-130 ramek (0,65 s w każdą stronę)
    for d in (1, 2, 4, 8, 16, 32):
        y = L.Conv2D(128, (5, 1), dilation_rate=(d, 1), padding="same",
                     use_bias=False)(x)
        y = L.BatchNormalization()(y)
        y = L.ReLU()(y)
        x = L.Add()([x, y])
    x = L.Conv2D(1, (1, 1), name="logit")(x)
    wyj = L.Reshape((Z.RAMKI_ZBOCZA,))(x)
    return keras.Model(wej, wyj, name="siec_zboczy")


def model_zboczy_5():
    """Dwa wejścia (20.10): mel [512, 32] jak zboczy_3 i obraz drobny
    [512, 32] (625-867 Hz co 7,8 Hz, nasza stacja na prążku 16). Każde
    zwijane po częstotliwości osobno do 1, kanały łączone (Concatenate —
    DPU je wykonuje), potem te same sploty po czasie co zboczy_3."""
    from tensorflow import keras
    L = keras.layers
    wej = keras.Input((Z.RAMKI_ZBOCZA, C.N_MELS, 1), name="widmo_5ms")
    wej2 = keras.Input((Z.RAMKI_ZBOCZA, Z.DROBNY_PRAZKI, 1), name="drobny_5ms")

    def galaz(x, filtry):
        for f in filtry:
            x = L.Conv2D(f, (5, 3), padding="same", use_bias=False)(x)
            x = L.BatchNormalization()(x)
            x = L.ReLU()(x)
            x = L.MaxPooling2D((1, 2))(x)
        return x

    a = galaz(wej, (32, 48, 64, 96, 128))          # 32 -> 1 pasmo
    b = galaz(wej2, (32, 48, 64, 64, 64))          # 32 -> 1 prążek
    x = L.Concatenate(axis=-1)([a, b])
    x = L.Conv2D(128, (1, 1), use_bias=False)(x)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)
    for d in (1, 2, 4, 8, 16, 32):
        y = L.Conv2D(128, (5, 1), dilation_rate=(d, 1), padding="same",
                     use_bias=False)(x)
        y = L.BatchNormalization()(y)
        y = L.ReLU()(y)
        x = L.Add()([x, y])
    x = L.Conv2D(1, (1, 1), name="logit")(x)
    wyj = L.Reshape((Z.RAMKI_ZBOCZA,))(x)
    return keras.Model([wej, wej2], wyj, name="siec_zboczy_5")


def trenuj(zbior: Path, katalog: Path, epoki: int, batch: int) -> None:
    import tensorflow as tf
    from tensorflow import keras
    X, X2, K, M = wczytaj(zbior)
    n = len(X)
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    nv = max(1000, n // 20)
    tr, va = idx[nv:], idx[:nv]
    print(f"zbiór {n} klipów: trening {len(tr)}, walidacja {nv}")

    def ds(ii, mieszaj):
        # CPU: inaczej from_tensor_slices ładuje zbiór do pamięci KARTY
        with tf.device("/cpu:0"):
            d = tf.data.Dataset.from_tensor_slices(
                ((X[ii], X2[ii]) if X2 is not None else X[ii], K[ii]))
        if mieszaj:
            d = d.shuffle(20000, reshuffle_each_iteration=True)
        obraz = lambda x: tf.cast(x, tf.float32)[..., None] / 255.0
        d = d.batch(batch).map(
            lambda x, k: ((obraz(x[0]), obraz(x[1])) if X2 is not None else obraz(x),
                          tf.cast(k, tf.float32) / 255.0),
            num_parallel_calls=tf.data.AUTOTUNE)
        return d.prefetch(tf.data.AUTOTUNE)

    m = model_zboczy_5() if X2 is not None else model_zboczy()
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss=keras.losses.BinaryCrossentropy(from_logits=True))
    print(f"sieć zboczy: {m.count_params()} parametrów")
    katalog.mkdir(parents=True, exist_ok=True)
    m.fit(ds(tr, True), validation_data=ds(va, False), epochs=epoki, verbose=2,
          callbacks=[
              keras.callbacks.ModelCheckpoint(str(katalog / "best.keras"),
                                              save_best_only=True),
              keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2),
              keras.callbacks.EarlyStopping(patience=5,
                                            restore_best_weights=True),
              keras.callbacks.CSVLogger(str(katalog / "log.csv"))])


# --------------------------------------------------------------------------
# ocena: sieć vs próg klasyczny, te same klipy
# --------------------------------------------------------------------------
def klasyczna(img: np.ndarray, ton: float) -> np.ndarray:
    """Przekrój przy tonie, moc, wygładzenie 15 ms, próg Otsu — jak grzebień."""
    db = C.DB_MIN + img * (C.DB_MAX - C.DB_MIN)
    p = (10 ** (db[:, sn.pasma_przy_tonie(ton)] / 10)).mean(axis=1)
    p = np.convolve(p, np.ones(3) / 3, mode="same")
    d = 10 * np.log10(p + 1e-12)
    return (d > tempo._prog_otsu(d)).astype(np.float32)


def _ze_znakiem(pp, tp, tol):
    out = []
    for x in tp:
        if pp.size:
            j = int(np.argmin(np.abs(pp - x)))
            if abs(pp[j] - x) < tol:
                out.append(float(pp[j] - x))
    return out


def ocen(sciezka_modelu: Path, n_na_przedzial: int, seed: int) -> None:
    from tensorflow import keras
    import tensorflow as tf
    m = keras.models.load_model(sciezka_modelu)
    rng = np.random.default_rng([seed, 99])
    print(f"model {sciezka_modelu}, {n_na_przedzial} klipów na przedział S/N "
          f"(błąd zbocza: |mediana| i 90%, przesunięcie narast./opad. ze znakiem)")
    for lo, hi in PRZEDZIALY_SN:
        imgs, drobs, kls, ms = [], [], [], []
        dwa = len(m.inputs) == 2
        while len(imgs) < n_na_przedzial:
            if dwa:
                img, drob, kl, mm = Z.klip_zboczy_5(rng)
            else:
                (img, kl, mm), drob = Z.klip_zboczy(rng), None
            if mm.get("text") and lo <= mm["snr_db"] < hi:
                imgs.append(img); drobs.append(drob); kls.append(kl); ms.append(mm)
        wej = ([np.array(imgs)[..., None], np.array(drobs)[..., None]] if dwa
               else np.array(imgs)[..., None])
        p = tf.sigmoid(m.predict(wej, verbose=0, batch_size=64)).numpy()
        for nazwa, wyj in (("klasyczna", [klasyczna(i, mm["tone"]) for i, mm in zip(imgs, ms)]),
                           ("sieć", list(p))):
            b, pom, fal, pr, nar, opa = [], 0, 0, 0, [], []
            for w, kl, mm in zip(wyj, kls, ms):
                u = 1.2 / mm["wpm"] / 0.005
                r = Z.blad_zboczy(w, kl, tol_ramek=0.5 * u)
                b += r["bledy"]; pom += r["pominiete"]; fal += r["falszywe"]
                pr += r["prawdziwych"]
                (tn, to), (pn, po) = Z.zbocza(kl), Z.zbocza(w)
                nar += _ze_znakiem(pn, tn, 0.5 * u); opa += _ze_znakiem(po, to, 0.5 * u)
            b = np.abs(np.array(b)) * 5.0
            print(f"S/N {lo:3d}..{hi:3d} dB {nazwa:9s}: mediana {np.median(b):5.1f} ms, "
                  f"90% {np.percentile(b, 90):5.1f} ms; pominięte {100 * pom / pr:4.1f}%, "
                  f"fałszywe {100 * fal / pr:5.1f}%; przesunięcie "
                  f"{5 * np.median(nar):+5.1f}/{5 * np.median(opa):+5.1f} ms")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generuj")
    g.add_argument("wyj", type=Path); g.add_argument("n", type=int)
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--procesy", type=int, default=None)
    g.add_argument("--wersja", type=int, default=3, choices=[3, 5])
    t = sub.add_parser("trenuj")
    t.add_argument("zbior", type=Path); t.add_argument("katalog", type=Path)
    t.add_argument("--epoki", type=int, default=20)
    t.add_argument("--batch", type=int, default=64)
    o = sub.add_parser("ocen")
    o.add_argument("model", type=Path)
    o.add_argument("--n", type=int, default=150)
    o.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    if a.cmd == "generuj":
        generuj(a.wyj, a.n, a.seed, a.procesy, a.wersja)
    elif a.cmd == "trenuj":
        trenuj(a.zbior, a.katalog, a.epoki, a.batch)
    else:
        ocen(a.model, a.n, a.seed)


if __name__ == "__main__":
    main()
