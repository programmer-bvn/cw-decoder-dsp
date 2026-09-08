"""TRENING: zbiór .npz -> model .keras.

    python -m tools.trainer
    python -m tools.trainer --epochs 60 --batch 128
    python -m tools.trainer --dataset out/inny.npz --out out/inny_model.keras

Trzy rzeczy zrobione inaczej niż w poprzednich trenerach:

1. ODCISK FRONT-ENDU jest sprawdzany PRZED treningiem. Jeśli zbiór powstał
   na innych parametrach niż obecny config.py, trening się nie zaczyna
   i wypisuje, które wartości się różnią. Wcześniej taki rozjazd przechodził
   bez śladu i ujawniał się dopiero na żywym sygnale.

2. DANE IDĄ PRZEZ tf.data, a nie jako jedna tablica w pamięci. Obrazy leżą
   w zbiorze jako uint8 (164 MB dla 40000 próbek) i są przeliczane na
   float32 partia po partii. Cały zbiór w float32 to 655 MB, plus tyle samo
   na kopię przy podziale na zbiory — stąd brał się "trainer_outofmemor".

3. PODZIAŁ JEST WARSTWOWY (stratified). Klasa 0 to 15% zbioru, a pozostałe
   36 klas dzielą się resztą po ~2,4%. Przy losowym podziale 10% na
   walidację rzadsza klasa może dostać kilka próbek i val_accuracy przestaje
   znaczyć cokolwiek dla tej klasy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsp import config as C


def load_dataset(path: Path):
    """Wczytuje zbiór, sprawdza odcisk, zwraca (X_uint8_lub_float, y)."""
    if not path.exists():
        raise SystemExit(f"nie ma zbioru: {path}\n"
                         f"Najpierw: python -m tools.generator")
    data = np.load(path, allow_pickle=False)
    if "fingerprint" in data:
        C.check_fingerprint(str(data["fingerprint"]), source=path.name)
        print(f"odcisk front-endu: zgodny z config.py")
    else:
        print("UWAGA: zbiór bez odcisku front-endu (stary format) — "
              "nie da się sprawdzić zgodności parametrów.")
    return np.asarray(data["X"]), np.asarray(data["y"]).astype(np.int32)


def stratified_split(y: np.ndarray, val_fraction: float, seed: int):
    """Podział warstwowy: każda klasa oddaje ten sam UDZIAŁ na walidację."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_fraction)))
        val_idx.append(idx[:n_val])
        train_idx.append(idx[n_val:])
    tr = np.concatenate(train_idx)
    va = np.concatenate(val_idx)
    rng.shuffle(tr)
    rng.shuffle(va)
    return tr, va


def class_weights(y: np.ndarray) -> np.ndarray:
    """Wagi wyrównujące udział klas: waga = n / (liczba_klas * licznik).

    PO CO. Generator daje 15% klipów "puste radio" (klasa 0) i po ~2,4% na
    każdy z 36 znaków. To sześciokrotna przewaga jednej klasy, a najtańsze
    minimum funkcji straty przy takim rozkładzie to "zawsze odpowiadaj 0" —
    daje 15% dokładności bez uczenia się czegokolwiek. Model dokładnie w to
    wpadał: 100% trafień na klasie 0 i 0% na znakach.

    Wagi są liczone TYLKO ze zbioru treningowego. Liczenie ich z całości
    przeniosłoby informację o rozkładzie walidacji do treningu.
    """
    counts = np.bincount(y, minlength=C.N_CLASSES).astype(np.float64)
    present = counts > 0
    w = np.zeros(C.N_CLASSES, dtype=np.float32)
    w[present] = len(y) / (present.sum() * counts[present])
    return w


def make_pipeline(X, y, idx, batch: int, training: bool,
                  weights: np.ndarray | None = None):
    """tf.data z konwersją uint8 -> float32 dopiero w partii.

    Wagi wchodzą jako trzeci element krotki, a nie przez class_weight= w fit().
    Dla zbiorów tf.data to droga pewna — class_weight bywa ignorowane albo
    zgłasza błąd, w zależności od wersji Kerasa.
    """
    import tensorflow as tf

    is_u8 = X.dtype == np.uint8
    parts = [X[idx], y[idx]]
    if weights is not None:
        parts.append(weights[y[idx]])
    ds = tf.data.Dataset.from_tensor_slices(tuple(parts))

    def prep(img, label, *rest):
        img = tf.cast(img, tf.float32)
        if is_u8:
            img = img / 255.0
        img = tf.expand_dims(img, -1)
        return (img, label, rest[0]) if rest else (img, label)

    if training:
        ds = ds.shuffle(min(len(idx), 20000), reshuffle_each_iteration=True)
    return (ds.map(prep, num_parallel_calls=tf.data.AUTOTUNE)
              .batch(batch)
              .prefetch(tf.data.AUTOTUNE))


def confusion_report(model, ds, y_true: np.ndarray) -> None:
    """Macierz pomyłek w formie listy — 37x37 tabela w terminalu jest
    nieczytelna, a interesuje nas i tak tylko to, co się myli."""
    probs = model.predict(ds, verbose=0)
    pred = probs.argmax(axis=1)

    acc = float(np.mean(pred == y_true))
    print(f"\nDokładność na walidacji: {acc*100:.2f}%")

    # Klasa 0 osobno: pomylenie ciszy z sygnałem to inny rodzaj błędu niż
    # pomylenie dwóch znaków, i w praktyce bardziej kosztowny.
    m0 = y_true == 0
    if m0.any():
        print(f"  klasa 0 (puste radio): "
              f"{np.mean(pred[m0] == 0)*100:.1f}% poprawnie "
              f"({m0.sum()} próbek)")
    if (~m0).any():
        print(f"  znaki:                 "
              f"{np.mean(pred[~m0] == y_true[~m0])*100:.1f}% poprawnie "
              f"({(~m0).sum()} próbek)")
        fa = float(np.mean(pred[~m0] == 0))
        print(f"  znak wzięty za ciszę:  {fa*100:.1f}%")

    pairs = {}
    for t, p in zip(y_true, pred):
        if t != p:
            pairs[(int(t), int(p))] = pairs.get((int(t), int(p)), 0) + 1
    if pairs:
        print("\nNajczęstsze pomyłki (prawda -> predykcja):")
        for (t, p), n in sorted(pairs.items(), key=lambda kv: -kv[1])[:15]:
            ch_t = "PUSTE" if t == 0 else C.ID_TO_CHAR[t]
            ch_p = "PUSTE" if p == 0 else C.ID_TO_CHAR[p]
            code_t = C.MORSE_DICT.get(ch_t, "")
            code_p = C.MORSE_DICT.get(ch_p, "")
            print(f"  {ch_t:>5} {code_t:<6} -> {ch_p:>5} {code_p:<6}  {n}x")
        print("\nPomyłki między znakami różniącymi się jednym elementem "
              "(np. U '..-' / V '...-') znaczą, że model gubi timing, "
              "a nie że brakuje mu pojemności.")


def plot_history(history, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = history.history
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(h["loss"], label="trening")
    ax[0].plot(h["val_loss"], label="walidacja")
    ax[0].set_title("strata"); ax[0].set_xlabel("epoka"); ax[0].legend()
    ax[1].plot(np.array(h["accuracy"]) * 100, label="trening")
    ax[1].plot(np.array(h["val_accuracy"]) * 100, label="walidacja")
    ax[1].set_title("dokładność [%]"); ax[1].set_xlabel("epoka"); ax[1].legend()
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"krzywe uczenia: {out_path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Trening modelu CW")
    ap.add_argument("--dataset", type=Path, default=C.DATASET_PATH)
    ap.add_argument("--out", type=Path, default=C.MODEL_PATH)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arch", choices=("dpu", "gru"), default="dpu",
                    help="dpu = sam splot, wdrażalny na KV260 (domyślnie); "
                         "gru = z rekurencją, tylko jako punkt odniesienia")
    ap.add_argument("--no-class-weights", action="store_true",
                    help="wyłącz wyrównywanie udziału klas")
    args = ap.parse_args(argv)

    print("=" * 70)
    print("TRENING")
    print("=" * 70)
    print(C.summary())
    print("-" * 70)

    X, y = load_dataset(args.dataset)
    print(f"zbiór: {X.shape} {X.dtype}  "
          f"({X.nbytes/1024/1024:.0f} MB w pamięci)")

    tr, va = stratified_split(y, args.val, args.seed)
    print(f"podział warstwowy: trening={len(tr)}  walidacja={len(va)}")

    import keras
    import tensorflow as tf
    from dsp.model import build_model, check_dpu_compatible

    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPU: {[g.name for g in gpus] if gpus else 'brak (CPU)'}")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)

    w = None
    if not args.no_class_weights:
        w = class_weights(y[tr])
        print(f"wagi klas: klasa 0 -> {w[0]:.3f}, "
              f"znaki -> {w[1:][w[1:] > 0].mean():.3f} "
              f"(stosunek {w[1:][w[1:] > 0].mean() / w[0]:.1f}:1)")

    ds_tr = make_pipeline(X, y, tr, args.batch, training=True, weights=w)
    ds_va = make_pipeline(X, y, va, args.batch, training=False)

    model = build_model(arch=args.arch)
    model.summary()
    check_dpu_compatible(model)

    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                      patience=args.patience,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=3, min_lr=1e-5, verbose=1),
    ]

    print("\nstart treningu")
    history = model.fit(ds_tr, validation_data=ds_va, epochs=args.epochs,
                        callbacks=cbs, verbose=1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"\nzapisano model: {args.out} "
          f"({args.out.stat().st_size/1024/1024:.1f} MB)")

    confusion_report(model, ds_va, y[va])
    plot_history(history, C.OUT_DIR / "krzywe.png")


if __name__ == "__main__":
    main()
