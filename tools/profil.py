"""PROFIL KOSZTU: co w modelu zajmuje czas i ile da się urwać.

    python -m tools.profil
    python -m tools.profil --steps 12 --batch 64

Liczy mnożenia na próbkę warstwa po warstwie i MIERZY czas kroku uczenia dla
kilku wariantów. Sens: bez tego "model jest za wolny" prowadzi do zgadywania,
co obciąć. Rozkład kosztu w sieci splotowej rzadko jest tam, gdzie się wydaje —
najdroższe są warstwy o dużej mapie, nie te o wielu kanałach.

Warianty do porównania są wybrane na podstawie POMIARU z diag.py (TEST 10):
silny ton podnosi 21 z 32 pasm mel, a 6 najmocniejszych pasm mieści tylko 46%
przyrostu energii. Czyli oś częstotliwości jest w dużej części redundantna:
pasmo 400-1200 Hz podzielone na 32 pasma daje 25 Hz na pasmo, przy
rozdzielczości STFT 15,6 Hz i rozmyciu okna Hanna szerszym niż to. Dlatego
zmniejszenie liczby pasm jest pierwszym kandydatem, a nie ostatnim.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsp import config as C


def layer_macs(model) -> list[tuple[str, int, tuple]]:
    """Mnożenia na jedną próbkę, warstwa po warstwie."""
    out = []
    for l in model.layers:
        t = type(l).__name__
        macs = 0
        osh = l.output.shape if hasattr(l, "output") else None
        if t == "Conv2D":
            oh, ow, oc = osh[1], osh[2], osh[3]
            k = l.kernel_size[0] * l.kernel_size[1]
            ic = l.input.shape[-1]
            macs = int(oh) * int(ow) * int(oc) * k * int(ic)
        elif t == "Dense":
            macs = int(l.input.shape[-1]) * int(l.units)
        elif t == "Bidirectional":
            # GRU: 3 bramki, macierz wejściowa i rekurencyjna, dwa kierunki
            inner = l.forward_layer
            u = inner.units
            f = int(l.input.shape[-1])
            steps = int(l.input.shape[1])
            macs = 2 * steps * 3 * (f * u + u * u)
        out.append((f"{l.name} ({t})", macs,
                    tuple(osh[1:]) if osh is not None else ()))
    return out


def build_variant(name: str, n_mels: int, frames: int, channels, pools):
    """Model o zadanych parametrach — bez dotykania config.py."""
    import keras
    from keras import layers

    # Nazwa modelu nie może mieć spacji ani polskich znaków — Keras używa
    # jej jako nazwy zakresu w grafie TensorFlow.
    name = "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")
    inputs = keras.Input(shape=(frames, n_mels, 1), name="obraz")
    x = inputs
    for filt, pool in zip(channels, pools):
        x = layers.Conv2D(filt, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        if pool is not None:
            # Przy małej liczbie pasm oś częstotliwości szybko dochodzi do 1
            # i kolejna redukcja 2x dałaby rozmiar 0. Wtedy redukujemy tylko
            # czas — inaczej wariant z 8 pasmami nie da się w ogóle zbudować.
            pf = pool[1] if x.shape[2] >= 2 * pool[1] else 1
            x = layers.MaxPooling2D((pool[0], pf))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(C.N_CLASSES, activation="softmax", dtype="float32")(x)
    m = keras.Model(inputs, out, name=name)
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def receptive_field(model) -> int:
    """Zasięg widzenia po osi CZASU, liczony ze stosu warstw."""
    rf, jump = 1, 1
    for l in model.layers:
        t = type(l).__name__
        if t == "Conv2D":
            rf += (l.kernel_size[0] - 1) * jump
            jump *= l.strides[0]
        elif t == "MaxPooling2D":
            rf += (l.pool_size[0] - 1) * jump
            jump *= l.strides[0]
    return rf


def time_steps(model, batch: int, steps: int, frames: int, n_mels: int
               ) -> float:
    """Średni czas kroku uczenia [s]. Pierwsze kroki pomijane — zawierają
    kompilację grafu i rozgrzewanie oneDNN."""
    import tensorflow as tf
    rng = np.random.default_rng(0)
    X = rng.random((batch, frames, n_mels, 1)).astype(np.float32)
    y = rng.integers(0, C.N_CLASSES, batch).astype(np.int32)

    for _ in range(3):                        # rozgrzewka
        model.train_on_batch(X, y)
    t0 = time.time()
    for _ in range(steps):
        model.train_on_batch(X, y)
    return (time.time() - t0) / steps


def main(argv=None):
    ap = argparse.ArgumentParser(description="Profil kosztu modelu")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--breakdown", action="store_true",
                    help="rozklad kosztu obecnego modelu (proces potomny)")
    ap.add_argument("--variant", type=int, default=None,
                    help="mierz tylko ten wariant — używane wewnętrznie, "
                         "każdy wariant idzie w osobnym procesie")
    ap.add_argument("--n-train", type=int, default=30000,
                    help="rozmiar zbioru do przeliczenia czasu epoki")
    args = ap.parse_args(argv)

    # Warianty do porównania. Zmiana liczby pasm jest pierwszym kandydatem,
    # bo pomiar z diag.py (TEST 10) pokazał, że oś częstotliwości jest
    # w dużej części redundantna: silny ton podnosi 21 z 32 pasm.
    P4 = [(2, 2), (2, 2), (2, 2), (2, 2), None, None]
    variants = [
        ("obecny  128x32", 32, 128, (32, 48, 64, 96, 128, 128), P4),
        ("16 pasm 128x16", 16, 128, (32, 48, 64, 96, 128, 128), P4),
        ("8 pasm  128x8 ", 8, 128, (32, 48, 64, 96, 128, 128), P4),
        ("16 pasm, chudy", 16, 128, (16, 32, 48, 64, 96, 96), P4),
        ("8 pasm,  chudy", 8, 128, (16, 32, 48, 64, 96, 96), P4),
    ]

    # --- PROCES POTOMNY: tylko jeden pomiar, nic więcej ---
    # Budowanie modelu odniesienia i wariantu w tym samym procesie
    # przepełniało pamięć, dlatego tu nie ma żadnych dodatkowych modeli
    # ani wypisywania nagłówków.
    if args.variant is not None:
        import gc
        import keras
        name, nm_, fr, ch, po = variants[args.variant]
        mv = build_variant(name.strip(), nm_, fr, ch, po)
        macs = sum(r[1] for r in layer_macs(mv))
        params = mv.count_params()
        rf = receptive_field(mv)
        dt = time_steps(mv, args.batch, args.steps, fr, nm_)
        print(f"ROW \t{macs}\t{params}\t{dt}\t{rf}", flush=True)
        del mv
        keras.backend.clear_session()
        gc.collect()
        return 0

    # --- PROCES POTOMNY: rozkład kosztu obecnego modelu ---
    if args.breakdown:
        from dsp.model import build_model
        m = build_model(arch="dpu")
        rows = layer_macs(m)
        total = sum(r[1] for r in rows)
        print(f"Obecny model ({m.name}): {total/1e6:.1f} mln mnożeń "
              f"na próbkę, {m.count_params()} parametrów")
        print(f"{'warstwa':38s} {'kszt. wyjścia':>16s} {'mln MAC':>9s} "
              f"{'udział':>7s}")
        print("-" * 74)
        for nm, macs, sh in rows:
            if macs == 0:
                continue
            print(f"{nm:38s} {str(sh):>16s} {macs/1e6:9.2f} "
                  f"{100.0*macs/total:6.1f}%")
        return 0

    # =====================================================================
    # PROCES NADRZĘDNY — czysty dyspozytor.
    # NIE importuje tu tensorflow ani kerasa. Gdy rodzic trzymał TF
    # w pamięci, proces potomny nie miał już miejsca na własny i kończył
    # się "OpenBLAS: Memory allocation failed". Na maszynie z 3 GB wolnej
    # pamięci mieści się dokładnie jeden TensorFlow naraz.
    # =====================================================================
    import subprocess

    def child(extra):
        return subprocess.run(
            [sys.executable, __file__, "--batch", str(args.batch),
             "--steps", str(args.steps)] + extra,
            capture_output=True, text=True, encoding="utf-8",
            errors="replace")

    print("=" * 74)
    print("PROFIL KOSZTU")
    print("=" * 74)
    print()

    r = child(["--breakdown"])
    print(r.stdout.rstrip() if r.returncode == 0 else
          f"rozkład kosztu nieudany (kod {r.returncode})")
    print("-" * 74)
    print("Najdroższe są warstwy o DUŻEJ MAPIE, nie o wielu kanałach —")
    print("dlatego obcinanie kanałów w głębi sieci prawie nic nie daje.")

    print(f"\n\nPomiar czasu kroku uczenia, batch {args.batch}, "
          f"{args.steps} kroków, CPU")
    print("=" * 74)
    print(f"{'wariant':16s} {'mln MAC':>8s} {'param':>9s} {'s/krok':>8s} "
          f"{'epoka':>8s} {'przyspiesz.':>11s}")
    print("-" * 74)

    # Każdy wariant mierzony jest w OSOBNYM PROCESIE. Zwalnianie sesji
    # w jednym procesie nie wystarcza: TensorFlow zostawia po sobie grafy
    # i bufory oneDNN, a przy 3 GB wolnej pamięci drugi wariant kończy się
    # "Arena alloc failed" wewnątrz protobufa. Osobny proces oddaje całą
    # pamięć systemowi.
    base = None
    for i, (name, *_rest) in enumerate(variants):
        r = child(["--variant", str(i)])
        line = next((l for l in r.stdout.splitlines()
                     if l.startswith("ROW ")), None)
        if line is None:
            err = (r.stderr or r.stdout).strip().splitlines()
            print(f"{name:16s} POMIAR NIEUDANY (kod {r.returncode}): "
                  f"{err[-1][:80] if err else 'brak wyjścia'}", flush=True)
            continue
        _, macs, params, dt, rf = line.split("\t")
        dt = float(dt)
        if base is None:
            base = dt
        epoch_s = dt * (args.n_train / args.batch)
        print(f"{name:16s} {float(macs)/1e6:8.1f} {int(params):9d} "
              f"{dt:8.3f} {epoch_s/60:7.1f}m "
              f"{base/dt:10.2f}x   zasięg {rf}f", flush=True)

    return _epilog(args, base or 0.0)

def _epilog(args, dt: float):
    """Podsumowanie: wymagany zasięg i czas epoki wobec rozmiaru zbioru.

    dt to zmierzony czas kroku dla OBECNEGO modelu — nie budujemy tu
    drugiego, bo w procesie nadrzędnym nie ma na to pamięci.
    """
    need_units = 19 + 2 * C.GAP_CHAR_UNITS              # znak '0' + przerwy
    need = need_units * C.dot_seconds(C.WPM) * C.frames_per_second()
    print("-" * 74)
    print(f"zasięg wymagany: {need:.0f} ramek = "
          f"{need/C.frames_per_second():.2f}s "
          f"(najdłuższy znak '0' z przerwami międzyznakowymi)")

    if dt <= 0:
        return 0

    # --- czas treningu w zależności od rozmiaru zbioru ---
    print(f"\n\nCzas EPOKI dla obecnego modelu ({dt:.3f} s/krok), "
          f"batch {args.batch}, CPU")
    print("=" * 74)
    print(f"{'próbek w zbiorze':>18s} {'kroków':>8s} {'epoka':>9s} "
          f"{'40 epok':>10s} {'100 epok':>10s}")
    print("-" * 74)
    for n in (10000, 30000, 60000, 200000):
        steps = n / args.batch
        e = dt * steps
        print(f"{n:>18d} {steps:8.0f} {e/60:8.1f}m "
              f"{e*40/3600:9.1f}h {e*100/3600:9.1f}h")
    print("-" * 74)
    print("TU JEST PRZYCZYNA 17 GODZIN: nie model, a rozmiar zbioru.")
    print("W README podałem 200 tys. próbek jako sensowne dla GPU —")
    print("na CPU to 12 minut na epokę i pierwszy wynik po nocy.")
    print("Na pierwszy przebieg wystarczy 30 tys. próbek i 40 epok.")

    print(f"\nZmiana liczby pasm zmienia ODCISK front-endu, więc zbiór")
    print("trzeba wygenerować od nowa — pilnuje tego "
          "config.check_fingerprint().")


if __name__ == "__main__":
    main()
