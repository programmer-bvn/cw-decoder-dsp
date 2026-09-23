#!/usr/bin/env python3
"""Co naprawdę wyznacza tempo treningu: karta czy potok wejściowy.

    python tools/przepustowosc.py
    python tools/przepustowosc.py --dataset 'czesci/morse_200000_*.npz'

PO CO TO ISTNIEJE. Od początku projektu w rachunkach przewija się liczba
„karta konsumuje 8000 próbek/s". Wzięła się z podzielenia czasu epoki
przez liczbę próbek — czyli z pomiaru CAŁOŚCI. Nigdy nie sprawdzono,
która część całości ją wyznacza.

23.09 operator zauważył, że maszyna jedzie na 100% procesora, a karta
pokazuje 5%. Jeśli to było w trakcie treningu, tamta liczba nie mówi nic
o karcie i wszystkie oparte na niej wnioski trzeba przeliczyć: „generowanie
w locie zagłodziłoby GPU 34-krotnie" brzmi inaczej, gdy GPU i tak stoi.

JAK TO ROZSTRZYGNĄĆ. Trzema pomiarami zamiast jednym:

    A. sam potok, bez modelu     — ile próbek na sekundę dowozi wejście
    B. sam model, bez potoku     — jedna partia w pamięci karty, w kółko
    C. trening, czyli oba naraz  — to, co widać w logu nocy

Wnioskowanie jest wtedy jednoznaczne:

    C ~ A  i  A << B   ->  wąskim gardłem jest POTOK. Karta czeka.
    C ~ B  i  B << A   ->  wąskim gardłem jest KARTA. Tak ma być.
    C << A i C << B    ->  traci się na styku (np. kopiowanie host->karta)

Pomiar B celowo omija tf.data: partia leży już na karcie i jest liczona
w kółko. To jest górna granica, jakiej ten model może dosięgnąć na tym
sprzęcie — i jedyna uczciwa odpowiedź na pytanie „ile potrafi karta".
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np

KORZEN = Path(__file__).resolve().parent.parent


def wczytaj_standalone():
    """train_rtx.py jest samodzielny — wczytujemy go po ścieżce."""
    p = KORZEN / "train_rtx.py"
    if not p.exists():
        raise SystemExit(f"nie ma {p}")
    spec = importlib.util.spec_from_file_location("_sa", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def zmierz(nazwa: str, krok, ile_krokow: int, na_krok: int,
           rozgrzewka: int = 10) -> float:
    """Zwraca próbki/s. Pierwsze kroki odrzucane.

    Rozgrzewka NIE jest kosmetyką: pierwsze wywołanie buduje graf,
    alokuje bufory i dobiera algorytmy splotu. Wliczone do pomiaru
    potrafi zaniżyć wynik kilkukrotnie i zrobić z szybkiego kodu wolny.
    """
    for _ in range(rozgrzewka):
        krok()
    t0 = time.perf_counter()
    for _ in range(ile_krokow):
        krok()
    dt = time.perf_counter() - t0
    szybkosc = ile_krokow * na_krok / dt
    print(f"  {nazwa:<34} {szybkosc:9.0f} próbek/s   "
          f"({dt / ile_krokow * 1000:.1f} ms/krok)")
    return szybkosc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="",
                    help="wzorzec zbioru; bez tego dane losowe (tempo "
                         "potoku i tak od treści nie zależy)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--kroki", type=int, default=120)
    ap.add_argument("--arch", default="dpu")
    ap.add_argument("--mixed", action="store_true", default=True)
    args = ap.parse_args(argv)

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    sa = wczytaj_standalone()
    import tensorflow as tf
    from tensorflow import keras

    karty = tf.config.list_physical_devices("GPU")
    print("=" * 70)
    print(" PRZEPUSTOWOŚĆ: co wyznacza tempo treningu")
    print("=" * 70)
    print(f"karta: {karty[0].name if karty else 'BRAK — pomiar bez sensu'}")
    if not karty:
        print("Bez karty ten pomiar nie odpowiada na pytanie, o które chodzi.")
        return 1
    if args.mixed:
        keras.mixed_precision.set_global_policy("mixed_float16")
    print(f"partia: {args.batch}, kroków na pomiar: {args.kroki}")
    print()

    # --- dane ---
    if args.dataset:
        X, y, _yf, opis = sa.load_dataset(args.dataset)
        print(f"zbiór: {X.shape} ({X.nbytes/1024/1024:.0f} MB)")
    else:
        # Tempo potoku zależy od ROZMIARU i TYPU, nie od treści. Losowe
        # dane dają ten sam wynik, a nie wymagają 800 MB na dysku.
        n = max(args.batch * (args.kroki + 40), 20000)
        rng = np.random.default_rng(0)
        X = rng.integers(0, 256, (n, sa.IMG_FRAMES, sa.IMG_BINS),
                         dtype=np.uint8)
        y = rng.integers(0, sa.N_CLASSES, n).astype(np.int32)
        print(f"dane losowe: {X.shape} ({X.nbytes/1024/1024:.0f} MB)")
    print()

    wyniki = {}

    # --- A. sam potok -----------------------------------------------------
    print("A. SAM POTOK (bez modelu — ile wejście dowozi)")
    ds = sa.make_pipeline(X, y, None, args.batch, True)
    it = iter(ds)

    def krok_potok():
        next(it)

    try:
        wyniki["potok"] = zmierz("potok wejściowy", krok_potok,
                                 args.kroki, args.batch)
    except StopIteration:
        print("  za mało danych na tyle kroków — zmniejsz --kroki")
        return 1
    print()

    # --- B. sam model -----------------------------------------------------
    #  Partia LEŻY JUŻ NA KARCIE i jest liczona w kółko. Żadnego tf.data,
    #  żadnego kopiowania z hosta. To górna granica tego modelu na tym
    #  sprzęcie.
    print("B. SAM MODEL (jedna partia w pamięci karty, w kółko)")
    # build_model sam woła compile() — nie robimy tego drugi raz, bo
    # powtórne compile kasuje stan optymalizatora.
    model = sa.build_model(arch=args.arch)
    with tf.device("/GPU:0"):
        xb = tf.constant(
            np.expand_dims(X[:args.batch].astype(np.float32) / 255.0, -1))
        yb = tf.constant(y[:args.batch])

    # Strata liczona wprost, a nie przez model.compiled_loss: tamten
    # atrybut zniknął w Kerasie 3, a tutaj chodzi o pomiar tempa, nie
    # o wierne odtworzenie pętli Kerasa.
    @tf.function
    def krok_modelu():
        with tf.GradientTape() as tape:
            strata = tf.reduce_mean(
                keras.losses.sparse_categorical_crossentropy(
                    yb, model(xb, training=True)))
        grad = tape.gradient(strata, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grad, model.trainable_variables))

    try:
        wyniki["model"] = zmierz("model na karcie", krok_modelu,
                                 args.kroki, args.batch)
    except Exception as e:
        print(f"  nie udało się: {type(e).__name__}: {e}")
        wyniki["model"] = 0.0
    print()

    # --- C. jedno i drugie ------------------------------------------------
    print("C. TRENING (potok + model, czyli to, co widać w logu nocy)")
    ds2 = sa.make_pipeline(X, y, None, args.batch, True)
    t0 = time.perf_counter()
    model.fit(ds2, epochs=1, steps_per_epoch=args.kroki, verbose=0)
    dt = time.perf_counter() - t0
    wyniki["trening"] = args.kroki * args.batch / dt
    print(f"  {'trening':<34} {wyniki['trening']:9.0f} próbek/s   "
          f"({dt / args.kroki * 1000:.1f} ms/krok)")
    print()

    # --- werdykt ----------------------------------------------------------
    print("=" * 70)
    a, b, c = wyniki["potok"], wyniki["model"], wyniki["trening"]
    print(f" potok {a:.0f}/s   model {b:.0f}/s   trening {c:.0f}/s")
    print("=" * 70)

    if b <= 0:
        print("Pomiar modelu się nie udał — werdyktu nie wystawiam.")
        return 1

    # Marginesy celowo szerokie: chodzi o rozstrzygnięcie "co rządzi",
    # a nie o procenty. Przy wyniku niejednoznacznym lepiej to powiedzieć,
    # niż wskazać winnego na siłę.
    if a < 0.7 * b:
        print("WĄSKIM GARDŁEM JEST POTOK WEJŚCIOWY.")
        print(f"  Karta potrafi {b:.0f} próbek/s, a wejście dowozi {a:.0f}.")
        print(f"  Karta stoi bezczynnie przez około "
              f"{100 * (1 - a / b):.0f}% czasu.")
        print("  Tu warto szukać: kolejność .batch()/.map(), liczba wątków")
        print("  tf.data, kopiowanie host->karta, wielkość partii.")
    elif b < 0.7 * a:
        print("WĄSKIM GARDŁEM JEST KARTA — tak ma być.")
        print(f"  Wejście dowozi {a:.0f} próbek/s, karta bierze {b:.0f}.")
        print("  Szybszy potok niczego nie zmieni. Żeby przyspieszyć,")
        print("  trzeba mniejszego modelu albo większej partii.")
    else:
        print("POTOK I KARTA SĄ W PARZE — żadne nie rządzi wyraźnie.")
        print(f"  {a:.0f} vs {b:.0f} próbek/s.")

    if c < 0.7 * min(a, b):
        print()
        print(f"  UWAGA: sam trening ({c:.0f}/s) jest wolniejszy niż")
        print(f"  wolniejszy ze składników ({min(a, b):.0f}/s). Traci się")
        print("  na styku — najczęściej na kopiowaniu partii na kartę")
        print("  albo na braku prefetch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
