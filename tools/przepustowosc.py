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

Pomiar B omija potok: jedna gotowa partia float32 krąży w kółko. To jest
górna granica, jakiej ten model może dosięgnąć na tym sprzęcie — i jedyna
uczciwa odpowiedź na pytanie „ile potrafi karta".

B I C IDĄ PRZEZ model.fit(), Z ROZGRZEWKĄ. Do 08.10 B liczyła własna
pętla (GradientTape + apply_gradients), a C mierzyło całe fit() od
pierwszego kroku. Noc 06.10 dała przez to (out/noc_20261006_0004_*):

    B  2593 próbek/s  (98,7 ms/krok)   własna pętla, bez XLA
    C  1275 próbek/s (200,7 ms/krok)   z kompilacją pierwszego kroku
    trening tej samej nocy: 31 ms/krok, czyli ~8258 próbek/s

Oba pomiary były 3–6 razy poniżej tego, co karta naprawdę robiła, i werdykt
„karta jest wąskim gardłem" wyszedł dobry przypadkiem. Przyczyny:
  - Keras 3 kompiluje krok treningowy przez XLA (jit_compile="auto"),
    a własna pętla w tf.function — nie. To był pomiar INNEGO kodu.
  - pierwszy krok fit() to budowa grafu i kompilacja XLA, kilka sekund;
    rozłożone na 120 kroków podwaja czas kroku.
Teraz oba liczy ten sam fit() co noc, a czas biegnie od końca kroków
rozgrzewki do końca ostatniego (Stoper niżej).
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


def wypisz(nazwa: str, ile_krokow: int, na_krok: int, dt: float) -> float:
    szybkosc = ile_krokow * na_krok / dt
    print(f"  {nazwa:<34} {szybkosc:9.0f} próbek/s   "
          f"({dt / ile_krokow * 1000:.1f} ms/krok)")
    return szybkosc


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
    return wypisz(nazwa, ile_krokow, na_krok, time.perf_counter() - t0)


def zmierz_fit(nazwa: str, model, ds, ile_krokow: int, na_krok: int,
               rozgrzewka: int = 10) -> float:
    """Próbki/s dla model.fit() — tego samego, który liczy noc.

    Jedno fit() na rozgrzewka + ile_krokow kroków, a czas mierzy Stoper:
    od końca ostatniego kroku rozgrzewki do końca ostatniego kroku. Dwa
    osobne fit() (rozgrzewka, potem pomiar) byłyby prostsze, ale każde
    fit() zaczyna od nowego iteratora, a nowy iterator najpierw napełnia
    bufor shuffle (50 tys. próbek) — to też weszłoby do pomiaru.

    float(logs["loss"]) w Stoperze czeka na wynik kroku. Bez tego czas
    mierzyłby WYSŁANIE kroku na kartę, nie jego wykonanie. Noc i tak
    czeka po każdym kroku (pasek postępu, CSVLogger), więc to jest
    zgodne z tym, co dzieje się w nocy.
    """
    from tensorflow import keras

    class Stoper(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.t = {}

        def on_train_batch_end(self, krok, logs=None):
            if logs and "loss" in logs:
                float(logs["loss"])
            self.t[krok] = time.perf_counter()

    st = Stoper()
    model.fit(ds, epochs=1, steps_per_epoch=rozgrzewka + ile_krokow,
              verbose=0, callbacks=[st])
    koniec = rozgrzewka + ile_krokow - 1
    if koniec not in st.t or (rozgrzewka - 1) not in st.t:
        raise RuntimeError(f"fit() nie doszedł do kroku {koniec + 1} — "
                           f"za mało danych?")
    return wypisz(nazwa, ile_krokow, na_krok,
                  st.t[koniec] - st.t[rozgrzewka - 1])


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
    #  Jedna gotowa partia float32, ta sama w kółko — żadnego tasowania,
    #  rzutowania ani cięcia na partie. Zostaje tylko kopiowanie partii
    #  host->karta (4 MB na krok, ułamek milisekundy na PCIe), którego
    #  przez tf.data nie da się ominąć.
    print("B. SAM MODEL (jedna gotowa partia w kółko, przez model.fit)")
    # build_model sam woła compile() — nie robimy tego drugi raz, bo
    # powtórne compile kasuje stan optymalizatora.
    model = sa.build_model(arch=args.arch)
    xb = np.expand_dims(X[:args.batch].astype(np.float32) / 255.0, -1)
    yb = y[:args.batch]
    with tf.device("/cpu:0"):
        ds_b = tf.data.Dataset.from_tensors((xb, yb)).repeat()
    try:
        wyniki["model"] = zmierz_fit("model na karcie", model, ds_b,
                                     args.kroki, args.batch)
    except Exception as e:
        print(f"  nie udało się: {type(e).__name__}: {e}")
        wyniki["model"] = 0.0
    print()

    # --- C. jedno i drugie ------------------------------------------------
    #  Potok z wagami klas, jak w nocy (train_rtx.py, droga bez rotacji).
    #  Model ŚWIEŻY: ten z B jest już skompilowany i rozgrzany, więc C nie
    #  pokazałoby, czy rozgrzewka w ogóle coś odcina.
    print("C. TRENING (potok + model, czyli to, co widać w logu nocy)")
    model = sa.build_model(arch=args.arch)
    ds_c = sa.make_pipeline(X, y, None, args.batch, True,
                            weights=sa.class_weights(y))
    try:
        wyniki["trening"] = zmierz_fit("trening", model, ds_c.repeat(),
                                       args.kroki, args.batch)
    except Exception as e:
        print(f"  nie udało się: {type(e).__name__}: {e}")
        wyniki["trening"] = 0.0
    print()

    # --- werdykt ----------------------------------------------------------
    print("=" * 70)
    a, b, c = wyniki["potok"], wyniki["model"], wyniki["trening"]
    print(f" potok {a:.0f}/s   model {b:.0f}/s   trening {c:.0f}/s")
    print("=" * 70)

    if b <= 0 or c <= 0:
        print("Pomiar B albo C się nie udał — werdyktu nie wystawiam.")
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
