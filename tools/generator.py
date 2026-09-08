"""GENERATOR TRENINGOWY: syntetyczne nadania CW -> plik .npz.

    python -m tools.generator                    # 40000 próbek, domyślne parametry
    python -m tools.generator --n 5000           # mniejszy zbiór na próbę
    python -m tools.generator --wpm 25 --jitter 3
    python -m tools.generator --out out/inny.npz

Zapisuje: X (obrazy), y (etykiety), fingerprint (odcisk front-endu),
meta (parametry wywołania). Odcisk jest po to, żeby trener i podgląd mogły
sprawdzić, czy zbiór powstał na tych samych parametrach, na jakich pracują.

Etykietą jest ŚRODKOWY znak z CHARS_PER_CLIP nadanych — nadanie jest
centrowane w klipie, a okno IMG_FRAMES wycinane ze środka, więc znak
docelowy jest w polu widzenia sieci wraz z sąsiadami. Sąsiedzi są celowo:
bez nich model nie widziałby nigdy przerwy międzyznakowej i na żywym
strumieniu nie umiałby rozdzielić znaków.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsp import config as C
from dsp import frontend, radio


def make_clip(rng: np.random.Generator, wpm: float, jitter: float,
              realism: bool = True) -> tuple[np.ndarray, int, dict]:
    """Jeden klip audio + etykieta + opis parametrów, na jakich powstał.

    Cała fizyka jest w dsp/radio.py — tutaj zostaje tylko decyzja, CO
    zostanie nadane i jaka jest etykieta.

    Opis wraca na zewnątrz i idzie do pliku zbioru, bo przy analizie pomyłek
    modelu trzeba wiedzieć, że dana próbka miała zanik 0,55 i drugą stację
    40 Hz od tonu. Przy pobranym korpusie tej informacji nie ma i zostaje
    zgadywanie.
    """
    is_signal = rng.random() > C.SILENCE_FRACTION

    if is_signal:
        chars = [str(rng.choice(list(C.ALPHABET[1:])))
                 for _ in range(C.CHARS_PER_CLIP)]
        text = "".join(chars)                 # bez spacji -> odstęp 3 jednostki
        target = C.CHAR_TO_ID[chars[C.LABEL_INDEX]]
    else:
        # Klasa 0 = "puste radio": tło i zakłócenia, ale bez stacji docelowej.
        # Uwaga: przy włączonym realizmie taki klip MOŻE zawierać QRM, czyli
        # obcą stację. To zamierzone — model ma się uczyć, że sygnał poza
        # spodziewanym tonem nie jest znakiem do odczytu.
        text, target = "", 0

    audio, meta = radio.receive(text, rng, n_samples=C.CLIP_SAMPLES,
                                sr=C.SR, wpm=wpm, realism=realism)
    meta["signal"] = is_signal
    return audio, target, meta


# Parametry kanału zapisywane obok każdej próbki. Bez nich nie da się
# odpowiedzieć na pytanie "dlaczego model pomylił się na próbce 17432".
_META_FIELDS = ("tone", "amp", "wpm", "fist", "drift", "qsb",
                "qrm", "qrn", "noise_rms", "noise_tilt", "peak",
                # Wady nadajnika i odbiornika — bez nich nie da się
                # odpowiedzieć, czy model przewraca się na chirpie,
                # na przydźwięku, czy na ARW.
                "fist_drift", "gap_jitter", "chirp", "sag", "hum",
                "agc_tau",
                # Położenie znaku Z ETYKIETY, w ramkach okna sieci.
                # Dzięki temu X-Ray potrafi go WSKAZAĆ na kafelku, a
                # telegrafista sprawdza etykietę, zamiast zgadywać, który
                # z trzech nadanych znaków jest tym opisanym. Liczone
                # z faktycznych granic po rozjeździe klucza.
                "lab_x0", "lab_x1")


def generate(n: int, seed: int, wpm: float, jitter: float,
             out_path: Path, realism: bool = True, verbose: bool = True):
    rng = np.random.default_rng(seed)
    store_u8 = C.STORE_DTYPE == "uint8"

    X = np.empty((n, C.IMG_FRAMES, C.IMG_BINS),
                 dtype=np.uint8 if store_u8 else np.float32)
    y = np.empty(n, dtype=np.int16)
    cols = {k: np.empty(n, dtype=np.float32) for k in _META_FIELDS}
    texts: list[str] = []
    n_clipped = 0

    t0 = time.time()
    for i in range(n):
        audio, target, meta = make_clip(rng, wpm, jitter, realism=realism)
        img = frontend.to_net_image(audio)          # JEDNA ścieżka front-endu

        X[i] = np.rint(img * 255.0).astype(np.uint8) if store_u8 else img
        y[i] = target

        # Próbki audio -> ramki okna. Przeliczenie siedzi we frontend, bo
        # zależy od dopełniania w librosie i od wycinania okna — dwóch
        # rzeczy, które łatwo policzyć o jedną ramkę za dużo.
        for src, dst in (("lab_a", "lab_x0"), ("lab_b", "lab_x1")):
            v = meta.get(src, np.nan)
            meta[dst] = (np.nan if not np.isfinite(v)
                         else frontend.sample_to_window_frame(v))

        for k in _META_FIELDS:
            cols[k][i] = meta.get(k, np.nan)
        texts.append(meta["text"])
        n_clipped += bool(meta.get("clipped"))

        if verbose and (i % 2000 == 0 or i == n - 1):
            done = i + 1
            el = time.time() - t0
            eta = el / done * (n - done)
            print(f"  {done}/{n}  {el:.0f}s  pozostało ~{eta:.0f}s",
                  flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             X=X, y=y,
             text=np.array(texts),
             fingerprint=C.fingerprint_str(),
             dtype_note=C.STORE_DTYPE,
             meta=(f"n={n};seed={seed};wpm={wpm};jitter={jitter};"
                   f"realism={int(realism)}"),
             **cols)

    mb = out_path.stat().st_size / 1024 / 1024
    counts = np.bincount(y, minlength=C.N_CLASSES)
    print(f"\nZapisano: {out_path}  ({mb:.1f} MB, {C.STORE_DTYPE})")
    print(f"Klasa 0 (puste radio): {counts[0]} próbek "
          f"({100.0*counts[0]/n:.1f}%)")
    print(f"Pozostałe klasy: min {counts[1:].min()}, "
          f"max {counts[1:].max()} próbek na znak")

    if realism:
        sig = np.asarray(y) != 0
        print(f"\nKanał radiowy (udział klipów):")
        print(f"  QRM (inna stacja):  "
              f"{100.0*np.mean(cols['qrm'] > 0):.0f}%")
        print(f"  QRN (trzaski):      "
              f"{100.0*np.mean(cols['qrn'] > 0):.0f}%")
        print(f"  zanik QSB > 0.3:    "
              f"{100.0*np.mean(cols['qsb'] > 0.3):.0f}%")
        print(f"  rozjazd klucza:     mediana "
              f"{np.nanmedian(cols['fist'][sig])*100:.0f}%")
        print(f"  dryf tonu:          mediana "
              f"{np.nanmedian(np.abs(cols['drift'][sig])):.1f} Hz")
        if n_clipped:
            print(f"  obcięte w torze:    {n_clipped} klipów "
                  f"({100.0*n_clipped/n:.1f}%)")
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Generator zbioru treningowego CW")
    ap.add_argument("--n", type=int, default=C.N_SAMPLES_DATASET,
                    help=f"liczba próbek (domyślnie {C.N_SAMPLES_DATASET})")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--wpm", type=float, default=C.WPM)
    ap.add_argument("--jitter", type=float, default=C.WPM_JITTER,
                    help="rozrzut WPM w zbiorze (uczy odporności na "
                         "ręczny klucz)")
    ap.add_argument("--out", type=Path, default=C.DATASET_PATH)
    ap.add_argument("--no-realism", action="store_true",
                    help="wyłącz model kanału (QSB/QRM/QRN/dryf/fist) — "
                         "daje czysty sygnał jak w v5.6")
    args = ap.parse_args(argv)

    realism = not args.no_realism
    print("=" * 70)
    print("GENERATOR TRENINGOWY CW")
    print("=" * 70)
    print(C.summary())
    print(f"próbek={args.n}  seed={args.seed}  "
          f"WPM={args.wpm}+/-{args.jitter}  "
          f"kanał radiowy={'TAK' if realism else 'NIE (czysty sygnał)'}")
    print("-" * 70)

    generate(args.n, args.seed, args.wpm, args.jitter, args.out,
             realism=realism)


if __name__ == "__main__":
    main()
