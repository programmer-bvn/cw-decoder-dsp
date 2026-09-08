"""PRZEGLĄDARKA X-RAY: pokazuje DOKŁADNIE to, co widzi sieć.

    python -m tools.xray --dataset out/morse_dataset.npz --random 12
    python -m tools.xray --dataset out/morse_dataset.npz --index 0 1 2 3
    python -m tools.xray --dataset out/morse_dataset.npz --class K --random 8
    python -m tools.xray --wav nagranie.wav
    python -m tools.xray --wav nagranie.wav --predict
    python -m tools.xray --text SOS --wpm 20        # podgląd syntezy bez zbioru

Obraz jest rysowany PRZEZ TEN SAM front-end, którym powstał zbiór — nie ma
osobnej ścieżki "do podglądu". Osie są opisane w jednostkach fizycznych
(sekundy, herce, decybele), bo bez tego nie da się stwierdzić, czy ton wypadł
w pasmie i czy timing jest zgodny z zadanym WPM.

Skala barw jest przypięta do [0, 1], czyli do [DB_MIN, DB_MAX] dB. Nie ma
automatycznego rozciągania kontrastu — gdyby było, sam szum wyglądałby jak
sygnał i podgląd kłamałby o tym, co dostaje sieć.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsp import config as C
from dsp import frontend, morse

import matplotlib
matplotlib.use("Agg")                       # zapis do pliku, bez okna
import matplotlib.pyplot as plt             # noqa: E402
import librosa                              # noqa: E402


# --------------------------------------------------------------------------
# Wczytywanie źródeł
# --------------------------------------------------------------------------
def load_dataset(path: Path):
    """Wczytuje zbiór i SPRAWDZA odcisk front-endu."""
    data = np.load(path, allow_pickle=False)
    if "fingerprint" in data:
        C.check_fingerprint(str(data["fingerprint"]), source=path.name)
    else:
        print(f"UWAGA: {path.name} nie ma odcisku front-endu — to stary "
              f"format, nie da się sprawdzić zgodności parametrów.")
    return data


def image_from_dataset(data, idx: int) -> np.ndarray:
    """Obraz ze zbioru, przeliczony z powrotem na [0, 1]."""
    img = np.asarray(data["X"][idx])
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    return np.squeeze(img)


# --------------------------------------------------------------------------
# Rysowanie
# --------------------------------------------------------------------------
def _band_centers() -> np.ndarray:
    """Częstotliwości środkowe pasm mel — do opisu osi pionowej."""
    return librosa.mel_frequencies(n_mels=C.N_MELS, fmin=C.FMIN,
                                   fmax=C.FMAX, htk=False)


def draw_panel(ax, img: np.ndarray, title: str = "",
               show_axes: bool = True, fontsize: int = 8) -> None:
    """Jeden obraz. Oś czasu w sekundach, oś pionowa w hercach."""
    seconds = C.IMG_FRAMES * C.HOP_LENGTH / C.SR
    im = ax.imshow(img.T, origin="lower", aspect="auto",
                   vmin=0.0, vmax=1.0, cmap="magma",
                   extent=[0.0, seconds, 0, C.N_MELS])

    if show_axes:
        centers = _band_centers()
        ticks = np.linspace(0, C.N_MELS - 1, 5).astype(int)
        ax.set_yticks(ticks + 0.5)
        ax.set_yticklabels([f"{centers[t]:.0f}" for t in ticks],
                           fontsize=fontsize - 1)
        ax.set_xlabel("czas [s]", fontsize=fontsize)
        ax.set_ylabel("Hz", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize - 1)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    if title:
        ax.set_title(title, fontsize=fontsize, pad=2)
    return im


# Dobór siatki: (górna granica liczby obrazów, kolumny, szerokość panelu
# w calach, czcionka tytułu). Przy dużej liczbie próbek panele muszą być
# małe, a osie i tytuły znikają — arkusz stykowy służy do oceny CAŁOŚCI
# na oko, nie do czytania pojedynczej próbki.
_GRID = (
    (1,    1,  6.0, 10),
    (4,    4,  3.4,  9),
    (16,   4,  2.6,  8),
    (40,   8,  1.9,  7),
    (80,  10,  1.5,  6),
    (200, 14,  1.15, 5),
    (10**6, 20, 0.9, 0),
)


def _grid_spec(n: int, cols_override: int | None = None):
    for limit, cols, size, fs in _GRID:
        if n <= limit:
            if cols_override:
                cols = cols_override
            return min(cols, n), size, fs
    raise AssertionError


def figure_grid(images, titles, out_path: Path, suptitle: str = "",
                cols_override: int | None = None) -> Path:
    """Arkusz stykowy: siatka obrazów + wspólna skala barw w decybelach.

    Liczba kolumn i rozmiar panelu dobierają się do liczby próbek, żeby
    64 albo 148 obrazów dało się obejrzeć jednym spojrzeniem. Przy małych
    panelach osie i tytuły są wyłączane — inaczej napisy zajmują więcej
    miejsca niż same obrazy.
    """
    n = len(images)
    cols, size, fs = _grid_spec(n, cols_override)
    rows = (n + cols - 1) // cols
    show_axes = size >= 2.6
    show_titles = fs > 0

    # Obraz jest 128x32, czyli 4:1 — panel niższy niż szeroki po transpozycji.
    fig, axes = plt.subplots(rows, cols,
                             figsize=(size * cols, size * 0.95 * rows),
                             squeeze=False, layout="constrained")

    im = None
    for k in range(rows * cols):
        ax = axes[k // cols][k % cols]
        if k < n:
            im = draw_panel(ax, images[k], titles[k] if show_titles else "",
                            show_axes=show_axes, fontsize=fs or 6)
        else:
            ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=10)

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7,
                            pad=0.02)
        cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels([f"{C.DB_MIN + t * C.db_span():.0f}"
                             for t in (0, 0.25, 0.5, 0.75, 1.0)])
        cbar.set_label("dB (ref=%.3g)" % C.DB_REF, fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# Statystyki — liczby, nie tylko obrazek
# --------------------------------------------------------------------------
def describe(img: np.ndarray, label: str = "") -> str:
    """Zwarty opis obrazu w decybelach. Sam obrazek nie wystarcza:
    "za jasne tło" trzeba umieć zmierzyć, a nie oceniać na oko."""
    db = frontend.denormalize_db(img)
    # Tło = dolny kwartyl, sygnał = górny percentyl. Dla klipu z sygnałem
    # różnica powinna wynosić 20-30 dB. Poniżej 10 dB znaczy, że sygnał
    # nie odróżnia się od szumu i próbka jest bezużyteczna.
    bg = float(np.percentile(db, 25))
    sig = float(np.percentile(db, 99.5))
    frac_clip_lo = float(np.mean(img <= 0.0))
    frac_clip_hi = float(np.mean(img >= 1.0))
    return (f"{label}tło={bg:6.1f} dB  szczyt={sig:6.1f} dB  "
            f"kontrast={sig - bg:5.1f} dB  "
            f"przy 0: {100*frac_clip_lo:4.1f}%  przy 1: {100*frac_clip_hi:4.1f}%")


# --------------------------------------------------------------------------
# Tryby pracy
# --------------------------------------------------------------------------
def mode_dataset(args):
    data = load_dataset(args.dataset)
    y = np.asarray(data["y"])
    n = len(y)
    print(f"Zbiór: {args.dataset.name}  próbek={n}")
    if "meta" in data:
        print(f"meta: {data['meta']}")

    pool = np.arange(n)
    if args.cls is not None:
        want = C.CHAR_TO_ID.get(args.cls.upper())
        if want is None:
            raise SystemExit(f"nieznany znak: {args.cls!r}")
        pool = pool[y[pool] == want]
        if pool.size == 0:
            raise SystemExit(f"brak próbek klasy {args.cls!r} w zbiorze")

    rng = np.random.default_rng(args.seed)

    if args.index:
        idx = [i for i in args.index if 0 <= i < n]
        mode = "wybrane numery"
    elif args.per_class:
        # Po K próbek z KAŻDEJ klasy, w kolejności alfabetu. Najlepszy widok
        # do oceny zbioru na oko: od razu widać, czy któryś znak wypada
        # inaczej niż pozostałe i czy klasa 0 rzeczywiście jest pusta.
        idx = []
        for ci in range(C.N_CLASSES):
            cand = np.flatnonzero(y == ci)
            if cand.size == 0:
                continue
            take = min(args.per_class, cand.size)
            idx.extend(rng.choice(cand, size=take, replace=False).tolist())
        mode = f"po {args.per_class} z każdej klasy"
    else:
        idx = rng.choice(pool, size=min(args.random, pool.size),
                         replace=False).tolist()
        mode = f"{len(idx)} losowych"

    images, titles = [], []
    stats = []
    for i in idx:
        img = image_from_dataset(data, i)
        ch = C.ID_TO_CHAR[int(y[i])]
        lab = "PUSTE" if int(y[i]) == 0 else f"'{ch}'"
        extra = ""
        if "tone" in data:
            t_hz = float(data["tone"][i])
            extra = (f" {t_hz:.0f}Hz" if np.isfinite(t_hz) else " ---")
            extra += f" a{float(data['amp'][i]):.2f}"
        # Znaczniki zakłóceń — po nich widać na oko, czy trudna próbka jest
        # trudna z powodu zaniku, obcej stacji czy trzasków.
        if "qsb" in data:
            marks = ""
            if float(data["qsb"][i]) > 0.3:
                marks += "~"                      # zanik
            if float(data["qrm"][i]) > 0:
                marks += "M"                      # inna stacja
            if float(data["qrn"][i]) > 0:
                marks += "N"                      # trzaski
            if float(data["fist"][i]) > 0.15:
                marks += "F"                      # rozjazd klucza
            if marks:
                extra += " " + marks
        full = str(data["text"][i]) if "text" in data else ""
        images.append(img)
        titles.append(f"#{i} {lab}{extra}")

        db = frontend.denormalize_db(img)
        stats.append((float(np.percentile(db, 25)),
                      float(np.percentile(db, 99.5))))
        if len(idx) <= 16:
            print(describe(img, label=f"#{i:<6} {lab:<8} " +
                           (f"nadano '{full}'  " if full else "")))

    # Przy dużej liczbie próbek pojedyncze wiersze są nieczytelne —
    # podajemy rozkład, bo o tym i tak się decyduje.
    if len(idx) > 16:
        bg = np.array([s[0] for s in stats])
        pk = np.array([s[1] for s in stats])
        con = pk - bg
        print(f"\nrozkład na {len(idx)} próbkach:")
        print(f"  tło:      {bg.min():7.1f} .. {bg.max():7.1f} dB  "
              f"(mediana {np.median(bg):.1f})")
        print(f"  szczyt:   {pk.min():7.1f} .. {pk.max():7.1f} dB  "
              f"(mediana {np.median(pk):.1f})")
        print(f"  kontrast: {con.min():7.1f} .. {con.max():7.1f} dB  "
              f"(mediana {np.median(con):.1f})")
        weak = int(np.sum(con < 15.0))
        if weak:
            print(f"  UWAGA: {weak} próbek ma kontrast poniżej 15 dB — "
                  f"na oko będą wyglądać jak sam szum")

    out = figure_grid(images, titles, args.out, cols_override=args.cols,
                      suptitle=f"X-RAY — {args.dataset.name} — {mode}   "
                               f"obraz {C.IMG_FRAMES}x{C.IMG_BINS}, "
                               f"skala {C.DB_MIN:.0f}..{C.DB_MAX:.0f} dB")
    print(f"\nZapisano: {out}")


def mode_wav(args):
    audio = frontend.load_audio(args.wav)
    dur = audio.size / C.SR
    print(f"Plik: {args.wav}  {dur:.2f}s  {C.SR} Hz  "
          f"RMS={np.sqrt(np.mean(audio**2)):.4f}  "
          f"szczyt={np.max(np.abs(audio)):.4f}")

    img = frontend.to_net_image(audio)
    print(describe(img, label="obraz: "))

    title = args.wav.name
    if args.predict:
        from dsp.model import load_model
        model = load_model(args.model)
        p = model.predict(img[np.newaxis, ..., np.newaxis], verbose=0)[0]
        top = np.argsort(p)[::-1][:5]
        print("\nPredykcja modelu:")
        for k in top:
            ch = "PUSTE" if k == 0 else C.ID_TO_CHAR[int(k)]
            print(f"  {ch:>6}  {p[k]*100:5.1f}%")
        title += f"   ->  '{C.ID_TO_CHAR[int(top[0])]}' ({p[top[0]]*100:.0f}%)"

    out = figure_grid([img], [title], args.out,
                      suptitle="X-RAY — plik wav")
    print(f"\nZapisano: {out}")


def mode_text(args):
    """Podgląd syntezy bez zbioru — sprawdzenie, czy generator daje obraz
    o właściwym timingu, zanim zmarnuje się godzinę na 40000 próbek."""
    rng = np.random.default_rng(args.seed)
    audio = rng.normal(0.0, C.NOISE_RMS, C.CLIP_SAMPLES).astype(np.float32)
    wave = morse.synth_cw(args.text, wpm=args.wpm, tone=C.TONE_CENTER,
                          amp=(C.SIGNAL_AMP_MIN + C.SIGNAL_AMP_MAX) / 2)
    if wave.size < audio.size:
        s = (audio.size - wave.size) // 2
        audio[s:s + wave.size] += wave
    else:
        s = (wave.size - audio.size) // 2
        audio += wave[s:s + audio.size]

    img = frontend.to_net_image(audio)
    code = morse.text_to_code(args.text)
    print(f"tekst='{args.text.upper()}'  kod={code}")
    print(f"WPM={args.wpm}  kropka={C.dot_seconds(args.wpm)*1000:.0f} ms  "
          f"nadanie={morse.total_units(args.text)} jednostek "
          f"= {morse.total_units(args.text)*C.dot_seconds(args.wpm):.2f} s")
    print(describe(img, label="obraz: "))

    out = figure_grid([img], [f"'{args.text.upper()}'  {args.wpm} WPM"],
                      args.out, suptitle="X-RAY — synteza")
    print(f"\nZapisano: {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="X-Ray: podgląd tego, co widzi sieć")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", type=Path, help="plik .npz ze zbiorem")
    src.add_argument("--wav", type=Path, help="plik audio")
    src.add_argument("--text", type=str, help="tekst do zsyntetyzowania")

    ap.add_argument("--index", type=int, nargs="+",
                    help="konkretne numery próbek ze zbioru")
    ap.add_argument("--random", type=int, default=48,
                    help="ile losowych próbek pokazać (domyślnie 48)")
    ap.add_argument("--per-class", type=int, metavar="K",
                    help="po K próbek z KAŻDEJ klasy, w kolejności alfabetu "
                         "— najlepszy widok do oceny zbioru na oko")
    ap.add_argument("--cols", type=int,
                    help="wymuś liczbę kolumn siatki")
    ap.add_argument("--class", dest="cls", type=str,
                    help="ogranicz do jednej klasy, np. --class K")
    ap.add_argument("--wpm", type=float, default=C.WPM)
    ap.add_argument("--predict", action="store_true",
                    help="dołóż predykcję modelu (tylko z --wav)")
    ap.add_argument("--model", type=Path, default=C.MODEL_PATH)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=C.OUT_DIR / "xray.png")
    args = ap.parse_args(argv)

    print("=" * 70)
    print("X-RAY")
    print("=" * 70)
    print(C.summary())
    print("-" * 70)

    if args.dataset:
        mode_dataset(args)
    elif args.wav:
        mode_wav(args)
    else:
        mode_text(args)


if __name__ == "__main__":
    main()
