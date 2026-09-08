"""X-RAY CAŁOŚCI: przegląd WSZYSTKICH danych treningowych, w jednolitej skali.

    python -m tools.xray_all --dataset out/ds30k.npz
    python -m tools.xray_all --dataset out/ds30k.npz --sort contrast
    python -m tools.xray_all --dataset out/ds30k.npz --pages       # w kolejności
    python -m tools.xray_all --wav-dir nagrania/                   # pliki wav
    python -m tools.xray_all --dataset out/ds30k.npz --wav-dir nagrania/

PO CO TO ISTNIEJE
-----------------
Odniesieniem w tym projekcie jest telegrafista. Jeśli operator nie potrafi
odczytać z obrazu, co sieć dostaje na wejściu, to nikt tego nie odczyta —
a wtedy nie da się stwierdzić, czy model myli się z powodu złych danych,
czy złej architektury. Podgląd ośmiu losowych próbek na to nie odpowiada.

Ten plik rysuje KOMPLET danych: każdą próbkę zbioru, w rozdzielczości 1:1
(jeden piksel obrazu = jedna ramka x jedno pasmo, bez interpolacji i bez
skalowania), pogrupowaną po klasach, na JEDNEJ skali decybelowej wspólnej
dla wszystkich źródeł — zbioru syntetycznego, plików wav i nagrań
z mikrofonu. Ta sama skala jest warunkiem porównywalności: obraz z pasma
i obraz z generatora muszą być rysowane tak samo, inaczej porównanie
"czy syntetyk wygląda jak radio" nie ma sensu.

CO WIDAĆ NA KAFELKU
-------------------
    oś pozioma   czas, 128 ramek = 2,56 s (1 piksel = 20 ms)
    oś pionowa   32 pasma mel od 400 do 1200 Hz (u dołu niskie)
    jasność      moc, liniowo od DB_MIN (czarne) do DB_MAX (białe)

Pod każdym kafelkiem jest paseczek 3 px ze znacznikami zakłóceń:
    niebieski   QSB, zanik głębszy niż 0,3
    czerwony    QRM, inna stacja w pasmie
    żółty       QRN, trzaski atmosferyczne
    zielony     rozjazd ręcznego klucza powyżej 15%
Czarny paseczek = próbka czysta.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsp import config as C
from dsp import frontend

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib import cm                           # noqa: E402


# Paski pod kafelkiem: wskaźnik znaku z etykiety, potem znaczniki zakłóceń.
LAB_H = 3             # biały odcinek = granice znaku opisanego etykietą
FLAG_H = 3            # kolory zakłóceń
GAP = 2               # odstęp między kafelkami

# Kolory znaczników. Dobrane tak, żeby były rozpoznawalne obok mapy magma,
# która jest czarno-czerwono-żółta — dlatego QSB jest niebieski, a nie
# czerwony, choć zanik "intuicyjnie" pasowałby do ciepłej barwy.
FLAG_COLORS = {
    "qsb": (0.20, 0.45, 1.00),
    "qrm": (1.00, 0.15, 0.15),
    "qrn": (1.00, 0.85, 0.10),
    "fist": (0.20, 0.85, 0.35),
}


# --------------------------------------------------------------------------
# Źródła danych — wszystkie sprowadzone do wspólnej postaci
# --------------------------------------------------------------------------
def load_npz(path: Path):
    """Zbiór syntetyczny. Zwraca (obrazy 0-1, etykiety, opisy zakłóceń)."""
    data = np.load(path, allow_pickle=False)
    if "fingerprint" in data:
        C.check_fingerprint(str(data["fingerprint"]), source=path.name)
    X = np.asarray(data["X"])
    if X.dtype == np.uint8:
        X = X.astype(np.float32) / 255.0
    X = np.squeeze(X)
    y = np.asarray(data["y"]).astype(int)

    flags = {}
    for k in ("qsb", "qrm", "qrn", "fist", "amp", "tone", "lab_x0", "lab_x1"):
        if k in data:
            flags[k] = np.asarray(data[k], dtype=np.float32)
    return X, y, flags, str(data["meta"]) if "meta" in data else ""


def load_wav_dir(path: Path, window_stride_s: float = 2.56):
    """Katalog z plikami wav -> okna obrazu, tą samą ścieżką front-endu.

    Etykiety nie ma (to nagrania z pasma), więc klasa jest -1 i takie próbki
    idą na osobną stronę. Skala jest ta sama, co dla zbioru syntetycznego —
    o to w tym narzędziu chodzi.
    """
    files = sorted([p for p in path.iterdir()
                    if p.suffix.lower() in (".wav", ".flac", ".ogg")])
    if not files:
        return np.zeros((0, C.IMG_FRAMES, C.IMG_BINS), np.float32), \
            np.zeros(0, int), {}, []

    imgs, names = [], []
    stride = max(1, int(round(window_stride_s * C.frames_per_second())))
    for f in files:
        audio = frontend.load_audio(f)
        full = frontend.normalize_db(
            frontend.power_to_db(frontend.melspec_power(audio)))
        if full.shape[0] < C.IMG_FRAMES:
            imgs.append(frontend.center_window(full))
            names.append(f"{f.name}@0.0s")
            continue
        for s in range(0, full.shape[0] - C.IMG_FRAMES + 1, stride):
            imgs.append(full[s:s + C.IMG_FRAMES])
            names.append(f"{f.name}@{s*C.HOP_LENGTH/C.SR:.1f}s")

    X = np.stack(imgs).astype(np.float32)
    return X, np.full(len(imgs), -1, dtype=int), {}, names


# --------------------------------------------------------------------------
# Składanie mozaiki
# --------------------------------------------------------------------------
def _label_strip(width: int, flags: dict, i: int) -> np.ndarray:
    """Wskaźnik znaku Z ETYKIETY: RGB [LAB_H, width, 3].

    Biały odcinek pokazuje DOKŁADNE granice tego znaku, który opisuje
    etykieta. Bez tego kafelek pokazuje trzy nadane znaki i nie da się
    stwierdzić, który z nich jest opisany — a przy rozjeździe ręcznego
    klucza środek okna wypada nawet o pół znaku obok właściwego.

    Ten pasek jest też kontrolą samego generatora: jeśli odcinek nie
    pokrywa się z grupą elementów widoczną na kafelku, to timing
    w generatorze jest policzony błędnie.
    """
    strip = np.zeros((LAB_H, width, 3), dtype=np.float32)
    x0 = flags.get("lab_x0")
    x1 = flags.get("lab_x1")
    if x0 is None or x1 is None:
        return strip
    a, b = float(x0[i]), float(x1[i])
    if not (np.isfinite(a) and np.isfinite(b)):
        return strip

    ia = int(np.clip(np.floor(a), 0, width - 1))
    ib = int(np.clip(np.ceil(b), 1, width))
    if ib <= ia:
        return strip
    strip[:, ia:ib] = (0.95, 0.95, 0.95)

    # Czerwone kreski na krawędziach, gdy znak wychodzi za kadr — wtedy
    # etykieta opisuje coś, czego sieć nie widzi w całości.
    if a < 0:
        strip[:, :2] = (1.0, 0.1, 0.1)
    if b > width:
        strip[:, -2:] = (1.0, 0.1, 0.1)
    return strip


def _flag_strip(width: int, flags: dict, i: int) -> np.ndarray:
    """Paseczek znaczników pod kafelkiem: RGB [FLAG_H, width, 3]."""
    strip = np.zeros((FLAG_H, width, 3), dtype=np.float32)
    active = []
    if flags.get("qsb") is not None and flags["qsb"][i] > 0.3:
        active.append("qsb")
    if flags.get("qrm") is not None and flags["qrm"][i] > 0:
        active.append("qrm")
    if flags.get("qrn") is not None and flags["qrn"][i] > 0:
        active.append("qrn")
    if flags.get("fist") is not None and flags["fist"][i] > 0.15:
        active.append("fist")
    if not active:
        return strip
    # Znaczniki dzielą szerokość kafelka na równe odcinki — po długości
    # kolorowego pola widać, ile zakłóceń naraz trafiło w próbkę.
    seg = width // len(active)
    for k, name in enumerate(active):
        a = k * seg
        b = width if k == len(active) - 1 else (k + 1) * seg
        strip[:, a:b] = FLAG_COLORS[name]
    return strip


def build_mosaic(X: np.ndarray, idx: np.ndarray, flags: dict,
                 cols: int, zoom: int = 1,
                 cmap_name: str = "magma") -> np.ndarray:
    """Mozaika RGB z kafelków.

    Kafelek jest transponowany (czas w poziomie, częstotliwość w pionie),
    bo tak wygląda wodospad na odbiorniku i tak operator czyta CW.

    zoom to POWIELANIE PIKSELI całkowitą krotnością (np.repeat), nie
    skalowanie. Żaden piksel nie jest interpolowany ani uśredniony, więc
    powiększony obraz pokazuje dokładnie te same liczby, jakie dostaje
    sieć — tylko większe. Przy zoom=1 kropka przy 20 WPM ma 3 piksele
    (60 ms / 20 ms na ramkę), co jest na granicy czytelności; zoom=3 daje
    9 pikseli i da się liczyć elementy wzrokiem.

    Składanie idzie wprost w tablicy numpy, nie przez matplotlib subplots —
    30 tysięcy osobnych osi zajęłoby kilkanaście minut i kilka gigabajtów.
    """
    cmap = (matplotlib.colormaps[cmap_name]
            if hasattr(matplotlib, "colormaps") else cm.get_cmap(cmap_name))

    z = max(1, int(zoom))
    tile_w, tile_h = C.IMG_FRAMES * z, C.IMG_BINS * z
    lab_h, flag_h, gap = LAB_H * z, FLAG_H * z, GAP * z
    cell_w = tile_w + gap
    cell_h = tile_h + lab_h + flag_h + gap
    rows = (len(idx) + cols - 1) // cols

    # uint8, nie float32: mozaika 5200 x 3000 px w RGB to 47 MB w uint8,
    # a 187 MB w float32. Przy komplecie 30 tysięcy próbek i kilku stronach
    # naraz ta różnica decyduje, czy render przechodzi na 12 GB pamięci.
    mosaic = np.full((rows * cell_h + gap, cols * cell_w + gap, 3),
                     26, dtype=np.uint8)                # tło mozaiki
    if len(idx) == 0:
        return mosaic

    def _up(a):
        """Powielenie pikseli; wynik uint8 bez zaokrągleń w dół po drodze."""
        a8 = np.rint(np.asarray(a) * 255.0).astype(np.uint8)
        return np.repeat(np.repeat(a8, z, axis=0), z, axis=1) if z > 1 else a8

    for k, i in enumerate(idx):
        r, c = divmod(k, cols)
        y0, x0 = gap + r * cell_h, gap + c * cell_w
        # imshow rysuje wiersz 0 u góry, a chcemy niskie pasma u dołu -> [::-1]
        mosaic[y0:y0 + tile_h, x0:x0 + tile_w] = \
            _up(cmap(np.clip(X[i].T[::-1], 0.0, 1.0))[..., :3])

        if flags:
            yl = y0 + tile_h
            mosaic[yl:yl + lab_h, x0:x0 + tile_w] = \
                _up(_label_strip(C.IMG_FRAMES, flags, int(i)))
            yf = yl + lab_h
            mosaic[yf:yf + flag_h, x0:x0 + tile_w] = \
                _up(_flag_strip(C.IMG_FRAMES, flags, int(i)))
    return mosaic


def _text_banner(width: int, title: str, subtitle: str = "",
                 height: int = 44) -> np.ndarray:
    """Pasek z podpisem jako tablica uint8 RGB o zadanej szerokości.

    Tekst rysuje matplotlib, ale TYLKO na tym pasku (kilka tysięcy na
    czterdzieści pikseli), nie na całej stronie. To rozwiązuje problem
    pamięci: matplotlib przy rysowaniu obrazu uint8 przelicza go na
    float32, więc strona 4164 x 22484 px kosztowałaby 1 GB tylko na tę
    jedną konwersję. Mozaika idzie do pliku osobno, wprost z uint8.
    """
    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.patch.set_facecolor("white")
    fig.text(0.004, 0.95, title, fontsize=11, color="black", va="top",
             family="monospace")
    if subtitle:
        fig.text(0.004, 0.42, subtitle, fontsize=8, color="#444444",
                 va="top", family="monospace")
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)

    if buf.shape[1] != width:                 # zabezpieczenie na zaokrąglenia
        buf = buf[:, :width] if buf.shape[1] > width else np.pad(
            buf, ((0, 0), (0, width - buf.shape[1]), (0, 0)),
            constant_values=255)
    return buf


def save_page(mosaic: np.ndarray, out_path: Path, title: str,
              subtitle: str = "") -> Path:
    """Zapis strony 1:1 — jeden piksel tablicy = jeden piksel pliku.

    Zapis idzie przez imsave, które dla tablicy uint8 oddaje ją wprost
    koderowi PNG. Bez tego strona z 4500 kafelkami się nie zapisuje.
    """
    banner = _text_banner(mosaic.shape[1], title, subtitle)
    page = np.vstack([banner, mosaic])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    matplotlib.image.imsave(str(out_path), page)
    return out_path


def save_legend(out_path: Path) -> Path:
    """Legenda: skala decybelowa i znaczniki zakłóceń. Jedna dla wszystkich
    stron, bo skala jest wspólna — powtarzanie jej na każdej stronie
    zabierałoby miejsce kafelkom."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 2.6),
                                   layout="constrained")

    grad = np.linspace(0, 1, 256)[None, :]
    ax1.imshow(grad, aspect="auto", cmap="magma", vmin=0, vmax=1,
               extent=[C.DB_MIN, C.DB_MAX, 0, 1])
    ax1.set_yticks([])
    ax1.set_xlabel(f"moc [dB], odniesienie ref={C.DB_REF:g}  —  "
                   f"skala WSPÓLNA dla wszystkich źródeł danych", fontsize=9)
    ax1.set_title(f"Skala jasności: {C.DB_MIN:.0f} dB (czarne) "
                  f"..  {C.DB_MAX:.0f} dB (białe), rozpiętość "
                  f"{C.db_span():.0f} dB", fontsize=10)

    ax2.set_xlim(0, 4); ax2.set_ylim(0, 1); ax2.axis("off")
    ax2.set_title("Znaczniki pod kafelkiem", fontsize=10)
    names = {"qsb": "QSB — zanik > 0,3", "qrm": "QRM — inna stacja",
             "qrn": "QRN — trzaski", "fist": "rozjazd klucza > 15%"}
    for k, (key, label) in enumerate(names.items()):
        ax2.add_patch(plt.Rectangle((k + 0.05, 0.45), 0.3, 0.25,
                                    color=FLAG_COLORS[key]))
        ax2.text(k + 0.05, 0.2, label, fontsize=8, family="monospace")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, facecolor="white")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# Porządkowanie próbek
# --------------------------------------------------------------------------
def sort_indices(idx: np.ndarray, X: np.ndarray, flags: dict,
                 how: str) -> np.ndarray:
    """Kolejność kafelków na stronie.

    contrast — od najsłabszych. Najtrudniejsze próbki zbierają się w lewym
    górnym narożniku, więc od nich zaczyna się przegląd. To ważniejsze niż
    wygląda: jeśli w zbiorze są próbki nieczytelne dla operatora, to model
    ma się uczyć na etykietach, których nie da się potwierdzić.
    """
    if how == "index":
        return idx
    if how == "amp" and flags.get("amp") is not None:
        return idx[np.argsort(flags["amp"][idx])]
    # contrast: różnica między szczytem a tłem, licząc w decybelach
    db = frontend.denormalize_db(X[idx])
    con = (np.percentile(db, 99.5, axis=(1, 2))
           - np.percentile(db, 25, axis=(1, 2)))
    return idx[np.argsort(con)]


# --------------------------------------------------------------------------
# Tryby
# --------------------------------------------------------------------------
def render_by_class(X, y, flags, out_dir: Path, cols: int, sort: str,
                    source: str, zoom: int = 1, per_page: int = 1024
                    ) -> list[tuple[str, Path, int]]:
    """Jedna strona na klasę: wszystkie próbki danego znaku razem.

    To najlepszy widok do oceny na oko. Operator patrzy na stronę z 700
    obrazami litery K i od razu widzi te, które nie wyglądają jak -.-

    Klasy liczniejsze niż per_page są dzielone na kolejne strony. Klasa 0
    ma 15% zbioru, czyli przy 30 tysiącach próbek 4500 kafelków — jedna
    strona miałaby 22 tysiące pikseli wysokości i nie dałoby się jej
    obejrzeć ani otworzyć.
    """
    pages = []
    for ci in range(C.N_CLASSES):
        all_idx = np.flatnonzero(y == ci)
        if all_idx.size == 0:
            continue
        all_idx = sort_indices(all_idx, X, flags, sort)
        ch = C.ID_TO_CHAR[ci]
        name = "PUSTE_RADIO" if ci == 0 else ch
        code = C.MORSE_DICT.get(ch, "")
        n_parts = (all_idx.size + per_page - 1) // per_page

        for part in range(n_parts):
            idx = all_idx[part * per_page:(part + 1) * per_page]
            mosaic = build_mosaic(X, idx, flags, cols, zoom)
            db = frontend.denormalize_db(X[idx])
            con = (np.percentile(db, 99.5, axis=(1, 2))
                   - np.percentile(db, 25, axis=(1, 2)))

            part_tag = f"  cz.{part+1}/{n_parts}" if n_parts > 1 else ""
            title = (f"KLASA {ci:2d}  '{name}'  {code:<6}   "
                     f"{idx.size} próbek{part_tag}   [{source}]")
            sub = (f"kolejność: {sort}   "
                   f"kontrast {con.min():.0f}..{con.max():.0f} dB "
                   f"(mediana {np.median(con):.0f})   "
                   f"kafelek {C.IMG_FRAMES}x{C.IMG_BINS} = "
                   f"2,56 s x 400-1200 Hz, powiekszenie {zoom}x")
            suffix = f"_{part+1:02d}" if n_parts > 1 else ""
            p = save_page(mosaic,
                          out_dir / f"klasa_{ci:02d}_{name}{suffix}.png",
                          title, sub)
            pages.append((f"{ci:2d}  '{name}'  {code}{part_tag}", p, idx.size))
            print(f"  klasa {ci:2d} '{name}'{part_tag}: {idx.size:5d} "
                  f"próbek -> {p.name}")
    return pages


def render_pages(X, y, flags, out_dir: Path, cols: int, per_page: int,
                 sort: str, source: str, zoom: int = 1) -> list[tuple[str, Path, int]]:
    """Strony w kolejności indeksu — przegląd zbioru jako całości."""
    order = sort_indices(np.arange(len(X)), X, flags, sort)
    pages = []
    n_pages = (len(order) + per_page - 1) // per_page
    for p in range(n_pages):
        idx = order[p * per_page:(p + 1) * per_page]
        mosaic = build_mosaic(X, idx, flags, cols, zoom)
        labels = "".join(C.ID_TO_CHAR[int(y[i])] if y[i] > 0
                         else ("." if y[i] == 0 else "?") for i in idx[:60])
        title = (f"STRONA {p+1}/{n_pages}   próbki {idx[0]}..{idx[-1]}   "
                 f"{idx.size} kafelków   [{source}]")
        sub = f"kolejność: {sort}   pierwsze etykiety: {labels}"
        f = save_page(mosaic, out_dir / f"strona_{p+1:03d}.png", title, sub)
        pages.append((f"strona {p+1}/{n_pages}", f, idx.size))
        print(f"  strona {p+1}/{n_pages}: {idx.size} kafelków -> {f.name}")
    return pages


def write_index(out_dir: Path, pages: list, legend: Path, header: str) -> Path:
    """Strona HTML spinająca wszystkie mozaiki — jedno przewijanie zamiast
    otwierania kilkudziesięciu plików po kolei."""
    rows = "\n".join(
        f'<h2>{lab} <span class="n">{n} próbek</span></h2>\n'
        f'<img src="{p.name}" alt="{lab}">'
        for lab, p, n in pages)
    html = f"""<!doctype html>
<meta charset="utf-8">
<title>X-Ray danych treningowych CW</title>
<style>
 body {{ background:#1a1a1a; color:#ddd; font:13px monospace;
         margin:0; padding:16px; }}
 h1 {{ font-size:18px; }}
 h2 {{ font-size:14px; margin:22px 0 4px; color:#8fd; }}
 .n {{ color:#888; font-weight:normal; }}
 img {{ display:block; max-width:100%; image-rendering:pixelated;
        border:1px solid #444; background:#fff; }}
 pre {{ color:#aaa; }}
</style>
<h1>X-Ray kompletu danych treningowych</h1>
<pre>{header}</pre>
<h2>Legenda — skala wspólna dla wszystkich źródeł</h2>
<img src="{legend.name}" alt="legenda">
{rows}
"""
    p = out_dir / "index.html"
    p.write_text(html, encoding="utf-8")
    return p


# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="X-Ray kompletu danych treningowych, jednolita skala")
    ap.add_argument("--dataset", type=Path, help="plik .npz ze zbiorem")
    ap.add_argument("--wav-dir", type=Path,
                    help="katalog z nagraniami wav — ta sama skala")
    ap.add_argument("--out", type=Path, default=C.OUT_DIR / "xray_all")
    ap.add_argument("--cols", type=int, default=16,
                    help="kafelków w wierszu (domyślnie 16 -> ~2100 px)")
    ap.add_argument("--pages", action="store_true",
                    help="strony w kolejności zamiast jednej strony na klasę")
    ap.add_argument("--per-page", type=int, default=1024)
    ap.add_argument("--zoom", type=int, default=2,
                    help="powiekszenie kafelka calkowita krotnoscia, "
                         "bez interpolacji (2 = domyslnie, 3-4 do czytania)")
    ap.add_argument("--sort", choices=("contrast", "index", "amp"),
                    default="contrast",
                    help="kolejność kafelków; contrast = od najsłabszych")
    ap.add_argument("--limit", type=int,
                    help="ogranicz liczbę próbek (do szybkiego sprawdzenia)")
    args = ap.parse_args(argv)

    if not args.dataset and not args.wav_dir:
        raise SystemExit("podaj --dataset i/lub --wav-dir")

    print("=" * 72)
    print("X-RAY KOMPLETU DANYCH")
    print("=" * 72)
    print(C.summary())
    print("-" * 72)

    args.out.mkdir(parents=True, exist_ok=True)
    legend = save_legend(args.out / "legenda.png")
    pages: list = []
    header_parts = [C.summary()]

    if args.dataset:
        X, y, flags, meta = load_npz(args.dataset)
        if args.limit:
            X, y = X[:args.limit], y[:args.limit]
            flags = {k: v[:args.limit] for k, v in flags.items()}
        print(f"\nzbiór {args.dataset.name}: {len(X)} próbek   {meta}")
        header_parts.append(f"zbiór: {args.dataset.name}  {len(X)} próbek  "
                            f"{meta}")
        src = args.dataset.name
        if args.pages:
            pages += render_pages(X, y, flags, args.out, args.cols,
                                  args.per_page, args.sort, src, args.zoom)
        else:
            pages += render_by_class(X, y, flags, args.out, args.cols,
                                     args.sort, src, args.zoom,
                                     args.per_page)

    if args.wav_dir:
        Xw, yw, fw, names = load_wav_dir(args.wav_dir)
        print(f"\nnagrania {args.wav_dir}: {len(Xw)} okien")
        header_parts.append(f"nagrania: {args.wav_dir}  {len(Xw)} okien "
                            f"(bez etykiet)")
        if len(Xw):
            order = sort_indices(np.arange(len(Xw)), Xw, fw, args.sort)
            mosaic = build_mosaic(Xw, order, fw, args.cols, args.zoom)
            p = save_page(mosaic, args.out / "nagrania_wav.png",
                          f"NAGRANIA Z PASMA   {len(Xw)} okien   "
                          f"[{args.wav_dir}]",
                          "bez etykiet; ta sama skala co zbiór syntetyczny — "
                          "porównaj wygląd kafelków")
            pages.append(("NAGRANIA Z PASMA (bez etykiet)", p, len(Xw)))

    idx_path = write_index(args.out, pages, legend, "\n".join(header_parts))
    print(f"\nstron: {len(pages)}")
    print(f"Otwórz: {idx_path}")


if __name__ == "__main__":
    main()
