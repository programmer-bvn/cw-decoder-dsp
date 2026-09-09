"""Koperta kompetencji modelu i rozkład jego błędów.

Odpowiada na trzy pytania, których nie da się odczytać z dokładności
walidacyjnej:

  1. W JAKIM ZAKRESIE TEMPA I TONU model w ogóle czyta. Poprzedni model
     miał 98,68% na walidacji i **zero zdarzeń** poza 18-22 WPM i
     670-830 Hz. Wysoka dokładność na zbiorze, którego rozkład jest
     wąski, nie mówi nic o pracy na pasmie.

  2. JAKI RODZAJ BŁĘDU dominuje. Pomyłki `6 -....` -> `B -...`,
     `5 .....` -> `H ....`, `0 -----` -> `O ---` to wszystko ZGUBIONY
     ELEMENT. To inna choroba niż zamiana kropki na kreskę.

  3. CZY GUBIENIE BIERZE SIĘ Z OBCINANIA OKNA. To jest to pytanie, przez
     które warto było napisać to narzędzie. Były dwie hipotezy:

       (a) okno 2,56 s przy wolnym tempie nie mieści znaku z przerwami,
           więc skrajne elementy wypadają za kadr;
       (b) sama sieć splotowa nie potrafi zliczać powtórzeń w czasie.

     Można je rozróżnić przez korelację błędów z tempem, ale to słaby
     test — tempo wpływa na wiele rzeczy naraz. Mocniejszy jest dostępny
     wprost: generator zapisuje `lab_a`/`lab_b`, czyli FAKTYCZNE granice
     znaku z etykiety po rozjeździe klucza. Da się więc policzyć, jaka
     CZĘŚĆ tego znaku mieści się w widocznym oknie, i zestawić to
     z błędem.

     Jeśli błędy kumulują się tam, gdzie widoczność < 1 — to okno,
     i lekarstwem jest CTC albo szersze pole widzenia. Jeśli są równie
     częste przy widoczności 1,0 — to zliczanie, i lekarstwem jest
     rekurencja albo inna głowa. Bez tej liczby przebudowa architektury
     byłaby strzelaniem.

Uruchomienie:

    python -m tools.koperta --model runs/cw2/best.keras
    python -m tools.koperta --model runs/cw2/best.keras --n 60 --out out/koperta.txt

Wynik idzie na ekran i, z --out, do pliku — w formie, którą da się
przeczytać po ciemku po powrocie od innej roboty.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):                      # uruchomienie wprost
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsp import config as C
from dsp import frontend, morse, radio

# Siatka pomiarowa. Tempo obejmuje zakres z config (13-27 WPM) z zapasem
# po obu stronach, żeby było widać, GDZIE model milknie, a nie tylko że
# w środku czyta. Ton to całe pasmo front-endu 400-1200 Hz.
WPM_SIATKA = (10.0, 13.0, 15.0, 18.0, 20.0, 22.0, 25.0, 27.0, 30.0)
TON_SIATKA = (450.0, 550.0, 650.0, 750.0, 850.0, 950.0, 1100.0)


# --------------------------------------------------------------------------
# SYNTEZA
# --------------------------------------------------------------------------
def _grupa(rng: np.random.Generator) -> tuple[str, str]:
    """Grupa znaków jak w generatorze treningowym: CHARS_PER_CLIP losowych
    liter, etykietą jest ta pod LABEL_INDEX. Musi być identycznie, bo
    inaczej mierzymy inny rozkład niż ten, na którym model się uczył."""
    znaki = [str(rng.choice(list(C.ALPHABET[1:])))
             for _ in range(C.CHARS_PER_CLIP)]
    return "".join(znaki), znaki[C.LABEL_INDEX]


def _wstaw(audio: np.ndarray, wave: np.ndarray) -> int:
    """Wstawienie nadania w klip, wyśrodkowane. Powtarza logikę
    radio.receive() — gdyby się rozjechały, koperta mierzyłaby coś innego
    niż to, co model widzi w treningu."""
    n = audio.size
    if wave.size < n:
        s = (n - wave.size) // 2
        audio[s:s + wave.size] += wave
        return s
    s = (wave.size - n) // 2
    audio += wave[s:s + n]
    return -s


def czysty(rng: np.random.Generator, wpm: float, tone: float,
           amp: float = 0.30) -> tuple[np.ndarray, str, float, float]:
    """Sygnał IDEALNY o zadanym tempie i tonie: bez zaniku, chirpu,
    rozjazdu klucza i przydźwięku. Szum tła zostaje, bo obraz bez szumu
    jest poza rozkładem treningowym i model widziałby coś, czego nigdy
    nie uczył się rozpoznawać.

    Zwraca (audio, znak_etykiety, lab_a, lab_b).
    """
    text, znak = _grupa(rng)
    wave, spans = morse.synth_cw(
        text, wpm=wpm, tone=tone, amp=amp, sr=C.SR,
        phase=float(rng.uniform(0, 2 * np.pi)),
        rng=rng, return_spans=True)

    audio = radio.band_noise(C.CLIP_SAMPLES, rng, rms=C.NOISE_RMS, tilt=0.0)
    offset = _wstaw(audio, wave)

    lab_a = lab_b = float("nan")
    for tag, a, b in spans:
        if tag == C.LABEL_INDEX:
            lab_a, lab_b = a + offset, b + offset
            break
    return audio, znak, lab_a, lab_b


# --------------------------------------------------------------------------
# GEOMETRIA OKNA
# --------------------------------------------------------------------------
def widocznosc(lab_a: float, lab_b: float) -> float:
    """Jaka część znaku z etykiety mieści się w oknie sieci: 1,0 = cały,
    0,5 = połowa wypadła za kadr, 0,0 = nie ma go wcale.

    To jest ta liczba, która rozstrzyga spór o przyczynę gubienia
    elementów. Okno ma IMG_FRAMES ramek, a `lab_a`/`lab_b` są w próbkach
    klipu, więc przeliczamy je na ramki OKNA — funkcją z frontend.py, nie
    własnym wzorem, bo dodatkowa ramka z center=True już raz kosztowała
    dwuramkowe przesunięcie.
    """
    if not np.isfinite(lab_a) or not np.isfinite(lab_b):
        return float("nan")
    fa = frontend.sample_to_window_frame(lab_a)
    fb = frontend.sample_to_window_frame(lab_b)
    if fb <= fa:
        return 0.0
    lo = max(fa, 0.0)
    hi = min(fb, float(C.IMG_FRAMES))
    return float(max(0.0, (hi - lo) / (fb - fa)))


# --------------------------------------------------------------------------
# RODZAJ BŁĘDU
# --------------------------------------------------------------------------
def rodzaj_bledu(prawda: str, pred: str) -> str:
    """Klasyfikacja pomyłki po DŁUGOŚCI kodu, nie po wyglądzie znaku.

    `6` to `-....`, `B` to `-...`. Z punktu widzenia liter to zupełnie
    inne znaki; z punktu widzenia sieci to ten sam wzór o jeden element
    krótszy. Nazwa błędu musi opisywać to drugie, bo to ono mówi, co
    naprawiać.
    """
    if pred == prawda:
        return "trafiony"
    if pred == " ":
        return "wzięty za ciszę"
    kp = C.MORSE_DICT.get(prawda, "")
    kq = C.MORSE_DICT.get(pred, "")
    if not kp or not kq:
        return "inne"
    if len(kq) == len(kp) - 1:
        return "zgubiony element"
    if len(kq) == len(kp) + 1:
        return "wstawiony element"
    if len(kq) == len(kp):
        return "kropka/kreska"
    return "inne"


RODZAJE = ("trafiony", "zgubiony element", "wstawiony element",
           "kropka/kreska", "wzięty za ciszę", "inne")


# --------------------------------------------------------------------------
# POMIARY
# --------------------------------------------------------------------------
def _przewiduj(model, obrazy: list[np.ndarray], batch: int = 256) -> np.ndarray:
    X = np.stack(obrazy)[..., None].astype(np.float32)
    p = model.predict(X, batch_size=batch, verbose=0)
    return p.argmax(1)


def mierz_koperte(model, n: int, seed: int = 20260909):
    """Siatka tempo x ton na sygnale idealnym. Zwraca (trafienia, liczby)."""
    rng = np.random.default_rng(seed)
    obrazy, prawdy, komorki = [], [], []

    for iw, wpm in enumerate(WPM_SIATKA):
        for it, tone in enumerate(TON_SIATKA):
            for _ in range(n):
                audio, znak, _, _ = czysty(rng, wpm, tone)
                obrazy.append(frontend.to_net_image(audio))
                prawdy.append(C.CHAR_TO_ID[znak])
                komorki.append((iw, it))

    pred = _przewiduj(model, obrazy)
    prawdy = np.asarray(prawdy)
    traf = np.zeros((len(WPM_SIATKA), len(TON_SIATKA)), dtype=np.int32)
    ile = np.zeros_like(traf)
    for (iw, it), p, t in zip(komorki, pred, prawdy):
        ile[iw, it] += 1
        if p == t:
            traf[iw, it] += 1
    return traf, ile


def mierz_bledy(model, n: int, seed: int = 20260910):
    """Pełny model kanału, tempo losowane jak w treningu. Dla każdej próbki
    zapisujemy rodzaj błędu, FAKTYCZNE tempo i widoczność znaku w oknie.

    Tempo bierzemy z meta, nie z zadanej wartości: receive() dokłada do
    niego WPM_JITTER, więc zadane 20 daje faktyczne 13-27. Binowanie po
    faktycznym jest jedyne sensowne.
    """
    rng = np.random.default_rng(seed)
    obrazy, prawdy, wpmy, widoki = [], [], [], []

    for _ in range(n):
        text, znak = _grupa(rng)
        audio, meta = radio.receive(text, rng, n_samples=C.CLIP_SAMPLES,
                                    realism=True)
        obrazy.append(frontend.to_net_image(audio))
        prawdy.append(znak)
        wpmy.append(float(meta["wpm"]))
        widoki.append(widocznosc(meta["lab_a"], meta["lab_b"]))

    pred_id = _przewiduj(model, obrazy)
    pred = [C.ALPHABET[i] for i in pred_id]
    rodzaje = [rodzaj_bledu(t, p) for t, p in zip(prawdy, pred)]
    return (np.asarray(rodzaje), np.asarray(wpmy, dtype=np.float64),
            np.asarray(widoki, dtype=np.float64), prawdy, pred)


# --------------------------------------------------------------------------
# RAPORT
# --------------------------------------------------------------------------
def _tabela_koperty(traf, ile) -> list[str]:
    w = ["KOPERTA: gdzie model czyta (sygnał idealny, % trafień)", ""]
    w.append("  WPM \\ ton   " + "".join(f"{t:7.0f}" for t in TON_SIATKA))
    w.append("  " + "-" * (12 + 7 * len(TON_SIATKA)))
    for iw, wpm in enumerate(WPM_SIATKA):
        wiersz = f"  {wpm:5.0f}      "
        for it in range(len(TON_SIATKA)):
            proc = 100.0 * traf[iw, it] / max(1, ile[iw, it])
            wiersz += "      ." if proc < 0.5 else f"{proc:7.0f}"
        w.append(wiersz)
    w.append("")
    w.append("  kropka = ZERO trafień. Nie \"gorzej\" — model milczy.")
    return w


def _zakres(traf, ile, prog: float = 50.0):
    """Przedział tempa i tonu, w którym model trzyma zadany próg. Liczony
    po brzegach, nie po najlepszej komórce — interesuje nas, gdzie jeszcze
    działa, a nie gdzie działa najlepiej."""
    proc = 100.0 * traf / np.maximum(1, ile)
    dobre_w = [WPM_SIATKA[i] for i in range(len(WPM_SIATKA))
               if proc[i, :].max() >= prog]
    dobre_t = [TON_SIATKA[j] for j in range(len(TON_SIATKA))
               if proc[:, j].max() >= prog]
    return dobre_w, dobre_t


def _tabela_widocznosci(rodzaje, widoki) -> list[str]:
    """TO JEST ROZSTRZYGAJĄCY POMIAR — patrz opis modułu."""
    w = ["PRZYCZYNA GUBIENIA: błąd wobec widoczności znaku w oknie", ""]
    w.append("  widoczność    próbek   trafień   zgubiony el.")
    w.append("  " + "-" * 48)
    kubelki = ((0.0, 0.5, "poniżej 50%"), (0.5, 0.9, "50-90%"),
               (0.9, 0.999, "90-100%"), (0.999, 1.01, "cały (100%)"))
    zebrane = []
    for lo, hi, opis in kubelki:
        m = (widoki >= lo) & (widoki < hi)
        ile = int(m.sum())
        if ile == 0:
            w.append(f"  {opis:12s} {ile:8d}         -              -")
            continue
        traf = 100.0 * float((rodzaje[m] == "trafiony").sum()) / ile
        gub = 100.0 * float((rodzaje[m] == "zgubiony element").sum()) / ile
        zebrane.append((opis, ile, traf, gub))
        w.append(f"  {opis:12s} {ile:8d}   {traf:6.1f}%        {gub:6.1f}%")
    return w, zebrane


def _wyrok(zebrane) -> list[str]:
    """Wniosek wypisany wprost, żeby nie trzeba było go wyprowadzać
    z tabeli o 23:00 po dniu pracy fizycznej."""
    caly = next((z for z in zebrane if z[0].startswith("cały")), None)
    obciete = [z for z in zebrane if not z[0].startswith("cały")]
    if caly is None or not obciete:
        return ["WYROK: za mało danych w kubełkach — podnieś --n-bledy"]

    traf_caly = caly[2]
    n_obc = sum(z[1] for z in obciete)
    traf_obc = (sum(z[1] * z[2] for z in obciete) / n_obc) if n_obc else 0.0
    udzial = 100.0 * n_obc / (n_obc + caly[1])

    w = ["WYROK", ""]
    w.append(f"  znak w całości w oknie:  {traf_caly:5.1f}% trafień "
             f"({caly[1]} próbek)")
    w.append(f"  znak obcięty oknem:      {traf_obc:5.1f}% trafień "
             f"({n_obc} próbek, {udzial:.0f}% zbioru)")
    w.append("")
    roznica = traf_caly - traf_obc
    if roznica >= 15.0:
        w.append(f"  Różnica {roznica:.0f} punktów procentowych. Dominuje")
        w.append("  OBCINANIE OKNA, nie niezdolność sieci do zliczania.")
        w.append("  Lekarstwem jest CTC albo szersze pole widzenia,")
        w.append("  a NIE rekurencja dla samego zliczania.")
    elif roznica <= 5.0:
        w.append(f"  Różnica tylko {roznica:.0f} punktów. Model myli się")
        w.append("  równie często, gdy widzi CAŁY znak — czyli problem jest")
        w.append("  w ZLICZANIU, nie w oknie. Wtedy warto porównać z 'gru':")
        w.append("  rekurencja jest tu kandydatem na lekarstwo.")
    else:
        w.append(f"  Różnica {roznica:.0f} punktów — obie przyczyny działają.")
        w.append("  Okno tłumaczy część, ale nie całość. Poszerzenie pola")
        w.append("  widzenia pomoże, samo nie wystarczy.")
    if udzial < 5.0:
        w.append("")
        w.append(f"  UWAGA: obciętych jest tylko {udzial:.0f}% zbioru, więc")
        w.append("  nawet duża różnica tłumaczy mały ułamek wszystkich błędów.")
    return w


def _tabela_wpm(rodzaje, wpmy, widoki) -> list[str]:
    w = ["ROZKŁAD WOBEC TEMPA", ""]
    w.append("  WPM        próbek  trafień  zgub.el.  wstaw.  kr/kre  "
             "obciętych")
    w.append("  " + "-" * 66)
    granice = (5, 12, 15, 18, 21, 24, 27, 61)
    for lo, hi in zip(granice[:-1], granice[1:]):
        m = (wpmy >= lo) & (wpmy < hi)
        ile = int(m.sum())
        if ile == 0:
            continue
        r = rodzaje[m]
        f = lambda nazwa: 100.0 * float((r == nazwa).sum()) / ile
        obc = 100.0 * float((widoki[m] < 0.999).sum()) / ile
        w.append(f"  {lo:2d}-{hi - 1:<2d}   {ile:9d}  {f('trafiony'):6.1f}%  "
                 f"{f('zgubiony element'):7.1f}%  "
                 f"{f('wstawiony element'):5.1f}%  "
                 f"{f('kropka/kreska'):5.1f}%  {obc:8.1f}%")
    return w


def _najczestsze(prawdy, pred, ile: int = 12) -> list[str]:
    from collections import Counter
    c = Counter((t, p) for t, p in zip(prawdy, pred) if t != p)
    w = ["NAJCZĘSTSZE POMYŁKI", ""]
    for (t, p), n in c.most_common(ile):
        kt = C.MORSE_DICT.get(t, "?")
        kp = C.MORSE_DICT.get(p, "?") if p != " " else "(cisza)"
        w.append(f"  {t} {kt:6s} -> {p} {kp:8s} {n:4d}x   "
                 f"{rodzaj_bledu(t, p)}")
    return w


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Koperta kompetencji modelu i rozkład błędów")
    ap.add_argument("--model", default="runs/cw2/best.keras",
                    help="plik modelu (.keras)")
    ap.add_argument("--n", type=int, default=40,
                    help="próbek na komórkę siatki tempo x ton")
    ap.add_argument("--n-bledy", type=int, default=3000,
                    help="próbek do analizy błędów (pełny model kanału)")
    ap.add_argument("--prog", type=float, default=50.0,
                    help="próg %% trafień uznawany za \"czyta\"")
    ap.add_argument("--out", default=None, help="zapis raportu do pliku")
    a = ap.parse_args(argv)

    sciezka = Path(a.model)
    if not sciezka.exists():
        print(f"BŁĄD: nie ma {sciezka}", file=sys.stderr)
        return 1

    from dsp.model import load_model
    model = load_model(sciezka)

    wiersze = ["=" * 70,
               f" KOPERTA I BŁĘDY: {sciezka}",
               "=" * 70,
               f" front-end: {C.fingerprint_str()}",
               f" siatka: {len(WPM_SIATKA)}x{len(TON_SIATKA)} komórek "
               f"po {a.n} próbek; błędy z {a.n_bledy} próbek",
               ""]

    traf, ile = mierz_koperte(model, a.n)
    wiersze += _tabela_koperty(traf, ile) + [""]

    dobre_w, dobre_t = _zakres(traf, ile, a.prog)
    if dobre_w and dobre_t:
        wiersze.append(f"  czyta (>= {a.prog:.0f}%): "
                       f"tempo {min(dobre_w):.0f}-{max(dobre_w):.0f} WPM, "
                       f"ton {min(dobre_t):.0f}-{max(dobre_t):.0f} Hz")
    else:
        wiersze.append(f"  UWAGA: nigdzie nie osiąga {a.prog:.0f}% — "
                       f"model albo nienauczony, albo zły plik")
    wiersze.append("")

    rodzaje, wpmy, widoki, prawdy, pred = mierz_bledy(model, a.n_bledy)
    tab, zebrane = _tabela_widocznosci(rodzaje, widoki)
    wiersze += tab + [""] + _wyrok(zebrane) + [""]
    wiersze += _tabela_wpm(rodzaje, wpmy, widoki) + [""]
    wiersze += _najczestsze(prawdy, pred) + [""]

    tekst = "\n".join(wiersze)
    print(tekst)
    if a.out:
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(tekst + "\n", encoding="utf-8")
        print(f"zapisane: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
