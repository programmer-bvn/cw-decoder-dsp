"""Odczyt PRAWDZIWYCH nagrań i porównanie z tym, co faktycznie nadano.

PO CO TO ISTNIEJE. Dokładność walidacyjna na syntetyku nie mówi nic
o pracy na antenie — to najdroższa lekcja tego projektu. Pierwszy model
miał 98,68% i na paśmie dawał fragmenty. Model z 15.09 ma 98,73% na
milionie próbek i dopóki nikt nie puści go na nagraniu z radia, ta liczba
znaczy dokładnie tyle samo, co tamta.

Cztery nagrania leżą w `probki/`, a obok każdego jest opis z polem
`nadane:`. To jedyna prawda naziemna, jaką ten projekt ma.

SKĄD SIĘ BIERZE OCZEKIWANY TEKST. Z pola `nadane:`, przez wyciągnięcie
z niego ciągów PISANYCH WIELKIMI LITERAMI. Reguła wygląda na sztuczkę,
ale jest trafna: telegrafia nie zna małych liter, więc wszystko, co
w opisie jest małą literą, to komentarz autora, a nie treść nadania.

    mic3:    "CQ CQ CQ DE SQ2BVN SQ2BVN"
             -> CQ, CQ, CQ, DE, SQ2BVN, SQ2BVN    (pełna prawda)
    mic2:    "litery zgodnie z definicja... (w odczycie rozpoznawalne
              CQ, SQ2BVN, 73)"
             -> CQ, SQ2BVN, 73                    (fragmenty)
    mic1:    "...(tresc nieodnotowana)"
             -> nic                                (tylko podgląd odczytu)
    radio1:  "nieznane — odbior z pasma"
             -> nic

Dzięki temu nie trzeba niczego dopisywać do opisów, a nowe nagranie
z zapisanym `nadane:` wejdzie do pomiaru samo.

CO MIERZYMY. Dla każdego oczekiwanego ciągu dwie rzeczy:

  PODCIĄG   — czy występuje w odczycie w całości, znak po znaku.
              To jest to, co widzi operator: "SQ2BVN" w zupie znaków.
  KOLEJNOŚĆ — czy występuje z zachowaną kolejnością, ale z wtrętami
              (np. "SXQ2BVN"). Dekoder gubi i wstawia znaki, więc sam
              podciąg bywa zbyt surowy, żeby pokazać postęp.

Różnica między tymi dwoma liczbami mówi, czy problemem jest
rozpoznawanie, czy śmieci pomiędzy.

Uruchomienie:

    python -m tools.nagrania --model runs/dpu_1000000/best.keras
    python -m tools.nagrania --model runs/dpu_1000000/best.keras \\
        --out out/nagrania.txt
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

if __package__ in (None, ""):                      # uruchomienie wprost
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsp import config as C
from tools import wav2net

# Ciągi krótsze niż dwa znaki odpadają: pojedyncza litera trafi się
# w każdym odczycie przez przypadek i nic nie mierzy.
TOKEN = re.compile(r"[A-Z0-9]{2,}")


def oczekiwane(opis: Path) -> list[str]:
    """Ciągi nadane, wyciągnięte z pola `nadane:` opisu próbki."""
    if not opis.exists():
        return []
    for linia in opis.read_text(encoding="utf-8").splitlines():
        if linia.startswith("nadane:"):
            return TOKEN.findall(linia.split(":", 1)[1])
    return []


def ile_podciagow(igla: str, stog: str) -> int:
    """Ile razy `igla` występuje w `stog` w całości, bez nakładania."""
    n, i = 0, 0
    while True:
        j = stog.find(igla, i)
        if j < 0:
            return n
        n += 1
        i = j + len(igla)


def ile_w_kolejnosci(igla: str, stog: str) -> int:
    """Ile razy da się wyłuskać `iglę` jako podciąg, ZUŻYWAJĄC stóg.

    Zużywanie jest tu istotne. Pierwsza wersja sprawdzała tylko "czy
    występuje", a `CQ` jest w mic3 oczekiwane TRZY razy — więc jedno
    wystąpienie w odczycie zaliczało się trzykrotnie i dawało 3/6 zamiast
    1/6. Miernik zawyżałby postęp, i to systematycznie, bo powtórzenia
    w telegrafii są normą: CQ nadaje się trzy razy, znak dwa razy.
    """
    n, i = 0, 0
    while i < len(stog):
        k = i
        for z in igla:
            k = stog.find(z, k)
            if k < 0:
                return n
            k += 1
        n += 1
        i = k
    return n


def scal_po_czasie(zdarzenia: list, okno_s: float) -> list:
    """Scala SĄSIEDNIE zdarzenia o tym samym znaku, jeśli dzieli je mniej
    niż `okno_s`. Zostawia to o wyższej pewności.

    PO CO. Okno sieci przesuwa się co 20 ms, a znak przy 20 WPM trwa około
    pół sekundy — więc ten sam znak widzi kilkanaście kolejnych okien.
    decode_stream scala tylko przebiegi SĄSIADUJĄCE w indeksie, a chwilowy
    spadek pewności albo wtręt klasy 0 rozbija taki przebieg na kilka
    zdarzeń. Stąd odczyty w rodzaju "CCQQQCCCCQQQQBBBVVV": informacja jest,
    ale utopiona w powtórzeniach i dla człowieka nieczytelna.

    PRÓG JEST ZMIERZONY, NIE ZGADNIĘTY. Ogranicza go od góry odstęp
    DWÓCH TAKICH SAMYCH LITER OBOK SIEBIE (jak LL w HELLO) — scalenie
    szersze niż ten odstęp zjadłoby podwojenie. Dla L (9 jednostek):

        13 WPM   dwa L co 1108 ms
        20 WPM   dwa L co  720 ms
        27 WPM   dwa L co  533 ms
        30 WPM   dwa L co  480 ms

    Od dołu ogranicza go to, ile trwają powtórzenia z przesuwanego okna:
    przy 20 WPM próg 0,15 s i 0,25 s NIE sklejał ich w całości, 0,35 s
    sklejał. Stąd domyślne 0,35 — mieści się pod 480 ms nawet przy
    30 WPM, więc jest bezpieczne w całym zakresie 13-30 WPM.

    Przy wolnym nadawaniu zostaną resztki, bo tam znak trwa dłużej niż
    próg. Pełne rozwiązanie to dekoder sekwencyjny CTC, który znosi całą
    tę klasę problemu — scalanie jest obejściem, nie naprawą, i dlatego
    siedzi w narzędziu POMIAROWYM, a nie w dekoderze produkcyjnym.

    Dlatego to narzędzie pokazuje odczyt SUROWY i SCALONY obok siebie,
    razem z obiema miarami — dopiero z tego widać, czy to pomaga na
    prawdziwym materiale.
    """
    if okno_s <= 0 or not zdarzenia:
        return zdarzenia
    wynik = [zdarzenia[0]]
    for z in zdarzenia[1:]:
        p = wynik[-1]
        if z[1] == p[1] and (z[0] - p[0]) < okno_s:
            if z[2] > p[2]:                 # zostaje pewniejsze
                wynik[-1] = z
        else:
            wynik.append(z)
    return wynik


def odczytaj(sciezka: Path, model, min_conf: float, min_windows: int,
             scal_s: float = 0.0):
    """Nagranie -> (odczyt_surowy, odczyt_scalony, opis_pętli, wzmocnienie)."""
    okna, srodki, wzm, lock = wav2net.wav_to_windows(
        sciezka, auto_gain=True, retune=True)
    zdarzenia, _, _, _ = wav2net.decode_stream(
        okna, srodki, model, min_conf=min_conf, min_windows=min_windows)
    surowy = "".join(C.ALPHABET[i] for _, i, _, _ in zdarzenia)
    scalony = "".join(C.ALPHABET[i]
                      for _, i, _, _ in scal_po_czasie(zdarzenia, scal_s))
    return surowy, scalony, lock, wzm


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Odczyt prawdziwych nagrań i porównanie z nadanym")
    ap.add_argument("--model", default="runs/dpu_1000000/best.keras")
    ap.add_argument("--probki", default="probki",
                    help="katalog z nagraniami i opisami")
    ap.add_argument("--min-conf", type=float, default=0.95)
    ap.add_argument("--min-windows", type=int, default=3)
    ap.add_argument("--scal", type=float, default=0.35,
                    help="scalaj powtorzenia tego samego znaku blizsze "
                         "niz tyle sekund; 0 wylacza")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    model_p = Path(a.model)
    if not model_p.exists():
        print(f"BŁĄD: nie ma modelu {model_p}", file=sys.stderr)
        return 1

    kat = Path(a.probki)
    nagrania = sorted(kat.glob("*.wav"))
    if not nagrania:
        print(f"BŁĄD: brak nagrań w {kat}/", file=sys.stderr)
        print("Nagrania nie są w repozytorium (106 MB) — wracają na HDD",
              file=sys.stderr)
        print("przez na_hdd.bat, a do pobrania są w wydaniu probki-v1.",
              file=sys.stderr)
        return 1

    from dsp.model import load_model
    model = load_model(model_p)

    w = ["=" * 70,
         f" ODCZYT PRAWDZIWYCH NAGRAŃ: {model_p}",
         "=" * 70,
         f" próg pewności {a.min_conf}, minimum {a.min_windows} okien",
         ""]

    razem_tok, razem_pod, razem_kol = 0, 0, 0
    razem_s_pod, razem_s_kol = 0, 0

    for wav in nagrania:
        opis = wav.with_suffix(".txt")
        chce = oczekiwane(opis)
        try:
            surowy, scalony, lock, wzm = odczytaj(
                wav, model, a.min_conf, a.min_windows, a.scal)
        except Exception as e:
            w.append(f"{wav.name}")
            w.append(f"  BŁĄD ODCZYTU: {type(e).__name__}: {e}")
            w.append("")
            continue

        w.append(f"{wav.name}   (korekta poziomu {wzm:+.1f} dB)")
        w.append(f"  ton: {lock}")
        w.append(f"  odczyt surowy:  {surowy if surowy else '(nic)'}")
        if a.scal > 0:
            w.append(f"  po scaleniu {a.scal:.2f}s: "
                     f"{scalony if scalony else '(nic)'}")

        if not chce:
            # Brak prawdy naziemnej to NIE jest brak wyniku. mic1 nadawano
            # bez zapisu treści, radio1 to obce stacje. Odczyt zostaje do
            # obejrzenia okiem — telegrafista rozpozna, czy to ma sens.
            w.append("  nadane: nieodnotowane — odczyt tylko do obejrzenia")
            w.append("")
            continue

        # Zliczanie z KROTNOSCIA: jesli CQ nadano trzy razy, jedno
        # wystapienie w odczycie zalicza sie raz, nie trzy.
        def ocen(tekst):
            ile_chce = Counter(chce)
            n_pod = n_kol = 0
            zn_pod, zn_kol = [], []
            for token, krotnosc in ile_chce.items():
                p = min(krotnosc, ile_podciagow(token, tekst))
                k = min(krotnosc, ile_w_kolejnosci(token, tekst))
                n_pod += p
                n_kol += k
                if p:
                    zn_pod.append(f"{token}x{p}" if p > 1 else token)
                if k:
                    zn_kol.append(f"{token}x{k}" if k > 1 else token)
            return n_pod, n_kol, zn_pod, zn_kol

        n_pod, n_kol, zn_pod, zn_kol = ocen(surowy)
        razem_tok += len(chce)
        razem_pod += n_pod
        razem_kol += n_kol

        w.append(f"  nadane: {' '.join(chce)}")
        w.append(f"  w całości:    {n_pod}/{len(chce)}"
                 + (f"   ({' '.join(zn_pod)})" if zn_pod else ""))
        w.append(f"  w kolejności: {n_kol}/{len(chce)}"
                 + (f"   ({' '.join(zn_kol)})" if zn_kol else ""))

        if a.scal > 0:
            s_pod, s_kol, _, _ = ocen(scalony)
            razem_s_pod += s_pod
            razem_s_kol += s_kol
            # Skrócenie odczytu jest tu osobną informacją: jeśli scalanie
            # tnie długość o połowę, a miary nie spadają, to znaczy że
            # wycięło same powtórzenia, nie treść.
            ile_krocej = (100.0 * (1 - len(scalony) / max(1, len(surowy))))
            w.append(f"  po scaleniu:  w całości {s_pod}/{len(chce)}, "
                     f"w kolejności {s_kol}/{len(chce)}, "
                     f"odczyt krótszy o {ile_krocej:.0f}%")
        w.append("")

    if razem_tok:
        w.append("-" * 70)
        w.append(f"RAZEM na nagraniach z zapisaną treścią: "
                 f"w całości {razem_pod}/{razem_tok}, "
                 f"w kolejności {razem_kol}/{razem_tok}")
        if a.scal > 0:
            w.append(f"RAZEM po scaleniu {a.scal:.2f}s: "
                     f"w całości {razem_s_pod}/{razem_tok}, "
                     f"w kolejności {razem_s_kol}/{razem_tok}")
        w.append("")
        w.append("To jest liczba, która ma znaczenie. Dokładność walidacyjna")
        w.append("na syntetyku mówi o syntetyku — pierwszy model tego projektu")
        w.append("miał 98,68% i na paśmie dawał fragmenty.")
    else:
        w.append("-" * 70)
        w.append("Żadne nagranie nie ma zapisanej treści nadania, więc nie ma")
        w.append("czego porównać. Odczyty wyżej są do obejrzenia okiem.")
        w.append("Warto uzupełnić pole 'nadane:' w probki/*.txt.")

    tekst = "\n".join(w)
    print(tekst)
    if a.out:
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(tekst + "\n", encoding="utf-8")
        print(f"zapisane: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
