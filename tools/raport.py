"""ZGŁOSZENIE BŁĘDNEGO ODCZYTU — audio wstecz + opis operatora.

    python -m tools.raport --wav probki/mic3_pamiec_15wpm.wav \
        --odczyt "CCLQ2BVQ2BVQ2BVS" --nadane "CQ CQ CQ DE SQ2BVN SQ2BVN"

    python -m tools.raport --wav nagranie.wav --sekundy 30 --zip

PO CO TO ISTNIEJE
-----------------
Model poprawia się od danych, których nie umie odczytać — nie od tych,
które odczytuje dobrze. A jedynym źródłem takich danych jest operator,
który SŁYSZY, co zostało nadane, i widzi, że dekoder wypisał coś innego.

KLUCZOWA WŁAŚCIWOŚĆ: zapis WSTECZ
---------------------------------
Operator zgłasza błąd dopiero po tym, jak go usłyszał. Nie można go prosić,
żeby wcześniej nacisnął "nagrywaj" — w chwili naciskania interesujący
fragment już minął. Dlatego audio musi lecieć do bufora cyklicznego
nieprzerwanie, a zgłoszenie zapisuje jego zawartość.

60 s przy 20 WPM to ok. 100-120 znaków, czyli kilka pełnych grup.
Jako wav 8 kHz mono 16 bit — 960 kB, więc mieści się w mailu i w
załączniku do zgłoszenia na GitHubie (limit 25 MB).

CO WCHODZI DO ZGŁOSZENIA
------------------------
Trzy warstwy, i wszystkie trzy są potrzebne:

  1. AUDIO — bez tego nie ma czego analizować.
  2. CO USŁYSZAŁ OPERATOR — to jest etykieta, i tylko on ją zna.
  3. POMIARY I WERSJE, liczone automatycznie — ton, tempo, szerokość,
     kontrast, poziom, odcisk front-endu, wersja modelu.

Trzeciej warstwy nie wolno zostawiać człowiekowi. Dziś kosztowało nas to
pół dnia: nagranie przyszło bez zapisanego tempa (okazało się 15 WPM przy
modelu uczonym na 20) i bez informacji, że szło przez mikrofon komputera
(przesterowanie o 12 dB). Oba fakty są mierzalne z pliku, więc mierzy je
program, a nie operator.

CZEGO TEN MODUŁ NIE ROBI
------------------------
NIE WYSYŁA nic samodzielnie. Buduje paczkę i pokazuje, gdzie leży.
Wysłanie jest decyzją operatora — to jego audio i jego znak wywoławczy.

ZNAK WYWOŁAWCZY: PO CO JEST I DLACZEGO NIE PUBLIKUJEMY GO WPROST
----------------------------------------------------------------
Znak jest w zgłoszeniu celowo i jest jego najmocniejszą stroną. To on
zastępuje nadmiarowość, której potrzebuje anonimowe etykietowanie w stylu
"zaznacz autobusy": nikt nie wyśle śmiecia pod własnym znakiem. Pozwala
też liczyć wiarygodność per operator i dopytać, gdy zgłoszenie jest
niejasne.

Ale znak identyfikuje w rejestrze licencyjnym z imieniem i adresem, więc
w zbiorze OPUBLIKOWANYM zamienia się go na trwały pseudonim (pseudonim()).
Mapowanie zostaje lokalnie. Statystyki per operator działają dalej, bo
pseudonim jest stały.

Czego to NIE załatwia, i trzeba to powiedzieć wprost: nagranie CW zawiera
znaki wywoławcze W SAMYM SYGNALE. Nie da się ich usunąć bez zniszczenia
danych — a właśnie te fragmenty są najcenniejsze do treningu. Przy
transmisji krótkofalarskiej to nie jest problem, bo nadanie jest publiczne
z natury i znaki trafiają rutynowo do logów i spotów. Pseudonim chroni
tożsamość ZGŁASZAJĄCEGO, nie treść nadania.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsp import config as C
from dsp import frontend, tune


# ==========================================================================
# PSEUDONIM OPERATORA
# ==========================================================================
def pseudonim(znak: str, sol: str = "AI_DSP") -> str:
    """Trwały pseudonim ze znaku wywoławczego, np. SQ2BVN -> op-3f9a2c.

    Trwały, czyli ten sam znak zawsze daje ten sam pseudonim — dzięki temu
    da się liczyć wiarygodność per operator w opublikowanym zbiorze, nie
    publikując, kto to jest.

    Nieodwracalny w praktyce, ale NIE tajny: przestrzeń znaków
    wywoławczych jest mała (rzędu milionów), więc mając listę można
    pseudonimy przemapować. To jest zamierzone i wystarczające — chodzi
    o to, żeby znak nie leżał wprost w pliku, który ktoś zaindeksuje,
    a nie o ochronę kryptograficzną. Kto chce, i tak usłyszy znak
    w nagraniu.

    Sól ma tylko tyle znaczenia, że pseudonimy z tego projektu nie
    zestawią się z pseudonimami z innego.

    ZNANE OGRANICZENIE: ZNAK NIE JEST TRWAŁYM IDENTYFIKATOREM OSOBY
    ---------------------------------------------------------------
    W Polsce nie ma przedłużania licencji — jest wydawanie od nowa,
    z podaniem żądanego znaku. Znak może więc po wygaśnięciu trafić do
    kogoś innego. Do tego zmiana okręgu zamieszkania historycznie
    zmieniała cyfrę (SQ1BVN -> SQ3BVN -> SQ2BVN u tego samego operatora).
    Czyli ten sam znak może z czasem oznaczać różne osoby, a ta sama
    osoba różne znaki.

    Świadomie NIE dokładam tu roku do pseudonimu, choć rozwiązałoby to
    pierwszy przypadek. Zepsułoby drugi i to gorzej: operator zgłaszający
    błędy przez trzy lata rozpadłby się na trzy tożsamości, a wtedy
    statystyka wiarygodności per operator — jedyny powód, żeby ten
    pseudonim istniał — przestaje działać. Ryzyko scalenia dwóch osób
    jest rzadkie, ryzyko rozbicia jednej byłoby stałe.

    Zamiast tego każde zgłoszenie nosi DATĘ (nazwa pliku i pole
    "zapisano"). Jeśli wiarygodność pseudonimu zmieni się skokowo,
    widać kiedy — i wtedy można to rozdzielić ręcznie.
    """
    import hashlib
    z = znak.strip().upper()
    if not z:
        return ""
    h = hashlib.sha256((sol + "|" + z).encode("utf-8")).hexdigest()
    return f"op-{h[:6]}"


# ==========================================================================
# BUFOR CYKLICZNY — audio "wstecz"
# ==========================================================================
class BuforWstecz:
    """Trzyma ostatnie `sekundy` audio, żeby dało się je zapisać po fakcie.

    Osobny od frontend.Waterfall, bo tamten trzyma dokładnie tyle, ile
    potrzebuje jedno okno sieci (2,56 s). Tutaj chodzi o materiał dla
    człowieka i do treningu, więc rząd wielkości jest inny.

    Pamięć: 60 s przy 8 kHz w float32 to 1,9 MB. Nieistotne.
    """

    def __init__(self, sekundy: float = 60.0, sr: int = C.SR):
        self.sr = sr
        self.dlugosc = int(round(sekundy * sr))
        self.buf = np.zeros(self.dlugosc, dtype=np.float32)
        self.zapisano = 0

    def dodaj(self, blok: np.ndarray) -> None:
        b = np.asarray(blok, dtype=np.float32).ravel()
        n = b.size
        if n >= self.dlugosc:
            self.buf[:] = b[-self.dlugosc:]
        else:
            self.buf[:-n] = self.buf[n:]
            self.buf[-n:] = b
        self.zapisano = min(self.dlugosc, self.zapisano + n)

    def pobierz(self, sekundy: float | None = None) -> np.ndarray:
        """Ostatnie `sekundy` audio (albo tyle, ile jest)."""
        n = self.zapisano if sekundy is None else min(
            self.zapisano, int(round(sekundy * self.sr)))
        return self.buf[-n:].copy() if n else np.zeros(0, dtype=np.float32)


# ==========================================================================
# OCENA: czy ten fragment wygląda na błąd wart zgłoszenia
# ==========================================================================
def sugeruj_zgloszenie(audio: np.ndarray,
                       zdarzenia: list | None = None) -> tuple[bool, list[str]]:
    """(czy_warto, powody). Program sam zauważa, że coś było nie tak.

    Sens: operator nie musi patrzeć na ekran. Wszystkie te wskaźniki i tak
    są liczone przy dekodowaniu, więc podpowiedź jest darmowa — a bez niej
    zgłoszenia przyjdą tylko od tych, którzy akurat patrzyli.

    Progi pochodzą z pomiarów na prawdziwych nagraniach, nie z założeń.
    """
    from tools.wav2net import TARGET_PEAK_DB, measure_peak_db

    powody = []

    # --- poziom: przesterowanie widzieliśmy w każdym nagraniu z mikrofonu ---
    szczyt = measure_peak_db(audio)
    if szczyt >= C.DB_MAX - 0.5:
        powody.append(f"obraz nasycony ({szczyt:.1f} dB przy DB_MAX="
                      f"{C.DB_MAX:.0f}) — poziom wejścia za wysoki")
    elif szczyt < C.DB_MIN + 10:
        powody.append(f"sygnał bardzo słaby ({szczyt:.1f} dB)")

    # --- ton: poza zakresem treningu model nie zwraca NIC ---
    t, f = tune.track_tone(audio)
    ok = np.isfinite(f)
    if ok.any():
        fv = f[ok]
        lo, hi = C.TONE_CENTER - C.TONE_SPREAD, C.TONE_CENTER + C.TONE_SPREAD
        udzial = float(np.mean((fv >= lo) & (fv <= hi)))
        if udzial < 0.8:
            powody.append(f"ton w zakresie czytelnym tylko "
                          f"{100*udzial:.0f}% czasu (mediana "
                          f"{np.median(fv):.0f} Hz, zakres {lo:.0f}-{hi:.0f})")
    else:
        powody.append("pętla nie zaczepiła się na żadnym tonie")

    # --- szerokość: czysty CW to ok. 75 Hz; 200 Hz to dwie stacje albo
    #     przesterowanie
    db = frontend.power_to_db(frontend.melspec_power(audio))
    prof = db.mean(axis=0)
    pasm = int(np.sum(prof > prof.max() - 10.0))
    szer = pasm * (C.FMAX - C.FMIN) / C.N_MELS
    if szer > 130:
        powody.append(f"sygnał rozlany na {szer:.0f} Hz (czysty CW to ok. "
                      f"75 Hz) — dwie stacje albo przesterowanie")

    # --- pewność dekodera, jeśli podano zdarzenia ---
    if zdarzenia:
        pewnosci = [e[2] for e in zdarzenia if len(e) > 2]
        if pewnosci:
            niskie = sum(1 for p in pewnosci if p < 0.98)
            if niskie > len(pewnosci) * 0.4:
                powody.append(f"{niskie} z {len(pewnosci)} znaków poniżej "
                              f"98% pewności")

    return bool(powody), powody


# ==========================================================================
# PACZKA ZGŁOSZENIA
# ==========================================================================
SZABLON = """\
# ZGŁOSZENIE BŁĘDNEGO ODCZYTU
#
# Wypełnij dwa pola poniżej. Reszta jest zmierzona automatycznie i nie
# trzeba jej ruszać. Bez pola "nadane" to nagranie jest bezużyteczne —
# tylko ty wiesz, co tam faktycznie było.

nadane:        {nadane}
odczytano:     {odczytano}

# --- kontekst, jeśli wiesz (pomaga, nie jest konieczny) ---
znak:          {znak}
nadajnik:      {nadajnik}
klucz:         {klucz}
pasmo:         {pasmo}
filtr:         {filtr}
tor:           {tor}
uwagi:         {uwagi}

"""

MARKER = "# ---- PONIŻEJ MIERZONE AUTOMATYCZNIE, NIE EDYTUJ ----"


def zapisz_raport(audio: np.ndarray, katalog: Path,
                  odczytano: str = "", nadane: str = "",
                  model: str = "", pola: dict | None = None,
                  sr: int = C.SR, zipuj: bool = False) -> Path:
    """Zapisuje wav + txt (i opcjonalnie zip). Zwraca ścieżkę do opisu."""
    import soundfile as sf
    from tools.probki import blok_mierzony, zmierz

    katalog.mkdir(parents=True, exist_ok=True)
    stempel = datetime.now().strftime("%Y%m%d_%H%M%S")
    baza = katalog / f"raport_{stempel}"
    wav = baza.with_suffix(".wav")

    sf.write(wav, np.asarray(audio, dtype=np.float32), sr, subtype="PCM_16")

    d = {"nadane": nadane, "odczytano": odczytano, "znak": "",
         "nadajnik": "", "klucz": "", "pasmo": "", "filtr": "",
         "tor": "", "uwagi": ""}
    d.update(pola or {})

    w = zmierz(wav)
    tekst = SZABLON.format(**d)
    tekst += MARKER + "\n"
    tekst += f"# model: {model or '(nie podano)'}\n"
    tekst += f"# odcisk front-endu: {C.fingerprint_str()}\n"
    tekst += f"# zapisano: {datetime.now().isoformat(timespec='seconds')}\n\n"
    # blok_mierzony sam dokłada swój marker, więc bierzemy jego treść
    # bez pierwszej linii.
    tekst += "\n".join(blok_mierzony(w).splitlines()[1:]) + "\n"

    warto, powody = sugeruj_zgloszenie(frontend.load_audio(wav))
    if powody:
        tekst += "\nco program uznał za podejrzane:\n"
        tekst += "".join(f"  - {p}\n" for p in powody)

    txt = baza.with_suffix(".txt")
    txt.write_text(tekst, encoding="utf-8")

    if zipuj:
        z = baza.with_suffix(".zip")
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(wav, wav.name)
            zf.write(txt, txt.name)
        print(f"paczka: {z}  ({z.stat().st_size/1024:.0f} kB)")

    return txt


# ==========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Zgłoszenie błędnie odczytanego fragmentu")
    ap.add_argument("--wav", type=Path, required=True)
    ap.add_argument("--sekundy", type=float, default=60.0,
                    help="ile ostatnich sekund wziąć (domyślnie 60)")
    ap.add_argument("--odczytano", default="", help="co wypisał dekoder")
    ap.add_argument("--nadane", default="",
                    help="co TWOIM ZDANIEM zostało nadane")
    ap.add_argument("--znak", default="")
    ap.add_argument("--pasmo", default="")
    ap.add_argument("--tor", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--out", type=Path, default=C.ROOT / "zgloszenia")
    ap.add_argument("--zip", action="store_true")
    args = ap.parse_args(argv)

    audio = frontend.load_audio(args.wav)
    n = min(audio.size, int(round(args.sekundy * C.SR)))
    wycinek = audio[-n:]

    print("=" * 70)
    print("ZGŁOSZENIE BŁĘDNEGO ODCZYTU")
    print("=" * 70)
    print(f"źródło:  {args.wav.name}")
    print(f"wycinek: ostatnie {n/C.SR:.1f} s "
          f"({n*2/1024:.0f} kB jako wav 16 bit)")

    warto, powody = sugeruj_zgloszenie(wycinek)
    print(f"\nprogram uznał fragment za "
          f"{'PODEJRZANY' if warto else 'wyglądający poprawnie'}:")
    for p in powody or ["  (nic nie wzbudziło wątpliwości)"]:
        print(f"  - {p}" if powody else p)

    txt = zapisz_raport(wycinek, args.out, odczytano=args.odczytano,
                        nadane=args.nadane, model=args.model,
                        pola={"znak": args.znak, "pasmo": args.pasmo,
                              "tor": args.tor},
                        zipuj=args.zip)
    print(f"\nzapisano: {txt}")
    if not args.nadane:
        print("\nUZUPEŁNIJ pole 'nadane' w tym pliku — bez niego zgłoszenie")
        print("nie ma wartości, bo nie wiadomo, co powinno wyjść.")


if __name__ == "__main__":
    main()
