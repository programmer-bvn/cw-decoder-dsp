"""JEDNO ŹRÓDŁO PRAWDY dla całego projektu.

Wszystkie cztery narzędzia czytają stałe wyłącznie stąd:

    tools/generator.py   generator treningowy   audio -> npz
    tools/xray.py        przeglądarka X-Ray     npz / wav -> obraz
    tools/wav2net.py     konwerter wav -> sieć  wav -> wejście modelu
    tools/mic2wav.py     konwerter mic -> wav   mikrofon -> wav

PO CO TO ISTNIEJE
-----------------
W poprzedniej wersji projektu każde narzędzie miało własną kopię parametrów
front-endu, a normalizacja dB była zapisana trzy razy inaczej:

    Generator_v5_6.py   power_to_db(ref=0.5),  (db + 30) / 30
    Algorytmv6.1.py     power_to_db(ref=1.0),  (db + 35) / 40
    LiveDecoderav5.5.py power_to_db(ref=1.0),  (db + 50) / 40

To nie jest kosmetyka. Model uczony na pierwszym wariancie, karmiony trzecim,
widzi obraz przesunięty o 20 dB i przeskalowany — i nie zgłasza żadnego błędu.
Po prostu myli się częściej. Taka usterka jest niewidoczna w metrykach
treningu (val_accuracy jest wysokie) i ujawnia się tylko na żywym sygnale.

ZASADA: żaden plik w projekcie nie ma prawa mieć własnej liczby. Jeśli
potrzebujesz innej wartości — zmień ją TUTAJ, wtedy zmieni się jednocześnie
w generatorze, w podglądzie i w dekoderze, i dane pozostaną spójne.
"""

from pathlib import Path

# ==========================================================================
# 1. SYGNAŁ
# ==========================================================================
SR = 8000                       # Hz. Pasmo CW to 300-1200 Hz, 8 kHz z zapasem.
                                # Wyjście USB z IC-7300 daje 48 kHz — decymacja
                                # jest w frontend.load_audio().
CLIP_SECONDS = 4.0              # długość fragmentu analizowanego jednorazowo
CLIP_SAMPLES = int(SR * CLIP_SECONDS)      # 32000

# ==========================================================================
# 2. FRONT-END: audio -> obraz dla sieci
#    Te liczby definiują KONTRAKT modelu. Zmiana którejkolwiek unieważnia
#    wszystkie wytrenowane wagi — dane treningowe trzeba wygenerować od nowa.
# ==========================================================================
N_FFT = 512                     # 64 ms okna przy 8 kHz
HOP_LENGTH = 160                # 20 ms kroku -> 50 ramek na sekundę
N_MELS = 32                     # liczba pasm mel = wysokość obrazu
FMIN = 400.0                    # dolna granica pasma [Hz]
FMAX = 1200.0                   # górna granica pasma [Hz]
                                # Pasmo obcięte wokół tonu CW. Poniżej 400 Hz
                                # siedzi przydźwięk sieci i szum wentylatora,
                                # które w obrazie dawały jasny pas u dołu.

# Kształt wejścia sieci: [ramki, pasma, 1]
IMG_FRAMES = 128                # 128 ramek * 20 ms = 2,56 s pola widzenia
IMG_BINS = N_MELS
INPUT_SHAPE = (IMG_FRAMES, IMG_BINS, 1)

# --- Normalizacja dB: JEDNA, STAŁA SKALA ----------------------------------
# ref=1.0 to bezwzględny punkt odniesienia. NIE używać ref=np.max — wtedy
# skala jeździ za najgłośniejszym pikselem w klipie i sam szum po
# rozciągnięciu wygląda jak sygnał ("różowe ściany" z v5.5).
DB_REF = 1.0

# WARTOŚCI ZMIERZONE, nie zgadnięte — python -m tools.calibrate
#
#   szum 0.015 RMS:   5 pct -33.3 dB | mediana -26.7 dB | 95 pct -21.5 dB
#   sygnał amp 0.3:   szczyt +17 dB  | najwyższy zmierzony +20.7 dB
#
# Nie da się tego wyliczyć na piechotę z amplitudy sygnału, bo librosa
# NIE normalizuje STFT przez sumę okna (poziom skaluje się z n_fft), a
# filtrbank mel z norm="slaney" dzieli każde pasmo przez jego szerokość
# w hercach (przy 32 pasmach na 400-1200 Hz to dzielenie przez ~48).
# Łącznie daje to ponad 40 dB przesunięcia względem rachunku z amplitudy —
# i dlatego wszystkie trzy poprzednie zestawy stałych (-30/0, -35/+5,
# -50/-10) były chybione, a obraz wychodził raz szary, raz przepalony.
DB_MIN = -30.0                  # to i wszystko cichsze -> 0.0
DB_MAX = 24.0                   # to i wszystko głośniejsze -> 1.0
#
# DB_MIN celowo NIE jest ustawione na 95 percentyl szumu (-21.5 dB), choć
# to dałoby idealnie czarne tło. Powód: prawdziwa słaba stacja bywa
# 20 dB cichsza od sygnału z generatora i przy progu -22 dB byłaby
# całkowicie niewidoczna. Przy -30 dB szum zajmuje dolne 6-16% skali —
# ciemno, ale nie zerowo, więc słaby sygnał ma gdzie się pokazać.
# Skala jest STAŁA, więc szum nie "puchnie" jak przy ref=np.max.
#
# DB_MAX z zapasem nad najsilniejszy sygnał w zbiorze, żeby szczyty nie
# były obcinane — inaczej różnice poziomu między stacjami przepadają.

# ==========================================================================
# 3. MORSE
# ==========================================================================
ALPHABET = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
N_CLASSES = len(ALPHABET)       # 37; indeks 0 = "puste radio" (tylko szum)

CHAR_TO_ID = {c: i for i, c in enumerate(ALPHABET)}
ID_TO_CHAR = {i: c for i, c in enumerate(ALPHABET)}

MORSE_DICT = {
    "A": ".-",    "B": "-...",  "C": "-.-.",  "D": "-..",   "E": ".",
    "F": "..-.",  "G": "--.",   "H": "....",  "I": "..",    "J": ".---",
    "K": "-.-",   "L": ".-..",  "M": "--",    "N": "-.",    "O": "---",
    "P": ".--.",  "Q": "--.-",  "R": ".-.",   "S": "...",   "T": "-",
    "U": "..-",   "V": "...-",  "W": ".--",   "X": "-..-",  "Y": "-.--",
    "Z": "--..",
    "0": "-----", "1": ".----", "2": "..---", "3": "...--", "4": "....-",
    "5": ".....", "6": "-....", "7": "--...", "8": "---..", "9": "----.",
}

# --- Timing ---------------------------------------------------------------
# Kropka [s] = 1.2 / WPM (słowo wzorcowe PARIS = 50 jednostek).
# Odstępy w jednostkach kropki: kreska 3, przerwa w znaku 1,
# przerwa między znakami 3, przerwa między słowami 7.
WPM = 20.0                      # tempo nominalne

# ZMIERZONE NA PRAWDZIWYM NAGRANIU — i to była najkosztowniejsza pomyłka
# w domyślnych wartościach tego projektu.
#
# Przy WPM_JITTER = 0 zbiór ma DOKŁADNIE jedno tempo: 20 WPM, kropka 60 ms.
# Pierwsze nagranie z pasma (Nagrywanie (3).wav, "CQ CQ CQ DE SQ2BVN") idzie
# 15 WPM — kropka 80 ms, czyli KAŻDY element o trzecią dłuższy niż cokolwiek,
# co model widział. Model o dokładności walidacyjnej 98,68% na syntetyku
# gubił na nim znaki krótkie (S ..., N -., D -.., E .), bo wyuczone wzorce
# były o trzecią za wąskie. Znak długi ma dość struktury, żeby przetrwać
# rozjazd tempa; krótki nie ma.
#
# 7 WPM rozrzutu daje 13-27 WPM, czyli zakres, w którym pracuje większość
# operatorów. Dolna granica nie jest przypadkowa: przy 13 WPM najdłuższy
# znak z przerwami zajmuje 2,31 s, a okno sieci ma 2,56 s. Poniżej 11,7 WPM
# przestaje się mieścić — pilnuje tego diag.py TEST 9.
WPM_JITTER = 7.0

DASH_UNITS = 3
GAP_ELEMENT_UNITS = 1
GAP_CHAR_UNITS = 3
GAP_WORD_UNITS = 7

# --- Kluczowanie ----------------------------------------------------------
# Narastanie/opadanie obwiedni tonu. Skok prostokątny daje trzaski (klucze),
# których widmo rozlewa się na dziesiątki herców po obu stronach tonu —
# w obrazie mel widać je jako pionowe smugi. 5 ms to wartość spotykana
# w transceiverach.
KEY_RAMP_MS = 5.0

# ==========================================================================
# 4. GENERATOR ZBIORU
# ==========================================================================
N_SAMPLES_DATASET = 40000       # liczba próbek w zbiorze
SILENCE_FRACTION = 0.15         # udział próbek "puste radio" (klasa 0)
CHARS_PER_CLIP = 3              # znaki w klipie; etykietą jest ŚRODKOWY
LABEL_INDEX = 1                 # który znak z CHARS_PER_CLIP jest etykietą

NOISE_RMS = 0.015               # RMS szumu tła — zawsze obecny, nie ma
                                # idealnej ciszy na żadnym odbiorniku

# ZMIANA WZGLĘDEM v5.6, gdzie było 0.18-0.40.
# Tamten zakres to sygnał o 22-28 dB nad szumem — bardzo czysty. Model
# uczony tylko na takim NIGDY nie widział słabej stacji i na pasmie
# zawodzi przy pierwszym sygnale z zaniku.
#
# Uwaga na rachunek SNR: liczy się SNR W PASMIE, nie szerokopasmowy.
# Pasmo mel przy 750 Hz ma ~48 Hz, a szum rozłożony jest na 4000 Hz,
# więc samo zwężenie pasma daje 10*log10(4000/48) = 19 dB zysku.
# Dlatego amplituda 0.03 (szerokopasmowo ~3 dB) w obrazie daje jeszcze
# ok. 24 dB kontrastu nad tłem — sygnał słaby, ale czytelny.
SIGNAL_AMP_MIN = 0.03           # słaba stacja
SIGNAL_AMP_MAX = 0.50           # mocna stacja lokalna

TONE_CENTER = 750.0             # środek pasma tonu [Hz]
TONE_SPREAD = 80.0              # +/- rozrzut, symuluje rozstrojenie odbiornika

SEED = 12345                    # ziarno generatora — powtarzalny zbiór

# ==========================================================================
# 4b. MODEL KANAŁU RADIOWEGO (dsp/radio.py)
#
# Sygnał ze stałą amplitudą, białym szumem, tonem co do herca stałym i
# maszynowym timingiem nie występuje na pasmie. Model uczony wyłącznie na
# takim materiale nie ma pojęcia o istnieniu zjawisk, które na falach
# krótkich są regułą.
#
# Każde zjawisko ma osobny parametr, żeby dało się zmierzyć, które z nich
# model przenosi, a które go przewraca — wystarczy wygenerować zbiór
# z jednym włączonym.
#
# Wyłączenie całości: generator --no-realism (daje zbiór jak v5.6).
# ==========================================================================

# --- Szum tła -------------------------------------------------------------
NOISE_TILT = (-1.0, 0.0)        # nachylenie widma szumu: -1 = różowy (1/f),
                                # 0 = biały. Szum odbiornika na niskich
                                # pasmach jest bliżej różowego.
NOISE_RMS_SPREAD = (0.5, 2.0)   # mnożnik NOISE_RMS na klip — poziom szumu
                                # zmienia się z pasmem, porą dnia i anteną

# --- QSB: zanik -----------------------------------------------------------
QSB_DEPTH = (0.0, 0.6)          # głębokość zaniku; 0.6 = spadek o 8 dB.
                                # Głębsze zaniki chowają sygnał pod szumem
                                # i etykieta przestaje odpowiadać obrazowi.

# --- QRM: inna stacja w pasmie -------------------------------------------
QRM_PROB = 0.35                 # udział klipów z drugą stacją
QRM_AMP = (0.05, 0.35)          # jej amplituda; bywa mocniejsza od naszej

# --- QRN: trzaski atmosferyczne ------------------------------------------
QRN_PROB = 0.30                 # udział klipów z trzaskami
QRN_MAX = 6                     # ile trzasków najwyżej w klipie
QRN_AMP = (0.1, 0.6)            # amplituda impulsu

# --- Nadajnik i operator --------------------------------------------------
# --- Nadajnik i operator ---------------------------------------------------
# Rozjazd timingu ma DWA źródła o różnym charakterze i nie wolno ich mieszać:
#
#   FIST         szum na pojedynczym elemencie, niezależny. Klucz sztorcowy
#                daje 0,1-0,3; manipulator elektroniczny praktycznie zero,
#                bo elementy odmierza układ.
#   FIST_DRIFT   WOLNE błądzenie tempa przez całe nadanie. "Jak łapa boli",
#                czyli operator zwalnia i przyspiesza w skali sekund.
#                To jest proces SKORELOWANY — modelowanie go jako szumu
#                niezależnego na elementach było błędem: dawało sygnał
#                nerwowy, ale o stałym tempie średnim, czego na kluczu
#                ręcznym nie ma.
#   GAP_JITTER   rozjazd PRZERW MIĘDZYZNAKOWYCH, osobno od elementów.
#                Przy manipulatorze z IC-7300 elementy idą równo, ale
#                przerwy między znakami zależą od tego, jak szybko operator
#                myśli. Więc jitter elementów i przerw muszą być rozdzielne.
FIST = (0.0, 0.25)              # szum na element (+/- udział długości)
FIST_DRIFT = (0.0, 0.15)        # zakres wolnego błądzenia tempa (+/- udział)
FIST_DRIFT_TAU_S = 2.0          # stała czasowa błądzenia [s]
GAP_JITTER = (0.0, 0.40)        # rozjazd przerw międzyznakowych

# --- Zapadanie zasilania nadajnika: CHIRP i spadek amplitudy --------------
# Naduszenie klucza obciąża zasilacz, kondensator się rozładowuje, napięcie
# spada, a generator LC przestraja się zgodnie z U=f(t). Efekt:
#   - ton spada W TRAKCIE elementu (750 Hz na początku kreski, 730 Hz
#     na końcu) i wraca w przerwie,
#   - z tego SAMEGO mechanizmu spada amplituda elementu.
# Oba objawy mają jedno źródło, więc modeluje je jeden stan zapadania.
#
# To NIE jest to samo co dryf VFO. Dryf jest wolny i niezależny od klucza;
# chirp jest szybki i skorelowany z kluczem. Poprzednia wersja miała tylko
# dryf liniowy przez cały klip, czyli zjawisko innego rodzaju.
#
# W raporcie RST opisuje to TRZECIA litera (Tone).
CHIRP_HZ = (0.0, 25.0)          # ile herców spada ton w trakcie elementu
SAG_DB = (0.0, 3.0)             # ile decybeli spada amplituda elementu
SAG_TAU_MS = (20.0, 120.0)      # stała czasowa zapadania i powrotu

# --- Przydźwięk sieci na obwiedni w.cz. -----------------------------------
# Puste kondensatory filtrujące w zasilaczu nadajnika przenoszą tętnienie
# na obwiednię nośnej. Przy prostowaniu dwupołówkowym to 100 Hz, przy
# jednopołówkowym 50 Hz.
HUM_PROB = 0.20                 # udział nadań z przydźwiękiem
HUM_DEPTH = (0.05, 0.35)        # głębokość modulacji
HUM_HZ = (50.0, 100.0)          # do wyboru: 50 albo 100 Hz

# --- Dryf VFO (wolny, niezależny od klucza) -------------------------------
DRIFT_HZ = 8.0                  # dryf tonu w hercach na długość klipu (+/-)

# --- ARW odbiornika --------------------------------------------------------
# Automatyczna regulacja wzmocnienia ma własną stałą czasową i ściska
# dynamikę. Przy szybkim ARW obwiednia pierwszego elementu po przerwie jest
# przesterowana, potem wzmocnienie schodzi. To kolejne, po QSB i zapadaniu
# zasilania, źródło bujania amplitudy — i inne pod względem skali czasu.
AGC_PROB = 0.5                  # udział klipów z symulowanym ARW
AGC_TAU_MS = (50.0, 500.0)      # stała czasowa ARW
AGC_DEPTH = (0.2, 0.8)          # jak mocno ARW ściska dynamikę

# Format zapisu obrazów w zbiorze. "uint8" kwantuje [0,1] na 256 poziomów,
# czyli 40 dB / 255 = 0,16 dB na krok — poniżej rozdzielczości, jaką ma
# jakikolwiek tor odbiorczy. Zysk jest realny: 40000 obrazów 128x32 to
# 655 MB w float32, a 164 MB w uint8. Stary zbiór v5_6 zajmował 1,3 GB
# i wymagał osobnego trenera "out of memory".
STORE_DTYPE = "uint8"           # "uint8" albo "float32"

# ==========================================================================
# 5. ŚCIEŻKI
# ==========================================================================
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "out"
DATASET_PATH = OUT_DIR / "morse_dataset.npz"
MODEL_PATH = OUT_DIR / "morse_model.keras"      # format .keras, nie .h5
                                                # (.h5 jest w Keras 3 przestarzały
                                                #  i nie zapisuje własnych warstw)

# ==========================================================================
# 6. MIKROFON
# ==========================================================================
MIC_DEVICE = None               # None = urządzenie domyślne systemu;
                                # numer/nazwę wypisze `python -m tools.mic2wav --list`
MIC_BLOCK = 800                 # próbki na blok callbacku (100 ms przy 8 kHz)
MIC_CHANNELS = 1


# ==========================================================================
# Funkcje pomocnicze wyprowadzone ze stałych — żeby nikt nie liczył ich sam
# ==========================================================================
def dot_seconds(wpm: float = WPM) -> float:
    """Długość kropki w sekundach."""
    return 1.2 / float(wpm)


def dot_samples(wpm: float = WPM, sr: int = SR) -> int:
    """Długość kropki w próbkach audio."""
    return int(round(dot_seconds(wpm) * sr))


def frames_per_second() -> float:
    """Ile ramek spektrogramu przypada na sekundę."""
    return SR / HOP_LENGTH


def img_seconds() -> float:
    """Ile sekund sygnału widzi sieć w jednym obrazie."""
    return IMG_FRAMES * HOP_LENGTH / SR


def db_span() -> float:
    """Rozpiętość skali normalizacji w dB."""
    return DB_MAX - DB_MIN


def fingerprint() -> dict:
    """Odcisk parametrów FRONT-ENDU — tych, których zmiana unieważnia dane.

    Generator zapisuje ten słownik w pliku zbioru, a trener i podgląd
    porównują go ze swoim. Jeżeli ktoś zmieni DB_MIN albo N_MELS i zapomni
    wygenerować zbiór od nowa, dostanie jasny komunikat zamiast modelu,
    który po cichu uczy się na niespójnych danych.

    Świadomie NIE ma tu N_SAMPLES_DATASET ani SEED — to parametry zbioru,
    nie kontraktu front-endu, i mogą się różnić bez szkody.
    """
    return {
        "SR": SR,
        "CLIP_SECONDS": CLIP_SECONDS,
        "N_FFT": N_FFT,
        "HOP_LENGTH": HOP_LENGTH,
        "N_MELS": N_MELS,
        "FMIN": FMIN,
        "FMAX": FMAX,
        "IMG_FRAMES": IMG_FRAMES,
        "IMG_BINS": IMG_BINS,
        "DB_REF": DB_REF,
        "DB_MIN": DB_MIN,
        "DB_MAX": DB_MAX,
        "ALPHABET": ALPHABET,
        "MEL_CONVENTION": "librosa/slaney",
    }


def fingerprint_str() -> str:
    """Odcisk jako jeden wiersz tekstu — do zapisu w npz i do porównań."""
    fp = fingerprint()
    return ";".join(f"{k}={fp[k]}" for k in sorted(fp))


def check_fingerprint(other: str, source: str = "zbiór") -> None:
    """Porównuje odcisk z zapisanym. Wypisuje RÓŻNICE, nie samo "nie zgadza się"."""
    mine = fingerprint_str()
    if other == mine:
        return

    def parse(s):
        return dict(kv.split("=", 1) for kv in s.split(";") if "=" in kv)

    a, b = parse(other), parse(mine)
    diffs = [f"  {k}: {source}={a.get(k, '-')}  config.py={b.get(k, '-')}"
             for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    raise ValueError(
        f"{source} powstał na innych parametrach front-endu niż obecny "
        f"config.py:\n" + "\n".join(diffs) +
        "\n\nAlbo przywróć te wartości w config.py, albo wygeneruj dane "
        "od nowa:\n  python -m tools.generator"
    )


def summary() -> str:
    """Zwarty opis konfiguracji — wypisywany przez każde narzędzie na starcie,
    żeby w logu treningu było widać, na jakich parametrach powstały dane."""
    return (
        f"SR={SR} Hz  klip={CLIP_SECONDS}s  "
        f"mel: n_fft={N_FFT} hop={HOP_LENGTH} n_mels={N_MELS} "
        f"pasmo={FMIN:.0f}-{FMAX:.0f} Hz\n"
        f"obraz={IMG_FRAMES}x{IMG_BINS} ({img_seconds():.2f}s, "
        f"{frames_per_second():.0f} ramek/s)  "
        f"dB: ref={DB_REF} zakres=[{DB_MIN}, {DB_MAX}]\n"
        f"morse: WPM={WPM} kropka={dot_seconds()*1000:.0f}ms "
        f"ton={TONE_CENTER:.0f}+/-{TONE_SPREAD:.0f}Hz  klas={N_CLASSES}"
    )
