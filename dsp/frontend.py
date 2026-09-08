"""FRONT-END: audio -> obraz dla sieci. Jedna implementacja dla całego projektu.

Wszystkie cztery narzędzia wołają tę samą funkcję to_net_input(). Nie ma
drugiej ścieżki, nie ma "wersji dla live" i "wersji dla treningu" — bo
właśnie z tego brały się rozbieżności normalizacji między generatorem
a dekoderem.

    audio (float32, SR Hz)
      -> melspec_power()      widmo mocy w pasmach mel
      -> power_to_db()        skala dB, STAŁE ref=DB_REF
      -> normalize_db()       liniowo z [DB_MIN, DB_MAX] na [0, 1], obcięte
      -> center_window()      wycięcie IMG_FRAMES ramek ze środka
      -> [IMG_FRAMES, IMG_BINS, 1]

DWIE PUŁAPKI, KTÓRE SIĘ TU ZAMYKA
---------------------------------
1. librosa.power_to_db() ma DOMYŚLNIE top_db=80, co obcina wynik względem
   MAKSIMUM danego klipu. To po cichu przywraca skalę ruchomą, nawet gdy
   podało się stałe ref=1.0. Tutaj jest jawnie top_db=None.

2. librosa.feature.melspectrogram() domyślnie używa skali Slaneya
   (htk=False) i normalizacji pasm norm="slaney". TensorFlow (tf.signal)
   używa HTK bez normalizacji. Te dwa filtrbanki dają RÓŻNE liczby.
   Trzymamy się domyślnych ustawień librosy, bo na nich powstały
   dotychczasowe zbiory — ale robimy to jawnie, nie przez przypadek.
"""

from __future__ import annotations

import numpy as np

from . import config as C
from . import melref

try:
    import librosa
except ImportError as exc:                            # pragma: no cover
    raise ImportError(
        "Brak librosy. Zainstaluj: pip install librosa soundfile\n"
        "(w tym projekcie librosa służy już tylko do wczytywania plików "
        "audio i jako wzorzec w diag.py TEST 14 — sam front-end liczy "
        "dsp/melref.py na numpy)"
    ) from exc


# --------------------------------------------------------------------------
# Wejście: plik
# --------------------------------------------------------------------------
def load_audio(path, sr: int = C.SR) -> np.ndarray:
    """Wczytuje plik audio jako mono float32 o częstotliwości sr.

    Miksowanie do mono i decymacja są tu, a nie w narzędziach — wyjście USB
    z IC-7300 to 48 kHz stereo, mikrofon bywa 44,1 kHz, a sieć oczekuje
    zawsze 8 kHz mono.
    """
    audio, _ = librosa.load(str(path), sr=sr, mono=True)
    return np.asarray(audio, dtype=np.float32)


# --------------------------------------------------------------------------
# Krok 1-2: widmo mocy w pasmach mel, potem dB o stałym odniesieniu
# --------------------------------------------------------------------------
def melspec_power(audio: np.ndarray, sr: int = C.SR) -> np.ndarray:
    """Widmo mocy w pasmach mel. Wynik [ramki, N_MELS] (czas w pierwszej osi).

    Liczone przez dsp/melref.py — na samym numpy, bez librosy. Powody:

      - środowisko treningowe (TF 2.10) librosy nie ma;
      - docelowy KV260 też jej nie będzie miał, a front-end musi tam dać
        TE SAME liczby;
      - jedna implementacja zamiast dwóch, które mogą się rozjechać.

    Zgodność z librosą jest sprawdzana niezależnie w diag.py (TEST 14):
    krawędzie pasm 0,0 Hz różnicy, cały obraz 1,2e-07 = 6e-06 dB.
    Librosa zostaje w projekcie tylko do WCZYTYWANIA plików audio
    (load_audio) i jako wzorzec w tym teście.
    """
    return melref.melspec_power(audio, sr)


def power_to_db(mel_power: np.ndarray) -> np.ndarray:
    """Moc -> dB względem STAŁEGO odniesienia DB_REF, bez obcięcia względnego.

    W melref.power_to_db nie ma parametru top_db — nie da się go włączyć
    przez pomyłkę. W librosie jest domyślnie na 80 i obcina wynik względem
    maksimum klipu, po cichu przywracając skalę ruchomą.
    """
    return melref.power_to_db(mel_power, ref=C.DB_REF)


# --------------------------------------------------------------------------
# Krok 3: normalizacja do [0, 1] na stałej skali
# --------------------------------------------------------------------------
def normalize_db(mel_db: np.ndarray) -> np.ndarray:
    """[DB_MIN, DB_MAX] dB -> [0, 1]. Wszystko poza zakresem jest obcinane.

    Skala jest BEZWZGLĘDNA: ten sam poziom sygnału daje ten sam odcień
    niezależnie od tego, co jeszcze jest w klipie. Dzięki temu cisza jest
    czarna, a nie rozciągnięta do szarości jak przy ref=np.max.
    """
    img = (mel_db - C.DB_MIN) / C.db_span()
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def denormalize_db(img: np.ndarray) -> np.ndarray:
    """Odwrotność normalize_db() — do opisu osi w decybelach w X-Ray."""
    return (np.asarray(img, dtype=np.float32) * C.db_span() + C.DB_MIN)


# --------------------------------------------------------------------------
# Krok 4: wycięcie okna
# --------------------------------------------------------------------------
def center_window(img: np.ndarray, frames: int = C.IMG_FRAMES) -> np.ndarray:
    """Wycina `frames` ramek ze ŚRODKA obrazu; krótszy dopełnia zerami.

    Wycinanie po środku, bo generator umieszcza znak docelowy w środku klipu.
    Dopełnianie zerami (a nie powtarzaniem krawędzi) jest zgodne z tym, co
    normalize_db() daje dla ciszy — zero to poziom DB_MIN, czyli czarne tło.
    """
    img = np.asarray(img, dtype=np.float32)
    n = img.shape[0]

    if n == frames:
        return img
    if n > frames:
        start = (n - frames) // 2
        return img[start:start + frames]

    pad = frames - n
    before = pad // 2
    after = pad - before
    return np.pad(img, ((before, after), (0, 0)), mode="constant")


def clip_frames(n_samples: int = C.CLIP_SAMPLES) -> int:
    """Ile ramek mel daje klip o zadanej długości.

    librosa z center=True dopełnia sygnał po obu stronach i zwraca
    1 + n//hop ramek. Ta jedna dodatkowa ramka jest źródłem pomyłek przy
    przeliczaniu czasu na numer ramki, dlatego wzór jest tu, a nie
    rozsiany po narzędziach.
    """
    return 1 + n_samples // C.HOP_LENGTH


def window_crop_start(n_samples: int = C.CLIP_SAMPLES) -> int:
    """Numer pierwszej ramki klipu, która trafia do okna sieci."""
    return max(0, (clip_frames(n_samples) - C.IMG_FRAMES) // 2)


def sample_to_window_frame(sample: float,
                           n_samples: int = C.CLIP_SAMPLES) -> float:
    """Próbka audio w klipie -> numer ramki w OKNIE SIECI (może być ujemny
    albo powyżej IMG_FRAMES, gdy zdarzenie wypadło za kadrem)."""
    return sample / C.HOP_LENGTH - window_crop_start(n_samples)


# --------------------------------------------------------------------------
# CAŁA ŚCIEŻKA — jedyne wejście dla wszystkich narzędzi
# --------------------------------------------------------------------------
def to_net_image(audio: np.ndarray, sr: int = C.SR) -> np.ndarray:
    """audio -> [IMG_FRAMES, IMG_BINS] float32 w zakresie [0, 1].

    To jest kanoniczna definicja "co widzi sieć". Generator zapisuje wynik
    tej funkcji do zbioru, X-Ray go rysuje, wav2net podaje modelowi.
    """
    mel = melspec_power(audio, sr)
    db = power_to_db(mel)
    img = normalize_db(db)
    return center_window(img)


def to_net_input(audio: np.ndarray, sr: int = C.SR) -> np.ndarray:
    """Jak to_net_image(), ale z osią partii i kanału: [1, ramki, pasma, 1].
    Gotowe do podania wprost do model.predict()."""
    return to_net_image(audio, sr)[np.newaxis, ..., np.newaxis]


def batch_to_net_input(images: np.ndarray) -> np.ndarray:
    """Stos obrazów [n, ramki, pasma] -> [n, ramki, pasma, 1]."""
    images = np.asarray(images, dtype=np.float32)
    if images.ndim == 4:
        return images
    return images[..., np.newaxis]


# --------------------------------------------------------------------------
# Tryb ciągły: przesuwany wodospad
# --------------------------------------------------------------------------
class Waterfall:
    """Bufor przesuwny dla pracy na żywo. Trzyma AUDIO, nie obraz.

    DLACZEGO NIE BUFOR OBRAZU
    -------------------------
    Naturalne wydaje się liczenie mel dla każdego bloku 100 ms i wsuwanie
    nowych ramek do obrazu (tak działał LiveDecoder). To nie daje tego
    samego wyniku, co przetworzenie ciągłego sygnału:

      - librosa domyślnie dopełnia sygnał zerami po obu stronach (center=True),
        więc KAŻDY blok dostaje sztuczne zerowe krawędzie w środku strumienia;
      - okno FFT ma 512 próbek, a blok 800 — element klucza wypadający na
        styku bloków jest widziany dwa razy, za każdym razem w połowie.

    Efekt: obraz na żywo różni się od obrazu, na którym model się uczył,
    i to najbardziej właśnie w miejscach zmiany klucza, czyli tam, gdzie
    siedzi cała informacja.

    Tutaj bufor przechowuje surowe audio o długości okna sieci (plus zapas
    na okno FFT), a mel liczy się od nowa przy każdym wywołaniu. Wynik jest
    IDENTYCZNY z tym, co dałby ten sam fragment wczytany z pliku — sprawdza
    to diag.py (TEST 6). Koszt: 128 ramek mel na blok, rzecz nieistotna
    przy 10 blokach na sekundę.
    """

    def __init__(self, frames: int = C.IMG_FRAMES, bins: int = C.IMG_BINS):
        self.frames = frames
        self.bins = bins
        # Długość bufora musi dawać DOKŁADNIE `frames` ramek, bez obcinania.
        # librosa z center=True zwraca 1 + n//hop ramek, więc n = (frames-1)*hop.
        # Gdyby bufor był dłuższy, center_window() obcięłoby po kilka ramek
        # z obu stron i ostatnia ramka "na żywo" nie odpowiadałaby ostatniej
        # ramce z pliku — obraz byłby przesunięty w czasie o te kilka ramek.
        self.buf_len = (frames - 1) * C.HOP_LENGTH
        self.audio = np.zeros(self.buf_len, dtype=np.float32)
        self.filled = 0

    def push(self, audio_block: np.ndarray, sr: int = C.SR) -> np.ndarray:
        """Dokłada blok audio i zwraca aktualny obraz [ramki, pasma]."""
        block = np.asarray(audio_block, dtype=np.float32).ravel()
        n = block.size
        if n >= self.buf_len:
            self.audio[:] = block[-self.buf_len:]
        else:
            self.audio[:-n] = self.audio[n:]
            self.audio[-n:] = block
        self.filled = min(self.buf_len, self.filled + n)
        return self.image(sr)

    def image(self, sr: int = C.SR) -> np.ndarray:
        """Obraz z aktualnej zawartości bufora — tą samą ścieżką co plik.

        Bez center_window(): długość bufora jest dobrana tak, że mel daje
        od razu `frames` ramek. Gdyby tu było obcinanie, ostatnia ramka
        obrazu nie leżałaby na końcu bufora i predykcja dotyczyłaby
        chwili sprzed kilkudziesięciu milisekund.
        """
        img = normalize_db(power_to_db(melspec_power(self.audio, sr)))
        if img.shape[0] != self.frames:
            raise RuntimeError(
                f"bufor dał {img.shape[0]} ramek, oczekiwano {self.frames} — "
                f"zmienił się HOP_LENGTH albo konwencja dopełniania w librosie"
            )
        return img

    def net_input(self, sr: int = C.SR) -> np.ndarray:
        return self.image(sr)[np.newaxis, ..., np.newaxis]

    def ready(self) -> bool:
        """Czy bufor jest już wypełniony sygnałem, a nie zerami startowymi.
        Dopóki nie jest, predykcje dotyczą w części sztucznej ciszy."""
        return self.filled >= self.buf_len

    def reset(self) -> None:
        self.audio[:] = 0.0
        self.filled = 0


# --------------------------------------------------------------------------
# Kontrola kontraktu
# --------------------------------------------------------------------------
def check_image(img: np.ndarray, where: str = "") -> None:
    """Sprawdza kształt i zakres obrazu. Wołane przez narzędzia po wczytaniu
    zbioru — niezgodny kształt lepiej wykryć tutaj niż po godzinie treningu."""
    tag = f" ({where})" if where else ""
    exp = (C.IMG_FRAMES, C.IMG_BINS)
    got = tuple(img.shape[-3:-1]) if img.ndim >= 3 else tuple(img.shape)
    if got != exp:
        raise ValueError(f"zły kształt obrazu{tag}: {got}, oczekiwano {exp}")
    lo, hi = float(np.min(img)), float(np.max(img))
    if lo < -1e-6 or hi > 1.0 + 1e-6:
        raise ValueError(
            f"obraz poza zakresem [0,1]{tag}: min={lo:.3f} max={hi:.3f} — "
            "to znak, że dane powstały inną normalizacją niż w config.py"
        )
