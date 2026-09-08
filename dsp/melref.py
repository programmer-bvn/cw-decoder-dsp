"""FRONT-END BEZ LIBROSY — wzory jawne, do portu na KV260.

PO CO TO ISTNIEJE
-----------------
Docelowo model idzie na Kria KV260. Na PetaLinuksie w produkcji nie będzie
librosy (ciągnie numbę, scipy, soundfile i lazy_loader), a front-end trzeba
policzyć na ARM-ie albo w logice programowalnej. Nowa implementacja musi
dawać DOKŁADNIE te same liczby co treningowa — inaczej wracamy do usterki,
od której cały ten projekt się zaczął, tylko tym razem na sprzęcie i bez
możliwości porównania.

Dlatego wszystkie wzory są tu wypisane, a diag.py porównuje wynik z librosą
próbka po próbce. Ten plik jest wzorcem do przepisania na C.

TRZY KONWENCJE, KTÓRE MUSZĄ SIĘ ZGADZAĆ CO DO CYFRY
---------------------------------------------------
1. SKALA MEL. librosa domyślnie używa skali SLANEYA (htk=False), nie HTK.
   Slaney jest LINIOWA poniżej 1000 Hz (mel = f / (200/3)) i logarytmiczna
   powyżej. HTK jest logarytmiczna wszędzie (mel = 1127*ln(1+f/700)).
   Przy pasmie 400-1200 Hz różnica rozkłada pasma inaczej — filtrbank HTK
   dałby inne liczby i model straciłby dokładność bez żadnego komunikatu.

2. NORMALIZACJA PASM. librosa domyślnie norm="slaney": każdy trójkąt jest
   dzielony przez swoją szerokość w hercach (2/(f_gora - f_dol)), więc
   pasma mają jednakowe POLE, nie jednakową wysokość. Filtrbank
   tf.signal.linear_to_mel_weight_matrix ma trójkąty o wysokości 1 i tego
   NIE robi — to drugi powód, dla którego nie wolno podmienić jednego
   na drugi.

3. OKNO I DOPEŁNIENIE. Okno Hanna w wariancie OKRESOWYM (periodic,
   fftbins=True), sygnał dopełniony po obu stronach o n_fft//2. Rodzaj
   dopełnienia (zerami czy odbiciem) jest ustalany empirycznie w
   diag.py — od librosy 0.10 domyślne jest dopełnienie zerami, ale to
   szczegół, który zmieniał się między wersjami, więc go sprawdzamy,
   a nie zakładamy.
"""

from __future__ import annotations

import numpy as np

from . import config as C

# --------------------------------------------------------------------------
# Skala mel Slaneya — dokładnie jak w librosa.mel_frequencies(htk=False)
# --------------------------------------------------------------------------
_F_SP = 200.0 / 3.0            # herców na mel w części liniowej
_MIN_LOG_HZ = 1000.0           # próg przejścia na logarytm
_MIN_LOG_MEL = _MIN_LOG_HZ / _F_SP                    # = 15.0
_LOGSTEP = np.log(6.4) / 27.0  # tak, żeby 6,4 kHz wypadło 27 mel wyżej


def hz_to_mel(f):
    """Herce -> mele w skali Slaneya. Liniowo do 1000 Hz, potem logarytm."""
    f = np.asarray(f, dtype=np.float64)
    mel = f / _F_SP
    hi = f >= _MIN_LOG_HZ
    if np.any(hi):
        mel = np.where(hi,
                       _MIN_LOG_MEL + np.log(np.maximum(f, 1e-30)
                                             / _MIN_LOG_HZ) / _LOGSTEP,
                       mel)
    return mel


def mel_to_hz(m):
    """Odwrotność hz_to_mel()."""
    m = np.asarray(m, dtype=np.float64)
    hz = m * _F_SP
    hi = m >= _MIN_LOG_MEL
    if np.any(hi):
        hz = np.where(hi, _MIN_LOG_HZ * np.exp(_LOGSTEP * (m - _MIN_LOG_MEL)),
                      hz)
    return hz


def mel_frequencies(n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """n_mels + 2 krawędzi pasm, równomiernie w skali mel."""
    return mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax),
                                 n_mels + 2))


# --------------------------------------------------------------------------
# Filtrbank
# --------------------------------------------------------------------------
def mel_filterbank(sr: int = C.SR, n_fft: int = C.N_FFT,
                   n_mels: int = C.N_MELS, fmin: float = C.FMIN,
                   fmax: float = C.FMAX) -> np.ndarray:
    """Macierz [n_mels, n_fft//2 + 1] — jak librosa.filters.mel(norm="slaney").

    Trójkąty liczone są w HERCACH (nie w melach — to różnica względem
    tf.signal), a potem dzielone przez szerokość pasma, żeby miały równe
    pole.
    """
    n_bins = n_fft // 2 + 1
    fft_hz = np.linspace(0.0, sr / 2.0, n_bins, dtype=np.float64)
    edges = mel_frequencies(n_mels, fmin, fmax)          # n_mels + 2

    fb = np.zeros((n_mels, n_bins), dtype=np.float64)
    diff = np.diff(edges)                                # szerokości odcinków
    ramps = edges[:, None] - fft_hz[None, :]             # [n_mels+2, n_bins]

    for i in range(n_mels):
        lower = -ramps[i] / diff[i]                      # zbocze narastające
        upper = ramps[i + 2] / diff[i + 1]               # zbocze opadające
        fb[i] = np.maximum(0.0, np.minimum(lower, upper))

    # norm="slaney": jednakowe POLE pasma, nie jednakowa wysokość.
    enorm = 2.0 / (edges[2:n_mels + 2] - edges[:n_mels])
    fb *= enorm[:, None]
    return fb.astype(np.float64)


# --------------------------------------------------------------------------
# STFT
# --------------------------------------------------------------------------
def hann_periodic(n: int) -> np.ndarray:
    """Okno Hanna OKRESOWE (fftbins=True) — takie bierze librosa do analizy.
    Wariant symetryczny (np.hanning) różni się jedną próbką i daje inne
    liczby."""
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


def stft_power(audio: np.ndarray, n_fft: int = C.N_FFT,
               hop: int = C.HOP_LENGTH, center: bool = True,
               pad_mode: str = "constant") -> np.ndarray:
    """Widmo MOCY, [ramki, n_fft//2 + 1].

    center=True dopełnia sygnał o n_fft//2 z obu stron, więc ramka i jest
    wyśrodkowana na próbce i*hop, a liczba ramek to 1 + len//hop.
    """
    x = np.asarray(audio, dtype=np.float64)
    if center:
        pad = n_fft // 2
        if pad_mode == "reflect":
            x = np.pad(x, pad, mode="reflect")
        else:
            x = np.pad(x, pad, mode="constant")

    n_frames = 1 + (x.size - n_fft) // hop
    win = hann_periodic(n_fft)

    # Ramkowanie bez kopiowania: widok o zadanym kroku w pamięci.
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n_frames, n_fft),
        strides=(x.strides[0] * hop, x.strides[0]))

    spec = np.fft.rfft(frames * win, n=n_fft, axis=1)
    return (spec.real ** 2 + spec.imag ** 2)


# --------------------------------------------------------------------------
# Cała ścieżka — odpowiednik frontend.to_net_image()
# --------------------------------------------------------------------------
def melspec_power(audio: np.ndarray, sr: int = C.SR) -> np.ndarray:
    """Widmo mocy w pasmach mel, [ramki, n_mels]."""
    p = stft_power(audio, C.N_FFT, C.HOP_LENGTH)
    fb = mel_filterbank(sr, C.N_FFT, C.N_MELS, C.FMIN, C.FMAX)
    return (p @ fb.T).astype(np.float32)


def power_to_db(mel_power: np.ndarray, ref: float = C.DB_REF,
                amin: float = 1e-10) -> np.ndarray:
    """Moc -> dB względem stałego odniesienia. BEZ top_db.

    librosa.power_to_db ma domyślnie top_db=80, co obcina wynik względem
    maksimum klipu i po cichu przywraca skalę ruchomą. Tutaj takiego
    parametru nie ma — nie da się go włączyć przez pomyłkę.
    """
    mp = np.maximum(amin, np.asarray(mel_power, dtype=np.float64))
    return (10.0 * np.log10(mp) - 10.0 * np.log10(max(amin, ref))
            ).astype(np.float32)


def to_net_image(audio: np.ndarray, sr: int = C.SR) -> np.ndarray:
    """audio -> [IMG_FRAMES, IMG_BINS] w [0, 1]. Bez librosy."""
    db = power_to_db(melspec_power(audio, sr))
    img = np.clip((db - C.DB_MIN) / C.db_span(), 0.0, 1.0).astype(np.float32)

    n = img.shape[0]
    if n == C.IMG_FRAMES:
        return img
    if n > C.IMG_FRAMES:
        s = (n - C.IMG_FRAMES) // 2
        return img[s:s + C.IMG_FRAMES]
    pad = C.IMG_FRAMES - n
    return np.pad(img, ((pad // 2, pad - pad // 2), (0, 0)),
                  mode="constant")
