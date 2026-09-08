"""ŚLEDZENIE TONU: zaczep się na jedną stację i TRZYMAJ ją.

Front-end patrzy w pasmo FMIN..FMAX, czyli 800 Hz szerokości. Na zapchanym
pasmie mieści się w nim kilka stacji — zmierzone na raw_radio_14_020MHz.wav:
845 Hz przy 0 dB i 790 Hz przy -5,4 dB, czyli 55 Hz od siebie. Model uczono,
że "nasza" stacja jest przy 750 +/- 80 Hz i że jest jedna. Przy dwóch
naraz odczyt rozsypuje się na serie '5' (.....).

DLACZEGO NIE "NAJMOCNIEJSZY TON W OKNIE"
----------------------------------------
Najprostsze rozwiązanie — w każdym oknie znajdź najmocniejszy prążek
i przesuń go na 750 Hz — jest BŁĘDNE, i to nie subtelnie. Na falach
krótkich zanik (QSB) regularnie sprawia, że korespondent słabnie,
a przypadkowa stacja obok wychodzi na prowadzenie w słyszalności.
Dekoder wybierający maksimum przeskoczyłby wtedy na obcą stację w środku
nadania. Operator tego nie robi: raz dostrojony trzyma się swojego tonu
i przeczekuje zanik.

Stąd konstrukcja jak w pętli fazowej, z trzema stanami:

    SZUKANIE  brak zaczepu; wybieramy najmocniejszy prążek w pasmie,
              ale tylko gdy wystaje ponad tło o ACQUIRE_SNR_DB
    ZACZEP    szukamy prążka WYŁĄCZNIE w okolicy +/-CAPTURE_HZ od
              bieżącej estymaty i uśredniamy ją wykładniczo. Stacja
              mocniejsza, ale odległa, jest ignorowana — to jest sedno
    WYBIEG    w oknie nie ma nic ponad progiem; TRZYMAMY ostatnią
              częstotliwość do HOLD_S sekund. Dłuższa cisza zwalnia zaczep

Wąskie pasmo zaczepu (CAPTURE_HZ) pozwala nadążyć za dryfem VFO — model
kanału zakłada do 8 Hz na klip 4 s, czyli 2 Hz/s — a jednocześnie nie daje
przeskoczyć na stację 55 Hz dalej.
"""

from __future__ import annotations

import numpy as np

from . import config as C

# --- parametry pętli ---------------------------------------------------
BLOCK_S = 0.20            # okno analizy częstotliwości [s]; 5 Hz rozdzielczości
STEP_S = 0.05             # krok analizy [s]
ACQUIRE_SNR_DB = 8.0      # ile ponad tło musi wystawać prążek, żeby się zaczepić
# Próg utrzymania zaczepu. Niżej niż przy szukaniu, bo sygnał w zaniku
# słabnie, a pętla ma go wtedy TRZYMAĆ. 6 dB nad medianą pasma to poziom,
# przy którym szum w przerwie nie utrzymuje zaczepu, a słaby sygnał jeszcze
# tak. (Podniesienie z 3 na 6 dB nie zmieniło zachowania na nagraniach
# testowych — zostaje jako wartość ostrożniejsza, nie jako naprawa.)
TRACK_SNR_DB = 6.0
# Połowa szerokości okna śledzenia. 12 Hz z zapasem pokrywa dryf VFO
# (model kanału zakłada 8 Hz na klip 4 s, czyli 2 Hz/s, a krok pętli to
# 0,05 s), a nie pozwala przejść na stację odległą o kilkadziesiąt herców.
# ZMIERZONE: przy 25 Hz pętla na raw_radio_14_020MHz.wav przewędrowała
# 95 Hz między dwiema stacjami (751-845 Hz) — czyli zrobiła dokładnie to,
# czego ta konstrukcja ma nie robić.
CAPTURE_HZ = 12.0
HOLD_S = 3.0              # jak długo trzymamy ton bez sygnału (wybieg)
SMOOTH = 0.25             # waga nowej estymaty; mniejsza = spokojniejsza pętla


def _spectrum(block: np.ndarray, sr: int, nfft: int):
    win = np.hanning(block.size)
    sp = np.abs(np.fft.rfft(block * win, n=nfft)) ** 2
    return sp, np.fft.rfftfreq(nfft, 1.0 / sr)


def track_tone(audio: np.ndarray, sr: int = C.SR,
               fmin: float = C.FMIN, fmax: float = C.FMAX,
               block_s: float = BLOCK_S, step_s: float = STEP_S,
               capture_hz: float = CAPTURE_HZ, hold_s: float = HOLD_S
               ) -> tuple[np.ndarray, np.ndarray]:
    """Zwraca (czasy, częstotliwość) — estymatę tonu w kolejnych krokach.

    Częstotliwość jest NaN, gdy pętla nie ma zaczepu. W czasie wybiegu
    zwracana jest ostatnia znana wartość, bo dokładnie tak zachowuje się
    operator: przy zaniku nie przestraja odbiornika.
    """
    audio = np.asarray(audio, dtype=np.float64)
    nb = max(64, int(round(block_s * sr)))
    ns = max(1, int(round(step_s * sr)))
    nfft = 1 << int(np.ceil(np.log2(nb * 4)))       # nadpróbkowanie widma

    times, freqs = [], []
    f_est = np.nan
    silent_s = 0.0

    for start in range(0, max(1, audio.size - nb), ns):
        block = audio[start:start + nb]
        if block.size < nb:
            break
        sp, f = _spectrum(block, sr, nfft)

        band = (f >= fmin) & (f <= fmax)
        if not band.any():
            break
        # Tło: mediana mocy w pasmie. Odporna na jeden silny prążek.
        # Podłoga 1e-30 nie wystarcza: dla sygnału syntetycznego, w którym
        # połowa pasma jest dokładnie zerowa, mediana wychodzi 0 i log10
        # daje dzielenie przez zero. Skalujemy podłogę do mocy sygnału.
        floor = float(np.median(sp[band]))
        if floor <= 0.0:
            floor = max(float(sp[band].max()), 1e-30) * 1e-12

        if np.isnan(f_est):
            # --- SZUKANIE: cały pasmo, wysoki próg ---
            k = int(np.flatnonzero(band)[np.argmax(sp[band])])
            snr = 10.0 * np.log10(max(float(sp[k]), 1e-30) / floor)
            if snr >= ACQUIRE_SNR_DB:
                f_est = _parabolic(sp, f, k)
                silent_s = 0.0
        else:
            # --- ZACZEP: tylko okolica bieżącej estymaty ---
            near = band & (np.abs(f - f_est) <= capture_hz)
            if near.any():
                k = int(np.flatnonzero(near)[np.argmax(sp[near])])
                snr = 10.0 * np.log10(max(float(sp[k]), 1e-30) / floor)
            else:
                snr = -99.0
            if snr >= TRACK_SNR_DB:
                f_new = _parabolic(sp, f, k)
                f_est = (1.0 - SMOOTH) * f_est + SMOOTH * f_new
                # Licznik wybiegu MALEJE, a nie zeruje się, żeby sporadyczne
                # trafienie w szczyt szumu nie utrzymywało zaczepu przez
                # długą ciszę.
                #
                # UCZCIWA UWAGA: dodałem to, szukając przyczyny szerokiego
                # zakresu tonu na 4-minutowym nagraniu (621-825 Hz).
                # Nie pomogło — bo przyczyny nie było w pętli. Wykres
                # pokazał, że w tych fragmentach nagranie NIE ZAWIERA
                # kluczowanego sygnału (poziom ciągły -5 dB zamiast skoków
                # -25/+20 dB), a pętla wiernie śledziła to, co tam jest.
                # Zmiana zostaje, bo jest poprawna sama w sobie, ale nie
                # rozwiązała problemu, którego nie było.
                silent_s = max(0.0, silent_s - step_s)
            else:
                # --- WYBIEG: trzymamy ostatnią częstotliwość ---
                silent_s += step_s
                if silent_s > hold_s:
                    f_est = np.nan

        times.append(start / sr)
        freqs.append(f_est)

    return np.asarray(times), np.asarray(freqs)


def smooth_track(times: np.ndarray, freqs: np.ndarray,
                 window_s: float = 5.0) -> np.ndarray:
    """Mediana bieżąca po estymacie częstotliwości.

    Odrzuca krótkie wyskoki estymaty, a nadąża za dryfem prawdziwym, bo ten
    jest wolny i monotoniczny. Sensowna higiena przed przestrajaniem:
    przesunięcie widma powinno być gładkie, inaczej wnosi własną modulację.

    UCZCIWA UWAGA. Dopisałem to, podejrzewając, że szeroki zakres tonu na
    4-minutowym nagraniu (621-825 Hz) bierze się z błądzenia pętli.
    Zmniejszyło zakres tylko z 204 do 181 Hz, co powinno było mnie
    ostrzec — i faktycznie hipoteza była błędna. Wykres pokazał, że
    w tych fragmentach nagranie nie zawiera kluczowanego sygnału, a pętla
    śledziła to, co tam jest. Filtr zostaje jako higiena, nie jako
    naprawa.
    """
    f = np.asarray(freqs, dtype=np.float64)
    if f.size == 0:
        return f
    step = float(np.median(np.diff(times))) if times.size > 1 else 0.05
    n = max(3, int(round(window_s / max(step, 1e-6))) | 1)   # nieparzyste

    out = np.full_like(f, np.nan)
    half = n // 2
    for i in range(f.size):
        a, b = max(0, i - half), min(f.size, i + half + 1)
        seg = f[a:b]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[i] = float(np.median(seg))
    return out


def _parabolic(sp: np.ndarray, f: np.ndarray, k: int) -> float:
    """Interpolacja paraboliczna szczytu — daje rozdzielczość poniżej
    odstępu prążków, bez wydłużania okna analizy."""
    if k <= 0 or k >= sp.size - 1:
        return float(f[k])
    a, b, c = np.log(sp[k - 1] + 1e-30), np.log(sp[k] + 1e-30), \
        np.log(sp[k + 1] + 1e-30)
    d = a - 2.0 * b + c
    if abs(d) < 1e-30:
        return float(f[k])
    delta = 0.5 * (a - c) / d
    delta = float(np.clip(delta, -0.5, 0.5))
    return float(f[k] + delta * (f[1] - f[0]))


def retune(audio: np.ndarray, times: np.ndarray, freqs: np.ndarray,
           target: float = C.TONE_CENTER, sr: int = C.SR,
           smooth_s: float = 5.0) -> np.ndarray:
    """Przesuwa śledzony ton na `target`, nadążając za dryfem.

    Przesunięcie widma sygnału RZECZYWISTEGO wymaga sygnału analitycznego:
    mnożenie samego x(t) przez cosinus dałoby dwie wstęgi (sumę i różnicę),
    czyli lustrzane odbicie w pasmie. Bierzemy więc transformatę Hilberta,
    mnożymy przez zespoloną eksponentę i wracamy do części rzeczywistej.

    Faza jest CAŁKĄ z chwilowego przesunięcia, nie iloczynem — inaczej
    zmienne w czasie przesunięcie rozjechałoby fazę i rozmyło ton.
    """
    from scipy.signal import hilbert

    x = np.asarray(audio, dtype=np.float64)
    # Estymata przepuszczona przez medianę bieżącą — bez tego błądzenie
    # losowe pętli przenosi się wprost na przestrojenie.
    fr = smooth_track(times, freqs, smooth_s) if smooth_s > 0 else freqs
    # Estymata na siatkę próbek; luki (brak zaczepu) = zero przesunięcia.
    f_s = np.interp(np.arange(x.size) / sr, times, fr,
                    left=np.nan, right=np.nan)
    shift = np.where(np.isfinite(f_s), target - f_s, 0.0)

    phase = 2.0 * np.pi * np.cumsum(shift) / sr
    return np.real(hilbert(x) * np.exp(1j * phase)).astype(np.float32)


def bandpass(audio: np.ndarray, center: float = C.TONE_CENTER,
             half_width: float = 60.0, sr: int = C.SR,
             taps: int = 511) -> np.ndarray:
    """Wąskie pasmo wokół tonu — programowy odpowiednik filtra CW.

    Szerokość jest kompromisem, który da się policzyć. Kluczowanie z tempem
    20 WPM ma element o długości 60 ms, czyli częstotliwość elementów
    ok. 17 Hz; obwiednia z narastaniem cosinusowym ma wstęgi boczne rzędu
    +/-40 Hz. Pasmo +/-60 Hz przepuszcza je z zapasem, a odrzuca stację
    odległą o 55 Hz dopiero częściowo — dlatego samo zwężenie nie zastąpi
    śledzenia, tylko je uzupełnia.

    Filtr o fazie liniowej (FIR, okno Hamminga) — filtr o fazie
    nieliniowej zniekształciłby kształt obwiedni, a to na niej opiera się
    cały odczyt.
    """
    from scipy.signal import filtfilt, firwin

    lo = max(20.0, center - half_width) / (sr / 2.0)
    hi = min(sr / 2.0 - 20.0, center + half_width) / (sr / 2.0)
    b = firwin(taps, [lo, hi], pass_zero=False, window="hamming")
    return filtfilt(b, [1.0], np.asarray(audio, dtype=np.float64)
                    ).astype(np.float32)


def lock_report(times: np.ndarray, freqs: np.ndarray,
                smooth_s: float = 5.0) -> str:
    """Opis zachowania pętli — do wypisania przez narzędzia.

    Podaje zakres SUROWY i PO WYGŁADZENIU, bo różnica między nimi mówi,
    ile z zakresu jest błądzeniem pętli, a ile prawdziwym dryfem nadajnika.
    """
    ok = np.isfinite(freqs)
    if not ok.any():
        return "pętla NIE ZACZEPIŁA SIĘ w żadnym oknie"
    f = freqs[ok]
    locks = int(np.sum(ok[1:] & ~ok[:-1])) + int(ok[0])

    # Sam ROZRZUT jest miarą myląca: na nagraniu, w którym część czasu nie
    # ma kluczowanego sygnału, pętla wiernie śledzi to, co tam jest, i
    # rozrzut wychodzi setki herców przy tonie faktycznie stałym.
    # Miarą użyteczną jest UDZIAŁ CZASU w zakresie, na którym model
    # potrafi czytać.
    lo = C.TONE_CENTER - C.TONE_SPREAD
    hi = C.TONE_CENTER + C.TONE_SPREAD
    w_zakresie = float(np.mean((f >= lo) & (f <= hi)))

    return (f"zaczep w {100.0*ok.mean():.0f}% czasu, {locks} raz(y) od nowa; "
            f"mediana tonu {np.median(f):.0f} Hz\n"
            f"        w zakresie czytelnym dla modelu "
            f"({lo:.0f}-{hi:.0f} Hz): {100.0*w_zakresie:.0f}% czasu "
            f"(zakres estymaty {f.min():.0f}-{f.max():.0f} Hz)")
