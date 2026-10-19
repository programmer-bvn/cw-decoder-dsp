"""ŚLEDZENIE TONU: która częstotliwość to NASZA stacja — i trzymanie jej.

Front-end patrzy w pasmo FMIN..FMAX (800 Hz). Na zapchanym paśmie mieści się
w nim kilka stacji (raw_radio_14_020MHz.wav: 845 Hz przy 0 dB i 790 Hz przy
-5,4 dB). Zanik (QSB) sprawia, że korespondent słabnie, a obca stacja wychodzi
na prowadzenie — dekoder „najmocniejszy prążek" przeskoczyłby na nią.
Operator raz dostrojony trzyma się swojego tonu.

TON KLUCZOWANY (od 19.10; HISTORIA „Pętla tonu: ton kluczowany"):
w oknie KL_OKNO_S dla każdej częstotliwości kontrast kluczowania — różnica
między głośnymi (KL_GORA percentyl) a cichymi (KL_DOL) chwilami. Stacja CW
na swojej częstotliwości raz jest głośna, raz cicha; szpilki szumu skaczą po
częstotliwościach; nośna, przydźwięk, radiolatarnia są głośne bez przerw.
Kontrast ODSIEWA, wybiera bliskość środka filtra (tam operator dostraja swoją
stację), bez przeskoku, gdy poprzedni wybór wciąż kandyduje.

Do 18.10 pętla fazowa (SZUKANIE / ZACZEP ±CAPTURE_HZ / WYBIEG) zaczepiała się
w pierwszym bloku na tym, co było najmocniejsze — przy nagraniu zaczynającym
się od szumu na szpilce szumu — i nie puszczała. Zestaw z prawdziwym tonem
(tools/test_tonu.py, 150 przypadków): stara 31,7% czasu na tonie, ta 90,3%;
na prawdziwych nagraniach słowa bez zmian (53/62, 14/14), więcej okien
przechodzi bramkę (EEEE 19:09: 3 -> 11; stara siedziała na 794 Hz zamiast 760).
"""

from __future__ import annotations

import numpy as np

from . import config as C

# --- parametry pętli ---------------------------------------------------
BLOCK_S = 0.20            # okno analizy częstotliwości [s]; 5 Hz rozdzielczości
STEP_S = 0.05             # krok analizy [s]
ACQUIRE_SNR_DB = 8.0      # ile ponad tło musi wystawać prążek, żeby się zaczepić
# Strefa „ten sam kandydat co poprzednio". 12 Hz z zapasem pokrywa dryf VFO
# (model kanału zakłada 8 Hz na klip 4 s, czyli 2 Hz/s, a krok pętli to
# 0,05 s), a nie pozwala przejść na stację odległą o kilkadziesiąt herców.
# ZMIERZONE: przy 25 Hz pętla na raw_radio_14_020MHz.wav przewędrowała
# 95 Hz między dwiema stacjami (751-845 Hz) — czyli zrobiła dokładnie to,
# czego ta konstrukcja ma nie robić.
CAPTURE_HZ = 12.0


def _spectrum(block: np.ndarray, sr: int, nfft: int):
    win = np.hanning(block.size)
    sp = np.abs(np.fft.rfft(block * win, n=nfft)) ** 2
    return sp, np.fft.rfftfreq(nfft, 1.0 / sr)


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


# Poniżej tego nie ma tonu CW (C.FMIN = 400 Hz); jest podłoga, przydźwięk
# sieci 50/100/150 Hz i falowanie mikrofonu.
RETUNE_FMIN_HZ = 200.0


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
    x = np.asarray(audio, dtype=np.float64)
    # Estymata przepuszczona przez medianę bieżącą — bez tego błądzenie
    # losowe pętli przenosi się wprost na przestrojenie.
    fr = smooth_track(times, freqs, smooth_s) if smooth_s > 0 else freqs
    # Estymata na siatkę próbek; luki (brak zaczepu) = zero przesunięcia.
    f_s = np.interp(np.arange(x.size) / sr, times, fr,
                    left=np.nan, right=np.nan)
    shift = np.where(np.isfinite(f_s), target - f_s, 0.0)

    phase = 2.0 * np.pi * np.cumsum(shift) / sr
    # Sygnał analityczny BEZ tego, co poniżej RETUNE_FMIN_HZ. 17.10 operator
    # wskazał na regeneracie mic1 (26,0-26,3 s) ton narastający i opadający
    # w ciszy między znakami. Źródło: wolne falowanie podłogi w nagraniu
    # (mikrofon, kilka Hz, -39 dB względem znaku) — przestrojenie o +53 Hz
    # robiło z niego ton ~53 Hz. Przy przesunięciu o kilkaset Hz (ton 400 Hz
    # -> 750) przydźwięk 50/100 Hz trafiłby prosto w pasmo CW.
    X = np.fft.fft(x)
    f = np.fft.fftfreq(x.size, 1.0 / sr)
    an = np.fft.ifft(X * np.where(f >= RETUNE_FMIN_HZ, 2.0, 0.0))
    return np.real(an * np.exp(1j * phase)).astype(np.float32)


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


# --------------------------------------------------------------------------
# TON KLUCZOWANY (18/19.10) — track_tone; opis na początku pliku
# --------------------------------------------------------------------------
# Stacja CW na swojej częstotliwości raz jest głośna, raz cicha (kluczowanie);
# szpilki szumu skaczą po częstotliwościach; nośna, przydźwięk i radiolatarnia
# są głośne bez przerw. Dla każdej częstotliwości: różnica między głośnymi
# (KL_GORA percentyl) a cichymi (KL_DOL) chwilami w oknie KL_OKNO_S — kontrast
# kluczowania. Przy podobnym kontraście: ta, która wygrywała dotąd, potem
# bliższa środka filtra (tam operator dostraja swoją stację).
# Zestaw z prawdą: tools/test_tonu.py (obecna pętla: 29% czasu na tonie).
KL_OKNO_S = 6.0
KL_KROK_S = 0.25
KL_GORA, KL_DOL = 90, 10
KL_KONTRAST_DB = 10.0
# Kandydaci: kluczowani, nie słabsi niż KL_REMIS_DB od najgłośniejszego;
# wygrywa BLIŻSZY ŚRODKA FILTRA (operator dostraja tam swoją stację i trzyma
# ją, choćby obok weszła mocniejsza). Przy 3 dB („głośniejszy wygrywa"):
# druga stacja obok — 33% czasu na tonie.
KL_REMIS_DB = 10.0
KL_WYGLADZ_HZ = 5.0
# Ton CW z radia (CI-V 0x14 0x09, 20.10): operator dostraja SWOJĄ stację tak,
# żeby słyszeć ją na tym tonie. Wtedy środek filtra = ton z radia, a kluczowana
# stacja w ±KL_RADIO_HZ od niego wygrywa bez względu na poziom (obca +12 dB
# wypychała naszą z okna KL_REMIS_DB). Test QRM, nasza dokładnie na tonie:
# pętla na celu 16/16 (bez radia 9/16); ton = środek filtra (nasza do 20 Hz
# obok): 14/16. Bez tonu z radia droga bez zmian (bit w bit).
KL_RADIO_HZ = 25.0


def track_tone(audio: np.ndarray, sr: int = C.SR,
                          fmin: float = C.FMIN, fmax: float = C.FMAX,
                          block_s: float = BLOCK_S, step_s: float = STEP_S,
                          ton_cw: float | None = None
                          ) -> tuple[np.ndarray, np.ndarray]:
    audio = np.asarray(audio, dtype=np.float64)
    if ton_cw is not None and not np.isfinite(ton_cw):
        ton_cw = None
    nb = max(64, int(round(block_s * sr)))
    ns = max(1, int(round(step_s * sr)))
    nfft = 1 << int(np.ceil(np.log2(nb * 4)))
    starty = list(range(0, max(1, audio.size - nb), ns))
    f = np.fft.rfftfreq(nfft, 1.0 / sr)
    band = (f >= fmin) & (f <= fmax)
    fb = f[band]
    win = np.hanning(nb)
    S = np.empty((len(starty), band.sum()))
    for i, s in enumerate(starty):
        S[i] = (np.abs(np.fft.rfft(audio[s:s + nb] * win, n=nfft)) ** 2)[band]
    times = np.array(starty) / sr
    if not len(starty):
        return times, np.zeros(0)
    k = max(1, int(round(KL_WYGLADZ_HZ / (f[1] - f[0]))))
    S = np.apply_along_axis(lambda r: np.convolve(r, np.ones(2 * k + 1) / (2 * k + 1), mode="same"), 1, S)
    # Odniesienie JEDNO dla całości, nie mediana w każdym bloku: kliknięcia
    # zboczy podnoszą całe pasmo w chwili kluczowania, mediana bloku skakała
    # 36 dB, a stała nośna względem niej wyglądała na kluczowaną (kontrast 30 dB).
    med = max(float(np.median(S)), 1e-30)
    D = 10.0 * np.log10(np.maximum(S, 1e-30) / med)
    # Środek filtra z SZUMU: 5. percentyl w czasie (przerwy), wygładzony
    # ±30 Hz, pas -6 dB. Pierwsza wersja (średnie widmo) wyznaczała „pas"
    # wokół najsilniejszej stacji — środkiem była stacja obca.
    tlo = np.percentile(S, 5, axis=0)
    kk = max(1, int(round(30.0 / (f[1] - f[0]))))
    tlo = np.convolve(tlo, np.ones(2 * kk + 1) / (2 * kk + 1), mode="same")
    # Odniesienie: „płaskowyż" szumu (60. percentyl wzdłuż pasma), nie szczyt —
    # stacje nadające bez przerwy (test QRM 19.10) podnosiły szczyt i środek
    # uciekał o 28 Hz (mediana); z 60. percentyla: 5,6 Hz (test_tonu: 18 -> 2,7).
    pas = fb[tlo >= np.percentile(tlo, 60) / 4.0]
    srodek = 0.5 * (pas.min() + pas.max()) if pas.size else C.TONE_CENTER
    if ton_cw is not None:
        srodek = float(ton_cw)
    ocena_t, ocena_f = [], []
    poprz = np.nan
    pol = KL_OKNO_S / 2.0
    for tc in np.arange(0.0, times[-1] + 1e-9, KL_KROK_S):
        w = (times >= tc - pol) & (times <= tc + pol)
        if w.sum() < 8:
            ocena_t.append(tc); ocena_f.append(poprz); continue
        gora = np.percentile(D[w], KL_GORA, axis=0)
        dol = np.percentile(D[w], KL_DOL, axis=0)
        # kontrast kluczowania tylko ODSIEWA (nośna, przydźwięk); wybiera
        # poziom w chwilach wciśnięcia. Pierwsza wersja (wybór po samym
        # kontraście): kliknięcia twardych zboczy podsłuchu IC-7300 są
        # „kluczowane" na całym paśmie — 59% przypadków bez zakłócenia.
        ok = (gora >= ACQUIRE_SNR_DB) & (gora - dol >= KL_KONTRAST_DB)
        if not ok.any():
            ocena_t.append(tc); ocena_f.append(poprz); continue
        poz = np.where(ok, gora, -np.inf)
        lm = np.flatnonzero((poz[1:-1] >= poz[:-2]) & (poz[1:-1] >= poz[2:]) & ok[1:-1]) + 1
        if not lm.size:
            lm = np.array([int(np.argmax(poz))])
        przy_radiu = lm[np.abs(fb[lm] - srodek) <= KL_RADIO_HZ] if ton_cw is not None else lm[:0]
        lm = przy_radiu if przy_radiu.size else lm[poz[lm] >= poz[lm].max() - KL_REMIS_DB]
        kand = fb[lm]
        j = lm[np.argmin(np.abs(kand - srodek))]
        # bez przeskoku: poprzedni zostaje, jeśli jest wśród kandydatów
        # i nie dalej od środka niż o CAPTURE_HZ więcej niż wybrany
        if np.isfinite(poprz) and np.any(np.abs(kand - poprz) <= CAPTURE_HZ):
            jp = lm[np.argmin(np.abs(kand - poprz))]
            if abs(fb[jp] - srodek) <= abs(fb[j] - srodek) + CAPTURE_HZ:
                j = jp
        fj = _parabolic(10.0 ** (gora / 10.0), fb, int(j))
        poprz = fj
        ocena_t.append(tc); ocena_f.append(fj)
    ocena_f = np.asarray(ocena_f, dtype=np.float64)
    # z ocen co KL_KROK_S na siatkę bloków: najbliższa ocena (bez mieszania NaN)
    idx = np.clip(np.round(times / KL_KROK_S).astype(int), 0, ocena_f.size - 1)
    return times, ocena_f[idx]
