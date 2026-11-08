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
    an = analityczny(x, sr, -float(shift.min()) if shift.size else 0.0)
    return np.real(an * np.exp(1j * phase)).astype(np.float32)


def analityczny(x: np.ndarray, sr: int, w_dol_hz: float = 0.0) -> np.ndarray:
    """Sygnał analityczny do przestrojenia o `w_dol_hz` W DÓŁ (największe zejście).

    BEZ tego, co poniżej RETUNE_FMIN_HZ. 17.10 operator wskazał na regeneracie
    mic1 (26,0-26,3 s) ton narastający i opadający w ciszy między znakami.
    Źródło: wolne falowanie podłogi w nagraniu (mikrofon, kilka Hz, -39 dB
    względem znaku) — przestrojenie o +53 Hz robiło z niego ton ~53 Hz. Przy
    przesunięciu o kilkaset Hz (ton 400 Hz -> 750) przydźwięk 50/100 Hz
    trafiłby prosto w pasmo CW.

    Przy zejściu w dół (ton 1490 -> 750, szukanie 29.10) składowe poniżej
    zejścia przechodzą przez zero i wracają lustrem na |f - zejście|; odcięcie
    pod zejście - (FMIN - 100) kładzie lustro poniżej 300 Hz, pod obrazem sieci.
    Dla tonów z FMIN..FMAX (zejście <= 450 Hz) odcięcie zostaje 200 Hz — droga
    bit w bit jak przedtem."""
    X = np.fft.fft(np.asarray(x, np.float64))
    f = np.fft.fftfreq(X.size, 1.0 / sr)
    odc = max(RETUNE_FMIN_HZ, w_dol_hz - (C.FMIN - 100.0))
    return np.fft.ifft(X * np.where(f >= odc, 2.0, 0.0))


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
# ...ale tylko prawdziwa stacja, nie garb szumu. 08.11, 14,005 MHz, CW-R:
# stacja 51 Hz od tonu z radia (811 przy 760 Hz); przy 760 Hz w filtrze sam
# szum — kontrast 90/10 10-13 dB (szum Rayleigha daje tyle sam z siebie),
# próg KL_KONTRAST_DB przechodził, szum wygrywał przez bliskość radia,
# a stację kasowało kasuj_obce jako obcą: P sieci ~0,05, tekst „E E E".
# Kontrastem tego nie rozdzielić: nasza stacja w teście QRM obok obcej
# +12 dB ma 12-13,5 dB (próg 13: QRM 15/16, sesja dalej na szumie).
# Rozdziela WYBITNOŚĆ: szczyt „góry" ponad medianę pierścienia 15-40 Hz
# obok (niższą z dwóch stron — obca stacja może stać po jednej):
# szum przy 760 Hz maks. 4,2 dB, nasza w QRM min. 13,6 dB.
KL_RADIO_WYBITNOSC_DB = 8.0
KL_RADIO_PIERSCIEN_HZ = (15.0, 40.0)


def _wybitnosc(gora: np.ndarray, fb: np.ndarray, j: int) -> float:
    a, b = KL_RADIO_PIERSCIEN_HZ
    d = fb - fb[j]
    boki = [gora[(d <= -a) & (d >= -b)], gora[(d >= a) & (d <= b)]]
    boki = [np.median(x) for x in boki if x.size]
    return float(gora[j] - min(boki)) if boki else float("inf")


# Ton poza pasmem obrazu (29.10, maciej_333 z elektrody: własne SDR na STM32,
# krok strojenia ~7 kHz, NCO bez przestrajania — ton CW wychodzi przypadkowo,
# u niego 1492 Hz). Pętla w FMIN..FMAX łapała wtedy szum ledwie nad progiem
# (10 dB nad tłem; stacja 30 dB) albo nic i odczyt kończył się NaN.
# Szersze pasmo dla wszystkich (20.10) psuło środek filtra z szumu (test_tonu
# 92,2 -> 91,0%), więc 300-3000 Hz tylko wtedy, gdy kluczowana stacja spoza
# FMIN..FMAX jest o PRZEWAGA_DB głośniejsza od najgłośniejszej w nim (albo
# w nim nic, a ona >= ACQUIRE_SNR_DB + PRZEWAGA_DB). Zmierzone na całych
# plikach (196: probki/, zgłoszenia, maciej_333): maciej +20,2 dB; poza nim
# najwyżej +4,4 dB (pile-up maciej_333 — prawdziwa stacja w paśmie), reszta
# <= +0,1 dB. Górę daje SR 8 kHz i filtr SSB radia (~3 kHz).
SZUKAJ_FMIN_HZ = 300.0
SZUKAJ_FMAX_HZ = 3000.0
PRZEWAGA_DB = 10.0


def _kluczowane(audio: np.ndarray, sr: int):
    """-> (częstotliwości, poziom w chwilach wciśnięcia [dB nad tłem], kluczowana?)
    w 300-3000 Hz z całego wejścia, albo None. Kryterium jak w _sledz, odniesienie
    mocy jedno dla całego zakresu. Bloki bez nakładania (0,2 s) i float32:
    15 min nagrania to ~50 MB, nie 400 (laptop SQ2MMX ma 7,7 GB)."""
    x = np.asarray(audio, dtype=np.float64)
    nb = max(64, int(round(BLOCK_S * sr)))
    nfft = 1 << int(np.ceil(np.log2(nb * 4)))
    f = np.fft.rfftfreq(nfft, 1.0 / sr)
    band = (f >= SZUKAJ_FMIN_HZ) & (f <= SZUKAJ_FMAX_HZ)
    fb = f[band]
    win = np.hanning(nb)
    S = np.array([(np.abs(np.fft.rfft(x[s:s + nb] * win, n=nfft)) ** 2)[band]
                  for s in range(0, x.size - nb + 1, nb)], dtype=np.float32)
    S = S[S.sum(1) > 0] if S.size else S                # cisza cyfrowa (nagranie z filmu)
    if len(S) < 8:
        return None
    k = max(1, int(round(KL_WYGLADZ_HZ / (f[1] - f[0]))))
    S = np.apply_along_axis(lambda r: np.convolve(r, np.ones(2 * k + 1) / (2 * k + 1), mode="same"), 1, S)
    D = 10.0 * np.log10(np.maximum(S, 1e-30) / max(float(np.median(S)), 1e-30))
    gora, dol = np.percentile(D, KL_GORA, axis=0), np.percentile(D, KL_DOL, axis=0)
    return fb, gora, (gora >= ACQUIRE_SNR_DB) & (gora - dol >= KL_KONTRAST_DB)


def poza_pasmem(audio: np.ndarray, sr: int = C.SR, fmin: float = C.FMIN,
                fmax: float = C.FMAX) -> float | None:
    """Ton stacji spoza fmin..fmax, gdy przeważa o PRZEWAGA_DB (z całego wejścia), inaczej None."""
    k = _kluczowane(audio, sr)
    if k is None:
        return None
    fb, gora, ok = k
    w = (fb >= fmin) & (fb <= fmax)
    if not (ok & ~w).any():
        return None
    g_w = float(gora[ok & w].max()) if (ok & w).any() else ACQUIRE_SNR_DB
    j = int(np.argmax(np.where(ok & ~w, gora, -np.inf)))
    return float(fb[j]) if gora[j] >= g_w + PRZEWAGA_DB else None


def stacja_w_pasmie(audio: np.ndarray, sr: int = C.SR) -> bool:
    """Kluczowana w FMIN..FMAX co najmniej ACQUIRE_SNR_DB + PRZEWAGA_DB nad tłem —
    prawdziwa stacja, nie szum. Szum łapany przez pętlę: ~10 dB (maciej_333,
    pierwsze 2 s: 575 Hz); stacje ze zgłoszeń SQ2MMX: 54-93 dB."""
    k = _kluczowane(audio, sr)
    if k is None:
        return False
    fb, gora, ok = k
    w = (fb >= C.FMIN) & (fb <= C.FMAX)
    return bool((ok & w & (gora >= ACQUIRE_SNR_DB + PRZEWAGA_DB)).any())


def track_tone(audio: np.ndarray, sr: int = C.SR,
               fmin: float = C.FMIN, fmax: float = C.FMAX,
               block_s: float = BLOCK_S, step_s: float = STEP_S,
               ton_cw: float | None = None, szukaj: bool = True
               ) -> tuple[np.ndarray, np.ndarray]:
    """-> (czasy bloków, ton naszej stacji albo NaN). `szukaj`: stacja spoza
    FMIN..FMAX, gdy wyraźnie przeważa (albo ton z radia poza nim) — 300-3000 Hz."""
    if ton_cw is not None and not np.isfinite(ton_cw):
        ton_cw = None
    if szukaj:
        # CI-V podaje 300-900 Hz: ton z radia poniżej FMIN to od razu szukanie wokół niego;
        # z radiem w paśmie zostajemy w nim (operator dostroił tam swoją stację)
        if ton_cw is not None:
            kotwica = ton_cw if not fmin <= ton_cw <= fmax else None
        else:
            kotwica = poza_pasmem(audio, sr, fmin, fmax)
        if kotwica is not None:
            # kotwica jak ton z radia: kluczowana w ±KL_RADIO_HZ wygrywa bez względu na poziom
            return _sledz(audio, sr, SZUKAJ_FMIN_HZ, SZUKAJ_FMAX_HZ, block_s, step_s, kotwica)
    return _sledz(audio, sr, fmin, fmax, block_s, step_s, ton_cw)


def _sledz(audio: np.ndarray, sr: int, fmin: float, fmax: float,
           block_s: float, step_s: float, ton_cw: float | None
           ) -> tuple[np.ndarray, np.ndarray]:
    audio = np.asarray(audio, dtype=np.float64)
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
        przy_radiu = np.array([j for j in lm if abs(fb[j] - srodek) <= KL_RADIO_HZ
                               and _wybitnosc(gora, fb, j) >= KL_RADIO_WYBITNOSC_DB], int)             if ton_cw is not None else lm[:0]
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
