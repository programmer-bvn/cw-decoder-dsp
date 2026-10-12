"""MODEL KANAŁU RADIOWEGO: to, co dzieje się z sygnałem między nadajnikiem
a wejściem karty dźwiękowej.

Rozdzielenie odpowiedzialności:
    morse.py  — kodowanie i kluczowanie (co nadaje operator)
    radio.py  — propagacja i tor odbiorczy (co z tego dochodzi)

CO TU JEST I PO CO
------------------
Sygnał z czystego generatora — stała amplituda, biały szum, ton co do herca
stały, timing maszynowy — nie występuje w przyrodzie. Model uczony wyłącznie
na takim materiale nie ma pojęcia o istnieniu zjawisk, które na pasmie są
regułą, i zawodzi przy pierwszym prawdziwym sygnale. Kolejno, od
najważniejszego:

  QSB   zanik — amplituda pływa w rytmie sekund. Na falach krótkich to
        norma, nie wyjątek. Model uczony na stałej amplitudzie traktuje
        spadek poziomu jako koniec nadania.
  QRM   inna stacja w pasmie. Filtr CW ma 300-500 Hz, a odstęp między
        stacjami w zapchanym pasmie bywa 50 Hz — druga stacja WCHODZI
        w to samo pasmo mel i wygląda jak drugi rząd kropek.
  QRN   trzaski atmosferyczne. Impuls jest szerokopasmowy, więc pojawia
        się we WSZYSTKICH pasmach mel jednocześnie jako pionowa kreska.
  fist  rozjazd ręcznego klucza (w morse.keying_envelope).
  dryf  niestabilny VFO (w morse.synth_cw).
  szum  na pasmie nie jest biały — ma nachylenie i wolne zmiany poziomu.

Każde zjawisko jest osobną funkcją z własnym parametrem siły, żeby dało się
zmierzyć, które z nich model przenosi, a które go przewraca — wystarczy
wygenerować zbiór testowy z jednym włączonym.
"""

from __future__ import annotations

import numpy as np

from . import config as C
from . import frontend
from . import morse
from . import sn


# --------------------------------------------------------------------------
# Szum tła
# --------------------------------------------------------------------------
def band_noise(n: int, rng: np.random.Generator, rms: float = C.NOISE_RMS,
               tilt: float = 0.0, sr: int = C.SR) -> np.ndarray:
    """Szum tła o zadanym RMS i nachyleniu widma.

    tilt = 0    biały (płaski)
    tilt = -1   różowy, 1/f — tak wygląda szum odbiornika na niskich pasmach
    tilt = +1   niebieski, rosnący z częstotliwością

    Kształtowanie w dziedzinie częstotliwości, potem powrót do RMS —
    inaczej samo nachylenie zmieniałoby poziom i mieszałoby się z SNR.
    """
    if abs(tilt) < 1e-9:
        x = rng.standard_normal(n)
    else:
        spec = (rng.standard_normal(n // 2 + 1)
                + 1j * rng.standard_normal(n // 2 + 1))
        f = np.arange(spec.size, dtype=np.float64)
        f[0] = 1.0
        spec *= f ** (tilt / 2.0)          # widmo MOCY ma iść jak f^tilt
        spec[0] = 0.0
        x = np.fft.irfft(spec, n=n)

    r = float(np.sqrt(np.mean(x * x)))
    return (x * (rms / r)).astype(np.float32) if r > 1e-12 \
        else np.zeros(n, dtype=np.float32)


# --------------------------------------------------------------------------
# QSB — zanik
# --------------------------------------------------------------------------
def apply_qsb(audio: np.ndarray, rng: np.random.Generator,
              depth: float, rate_hz: tuple[float, float] = (0.15, 1.5),
              sr: int = C.SR) -> np.ndarray:
    """Zanik: wolna modulacja amplitudy o głębokości `depth` (0..1).

    Suma dwóch składowych o różnych okresach, nie jeden sinus — pojedynczy
    sinus daje regularne pulsowanie, którego model może się nauczyć jako
    wzoru. Prawdziwy zanik jest nieregularny.

    depth = 0.6 znaczy, że amplituda schodzi do 40% wartości szczytowej,
    czyli o 8 dB. Zaniki 20 dB też się zdarzają, ale wtedy sygnał na chwilę
    ginie pod szumem i etykieta przestaje odpowiadać temu, co widać.
    """
    if depth <= 0.0 or audio.size == 0:
        return audio
    t = np.arange(audio.size, dtype=np.float64) / sr
    f1 = rng.uniform(*rate_hz)
    f2 = rng.uniform(*rate_hz) * 2.7
    p1, p2 = rng.uniform(0, 2 * np.pi, size=2)
    m = 0.65 * np.sin(2 * np.pi * f1 * t + p1) + \
        0.35 * np.sin(2 * np.pi * f2 * t + p2)
    gain = 1.0 - depth * 0.5 * (1.0 - m)        # w [1-depth, 1]
    return (audio * np.clip(gain, 0.0, 1.0)).astype(np.float32)


# --------------------------------------------------------------------------
# QRN — trzaski atmosferyczne
# --------------------------------------------------------------------------
def apply_agc(audio: np.ndarray, tau_ms: float, depth: float,
              sr: int = C.SR) -> np.ndarray:
    """ARW odbiornika: ściska dynamikę z własną stałą czasową.

    To TRZECIE, niezależne źródło bujania amplitudy, obok zaniku i zapadania
    zasilacza nadajnika — i o innej skali czasu. Zanik działa w sekundach,
    zapadanie w dziesiątkach milisekund, ARW gdzieś pomiędzy.

    Skutek słyszalny i widoczny: pierwszy element po dłuższej przerwie jest
    przesterowany, bo wzmocnienie jest jeszcze wysokie, a potem schodzi.
    Model uczony bez ARW traktuje ten wyższy pierwszy element jako inny
    kształt niż pozostałe.

    depth = 0 to brak regulacji, 1 to pełne wyrównanie poziomu.
    """
    from scipy.signal import lfilter
    x = np.asarray(audio, dtype=np.float64)
    if depth <= 0.0 or x.size == 0:
        return audio

    # Detektor obwiedni z zadaną stałą czasową.
    a = float(np.exp(-1.0 / (max(1e-3, tau_ms) * 1e-3 * sr)))
    envd = lfilter([1.0 - a], [1.0, -a], np.abs(x))
    ref = float(np.percentile(envd, 95)) + 1e-9
    # Wzmocnienie odwrotnie proporcjonalne do obwiedni, w stopniu `depth`.
    gain = (ref / np.maximum(envd, ref * 1e-3)) ** float(depth)
    return (x * np.clip(gain, 0.0, 20.0)).astype(np.float32)


def add_qrn(audio: np.ndarray, rng: np.random.Generator,
            n_crashes: int, amp: float, sr: int = C.SR) -> np.ndarray:
    """Trzaski: krótkie impulsy szerokopasmowe o wykładniczym zaniku.

    Impuls jest szerokopasmowy, więc w obrazie mel daje PIONOWĄ kreskę
    przez wszystkie pasma — inaczej niż sygnał CW, który jest poziomy.
    To dobra wiadomość: te dwie rzeczy są dla splotu łatwo rozróżnialne,
    o ile model kiedykolwiek widział trzask.
    """
    if n_crashes <= 0 or audio.size == 0:
        return audio
    out = audio.copy()
    for _ in range(n_crashes):
        pos = int(rng.integers(0, audio.size))
        length = int(rng.integers(int(0.0005 * sr), int(0.006 * sr) + 1))
        length = min(length, audio.size - pos)
        if length <= 1:
            continue
        decay = np.exp(-np.arange(length) / max(1.0, length / 3.0))
        pulse = rng.standard_normal(length) * decay * amp * rng.uniform(0.4, 1.0)
        out[pos:pos + length] += pulse.astype(np.float32)
    return out


# --------------------------------------------------------------------------
# QRM — inna stacja w pasmie
# --------------------------------------------------------------------------
def add_qrm(audio: np.ndarray, rng: np.random.Generator,
            amp: float, sr: int = C.SR,
            alphabet: str = C.ALPHABET) -> np.ndarray:
    """Druga stacja CW: inny ton, inne tempo, inny tekst, inna faza.

    Ton jest losowany W CAŁYM pasmie analizy, także blisko tonu stacji
    docelowej — bo tak właśnie wygląda zapchane pasmo. Jeśli wypadnie
    bardzo blisko, dwie stacje nakładają się w tych samych pasmach mel
    i to jest przypadek trudny również dla człowieka.
    """
    if amp <= 0.0 or audio.size == 0:
        return audio

    tone = float(rng.uniform(C.FMIN + 30.0, C.FMAX - 30.0))
    wpm = float(rng.uniform(15.0, 35.0))
    text = "".join(str(rng.choice(list(alphabet[1:]))) for _ in range(6))

    wave = morse.synth_cw(text, wpm=wpm, tone=tone, amp=amp, sr=sr,
                          phase=float(rng.uniform(0, 2 * np.pi)),
                          fist=float(rng.uniform(0.0, 0.2)), rng=rng)
    if wave.size == 0:
        return audio

    out = audio.copy()
    # Druga stacja startuje w losowym momencie i może wchodzić i wychodzić
    # z kadru — nie jest wyśrodkowana, bo nie ma powodu, żeby była.
    #
    # ALE musi ZAHACZAĆ o okno widziane przez sieć. Front-end wycina środkowe
    # IMG_FRAMES ramek klipu, więc pierwsze i ostatnie 0,72 s klipu nie
    # trafiają do obrazu. Przy losowaniu startu z całego klipu ok. 12%
    # nadań QRM wypadało w tych odrzuconych brzegach: zbiór miał wtedy
    # znacznik "jest obca stacja", a na obrazie jej nie było. Etykieta
    # opisująca coś, czego nie widać, jest gorsza niż brak etykiety.
    vis_a = frontend.window_crop_start(audio.size) * C.HOP_LENGTH
    vis_b = vis_a + C.IMG_FRAMES * C.HOP_LENGTH
    lo = vis_a - wave.size + 1          # koniec nadania wpada w okno
    hi = vis_b                          # początek nadania wpada w okno
    start = int(rng.integers(lo, hi))

    a0, a1 = max(0, start), min(audio.size, start + wave.size)
    if a1 > a0:
        out[a0:a1] += wave[a0 - start:a1 - start]
    return out


# --------------------------------------------------------------------------
# Prawdziwy szum odbiornika — bank nagrań pustego pasma (probki/szum)
# --------------------------------------------------------------------------
# Po co: opis przy REAL_NOISE_PROB w config.py. Bank wczytywany RAZ na
# proces (generator chodzi na kilku). Pliki POSORTOWANE po nazwie: ta sama
# funkcja jest w train_rtx.py i oba generatory muszą losować z tej samej
# listy — inaczej TEST 12 w diag.py nie dostanie tych samych obrazów.
_BANK = None


def srodek_pasa(a: np.ndarray, sr: int = C.SR) -> float:
    """Środek pasa szumu w Hz: środek przedziału -6 dB od szczytu widma
    uśrednionego po całym nagraniu. Liczone, nie wpisane — szerokość
    i położenie filtra zależą od ustawień radia (09.10: FIL2 = 200 Hz
    przy 699-850 Hz, FIL1 = 1240 Hz przy 211-1441 Hz)."""
    n = 1024
    w = np.hanning(n)
    moc = np.zeros(n // 2 + 1)
    for k in range(0, a.size - n, n // 2):
        moc += np.abs(np.fft.rfft(a[k:k + n] * w)) ** 2
    # Wygładzenie ~70 Hz (9 prążków po 7,8 Hz): 10.10 w pliku z 7025 kHz
    # wąski prążek stacji był 7,5 dB nad pasem i „środek pasa" wychodził
    # w nim (szerokość -6 dB: 8 Hz zamiast 230). Pas filtra jest szeroki,
    # więc wygładzenie go nie przesuwa, a prążek rozmywa.
    moc = np.convolve(moc, np.ones(9) / 9.0, mode="same")
    f = np.fft.rfftfreq(n, 1.0 / sr)
    m = (f > 150.0) & (f < 3500.0)
    p = f[m][moc[m] > moc[m].max() / 4.0]
    return float(0.5 * (p.min() + p.max()))


def bank_szumu(katalog=None) -> list:
    """[(nazwa, próbki float64, środek pasa Hz)], posortowane po nazwie.

    Pusty, gdy katalogu nie ma (np. inna maszyna) — receive() zostaje wtedy
    przy szumie syntetycznym, zamiast się wywracać. Ile plików weszło,
    widać w meta ("szum_real" = numer pliku albo -1).
    """
    global _BANK
    if _BANK is None or katalog is not None:
        from pathlib import Path
        from scipy.io import wavfile
        kat = Path(katalog) if katalog is not None else C.ROOT / C.REAL_NOISE_DIR
        bank = []
        for p in sorted(kat.glob("*.wav")):
            sr, a = wavfile.read(p)
            if a.ndim > 1:
                a = a[:, 0]
            if a.dtype.kind in "iu":
                a = a.astype(np.float64) / (np.iinfo(a.dtype).max + 1.0)
            else:
                a = a.astype(np.float64)
            if sr != C.SR or a.size < 2 * C.CLIP_SAMPLES:
                continue
            bank.append((p.name, a, srodek_pasa(a, sr)))
        if katalog is not None:
            return bank
        _BANK = bank
    return _BANK


def szum_prawdziwy(n: int, rng: np.random.Generator, srodek_hz: float,
                   rms: float = C.REAL_NOISE_RMS, sr: int = C.SR,
                   bank: list | None = None):
    """Wycinek z banku, przesunięty tak, by środek pasa wypadł w srodek_hz.

    Przesunięcie sygnałem analitycznym (hilbert * e^jwt), czyli tak, jak
    robi to tune.retune przy odczycie. Losowo odwrócony w czasie: szum nie
    ma kierunku, a bank ma ledwie kilka minut. Zwraca (szum, numer pliku).
    """
    from scipy.signal import hilbert
    bank = bank_szumu() if bank is None else bank
    # Każdy PLIK z równą szansą, nie każda sekunda: z wagą długości 2/3
    # klipów dostawało szeroki filtr (1250 Hz = 279 s z 420), a nagrania
    # z pasma są głównie z wąskim (FIL2, 200 Hz) — porównanie obrazów 10.10.
    i = int(rng.integers(0, len(bank)))
    _, a, c = bank[i]
    s = int(rng.integers(0, a.size - n))
    x = a[s:s + n]
    if rng.random() < 0.5:
        x = x[::-1]
    t = np.arange(n, dtype=np.float64) / sr
    x = np.real(hilbert(x) * np.exp(2j * np.pi * (srodek_hz - c) * t))
    r = float(np.sqrt(np.mean(x * x)))
    return (x * (rms / max(r, 1e-12))).astype(np.float32), i


# --------------------------------------------------------------------------
# Cały tor: stacja + kanał + odbiornik
# --------------------------------------------------------------------------
def receive(text: str, rng: np.random.Generator,
            n_samples: int = C.CLIP_SAMPLES,
            sr: int = C.SR,
            wpm: float | None = None,
            realism: bool = True) -> tuple[np.ndarray, dict]:
    """Nadanie `text` przepuszczone przez kanał — zwraca (audio, opis).

    Opis zawiera WSZYSTKIE wylosowane parametry. To nie ozdoba: gdy model
    myli się na jakiejś próbce, jedyny sposób dowiedzieć się dlaczego, to
    zobaczyć, że miała zanik 0,55 i QRM 40 Hz od tonu. Przy pobranym
    korpusie tej informacji nie ma i zostaje zgadywanie.
    """
    wpm = C.WPM if wpm is None else wpm
    meta: dict = {}

    # --- tor odbiorczy: szum tła ---
    # Prawdziwe tło z banku (opis przy REAL_NOISE_PROB w config.py) albo
    # syntetyczne. Prawdziwe powstaje PO stacji, bo jego pas ma stanąć
    # przy jej tonie, a ton losuje się dopiero niżej.
    prawdziwy = bool(realism and rng.random() < C.REAL_NOISE_PROB
                     and len(bank_szumu()) > 0)
    if prawdziwy:
        szum = None
        # QRM i QRN mają amplitudy dobrane do NOISE_RMS; przy innym poziomie
        # tła skalujemy je tak, żeby stosunek do szumu został ten sam.
        skala = C.REAL_NOISE_RMS / C.NOISE_RMS
        meta.update(noise_rms=C.REAL_NOISE_RMS, noise_tilt=float("nan"))
    else:
        skala = 1.0
        tilt = float(rng.uniform(*C.NOISE_TILT)) if realism else 0.0
        szum = band_noise(n_samples, rng, rms=C.NOISE_RMS, tilt=tilt, sr=sr)
        meta.update(noise_rms=C.NOISE_RMS, noise_tilt=tilt)
    # Stacja powstaje z amplitudą 1 i dostaje swoją dopiero, gdy szum jest
    # znany: amplituda wynika z wylosowanego S/N (jedna miara, dsp/sn.py),
    # nie odwrotnie. Do 12.10 syntetyk losował amplitudę, a bank jasność
    # szumu — dwa różne światy (HISTORIA „Poziom zamiast normalizacji").
    stacja = np.zeros(n_samples, dtype=np.float32)
    snr = float(rng.uniform(*C.SN_DB)) if realism else C.SN_DB[1]

    # --- stacja docelowa ---
    if text:
        tone = C.TONE_CENTER + float(rng.uniform(-C.TONE_SPREAD, C.TONE_SPREAD))
        clip_wpm = wpm + (float(rng.uniform(-C.WPM_JITTER, C.WPM_JITTER))
                          if C.WPM_JITTER > 0 else 0.0)
        clip_wpm = float(np.clip(clip_wpm, 5.0, 60.0))
        # --- operator i klucz ---
        fist = float(rng.uniform(*C.FIST)) if realism else 0.0
        fist_drift = float(rng.uniform(*C.FIST_DRIFT)) if realism else 0.0
        gap_jit = float(rng.uniform(*C.GAP_JITTER)) if realism else 0.0
        # --- nadajnik ---
        drift = float(rng.uniform(-C.DRIFT_HZ, C.DRIFT_HZ)) if realism else 0.0
        chirp = float(rng.uniform(*C.CHIRP_HZ)) if realism else 0.0
        sag = float(rng.uniform(*C.SAG_DB)) if realism else 0.0
        sag_tau = float(rng.uniform(*C.SAG_TAU_MS)) if realism else 60.0
        hum = 0.0
        hum_f = 100.0
        if realism and rng.random() < C.HUM_PROB:
            hum = float(rng.uniform(*C.HUM_DEPTH))
            hum_f = float(rng.choice(C.HUM_HZ))

        wave, spans = morse.synth_cw(
            text, wpm=clip_wpm, tone=tone, amp=1.0, sr=sr,
            phase=float(rng.uniform(0, 2 * np.pi)),
            fist=fist, fist_drift=fist_drift, gap_jitter=gap_jit,
            drift_hz=drift, chirp_hz=chirp, sag_db=sag, sag_tau_ms=sag_tau,
            hum_depth=hum, hum_hz=hum_f,
            rng=rng, return_spans=True)

        qsb = float(rng.uniform(*C.QSB_DEPTH)) if realism else 0.0
        wave = apply_qsb(wave, rng, depth=qsb, sr=sr)

        # Wstawienie w klip: wyśrodkowane, bo etykietą jest środkowy znak.
        if wave.size < n_samples:
            s = (n_samples - wave.size) // 2
            stacja[s:s + wave.size] += wave
            offset = s                 # nadanie zaczyna się w próbce s klipu
        else:
            s = (wave.size - n_samples) // 2
            stacja += wave[s:s + n_samples]
            offset = -s                # klip zaczyna się w próbce s nadania

        # Położenie znaku Z ETYKIETY w próbkach klipu. Liczone z FAKTYCZNYCH
        # granic, po rozjeździe klucza — nie z nominalnego timingu, bo przy
        # fist=0.25 nominalny wypada nawet o pół znaku obok.
        lab_a, lab_b = float("nan"), float("nan")
        for tag, a, b in spans:
            if tag == C.LABEL_INDEX:
                lab_a, lab_b = a + offset, b + offset
                break

        meta.update(tone=tone, wpm=clip_wpm, fist=fist,
                    fist_drift=fist_drift, gap_jitter=gap_jit,
                    drift=drift, chirp=chirp, sag=sag, hum=hum,
                    qsb=qsb, text=text,
                    lab_a=lab_a, lab_b=lab_b, spans=spans, offset=offset)
    else:
        meta.update(tone=float("nan"), wpm=wpm, fist=0.0,
                    fist_drift=0.0, gap_jitter=0.0,
                    drift=0.0, chirp=0.0, sag=0.0, hum=0.0,
                    qsb=0.0, text="",
                    lab_a=float("nan"), lab_b=float("nan"),
                    spans=[], offset=0)

    szum_i = -1
    if prawdziwy:
        srodek = ((meta["tone"] if text else C.TONE_CENTER)
                  + float(rng.uniform(-C.REAL_NOISE_OFFSET_HZ,
                                      C.REAL_NOISE_OFFSET_HZ)))
        szum, szum_i = szum_prawdziwy(n_samples, rng, srodek, sr=sr)
    # Amplituda z S/N. Pusty klip liczy ją tak samo („stacja, która
    # akurat nie nadaje") — wtedy gałka AF niżej ustawia tło na tym samym
    # poziomie co w klipach ze stacją, a nie podkręca samego szumu.
    amp = sn.amp_dla_sn(snr, szum, meta["tone"] if text else C.TONE_CENTER,
                        sr=sr)
    audio = (szum + amp * stacja).astype(np.float32)
    meta.update(amp=amp if text else 0.0, szum_real=szum_i,
                snr_db=snr if text else float("nan"))

    # --- zakłócenia w pasmie ---
    qrm_amp = 0.0
    if realism and rng.random() < C.QRM_PROB:
        qrm_amp = float(rng.uniform(*C.QRM_AMP)) * skala
        audio = add_qrm(audio, rng, amp=qrm_amp, sr=sr)
    meta["qrm"] = qrm_amp

    n_crashes = 0
    if realism and rng.random() < C.QRN_PROB:
        n_crashes = int(rng.integers(1, C.QRN_MAX + 1))
        audio = add_qrn(audio, rng, n_crashes=n_crashes,
                        amp=float(rng.uniform(*C.QRN_AMP)) * skala, sr=sr)
    meta["qrn"] = n_crashes

    # --- ARW odbiornika ---
    # Ostatni etap przed ogranicznikiem, bo taka jest kolejność w torze:
    # antena -> mieszacz -> filtr -> ARW -> wyjście AF -> karta dźwiękowa.
    # Przy prawdziwym tle ARW ZAWSZE: radio przez USB zawsze je ma, a to ono
    # podciąga szum w przerwach między elementami (09.10, SN0FMT).
    agc_tau = 0.0
    if realism and (prawdziwy or rng.random() < C.AGC_PROB):
        agc_tau = float(rng.uniform(*C.AGC_TAU_MS))
        audio = apply_agc(audio, agc_tau,
                          float(rng.uniform(*C.AGC_DEPTH)), sr=sr)
    meta["agc_tau"] = agc_tau

    # --- gałka AF: poziom bezwzględny, niezależny od S/N ---
    # Losowany jest poziom TONU STACJI w obrazie (jak gałka AF albo tłumik),
    # a tło ustawia się samo, S/N niżej. Przy odczycie auto_gain działa
    # na naszych nagraniach tylko jako tłumik (12.10: korekta -2,8...-12,7
    # dB na wszystkich 16), więc sieć ma znać poziomy od cichych do ~17 dB,
    # a nie jeden. Szeroki szum przy małym S/N wymagałby wzmocnienia ponad
    # pełną skalę — tam „operator ścisza" (POZIOM_RMS_MAX).
    poziom = float(rng.uniform(*C.POZIOM_DB)) if realism else 17.0
    g = 10.0 ** ((poziom - C.TON_DB_AMP1) / 20.0) / max(amp, 1e-9)
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    g = min(g, C.POZIOM_RMS_MAX / max(rms, 1e-12))
    audio = (audio * g).astype(np.float32)
    # Tłumik jak przy odczycie: najsilniejsza rzecz w klipie (stacja,
    # druga stacja, trzask) nie wyżej niż TLUMIK_DB.
    szczyt = float(np.percentile(
        frontend.power_to_db(frontend.melspec_power(audio, sr)), 99.9))
    if szczyt > C.TLUMIK_DB:
        t = 10.0 ** ((C.TLUMIK_DB - szczyt) / 20.0)
        audio = (audio * t).astype(np.float32)
        g *= t
    meta["poziom_db"] = float(C.TON_DB_AMP1 + 20.0 * np.log10(max(amp * g, 1e-12)))

    # --- ogranicznik toru wejściowego ---
    # Karta dźwiękowa obcina powyżej pełnej skali. Bez tego suma stacji,
    # trzasków i szumu mogłaby przekroczyć 1.0 i dawać poziomy, jakich
    # na prawdziwym wejściu nie da się uzyskać.
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    meta["peak"] = peak
    meta["clipped"] = peak > 1.0
    np.clip(audio, -1.0, 1.0, out=audio)

    return audio, meta
