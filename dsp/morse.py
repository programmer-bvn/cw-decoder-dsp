"""Synteza sygnału CW: tekst -> kluczowany ton.

Różnice względem generatora z poprzedniej wersji projektu:

1. OBWIEDNIA KLUCZOWANIA jest podniesionym cosinusem, nie odcinkiem liniowym.
   Liniowe narastanie ma nieciągłą pochodną w obu końcach, co daje widmo
   opadające jak 1/f^2 — na spektrogramie widoczne jako pionowe smugi przy
   każdym elemencie. Cosinus podniesiony ma pochodną ciągłą, widmo opada
   szybciej i obraz jest taki, jaki daje prawdziwy transceiver.

2. TIMING liczony jest w PRÓBKACH z jednego wzoru, a nie przez sklejanie
   np.linspace na każdy element. Sklejanie kumulowało błąd zaokrąglenia:
   przy 20 WPM i 3 znakach różnica dochodziła do kilku milisekund, więc
   elementy w klipie nie były na siatce jednostek.

3. Przerwa PO ostatnim elemencie nie jest dodawana wewnątrz znaku — długość
   ciszy między znakami i słowami ustala funkcja wyższego poziomu, tak żeby
   nie sumowały się dwie przerwy pod rząd.
"""

from __future__ import annotations

import numpy as np

from . import config as C

# Odwrotny słownik — do dekodowania ciągu kropek i kresek.
CODE_TO_CHAR = {code: ch for ch, code in C.MORSE_DICT.items()}


# --------------------------------------------------------------------------
# Tekst -> elementy
# --------------------------------------------------------------------------
def text_to_code(text: str) -> str:
    """Tekst -> zapis kropkowo-kreskowy, znaki rozdzielone spacją,
    słowa ukośnikiem: "SOS OK" -> "... --- ... / --- -.-" """
    words = []
    for word in text.upper().split():
        letters = [C.MORSE_DICT[ch] for ch in word if ch in C.MORSE_DICT]
        if letters:
            words.append(" ".join(letters))
    return " / ".join(words)


def code_to_text(code: str) -> str:
    """Odwrotność text_to_code(). Nieznane sekwencje dają znak zapytania."""
    out = []
    for word in code.strip().split(" / "):
        out.append("".join(CODE_TO_CHAR.get(sym, "?")
                           for sym in word.split() if sym))
    return " ".join(out)


def text_to_units(text: str, with_tags: bool = False):
    """Tekst -> lista (czy_ton, długość_w_jednostkach_kropki).

    Zwraca ciąg przemienny: element, przerwa, element, przerwa... Przerwa
    końcowa NIE jest dodawana — dokłada ją wywołujący, jeśli jej potrzebuje.

    with_tags=True dokłada trzecie pole: numer znaku w tekście, do którego
    należy dany element (przerwy międzyznakowe i międzysłowne mają -1).
    Po to, żeby dało się wskazać na obrazie DOKŁADNE położenie znaku
    z etykiety — bez tego telegrafista patrzący na kafelek nie ma jak
    sprawdzić, który z trzech nadanych znaków jest tym opisanym.
    """
    seq: list[tuple] = []
    words = [w for w in text.upper().split() if w]
    char_no = 0

    for wi, word in enumerate(words):
        chars = [ch for ch in word if ch in C.MORSE_DICT]
        for ci, ch in enumerate(chars):
            code = C.MORSE_DICT[ch]
            for si, sym in enumerate(code):
                seq.append((True, C.DASH_UNITS if sym == "-" else 1, char_no))
                if si < len(code) - 1:
                    seq.append((False, C.GAP_ELEMENT_UNITS, char_no))
            char_no += 1
            if ci < len(chars) - 1:
                seq.append((False, C.GAP_CHAR_UNITS, -1))
        if wi < len(words) - 1:
            seq.append((False, C.GAP_WORD_UNITS, -1))

    return seq if with_tags else [(m, n) for m, n, _ in seq]


def total_units(text: str) -> int:
    """Łączna długość nadania w jednostkach kropki — do zaplanowania klipu
    przed jego wygenerowaniem."""
    return sum(n for _, n in text_to_units(text))


# --------------------------------------------------------------------------
# Elementy -> obwiednia
# --------------------------------------------------------------------------
def keying_envelope(text: str, wpm: float = C.WPM, sr: int = C.SR,
                    ramp_ms: float = C.KEY_RAMP_MS,
                    fist: float = 0.0,
                    drift: float = 0.0,
                    gap_jitter: float = 0.0,
                    rng: np.random.Generator | None = None,
                    return_spans: bool = False):
    """Obwiednia kluczowania w zakresie [0, 1], próbkowana z sr.

    Timing: pozycje granic elementów liczone są w jednostkach, a na próbki
    przeliczane JEDNORAZOWO przez round(pozycja * dot_samples). Błąd
    zaokrąglenia nie kumuluje się, bo każda granica jest liczona od zera.

    fist — rozjazd ręcznego klucza. 0.0 to timing maszynowy (klucz
    elektroniczny, generator komputerowy). 0.15 znaczy, że każdy element
    i każda przerwa dostają losową długość +/-15%. Prawdziwy operator z
    ręcznym kluczem ma rozjazd rzędu 0,1-0,3, i to on odpowiada za większość
    trudności w dekodowaniu — model uczony wyłącznie na maszynowym timingu
    nie ma pojęcia o istnieniu tego zjawiska.

    return_spans=True zwraca (env, spans), gdzie spans to lista
    (numer_znaku, próbka_początkowa, próbka_końcowa) — granice KAŻDEGO
    nadanego znaku, po uwzględnieniu rozjazdu. Potrzebne, żeby wskazać na
    obrazie, który znak jest tym z etykiety.
    """
    seq = text_to_units(text, with_tags=True)
    if not seq:
        empty = np.zeros(0, dtype=np.float32)
        return (empty, []) if return_spans else empty

    if fist > 0.0 or drift > 0.0 or gap_jitter > 0.0:
        r = rng if rng is not None else np.random.default_rng()

        # WOLNE błądzenie tempa — proces skorelowany, nie szum niezależny.
        # "Jak łapa boli": operator zwalnia i przyspiesza w skali sekund,
        # więc kolejne elementy mają PODOBNE odchylenie, a nie losowe.
        # Realizacja: błądzenie losowe wygładzone filtrem pierwszego rzędu,
        # o stałej czasowej FIST_DRIFT_TAU_S przeliczonej na elementy.
        n_el = len(seq)
        if drift > 0.0 and n_el > 1:
            # Ile elementów mieści się w stałej czasowej: przy 20 WPM
            # element trwa ok. 0,12 s, więc 2 s to ok. 17 elementów.
            el_s = C.dot_seconds(wpm) * 2.0
            alpha = float(np.exp(-el_s / max(0.2, C.FIST_DRIFT_TAU_S)))
            w = r.standard_normal(n_el)
            slow = np.empty(n_el)
            slow[0] = w[0]
            for i in range(1, n_el):
                slow[i] = alpha * slow[i - 1] + np.sqrt(1 - alpha ** 2) * w[i]
            slow *= drift / 2.0        # +/-2 sigma mieści się w `drift`
        else:
            slow = np.zeros(n_el)

        out = []
        for i, (is_m, n, tag) in enumerate(seq):
            scale = 1.0 + slow[i]
            # Szum na pojedynczym elemencie — niezależny, to drżenie ręki.
            if fist > 0.0:
                scale *= 1.0 + r.uniform(-fist, fist)
            # Przerwy MIĘDZYZNAKOWE mają własny, większy rozjazd: przy
            # manipulatorze elementy idą równo, ale czas na pomyślenie
            # kolejnego znaku jest różny.
            if gap_jitter > 0.0 and not is_m and tag < 0:
                scale *= 1.0 + r.uniform(-gap_jitter, gap_jitter)
            out.append((is_m, max(0.35, n * scale), tag))
        seq = out

    dot = C.dot_seconds(wpm) * sr                 # próbki na jednostkę (float)
    total = sum(n for _, n, _ in seq)
    env = np.zeros(int(round(total * dot)), dtype=np.float32)

    # ramp_ms = 0 wyłącza kształtowanie całkowicie (kluczowanie prostokątne).
    # Używa tego diag.py, żeby mierzyć timing bez wpływu narastania na próg.
    ramp_len = int(round(ramp_ms * 1e-3 * sr))
    bounds: dict[int, list[int]] = {}
    pos_units = 0.0
    for is_mark, units, tag in seq:
        a = int(round(pos_units * dot))
        b = int(round((pos_units + units) * dot))
        pos_units += units

        # Granice znaku liczymy z ELEMENTÓW, nie z przerw — znak zaczyna się
        # pierwszym tonem i kończy ostatnim, przerwa wewnętrzna jest w środku.
        if is_mark and tag >= 0:
            if tag in bounds:
                bounds[tag][1] = b
            else:
                bounds[tag] = [a, b]

        if not is_mark:
            continue

        n = b - a
        if n <= 0:
            continue
        block = np.ones(n, dtype=np.float32)
        # Narastanie i opadanie nie mogą być dłuższe niż połowa elementu,
        # bo inaczej nachodzą na siebie i szczyt nie osiąga jedności.
        r = min(ramp_len, n // 2)
        if r > 0:
            ramp = _raised_cosine(r)
            block[:r] = ramp
            block[-r:] = ramp[::-1]
        env[a:b] = block

    if not return_spans:
        return env
    spans = [(t, ab[0], ab[1]) for t, ab in sorted(bounds.items())]
    return env, spans


def _raised_cosine(n: int) -> np.ndarray:
    """Narastanie cosinusem podniesionym: 0 -> 1 na n próbkach.
    0.5*(1 - cos(pi*t)) ma zerową pochodną na oba końce."""
    t = np.arange(1, n + 1, dtype=np.float64) / (n + 1)
    return (0.5 * (1.0 - np.cos(np.pi * t))).astype(np.float32)


# --------------------------------------------------------------------------
# Obwiednia -> audio
# --------------------------------------------------------------------------
def psu_sag(env: np.ndarray, sr: int = C.SR,
            tau_ms: float = 60.0) -> np.ndarray:
    """Stan zapadania zasilania nadajnika, w zakresie [0, 1].

    Naduszenie klucza obciąża zasilacz, kondensator się rozładowuje, napięcie
    spada. Stan podąża za obwiednią klucza z opóźnieniem: rośnie przy
    naduszeniu, opada w przerwie. Filtr pierwszego rzędu, bo takie jest
    rozładowanie i doładowanie pojemności.

    Skutek: dla DŁUGIEJ kreski stan dochodzi blisko jedynki, więc odchylenie
    jest największe na jej końcu — dokładnie jak w opisie "naduszenie
    750 Hz, koniec kreski 730 Hz". Dla serii kresek stan nie wraca do zera
    w przerwach i zapadanie się kumuluje.
    """
    from scipy.signal import lfilter
    a = float(np.exp(-1.0 / (max(1e-3, tau_ms) * 1e-3 * sr)))
    return lfilter([1.0 - a], [1.0, -a],
                   np.asarray(env, dtype=np.float64))


def synth_cw(text: str, wpm: float = C.WPM, tone: float = C.TONE_CENTER,
             amp: float = 0.3, sr: int = C.SR,
             ramp_ms: float = C.KEY_RAMP_MS,
             phase: float = 0.0,
             fist: float = 0.0,
             fist_drift: float = 0.0,
             gap_jitter: float = 0.0,
             drift_hz: float = 0.0,
             chirp_hz: float = 0.0,
             sag_db: float = 0.0,
             sag_tau_ms: float = 60.0,
             hum_depth: float = 0.0,
             hum_hz: float = 100.0,
             rng: np.random.Generator | None = None,
             return_spans: bool = False):
    """Tekst -> nadanie CW jako float32. Bez szumu — szum dodaje generator.

    DWA RÓŻNE ZJAWISKA CZĘSTOTLIWOŚCIOWE, których nie wolno mieszać:

    drift_hz — WOLNY dryf VFO, liniowy przez całe nadanie, NIEZALEŻNY od
               klucza. Grzejąca się cewka, niestabilne zasilanie.
    chirp_hz — SZYBKI spad tonu W TRAKCIE elementu, SKORELOWANY z kluczem,
               wracający w przerwie: naduszenie 750 Hz, koniec kreski
               730 Hz. Zapadanie zasilacza przestraja generator LC.
               To opisuje trzecia litera raportu RST (Tone).

    Poprzednia wersja miała tylko drift_hz, czyli modelowała zjawisko
    innego RODZAJU niż to, które faktycznie psuje odczyt.

    sag_db    — spadek amplitudy w trakcie elementu. Ten sam mechanizm co
                chirp, więc liczony z tego samego stanu zapadania.
    hum_depth — przydźwięk sieci na obwiedni nośnej (puste kondensatory
                w zasilaczu nadajnika).
    """
    out = keying_envelope(text, wpm, sr, ramp_ms, fist=fist,
                          drift=fist_drift, gap_jitter=gap_jitter, rng=rng,
                          return_spans=return_spans)
    env, spans = out if return_spans else (out, [])
    if env.size == 0:
        return (env, spans) if return_spans else env

    t = np.arange(env.size, dtype=np.float64) / sr
    env = np.asarray(env, dtype=np.float64)

    # Jeden stan zapadania zasilania opisuje NARAZ chirp i spadek amplitudy,
    # bo w rzeczywistości mają jedną przyczynę.
    sag = psu_sag(env, sr, sag_tau_ms) \
        if (chirp_hz > 0.0 or sag_db > 0.0) else None

    # --- częstotliwość chwilowa ---
    f_inst = np.full(env.size, float(tone))
    if abs(drift_hz) > 1e-9:
        span = t[-1] if t.size > 1 else 1.0
        f_inst = f_inst + float(drift_hz) * (t / span)
    if chirp_hz > 0.0:
        f_inst = f_inst - float(chirp_hz) * sag

    # Faza to CAŁKA z częstotliwości chwilowej. Wstawienie f(t) wprost do
    # sin(2*pi*f(t)*t) dałoby przesunięcie dwukrotnie większe od zadanego,
    # a przy chirpie skorelowanym z kluczem — skoki fazy na każdym elemencie.
    ph = 2.0 * np.pi * np.cumsum(f_inst) / sr + phase

    # --- obwiednia amplitudy ---
    a_env = env
    if sag_db > 0.0:
        a_env = a_env * (10.0 ** (-float(sag_db) / 20.0 * sag))
    if hum_depth > 0.0:
        # Przydźwięk moduluje NOŚNĄ, więc mnoży obwiednię — jest słyszalny
        # tylko wtedy, gdy klucz jest naduszony.
        a_env = a_env * (1.0 + float(hum_depth)
                         * np.sin(2.0 * np.pi * float(hum_hz) * t))

    wave = (amp * a_env * np.sin(ph)).astype(np.float32)
    return (wave, spans) if return_spans else wave


# --------------------------------------------------------------------------
# Dekodowanie z obwiedni (klasyczne, bez sieci) — używane przez diag.py
# jako niezależna kontrola, że wygenerowany sygnał ma poprawny timing.
# --------------------------------------------------------------------------
def envelope_to_code(env: np.ndarray, dot_len: float,
                     threshold: float = 0.5) -> str:
    """Obwiednia + znana długość kropki -> zapis kropkowo-kreskowy.

    Progi w jednostkach kropki: element > 2 to kreska, przerwa > 2 kończy
    znak, przerwa > 5 kończy słowo. Wartości pośrednie, nie nominalne —
    mieszczą rozjazd ręcznego klucza.
    """
    key = np.asarray(env, dtype=np.float32) > threshold
    if key.size == 0:
        return ""

    out: list[str] = []
    for is_mark, length in _runs(key):
        u = length / dot_len
        if is_mark:
            out.append("-" if u > 2.0 else ".")
        else:
            if u > 5.0:
                out.append(" / ")
            elif u > 2.0:
                out.append(" ")
            # przerwa <= 2 jednostki to przerwa w znaku — nic nie dopisujemy
    return "".join(out).strip()


def _runs(mask: np.ndarray) -> list[tuple[bool, int]]:
    """Kodowanie długościami serii: [(stan, długość), ...]."""
    if mask.size == 0:
        return []
    change = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    bounds = np.concatenate([[0], change, [mask.size]])
    return [(bool(mask[bounds[i]]), int(bounds[i + 1] - bounds[i]))
            for i in range(bounds.size - 1)]
