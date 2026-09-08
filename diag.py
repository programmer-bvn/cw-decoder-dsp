"""DIAGNOSTYKA: kontrola liczbowa całego łańcucha, bez treningu.

    python diag.py

Każdy test podaje wartość OCZEKIWANĄ i OTRZYMANĄ. Kod wyjścia jest różny od
zera, gdy cokolwiek nie przejdzie — nadaje się do wywołania przed treningiem.

Sens tego pliku: usterki front-endu audio nie dają wyjątków. Rozjechana
normalizacja, ton poza pasmem, timing przesunięty o pół kropki — wszystko to
przechodzi bez śladu i objawia się tylko gorszą skutecznością modelu, której
nie da się odróżnić od "sieć jest za mała". Te testy rozdzielają jedno
od drugiego PRZED wydaniem godzin na trening.
"""

from __future__ import annotations

import sys
import traceback

import numpy as np

from dsp import config as C
from dsp import frontend, morse, radio

_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((name, ok, detail))
    print(f"  [{'OK ' if ok else 'BŁĄD'}] {name}")
    if detail:
        for line in detail.splitlines():
            print(f"         {line}")
    return ok


def section(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


# ==========================================================================
# TEST 1: timing Morse'a — czy elementy leżą na siatce jednostek
# ==========================================================================
def test_timing():
    section("TEST 1: timing kluczowania")
    text = "PARIS"
    wpm = 20.0
    env = morse.keying_envelope(text, wpm=wpm, sr=C.SR, ramp_ms=0.0)

    # Słowo PARIS ma z definicji 50 jednostek, licząc przerwę końcową.
    # text_to_units() nie dodaje przerwy końcowej, więc oczekujemy 50 - 7 = 43.
    units = morse.total_units(text)
    check("PARIS ma 43 jednostki (50 minus przerwa końcowa)",
          units == 43, f"oczekiwano 43, otrzymano {units}")

    dot = C.dot_seconds(wpm) * C.SR
    expected_len = int(round(43 * dot))
    check("długość obwiedni zgodna z 1.2/WPM",
          abs(env.size - expected_len) <= 1,
          f"oczekiwano {expected_len} próbek, otrzymano {env.size} "
          f"(kropka = {dot:.1f} próbek = {C.dot_seconds(wpm)*1000:.0f} ms)")

    # Każda seria tonu musi mieć długość 1 albo 3 jednostek z dokładnością
    # do jednej próbki. Odchylenie znaczy, że timing kumuluje zaokrąglenia.
    runs = morse._runs(env > 0.5)
    marks = [n for is_m, n in runs if is_m]
    err = max(abs(n / dot - round(n / dot)) for n in marks)
    lens = sorted({round(n / dot) for n in marks})
    check("każdy element ma długość 1 lub 3 jednostek",
          err < 0.02 and lens == [1, 3],
          f"zmierzone długości: {lens}, największe odchylenie "
          f"{err:.4f} jednostki")


# ==========================================================================
# TEST 2: obwiednia -> kod -> tekst (obieg zamknięty)
# ==========================================================================
def test_roundtrip():
    section("TEST 2: obieg zamknięty tekst -> obwiednia -> tekst")
    for text, wpm in (("SOS", 20.0), ("CQ DE SP1ABC", 25.0),
                      ("PARIS", 12.0), ("EEE", 30.0)):
        env = morse.keying_envelope(text, wpm=wpm, sr=C.SR, ramp_ms=0.0)
        dot = C.dot_seconds(wpm) * C.SR
        code = morse.envelope_to_code(env, dot_len=dot)
        got = morse.code_to_text(code)
        want = text.upper()
        check(f"'{want}' przy {wpm:.0f} WPM",
              got == want, f"odczytano '{got}'")


# ==========================================================================
# TEST 3: normalizacja dB — obieg zamknięty i granice
# ==========================================================================
def test_normalize():
    section("TEST 3: normalizacja dB")
    db = np.linspace(C.DB_MIN, C.DB_MAX, 101, dtype=np.float32)
    back = frontend.denormalize_db(frontend.normalize_db(db))
    err = float(np.max(np.abs(back - db)))
    check("normalize_db -> denormalize_db odtwarza decybele",
          err < 1e-3, f"największy błąd {err:.2e} dB")

    lo = frontend.normalize_db(np.array([C.DB_MIN - 20.0], dtype=np.float32))
    hi = frontend.normalize_db(np.array([C.DB_MAX + 20.0], dtype=np.float32))
    check("wartości poza zakresem są obcinane do 0 i 1",
          lo[0] == 0.0 and hi[0] == 1.0,
          f"DB_MIN-20 -> {lo[0]:.3f} (oczekiwano 0.0), "
          f"DB_MAX+20 -> {hi[0]:.3f} (oczekiwano 1.0)")


# ==========================================================================
# TEST 4: skala BEZWZGLĘDNA — najważniejszy test w tym pliku
# ==========================================================================
def test_absolute_scale():
    section("TEST 4: skala bezwzględna (kontrola błędu z v5.5)")
    rng = np.random.default_rng(0)

    # Ten sam ton, ta sama amplituda, w dwóch klipach: jednym cichym,
    # drugim z domieszką silnego sygnału poza pasmem.
    quiet = rng.normal(0, C.NOISE_RMS, C.CLIP_SAMPLES).astype(np.float32)
    tone = morse.synth_cw("E", wpm=10.0, tone=C.TONE_CENTER, amp=0.3)
    s = (C.CLIP_SAMPLES - tone.size) // 2
    quiet[s:s + tone.size] += tone

    loud = quiet.copy()
    t = np.arange(C.CLIP_SAMPLES) / C.SR
    loud += (0.9 * np.sin(2 * np.pi * 2500.0 * t)).astype(np.float32)

    img_q = frontend.to_net_image(quiet)
    img_l = frontend.to_net_image(loud)

    # Porównujemy tylko pasmo tonu, w oknie czasowym elementu.
    peak_q = float(img_q.max())
    peak_l = float(img_l.max())
    diff = abs(peak_q - peak_l)
    check("szczyt tego samego tonu nie zależy od reszty klipu",
          diff < 0.02,
          f"klip cichy: {peak_q:.4f}, klip z silnym sygnałem 2500 Hz: "
          f"{peak_l:.4f}, różnica {diff:.4f}\n"
          f"Przy ref=np.max albo top_db=80 różnica byłaby rzędu 0,3-0,5 —\n"
          f"i model widziałby inny obraz w zależności od tła.")

    # Tło musi wypadać przy zerze: szum 0.015 RMS jest poniżej DB_MIN.
    bg = float(np.percentile(frontend.denormalize_db(img_q), 25))
    check("tło szumowe wypada na dolnym końcu skali",
          bg <= C.DB_MIN + 2.0,
          f"tło {bg:.1f} dB, DB_MIN={C.DB_MIN} dB")


# ==========================================================================
# TEST 5: wycinanie okna
# ==========================================================================
def test_window():
    section("TEST 5: wycinanie okna")
    long_img = np.tile(np.arange(200, dtype=np.float32)[:, None],
                       (1, C.IMG_BINS)) / 200.0
    w = frontend.center_window(long_img)
    check("okno ma zadaną liczbę ramek",
          w.shape == (C.IMG_FRAMES, C.IMG_BINS),
          f"otrzymano {w.shape}, oczekiwano {(C.IMG_FRAMES, C.IMG_BINS)}")
    start = (200 - C.IMG_FRAMES) // 2
    check("okno jest wycięte ze środka",
          abs(float(w[0, 0]) - start / 200.0) < 1e-6,
          f"pierwsza ramka = {float(w[0,0])*200:.0f}, oczekiwano {start}")

    short = np.ones((40, C.IMG_BINS), dtype=np.float32)
    w2 = frontend.center_window(short)
    check("krótszy obraz jest dopełniony zerami i wycentrowany",
          w2.shape == (C.IMG_FRAMES, C.IMG_BINS) and w2[0, 0] == 0.0
          and w2[C.IMG_FRAMES // 2, 0] == 1.0,
          f"kształt {w2.shape}, brzeg={w2[0,0]}, środek={w2[C.IMG_FRAMES//2,0]}")


# ==========================================================================
# TEST 6: ścieżka na żywo == ścieżka plikowa
# ==========================================================================
def test_waterfall():
    section("TEST 6: Waterfall daje ten sam obraz co plik")
    rng = np.random.default_rng(1)
    audio = rng.normal(0, C.NOISE_RMS, C.CLIP_SAMPLES).astype(np.float32)
    wave = morse.synth_cw("SOS", wpm=20.0, tone=C.TONE_CENTER, amp=0.3)
    s = (C.CLIP_SAMPLES - wave.size) // 2
    audio[s:s + wave.size] += wave

    wf = frontend.Waterfall()
    for i in range(0, audio.size, C.MIC_BLOCK):
        wf.push(audio[i:i + C.MIC_BLOCK])

    # 6a. Bufor musi zawierać dokładnie ogon strumienia, bez przesunięcia.
    check("bufor audio zawiera dokładnie ogon strumienia",
          np.array_equal(wf.audio, audio[-wf.buf_len:]),
          f"bufor={wf.buf_len} próbek, "
          f"największa różnica "
          f"{float(np.max(np.abs(wf.audio - audio[-wf.buf_len:]))):.2e}")

    # 6b. Obraz na żywo vs obraz z CAŁEGO pliku. Porównujemy ogon, bo lewa
    #     krawędź bufora ma zerowe dopełnienie (center=True w librosie),
    #     a w pliku w tym miejscu jest prawdziwy sygnał. Prawa krawędź
    #     kończy się na tej samej próbce, więc te ramki muszą się zgadzać
    #     co do bitu.
    live = wf.image()
    full = frontend.normalize_db(frontend.power_to_db(
        frontend.melspec_power(audio)))
    k = 100
    err = float(np.max(np.abs(live[-k:] - full[-k:])))
    check(f"ostatnie {k} ramek na żywo zgadza się z obrazem z pliku",
          err < 1e-5,
          f"największa różnica {err:.2e}\n"
          f"Bufor OBRAZU zamiast bufora audio dawał tu ~0,1-0,3, "
          f"i to najbardziej przy zmianach klucza.")

    check("Waterfall zgłasza gotowość po wypełnieniu bufora",
          wf.ready(), f"filled={wf.filled}/{wf.buf_len}")


# ==========================================================================
# TEST 7: ton wypada w pasmie i w oczekiwanym prążku mel
# ==========================================================================
def test_tone_in_band():
    section("TEST 7: położenie tonu w pasmie mel")
    import librosa
    centers = librosa.mel_frequencies(n_mels=C.N_MELS, fmin=C.FMIN,
                                      fmax=C.FMAX, htk=False)

    for f0 in (C.TONE_CENTER - C.TONE_SPREAD, C.TONE_CENTER,
               C.TONE_CENTER + C.TONE_SPREAD):
        audio = morse.synth_cw("T", wpm=10.0, tone=f0, amp=0.5)
        img = frontend.to_net_image(audio)
        band = int(np.argmax(img.sum(axis=0)))
        got = centers[band]
        # Rozdzielczość pasm mel w tym zakresie to ~25 Hz, dopuszczamy 2 pasma.
        ok = abs(got - f0) < 60.0
        check(f"ton {f0:.0f} Hz trafia w pasmo {band} ({got:.0f} Hz)",
              ok, f"odchyłka {got - f0:+.0f} Hz")

    check("pasmo obejmuje cały rozrzut tonu generatora",
          C.FMIN < C.TONE_CENTER - C.TONE_SPREAD and
          C.TONE_CENTER + C.TONE_SPREAD < C.FMAX,
          f"pasmo {C.FMIN:.0f}-{C.FMAX:.0f} Hz, "
          f"ton {C.TONE_CENTER-C.TONE_SPREAD:.0f}-"
          f"{C.TONE_CENTER+C.TONE_SPREAD:.0f} Hz")


# ==========================================================================
# TEST 8: odcisk front-endu wykrywa zmianę parametrów
# ==========================================================================
def test_fingerprint():
    section("TEST 8: odcisk front-endu")
    fp = C.fingerprint_str()
    try:
        C.check_fingerprint(fp)
        ok_same = True
    except ValueError:
        ok_same = False
    check("odcisk zgodny z samym sobą przechodzi", ok_same)

    tampered = fp.replace(f"DB_MIN={C.DB_MIN}", "DB_MIN=-60.0")
    try:
        C.check_fingerprint(tampered, source="zbiór testowy")
        caught = ""
    except ValueError as e:
        caught = str(e)
    check("zmiana DB_MIN jest wykrywana",
          "DB_MIN" in caught,
          "komunikat: " + (caught.splitlines()[1].strip()
                           if caught else "BRAK — zmiana przeszła niezauważona"))


# ==========================================================================
# TEST 9: klip z generatora ma sygnał w polu widzenia sieci
# ==========================================================================
def test_generator_clip():
    section("TEST 9: klip generatora")
    from tools.generator import make_clip
    rng = np.random.default_rng(C.SEED)

    contrasts, silent_ok, n_silent = [], True, 0
    for _ in range(40):
        audio, target, meta = make_clip(rng, C.WPM, 0.0)
        img = frontend.to_net_image(audio)
        frontend.check_image(img, where="klip generatora")
        db = frontend.denormalize_db(img)
        contrast = float(np.percentile(db, 99.5) - np.percentile(db, 25))

        if meta["signal"]:
            contrasts.append(contrast)
        else:
            n_silent += 1
            silent_ok &= (target == 0)

    check(f"klipy bez sygnału mają etykietę 0 ({n_silent} sztuk)", silent_ok)

    n_vis = sum(1 for c in contrasts if c > 10.0)
    check("sygnał jest widoczny w oknie sieci (kontrast > 10 dB)",
          n_vis == len(contrasts),
          f"{n_vis}/{len(contrasts)} klipów z sygnałem; kontrast "
          f"min={min(contrasts):.1f} dB, średnio={np.mean(contrasts):.1f} dB")

    # W oknie sieci musi się zmieścić ZNAK DOCELOWY wraz z przerwami
    # międzyznakowymi po obu stronach — to one wyznaczają jego granice.
    # Sąsiedzi mogą być obcięci: są kontekstem, nie etykietą.
    longest = max(C.MORSE_DICT.values(), key=len)          # "-----" (0)
    char_units = morse.total_units("0")                    # 19 jednostek
    need_units = char_units + 2 * C.GAP_CHAR_UNITS         # 25 jednostek
    span = need_units * C.dot_seconds(C.WPM)
    win = C.img_seconds()

    check("znak docelowy z przerwami mieści się w oknie sieci",
          span <= win,
          f"najdłuższy znak '0' = {longest} = {char_units} jednostek, "
          f"z przerwami {need_units} jednostek = {span:.2f}s\n"
          f"okno sieci = {win:.2f}s  (zapas {win - span:+.2f}s)")

    # Granica tempa: poniżej tego WPM najdłuższy znak przestaje się mieścić.
    # Liczy się NAJNIŻSZE tempo w zbiorze, czyli WPM - WPM_JITTER, a nie
    # nominalne. Przy WPM_JITTER = 0 to była ta sama liczba i różnica nie
    # miała znaczenia; od kiedy zbiór ma rozrzut tempa, ma.
    wpm_floor = need_units * 1.2 / win
    wpm_min = C.WPM - C.WPM_JITTER
    span_min = need_units * C.dot_seconds(wpm_min)
    check(f"NAJNIŻSZE tempo w zbiorze ({wpm_min:.0f} WPM) mieści się w oknie",
          wpm_min >= wpm_floor,
          f"zakres tempa: {wpm_min:.0f}-{C.WPM + C.WPM_JITTER:.0f} WPM\n"
          f"przy {wpm_min:.0f} WPM najdłuższy znak z przerwami zajmuje "
          f"{span_min:.2f}s, okno ma {win:.2f}s\n"
          f"granica dla tego okna: {wpm_floor:.1f} WPM — niżej trzeba "
          f"zwiększyć IMG_FRAMES (teraz {C.IMG_FRAMES})")

    # Rozrzut tempa MUSI być niezerowy. Zbiór o jednym tempie daje model,
    # który czyta tylko to jedno tempo — sprawdzone na nagraniu z pasma
    # idącym 15 WPM przy modelu uczonym wyłącznie na 20 WPM.
    check("zbiór ma rozrzut tempa (WPM_JITTER > 0)",
          C.WPM_JITTER > 0,
          f"WPM_JITTER = {C.WPM_JITTER}\n"
          f"Przy zerze model czyta wyłącznie {C.WPM:.0f} WPM. Prawdziwi "
          f"operatorzy pracują w 13-27 WPM,\na kropka przy 15 WPM ma 80 ms "
          f"zamiast 60 ms — o trzecią dłużej niż wszystko, co widział.")

    # Informacyjnie: całe nadanie 3 znaków zwykle NIE mieści się w oknie
    # i tak ma być — sąsiedzi są przycięci celowo.
    group = morse.total_units("0" * C.CHARS_PER_CLIP) * C.dot_seconds(C.WPM)
    print(f"         (informacyjnie: całe nadanie {C.CHARS_PER_CLIP} "
          f"najdłuższych znaków = {group:.2f}s, czyli sąsiedzi są "
          f"przycięci — to zamierzone)")


# ==========================================================================
# TEST 10: model kanału radiowego
# ==========================================================================
def test_radio():
    section("TEST 10: model kanału radiowego")
    rng = np.random.default_rng(7)
    n = C.CLIP_SAMPLES

    # --- szum: RMS zachowany niezależnie od nachylenia widma ---
    for tilt in (0.0, -1.0, 1.0):
        x = radio.band_noise(n, rng, rms=0.02, tilt=tilt)
        r = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
        check(f"szum o nachyleniu {tilt:+.0f} ma zadany RMS",
              abs(r - 0.02) / 0.02 < 0.02,
              f"oczekiwano 0.0200, otrzymano {r:.4f}")

    # Nachylenie widma musi być faktycznie ujemne dla szumu różowego.
    # Bez tej kontroli literówka w wykładniku (tilt zamiast tilt/2) przeszłaby
    # niezauważona — RMS byłby poprawny, a widmo nie.
    pink = radio.band_noise(n, rng, rms=0.02, tilt=-1.0)
    spec = np.abs(np.fft.rfft(pink.astype(np.float64))) ** 2
    f = np.fft.rfftfreq(n, 1.0 / C.SR)
    band = (f > 100) & (f < 3000)
    slope = np.polyfit(np.log10(f[band]), np.log10(spec[band] + 1e-30), 1)[0]
    check("szum różowy ma widmo mocy opadające jak 1/f",
          -1.35 < slope < -0.65,
          f"nachylenie {slope:.2f} dekady/dekadę, oczekiwano -1.00")

    # --- QSB: zmierzona głębokość zaniku zgodna z zadaną ---
    tone = morse.synth_cw("T" * 12, wpm=12.0, tone=C.TONE_CENTER, amp=0.5)
    for depth in (0.0, 0.3, 0.6):
        faded = radio.apply_qsb(tone, np.random.default_rng(3), depth=depth)
        # Obwiednia szczytowa w oknach po 0,25 s.
        w = int(0.25 * C.SR)
        peaks = np.array([np.max(np.abs(faded[i:i + w]))
                          for i in range(0, faded.size - w, w)])
        peaks = peaks[peaks > 1e-6]
        got = 1.0 - peaks.min() / peaks.max() if peaks.size else 0.0
        # Zanik mierzy się na obwiedni, a ta jest próbkowana rzadko, więc
        # dopuszczamy spory margines — chodzi o rząd wielkości, nie o wartość.
        ok = abs(got - depth) < 0.25
        check(f"QSB o głębokości {depth:.1f} daje zmierzony spadek "
              f"{got:.2f}", ok)

    # --- QRN: trzask jest SZEROKOPASMOWY, czyli pionowy w obrazie ---
    # Mierzymy na czystym szumie, żeby nie mieszał się sygnał CW.
    # Kryterium: w ramce trafionej trzaskiem podnosi się WIELE pasm naraz.
    # Nie używamy minimum po pasmach — trzask trwa 0,5-6 ms, a jedna ramka
    # mel obejmuje okno 64 ms, więc energia impulsu rozkłada się i najsłabsze
    # pasmo rośnie tylko o ułamek decybela.
    quiet = radio.band_noise(n, rng, rms=0.005, tilt=0.0)
    crashed = radio.add_qrn(quiet, np.random.default_rng(11),
                            n_crashes=3, amp=0.6)
    img_q = frontend.to_net_image(quiet)
    img_c = frontend.to_net_image(crashed)

    rise = img_c - img_q                      # [ramki, pasma]
    bands_up = (rise > 0.05).sum(axis=1)      # ile pasm wzrosło w ramce
    best = int(bands_up.max())
    frames_hit = int(np.sum(bands_up >= C.N_MELS // 2))
    check("trzask podnosi wiele pasm w jednej ramce (kreska pionowa)",
          best >= C.N_MELS // 2,
          f"najwięcej pasm wzrosłych w jednej ramce: {best} z {C.N_MELS}; "
          f"ramek z ponad połową pasm: {frames_hit}")

    # Dla kontrastu: ton CW jest wąskopasmowy. Ale liczenie pasm powyżej
    # progu tego NIE rozdziela — silny ton przy DB_MIN=-30 dB podnosi ponad
    # próg 21 z 32 pasm, bo rozmycie widmowe okna Hanna i nakładające się
    # trójkąty mel rozlewają energię, a niska podłoga skali czyni widocznym
    # nawet upływ 40 dB poniżej szczytu.
    #
    # Rozdziela je natomiast SKUPIENIE energii: trzask podnosi wszystkie
    # pasma podobnie, ton ma ostry szczyt. Miara bezprogowa: jaka część
    # całego przyrostu mieści się w 6 najmocniejszych pasmach.
    tone_wave = np.zeros(n, dtype=np.float32)
    w = morse.synth_cw("T", wpm=10.0, tone=C.TONE_CENTER, amp=0.5)
    s = (n - w.size) // 2
    tone_wave[s:s + w.size] = w
    rise_t = frontend.to_net_image(quiet + tone_wave) - img_q

    def _skupienie(rise_img: np.ndarray, k: int = 6) -> float:
        prof = rise_img[np.argmax(rise_img.sum(axis=1))]
        prof = np.maximum(prof, 0.0)
        tot = prof.sum()
        return float(np.sort(prof)[-k:].sum() / tot) if tot > 1e-9 else 0.0

    sk_tone, sk_crash = _skupienie(rise_t), _skupienie(rise)
    check("ton skupia energię w kilku pasmach, trzask rozkłada po wszystkich",
          sk_tone > sk_crash + 0.15,
          f"6 najmocniejszych pasm z 32 mieści: ton {sk_tone*100:.0f}% "
          f"przyrostu, trzask {sk_crash*100:.0f}%\n"
          f"Liczenie pasm powyżej progu tego nie rozdziela — ton podnosi "
          f"ponad próg {int((rise_t > 0.05).sum(axis=1).max())} z "
          f"{C.N_MELS} pasm z powodu rozmycia widmowego.")

    # --- QRM: druga stacja wnosi energię DO OKNA WIDZIANEGO PRZEZ SIEĆ ---
    # Baza to czysty szum, bez stacji docelowej — inaczej trudno rozdzielić,
    # które pasma należą do której stacji. Sprawdzamy na 16 ziarnach, bo
    # położenie nadania jest losowe, a chodzi o to, żeby ZAWSZE zahaczało
    # o okno; wcześniej losowanie z całego klipu dawało 12% nadań
    # niewidocznych, przy zapisanym znaczniku "jest QRM".
    base = radio.band_noise(n, np.random.default_rng(21), rms=C.NOISE_RMS)
    e_b = frontend.to_net_image(base).sum(axis=0)
    visible, gains = 0, []
    for seed in range(16):
        e_m = frontend.to_net_image(
            radio.add_qrm(base, np.random.default_rng(seed), amp=0.3)
        ).sum(axis=0)
        g = e_m - e_b
        gains.append(float(g.max()))
        visible += int(np.sum(g > 2.0) >= 2)
    check("druga stacja jest widoczna w oknie sieci za każdym razem",
          visible == 16,
          f"widoczna w {visible}/16 prób; przyrost energii "
          f"min {min(gains):.1f}, mediana {np.median(gains):.1f}")

    # --- receive(): opis zawiera wszystkie parametry ---
    audio, meta = radio.receive("SOS", np.random.default_rng(1))
    need = {"tone", "amp", "wpm", "fist", "drift", "qsb", "qrm", "qrn",
            "noise_rms", "noise_tilt", "peak", "clipped", "text"}
    missing = need - set(meta)
    check("receive() zwraca pełny opis parametrów klipu",
          not missing and audio.size == C.CLIP_SAMPLES,
          f"brakuje: {sorted(missing) if missing else 'nic'}; "
          f"długość {audio.size} (oczekiwano {C.CLIP_SAMPLES})")
    check("tor wejściowy nie przepuszcza poziomu powyżej pełnej skali",
          float(np.max(np.abs(audio))) <= 1.0)


# ==========================================================================
# TEST 11: rozjazd ręcznego klucza nie psuje kolejności elementów
# ==========================================================================
def test_fist():
    section("TEST 11: rozjazd ręcznego klucza (fist)")
    dot = C.dot_seconds(20.0) * C.SR

    # Przy rozjeździe do 20% progi decyzyjne (2 i 5 kropek) nadal rozdzielają
    # kropkę od kreski, więc tekst musi się odczytać bez błędu.
    ok_all, bad = True, []
    for seed in range(12):
        rng = np.random.default_rng(seed)
        env = morse.keying_envelope("CQ TEST", wpm=20.0, sr=C.SR,
                                    ramp_ms=0.0, fist=0.20, rng=rng)
        got = morse.code_to_text(morse.envelope_to_code(env, dot_len=dot))
        if got != "CQ TEST":
            ok_all = False
            bad.append(f"seed {seed}: '{got}'")
    check("rozjazd 20% nadal daje poprawny odczyt klasycznym dekoderem",
          ok_all, "\n".join(bad) if bad else "12 z 12 prób poprawnie")

    # Przy 45% progi muszą się zacząć mylić — gdyby nie, znaczyłoby to, że
    # parametr fist w ogóle nie działa.
    fails = 0
    for seed in range(12):
        rng = np.random.default_rng(100 + seed)
        env = morse.keying_envelope("CQ TEST", wpm=20.0, sr=C.SR,
                                    ramp_ms=0.0, fist=0.45, rng=rng)
        if morse.code_to_text(morse.envelope_to_code(env, dot_len=dot)) \
                != "CQ TEST":
            fails += 1
    check("rozjazd 45% faktycznie psuje odczyt (dowód, że parametr działa)",
          fails > 0, f"{fails} z 12 prób odczytanych błędnie")


# ==========================================================================
# TEST 11b: wady nadajnika — chirp, zapadanie amplitudy, przydźwięk
# ==========================================================================
def test_nadajnik():
    section("TEST 11b: wady nadajnika (trzecia litera RST)")
    from scipy.signal import hilbert

    def chwilowa(w):
        """Częstotliwość i obwiednia chwilowa tam, gdzie klucz naduszony."""
        z = hilbert(np.asarray(w, dtype=np.float64))
        f = np.diff(np.unwrap(np.angle(z))) / (2 * np.pi) * C.SR
        env = np.abs(z)[1:]
        m = env > 0.05 * env.max()
        return f[m], env[m]

    # --- CHIRP: ton spada W TRAKCIE kreski i wraca w przerwie ---
    # Kreska przy 15 WPM trwa 240 ms, więc stan zapadania (tau 60 ms)
    # dochodzi blisko jedynki i odchylenie jest prawie pełne.
    w = morse.synth_cw("T", wpm=15.0, tone=750.0, amp=0.5,
                       chirp_hz=25.0, sag_tau_ms=60.0)
    f, _ = chwilowa(w)
    n = max(1, f.size // 10)
    f0, f1 = float(np.median(f[:n])), float(np.median(f[-n:]))
    check("ton spada w trakcie kreski (chirp)",
          8.0 < f0 - f1 < 26.0,
          f"początek {f0:.1f} Hz, koniec {f1:.1f} Hz, spadek {f0-f1:.1f} Hz\n"
          f"Zadano chirp_hz=25; pełny spadek wymaga kreski dłuższej niż "
          f"kilka stałych czasowych.\n"
          f"To opisuje trzecia litera raportu RST (Tone).")

    # Bez chirpu ton musi być stały — inaczej test powyżej nic nie znaczy.
    f_ref, _ = chwilowa(morse.synth_cw("T", wpm=15.0, tone=750.0, amp=0.5))
    n2 = max(1, f_ref.size // 10)
    d_ref = abs(float(np.median(f_ref[:n2])) - float(np.median(f_ref[-n2:])))
    check("bez chirpu ton jest stały",
          d_ref < 1.0, f"zmiana {d_ref:.2f} Hz (oczekiwano poniżej 1 Hz)")

    # --- CHIRP to NIE dryf VFO: dryf idzie w jedną stronę przez całe
    #     nadanie, chirp wraca w każdej przerwie ---
    w2 = morse.synth_cw("TTTTT", wpm=15.0, tone=750.0, amp=0.5,
                        chirp_hz=25.0, sag_tau_ms=60.0)
    env2 = np.abs(hilbert(w2.astype(np.float64)))
    key = env2 > 0.5 * env2.max()
    runs = morse._runs(key)
    starts = []
    pos = 0
    for is_m, ln in runs:
        if is_m and ln > 200:
            starts.append(pos)
        pos += ln
    f2 = np.diff(np.unwrap(np.angle(hilbert(w2.astype(np.float64))))) \
        / (2 * np.pi) * C.SR
    poczatki = [float(np.median(f2[s + 20:s + 60])) for s in starts[:4]
                if s + 60 < f2.size]
    check("chirp wraca na początku każdej kreski (a dryf by nie wrócił)",
          len(poczatki) >= 3 and max(poczatki) - min(poczatki) < 12.0,
          f"ton na początku kolejnych kresek: "
          f"{', '.join(f'{p:.0f}' for p in poczatki)} Hz\n"
          f"Rozrzut {max(poczatki)-min(poczatki):.1f} Hz — gdyby to był "
          f"dryf, rosłby monotonicznie.")

    # --- ZAPADANIE AMPLITUDY: z tego samego stanu co chirp ---
    w3 = morse.synth_cw("T", wpm=15.0, tone=750.0, amp=0.5, sag_db=3.0,
                        sag_tau_ms=60.0)
    _, e3 = chwilowa(w3)
    n3 = max(1, e3.size // 10)
    spadek = 20.0 * np.log10(float(np.median(e3[-n3:]))
                             / float(np.median(e3[:n3])))
    check("amplituda kreski spada (to samo zapadanie zasilania)",
          -3.5 < spadek < -0.8,
          f"spadek {spadek:.2f} dB przy zadanym sag_db=3.0")

    # --- PRZYDŹWIĘK: modulacja obwiedni 100 Hz ---
    w4 = morse.synth_cw("TTTT", wpm=12.0, tone=750.0, amp=0.5,
                        hum_depth=0.3, hum_hz=100.0)
    env4 = np.abs(hilbert(w4.astype(np.float64)))
    seg = env4[env4 > 0.3 * env4.max()]
    if seg.size > 512:
        sp = np.abs(np.fft.rfft(seg - seg.mean())) ** 2
        fr = np.fft.rfftfreq(seg.size, 1.0 / C.SR)
        band = (fr > 60) & (fr < 140)
        peak_f = float(fr[band][np.argmax(sp[band])])
        check("przydźwięk widoczny na obwiedni przy 100 Hz",
              abs(peak_f - 100.0) < 8.0,
              f"szczyt modulacji obwiedni przy {peak_f:.1f} Hz")
    else:
        check("przydźwięk widoczny na obwiedni przy 100 Hz", False,
              "za krótki odcinek do analizy")


# ==========================================================================
# TEST 11c: rozjazd tempa jest SKORELOWANY, nie niezależny
# ==========================================================================
def test_fist_drift():
    section("TEST 11c: wolne błądzenie tempa (klucz sztorcowy)")
    dot = C.dot_seconds(20.0) * C.SR

    def dlugosci_kropek(env):
        """Długości elementów krótkich, w jednostkach kropki."""
        runs = morse._runs(np.asarray(env) > 0.5)
        marks = [n / dot for m, n in runs if m]
        return np.array([m for m in marks if m < 2.0])

    text = "PARIS PARIS PARIS PARIS"

    # Szum niezależny: kolejne kropki nieskorelowane -> autokorelacja ~0.
    k_ind = dlugosci_kropek(morse.keying_envelope(
        text, wpm=20.0, ramp_ms=0.0, fist=0.20, drift=0.0,
        rng=np.random.default_rng(5)))
    # Wolne błądzenie: kolejne kropki podobne -> autokorelacja wyraźnie > 0.
    k_slow = dlugosci_kropek(morse.keying_envelope(
        text, wpm=20.0, ramp_ms=0.0, fist=0.0, drift=0.15,
        rng=np.random.default_rng(5)))

    def autokor(x):
        x = np.asarray(x, dtype=np.float64)
        if x.size < 6:
            return 0.0
        x = x - x.mean()
        d = float(np.dot(x, x))
        return float(np.dot(x[:-1], x[1:]) / d) if d > 1e-12 else 0.0

    a_ind, a_slow = autokor(k_ind), autokor(k_slow)
    check("wolne błądzenie daje SKORELOWANE długości elementów",
          a_slow > a_ind + 0.25,
          f"autokorelacja kolejnych kropek: szum niezależny {a_ind:+.2f}, "
          f"wolne błądzenie {a_slow:+.2f}\n"
          f"({k_ind.size} i {k_slow.size} kropek)\n"
          f"Klucz ręczny zwalnia i przyspiesza w skali sekund, więc kolejne "
          f"elementy MUSZĄ być podobne.\nSzum niezależny dawał sygnał "
          f"nerwowy, ale o stałym tempie średnim — czego na kluczu nie ma.")

    # Przerwy międzyznakowe muszą mieć własny, niezależny rozjazd.
    env_g = morse.keying_envelope(text, wpm=20.0, ramp_ms=0.0, fist=0.0,
                                  drift=0.0, gap_jitter=0.40,
                                  rng=np.random.default_rng(7))
    runs = morse._runs(np.asarray(env_g) > 0.5)
    gaps = np.array([n / dot for m, n in runs if not m])
    dlugie = gaps[gaps > 2.0]                # przerwy międzyznakowe (3 jedn.)
    check("przerwy międzyznakowe mają własny rozjazd",
          dlugie.size >= 3 and dlugie.std() > 0.15,
          f"{dlugie.size} przerw międzyznakowych, "
          f"{dlugie.min():.2f}-{dlugie.max():.2f} jednostki "
          f"(odchylenie {dlugie.std():.2f})\n"
          f"Przy manipulatorze elementy idą równo, a przerwy zależą od tego, "
          f"jak szybko operator myśli.")


# ==========================================================================
# TEST 12: plik samodzielny train_rtx.py nie rozjechał się z projektem
# ==========================================================================
def test_standalone():
    section("TEST 12: zgodność train_rtx.py z projektem")
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent / "train_rtx.py"
    if not path.exists():
        check("train_rtx.py istnieje", False, f"brak pliku {path}")
        return

    spec = importlib.util.spec_from_file_location("_standalone", path)
    sa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sa)

    # --- 12a. odcisk front-endu ---
    mine, theirs = C.fingerprint_str(), sa.FINGERPRINT
    if mine == theirs:
        check("odcisk front-endu identyczny", True)
    else:
        pa = dict(kv.split("=", 1) for kv in theirs.split(";") if "=" in kv)
        pb = dict(kv.split("=", 1) for kv in mine.split(";") if "=" in kv)
        diffs = [f"{k}: train_rtx={pa.get(k)}  config={pb.get(k)}"
                 for k in sorted(set(pa) | set(pb)) if pa.get(k) != pb.get(k)]
        check("odcisk front-endu identyczny", False, "\n".join(diffs))

    # --- 12b. stałe kanału radiowego ---
    same, bad = True, []
    for name in ("SILENCE_FRACTION", "CHARS_PER_CLIP", "LABEL_INDEX",
                 "NOISE_RMS", "SIGNAL_AMP_MIN", "SIGNAL_AMP_MAX",
                 "TONE_CENTER", "TONE_SPREAD", "WPM", "WPM_JITTER",
                 "KEY_RAMP_MS", "QRM_PROB", "QRN_PROB", "QRN_MAX",
                 "DRIFT_HZ", "NOISE_TILT", "NOISE_RMS_SPREAD",
                 "QSB_DEPTH", "QRM_AMP", "QRN_AMP", "FIST",
                 # Dołożone po rozbudowie modelu nadajnika i odbiornika.
                 # Bez nich zabezpieczenie przed rozjazdem miałoby dziurę
                 # dokładnie w tej części, którą właśnie się zmieniło.
                 "FIST_DRIFT", "FIST_DRIFT_TAU_S", "GAP_JITTER",
                 "CHIRP_HZ", "SAG_DB", "SAG_TAU_MS",
                 "HUM_PROB", "HUM_DEPTH", "HUM_HZ",
                 "AGC_PROB", "AGC_TAU_MS", "AGC_DEPTH"):
        a, b = getattr(C, name), getattr(sa, name, None)
        if a != b:
            same = False
            bad.append(f"{name}: config={a}  train_rtx={b}")
    check("stałe generatora i kanału identyczne", same,
          "\n".join(bad) if bad else "33 stałe zgodne")

    # --- 12c. TEN SAM OBRAZ z tego samego ziarna ---
    # To jest mocniejsze niż porównanie stałych: sprawdza, że cała ścieżka —
    # kolejność losowań, kluczowanie, kanał, mel, normalizacja — daje
    # identyczny wynik. Gdyby ktoś poprawił wzór w jednym pliku, a w drugim
    # nie, ten test to wyłapie, nawet gdy wszystkie stałe się zgadzają.
    from tools.generator import make_clip as make_project

    worst, bad_idx = 0.0, []
    for seed in range(6):
        a_audio, a_y, _ = make_project(np.random.default_rng(seed), C.WPM, 0.0)
        b_audio, b_y, _ = sa.make_clip(np.random.default_rng(seed))
        img_a = frontend.to_net_image(a_audio)
        img_b = sa.to_net_image(b_audio)
        d = float(np.max(np.abs(img_a - img_b)))
        worst = max(worst, d)
        if d > 1e-6 or a_y != b_y:
            bad_idx.append(f"ziarno {seed}: różnica {d:.2e}, "
                           f"etykiety {a_y} vs {b_y}")

    check("obrazy z tego samego ziarna identyczne co do bitu",
          not bad_idx,
          "\n".join(bad_idx) if bad_idx
          else f"6 ziaren, największa różnica {worst:.2e}")


# ==========================================================================
# TEST 13: model nadaje się na KV260 (DPUCZDX8G)
# ==========================================================================
def test_dpu_model():
    section("TEST 13: zgodność modelu z DPU na KV260")
    from dsp.model import DPU_SAFE_LAYERS, build_model

    m = build_model(arch="dpu")
    bad = [f"{l.name} ({type(l).__name__})" for l in m.layers
           if type(l).__name__ not in DPU_SAFE_LAYERS]
    check("architektura 'dpu' używa tylko warstw obsługiwanych przez "
          "DPUCZDX8G",
          not bad,
          "\n".join(bad) if bad else
          f"{len(m.layers)} warstw, wszystkie z listy dozwolonych; "
          f"{m.count_params()} parametrów")

    # Kontrola odwrotna: gdyby lista dozwolonych warstw kiedyś przypadkiem
    # objęła rekurencję, ten test przestałby cokolwiek znaczyć.
    g = build_model(arch="gru")
    bad_g = [type(l).__name__ for l in g.layers
             if type(l).__name__ not in DPU_SAFE_LAYERS]
    check("architektura 'gru' jest wykrywana jako niewdrażalna",
          bool(bad_g),
          f"wykryte warstwy poza listą: {sorted(set(bad_g))}\n"
          f"DPUCZDX8G nie obsługuje rekurencji — RNN w Vitis AI ma osobne "
          f"nakładki (U25, U50LV), których na Zynq nie ma.")

    # --- zasięg widzenia po osi czasu ---
    # Liczony ze stosu warstw, nie przepisany z komentarza. Musi pokryć
    # najdłuższy znak wraz z przerwami międzyznakowymi, inaczej neuron
    # w ostatniej warstwie nie widzi obu granic znaku naraz.
    rf, jump = 1, 1
    for layer in m.layers:
        t = type(layer).__name__
        if t == "Conv2D":
            k = layer.kernel_size[0]
            rf += (k - 1) * jump
            jump *= layer.strides[0]
        elif t == "MaxPooling2D":
            k = layer.pool_size[0]
            rf += (k - 1) * jump
            jump *= layer.strides[0]

    need_units = morse.total_units("0") + 2 * C.GAP_CHAR_UNITS
    need_frames = need_units * C.dot_seconds(C.WPM) * C.frames_per_second()
    check("zasięg widzenia pokrywa najdłuższy znak z przerwami",
          rf >= need_frames,
          f"zasięg {rf} ramek = {rf/C.frames_per_second():.2f}s, "
          f"potrzeba {need_frames:.0f} ramek = "
          f"{need_frames/C.frames_per_second():.2f}s "
          f"(znak '0' z przerwami)")

    # --- wejście modelu a format zbioru ---
    # Kwantyzacja Vitis AI do int8 działa najlepiej, gdy wejście ma stały,
    # znany zakres. Nasze obrazy to uint8 o STAŁEJ skali decybelowej, więc
    # kalibracja kwantyzatora nie musi niczego zgadywać. To bezpośredni
    # skutek pomiaru DB_MIN/DB_MAX, nie przypadek.
    shape = tuple(m.inputs[0].shape[1:])
    check("kształt wejścia modelu zgodny z config.INPUT_SHAPE",
          shape == C.INPUT_SHAPE,
          f"model {shape}, config {C.INPUT_SHAPE}")


# ==========================================================================
# TEST 14: front-end bez librosy daje te same liczby (port na KV260)
# ==========================================================================
def test_melref():
    section("TEST 14: front-end bez librosy (dsp/melref.py)")
    from dsp import melref
    import librosa

    # --- krawędzie pasm: skala Slaneya, nie HTK ---
    ea = librosa.mel_frequencies(n_mels=C.N_MELS + 2, fmin=C.FMIN,
                                 fmax=C.FMAX, htk=False)
    eb = melref.mel_frequencies(C.N_MELS, C.FMIN, C.FMAX)
    d_edges = float(np.abs(ea - eb).max())
    check("krawędzie pasm mel zgodne z librosą (skala Slaneya)",
          d_edges < 1e-9,
          f"największa różnica {d_edges:.2e} Hz\n"
          f"Skala HTK (1127*ln(1+f/700)) dałaby tu różnicę kilkudziesięciu "
          f"herców — filtrbank tf.signal jest w HTK i NIE jest wymienny.")

    # --- macierz filtrbanku: z normalizacją pola pasma ---
    fa = librosa.filters.mel(sr=C.SR, n_fft=C.N_FFT, n_mels=C.N_MELS,
                             fmin=C.FMIN, fmax=C.FMAX)
    fb = melref.mel_filterbank()
    d_fb = float(np.abs(fa - fb).max())
    check("macierz filtrbanku zgodna z librosą (norm=slaney)",
          d_fb < 1e-6,
          f"największa różnica {d_fb:.2e}, wartość szczytowa "
          f"{fa.max():.4f}\n"
          f"Trójkąty mają jednakowe POLE, nie jednakową wysokość — "
          f"tf.signal ma wysokość 1 i tego nie robi.")

    # --- STFT: dopełnienie zerami, nie odbiciem ---
    audio, _ = radio.receive("SOS", np.random.default_rng(3))
    pa = np.abs(librosa.stft(audio.astype(np.float64), n_fft=C.N_FFT,
                             hop_length=C.HOP_LENGTH)).T ** 2
    pc = melref.stft_power(audio, pad_mode="constant")
    pr = melref.stft_power(audio, pad_mode="reflect")
    rel_c = float(np.abs(pa - pc).max() / max(pa.max(), 1e-30))
    rel_r = float(np.abs(pa - pr).max() / max(pa.max(), 1e-30))
    check("STFT zgodny z librosą przy dopełnieniu ZERAMI",
          rel_c < 1e-12 and rel_r > 1e-6,
          f"pad_mode=constant: różnica względna {rel_c:.2e}\n"
          f"pad_mode=reflect:  różnica względna {rel_r:.2e}  <- nie pasuje\n"
          f"Rodzaj dopełnienia zmieniał się między wersjami librosy, "
          f"więc jest sprawdzany, a nie zakładany.")

    # --- cała ścieżka: obraz z melref (używany w produkcji) vs librosa ---
    # frontend.to_net_image() woła już melref, więc porównujemy melref
    # z NIEZALEŻNIE policzoną ścieżką librosy — inaczej test porównywałby
    # kod z samym sobą.
    def _librosa_image(au):
        mel = librosa.feature.melspectrogram(
            y=np.asarray(au, dtype=np.float32), sr=C.SR, n_fft=C.N_FFT,
            hop_length=C.HOP_LENGTH, n_mels=C.N_MELS, fmin=C.FMIN,
            fmax=C.FMAX, htk=False, norm="slaney", power=2.0).T
        db = librosa.power_to_db(mel, ref=C.DB_REF, top_db=None)
        img = np.clip((db - C.DB_MIN) / C.db_span(), 0.0, 1.0)
        return frontend.center_window(img.astype(np.float32))

    worst = 0.0
    for seed in range(8):
        au, _ = radio.receive("CQK", np.random.default_rng(seed))
        worst = max(worst, float(np.abs(_librosa_image(au)
                                        - melref.to_net_image(au)).max()))
    check("obraz z melref identyczny z niezależnym rachunkiem librosy",
          worst < 1e-5,
          f"największa różnica {worst:.2e} = {worst*C.db_span():.2e} dB "
          f"= {worst*255:.2e} kroku uint8 (8 ziaren)\n"
          f"To zapas float32, nie różnica metody. melref jest ścieżką\n"
          f"PRODUKCYJNĄ (woła ją frontend) i wzorcem do przepisania na C\n"
          f"dla KV260; librosa jest tu tylko niezależnym rachunkiem.")

    # --- to samo dla frontend, czyli dla tego, czym powstaje zbiór ---
    d_fe = max(float(np.abs(frontend.to_net_image(au)
                            - melref.to_net_image(au)).max())
               for au, _ in [radio.receive("SOS", np.random.default_rng(k))
                             for k in range(3)])
    check("frontend liczy front-end przez melref (bez librosy)",
          d_fe == 0.0,
          f"różnica {d_fe:.2e} — powinna być dokładnie 0, bo to ten sam kod")


# ==========================================================================
def main() -> int:
    print("=" * 70)
    print("DIAGNOSTYKA ŁAŃCUCHA DSP")
    print("=" * 70)
    print(C.summary())

    tests = (test_timing, test_roundtrip, test_normalize, test_absolute_scale,
             test_window, test_waterfall, test_tone_in_band, test_fingerprint,
             test_generator_clip, test_radio, test_fist, test_nadajnik,
             test_fist_drift, test_standalone,
             test_dpu_model, test_melref)

    for t in tests:
        try:
            t()
        except Exception:
            section(f"{t.__name__}: WYJĄTEK")
            traceback.print_exc()
            _results.append((t.__name__, False, "wyjątek"))

    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print("\n" + "=" * 70)
    print(f"WYNIK: {passed}/{total} kontroli przeszło")
    failed = [n for n, ok, _ in _results if not ok]
    if failed:
        print("Nie przeszły:")
        for n in failed:
            print(f"  - {n}")
    print("=" * 70)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
