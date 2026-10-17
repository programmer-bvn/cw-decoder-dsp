"""TESTER GRZEBIENIA (17.10). Nagrania: grzebień (tempo, pewność) i TRZY odczyty — sieć na surowym,
sieć na zregenerowanym, tabela Morse'a z nut. Zapisuje regenerowane WAV
do posłuchania.

    python tools/tester_grzebien.py runs/MODEL/best.keras out/regenerator

PODZIAŁ WINY: SIEĆ = sieć(regenerat) vs tabela z tych samych nut (wina
sieci — bez znajomości treści); SYGNAŁ = sieć(surowy) vs sieć(regenerat).
Pierwsza weryfikacja uchem (operator, 17.10): w SN0FMT_c ciągi „DDDD"
z tabeli SĄ nadane (ktoś stroił nadajnik) — sieć na surowym czytała tam
przypadkowe znaki, na regeneracie prawie same „D"."""
import os
import sys
from collections import Counter

REPO = "/mnt/w/PyCharmMiscProject/cw-decoder-dsp"
sys.path.insert(0, REPO)
os.chdir(REPO)
import numpy as np
import soundfile as sf
from pathlib import Path
from dsp import config as C, frontend, grzebien, tempo, tune
from dsp.model import load_model
from tools import nagrania, wav2net

MODEL = sys.argv[1] if len(sys.argv) > 1 else "runs/fcn_4000000_czesci_tempo/best.keras"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("out/regenerator")
OUT.mkdir(parents=True, exist_ok=True)
m = load_model(Path(MODEL))


def ocen(tekst, chce):
    pod = kol = 0
    for tok, kr in Counter(chce).items():
        pod += min(kr, nagrania.ile_podciagow(tok, tekst))
        kol += min(kr, nagrania.ile_w_kolejnosci(tok, tekst))
    return pod, kol


def cer(a, b):
    """Błąd znakowy a względem b (odległość edycyjna / len(b))."""
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (ca != cb))
    return d[len(b)] / max(1, len(b))


def odczyt_sieci(a):
    db, _ = tempo.obraz_w_tempie_wzorcowym(a)
    z, _ = wav2net.decode_fcn(frontend.normalize_db(db).astype(np.float32), m)
    return "".join(C.ALPHABET[i] for _, i, _, _ in z)


sumy = np.zeros((3, 2), int); razem = 0
print(f"model {MODEL}")
print("PODZIAŁ WINY (błąd znakowy [%]):  SIEĆ = sieć(regener.) vs tabela | "
      "SYGNAŁ = sieć(surowy) vs sieć(regener.)")
print(f"{'nagranie':40s} {'WPM':>5s} {'pewn':>5s}  {'SIEĆ':>5s} {'SYGNAŁ':>6s}   "
      f"słowa: surowy  regener.  tabela")
for p in sorted(Path("probki").glob("*.wav")):
    chce = nagrania.oczekiwane(p.with_suffix(".txt"))
    a = frontend.load_audio(p)
    t, f = tune.track_tone(a)
    a = tune.retune(a, t, f)
    g = wav2net.auto_gain_db(a)
    a = np.clip(a * 10 ** (g / 20), -1, 1).astype(np.float32)
    # grzebień w oknach 8 s co 4 s; nuty z każdego okna, sklejone po czasie
    n8, k4 = 8 * C.SR, 4 * C.SR
    wszystkie, pewn, wpm = [], [], []
    starty = list(range(0, max(1, a.size - n8 + 1), k4))
    if starty[-1] < a.size - n8:
        starty.append(a.size - n8)            # okno wyrównane do końca
    granica = 0.0                             # do tej chwili [s] znaki już są
    for i, s in enumerate(starty):
        r = grzebien.zmierz(a[s:s + n8])
        ostatnie = i == len(starty) - 1
        # Każde okno oddaje znaki od miejsca, gdzie skończyło poprzednie, do
        # 6 s od swojego początku (ostatnie — do końca): bez dziur i dubli.
        od = granica
        do = (a.size / C.SR) if ostatnie else s / C.SR + 6.0
        granica = do
        if not np.isfinite(r["u_s"]):
            continue
        # BRAMKA „czy nadaje stacja" (17.10 02:45): bez niej grzebień składał
        # z migotania szumu ciągi „DDDD" i regenerował je na czysto. Okno daje
        # nuty tylko przy ważnym S/N >= 3 dB (pomiar z koncentracją, dsp/sn)
        # i pewności grzebienia >= 0,85.
        from dsp import sn as _sn
        s_n = _sn.zmierz_sn(a[s:s + n8], C.TONE_CENTER)["sn_db"]
        if not (np.isfinite(s_n) and s_n >= 3.0 and r["pewnosc"] >= 0.85):
            continue
        pewn.append(r["pewnosc"]); wpm.append(r["wpm"])
        for st, el in grzebien.nuty(r["on_off"], r["u_s"], r["waga"] * r["u_s"]):
            ts = s / C.SR + st
            if od <= ts < do:
                wszystkie.append((ts, el, r["u_s"]))
    wszystkie.sort(key=lambda x: x[0])
    nuty_, ost = [], -1.0
    for ts, el, u in wszystkie:
        if ts > ost + 1e-3:
            nuty_.append((ts, el, u)); ost = ts
    u_med = float(np.median([u for _, _, u in nuty_])) if nuty_ else 0.06
    reg = grzebien.regeneruj([(ts, el) for ts, el, _ in nuty_], u_med, dlugosc_s=a.size / C.SR)
    sf.write(OUT / f"{p.stem}_regenerowany.wav", reg, C.SR)
    tab = grzebien.odczyt_klasyczny([(ts, el) for ts, el, _ in nuty_])
    # Regenerat do SIECI: czysty takt, ale z szumem W ZAKRESIE TRENINGU (S/N
    # 25 dB w 500 Hz, szum syntetyczny) — obraz bez szumu (tło ~-40 dB) jest
    # poza tym, co sieć kiedykolwiek widziała, i fałszowałby kolumnę SIEĆ
    # (pierwszy przebieg testera 17.10 02:26: SN0FMT 37-68% mimo tabeli 4/4).
    from dsp import radio as _radio, sn as _sn
    szum = _radio.band_noise(reg.size, np.random.default_rng(1), rms=0.015,
                             tilt=-0.5).astype(np.float64)
    amp = _sn.amp_dla_sn(25.0, szum, C.TONE_CENTER) / 0.3
    reg_siec = (szum + amp * reg).astype(np.float32)
    g2 = wav2net.auto_gain_db(reg_siec)
    reg_siec = np.clip(reg_siec * 10 ** (min(0.0, g2) / 20), -1, 1).astype(np.float32)
    odczyty = (odczyt_sieci(a), odczyt_sieci(reg_siec), tab)
    w = ""
    if chce:
        razem += len(chce)
        for k, o in enumerate(odczyty):
            sumy[k] += ocen(o, chce)
        w = "   ".join(f"{ocen(o, chce)[0]}/{len(chce)} {ocen(o, chce)[1]}/{len(chce)}" for o in odczyty)
    siec = 100 * cer(odczyty[1], odczyty[2]) if odczyty[2] else float("nan")
    sygn = 100 * cer(odczyty[0], odczyty[1]) if odczyty[1] else float("nan")
    print(f"{p.name[:40]:40s} {np.median(wpm) if wpm else float('nan'):5.1f} "
          f"{np.median(pewn) if pewn else float('nan'):5.2f}  {siec:5.0f} {sygn:6.0f}   {w}")
    if not chce:
        print(f"    tabela: {tab[:70]}")
print(f"{'RAZEM (w całości / w kolejności)':40s} {'':11s}   " + "   ".join(f"{a_}/{razem} {b_}/{razem}" for a_, b_ in sumy))
print(f"regenerowane WAV: {OUT}")
