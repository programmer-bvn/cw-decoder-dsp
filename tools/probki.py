"""BIBLIOTEKA PRÓBEK: nazwa.wav + nazwa.txt, część opisu mierzona automatycznie.

    python -m tools.probki                    # zmierz wszystko w probki/
    python -m tools.probki --dir probki       # inny katalog
    python -m tools.probki --list             # zbiorcza tabela
    python -m tools.probki --wav plik.wav     # jedna próbka

DLACZEGO OPIS MA DWIE CZĘŚCI
----------------------------
Część pisana ręcznie zawiera to, czego z pliku nie da się odczytać: co
zostało nadane, jakim kluczem, z jakiego radia, jakim torem. Część mierzona
zawiera to, co da się policzyć — i której NIE wolno wpisywać z pamięci, bo
w tym projekcie każda taka liczba okazała się inna, niż zakładałem:

    "nagranie jest ciche"        -> szczyt 24,0 dB, czyli sufit skali
    "korespondent idzie 20 WPM"  -> 15 WPM, i model nie czytał nic
    "to jedna stacja"            -> dwie, 55 Hz od siebie
    "ton jest w pasmie"          -> mediana 839 Hz, poza zakresem modelu

Dlatego część mierzona jest generowana przy każdym uruchomieniu, a część
ręczna zachowywana bez zmian.

CO ZNACZĄ POLA W CZĘŚCI RĘCZNEJ
-------------------------------
`tor` istnieje, bo nagranie mikrofonem PC (głośnik -> powietrze -> mikrofon
-> ARW karty) rozmyło ton na 200 Hz i podniosło tło o 15 dB. Wyjście USB
z radia albo karta SD nie mają tego problemu i to jest różnica między
próbką użyteczną a bezużyteczną.

`tempo zadane` istnieje, bo model czyta tylko wąski zakres tempa. Wpisanie
go przy nagrywaniu oszczędza późniejszego dochodzenia.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsp import config as C
from dsp import frontend, morse, tune

MARKER = "# ---- PONIŻEJ MIERZONE AUTOMATYCZNIE, NIE EDYTUJ ----"

SZABLON = """\
# OPIS PRÓBKI — tę część wypełniasz ręcznie, narzędzie jej nie nadpisuje.

nadane:        {nadane}
nadajnik:      {nadajnik}
klucz:         {klucz}
tempo zadane:  {tempo}
pasmo:         {pasmo}
filtr:         {filtr}
tor:           {tor}
uwagi:         {uwagi}

"""


def zmierz(path: Path) -> dict:
    """Wszystkie liczby, które da się policzyć z pliku."""
    import soundfile as sf
    from tools.wav2net import TARGET_PEAK_DB, measure_peak_db

    info = sf.info(str(path))
    a = frontend.load_audio(path)
    w: dict = {
        "plik": path.name,
        "zrodlo": f"{info.samplerate} Hz, {info.channels} kan., "
                  f"{info.duration:.1f} s",
        "probek": a.size,
    }

    # --- poziom w skali, w której patrzy model ---
    peak = measure_peak_db(a)
    w["szczyt_db"] = peak
    w["gain_db"] = TARGET_PEAK_DB - peak
    w["nasycone"] = peak >= C.DB_MAX - 0.5

    # --- ton: mediana i udział czasu w zakresie czytelnym ---
    t, f = tune.track_tone(a)
    ok = np.isfinite(f)
    if ok.any():
        fv = f[ok]
        lo, hi = C.TONE_CENTER - C.TONE_SPREAD, C.TONE_CENTER + C.TONE_SPREAD
        w["ton_mediana"] = float(np.median(fv))
        w["ton_zakres"] = (float(fv.min()), float(fv.max()))
        w["ton_w_zakresie"] = float(np.mean((fv >= lo) & (fv <= hi)))
    else:
        w["ton_mediana"] = float("nan")
        w["ton_zakres"] = (float("nan"), float("nan"))
        w["ton_w_zakresie"] = 0.0

    # --- szerokość sygnału i kontrast ---
    db = frontend.power_to_db(frontend.melspec_power(a))
    prof = db.mean(axis=0)
    band = int(np.argmax(prof))
    w["pasm_10db"] = int(np.sum(prof > prof.max() - 10.0))
    w["szerokosc_hz"] = w["pasm_10db"] * (C.FMAX - C.FMIN) / C.N_MELS

    env = db[:, band]
    tlo, szczyt = float(np.percentile(env, 20)), float(np.percentile(env, 98))
    w["tlo_db"], w["kontrast_db"] = tlo, szczyt - tlo

    # --- tempo z rozkładu długości elementów ---
    key = env > (tlo + szczyt) / 2.0
    runs = morse._runs(key)
    marks = np.array([n for m, n in runs if m], dtype=float)
    fps = C.frames_per_second()
    if marks.size >= 5:
        krotki = float(np.percentile(marks, 10)) / fps
        w["wpm"] = 1.2 / krotki if krotki > 0 else float("nan")
        w["element_ms"] = (float(np.percentile(marks, 10)) / fps * 1000,
                           float(np.median(marks)) / fps * 1000,
                           float(np.percentile(marks, 90)) / fps * 1000)
        w["stosunek"] = (w["element_ms"][2] / w["element_ms"][0]
                         if w["element_ms"][0] > 0 else float("nan"))
        w["elementow"] = int(marks.size)
        w["klucz_udzial"] = float(key.mean())
    else:
        w["wpm"] = float("nan")
        w["element_ms"] = (float("nan"),) * 3
        w["stosunek"] = float("nan")
        w["elementow"] = int(marks.size)
        w["klucz_udzial"] = float(key.mean())
    return w


def ocena(w: dict) -> list[str]:
    """Czy ta próbka jest materiałem, na którym model może cokolwiek odczytać.

    Progi wynikają z pomiarów z tego projektu, nie z ogólnych zasad:
    czysty ton CW zajmuje ok. 75 Hz (3 pasma), a 200 Hz to już przesterowanie
    albo dwie stacje; stosunek 90/10 percentyla długości elementów bliski 3
    to poprawny rozkład kropka-kreska.
    """
    u = []
    if w["nasycone"]:
        u.append(f"PRZESTEROWANE: szczyt {w['szczyt_db']:.1f} dB dochodzi "
                 f"do DB_MAX={C.DB_MAX:.0f}; potrzeba {w['gain_db']:+.0f} dB")
    elif abs(w["gain_db"]) > 6:
        u.append(f"poziom do korekty: {w['gain_db']:+.0f} dB "
                 f"(szczyt {w['szczyt_db']:.1f} dB, cel 17 dB)")

    if w["szerokosc_hz"] > 130:
        u.append(f"ton ROZLANY na {w['szerokosc_hz']:.0f} Hz "
                 f"(czysty CW to ok. 75 Hz) — przesterowanie albo dwie stacje")

    if w["kontrast_db"] < 25:
        u.append(f"niski kontrast {w['kontrast_db']:.0f} dB "
                 f"(tło {w['tlo_db']:.0f} dB)")

    if w["ton_w_zakresie"] < 0.8:
        u.append(f"ton w zakresie czytelnym tylko {100*w['ton_w_zakresie']:.0f}% "
                 f"czasu (mediana {w['ton_mediana']:.0f} Hz) — "
                 f"konieczne przestrojenie")

    if np.isfinite(w["wpm"]):
        lo, hi = C.WPM - C.WPM_JITTER, C.WPM + C.WPM_JITTER
        if not (lo <= w["wpm"] <= hi):
            u.append(f"tempo {w['wpm']:.0f} WPM POZA zakresem treningu "
                     f"({lo:.0f}-{hi:.0f} WPM)")
        if np.isfinite(w["stosunek"]) and not (2.0 < w["stosunek"] < 4.5):
            u.append(f"stosunek długości elementów {w['stosunek']:.1f} "
                     f"(kropka:kreska powinno dać ok. 3) — możliwe dwie "
                     f"stacje albo trzaski")
    else:
        u.append("nie udało się zmierzyć tempa — za mało elementów")

    return u or ["nadaje się jako materiał testowy"]


def blok_mierzony(w: dict) -> str:
    e = w["element_ms"]
    linie = [
        MARKER,
        f"# zmierzone dla config: SR={C.SR}, pasmo {C.FMIN:.0f}-{C.FMAX:.0f} Hz,",
        f"#                       ton {C.TONE_CENTER:.0f}+/-{C.TONE_SPREAD:.0f} Hz,",
        f"#                       tempo {C.WPM-C.WPM_JITTER:.0f}-"
        f"{C.WPM+C.WPM_JITTER:.0f} WPM",
        "",
        f"zrodlo:        {w['zrodlo']}",
        f"szczyt obrazu: {w['szczyt_db']:.1f} dB   "
        f"(korekta poziomu {w['gain_db']:+.1f} dB do celu 17 dB)",
        f"tlo:           {w['tlo_db']:.1f} dB   kontrast {w['kontrast_db']:.1f} dB",
        f"szerokosc:     {w['szerokosc_hz']:.0f} Hz "
        f"({w['pasm_10db']} z {C.N_MELS} pasm w granicach 10 dB)",
        f"ton:           mediana {w['ton_mediana']:.0f} Hz, "
        f"zakres {w['ton_zakres'][0]:.0f}-{w['ton_zakres'][1]:.0f} Hz, "
        f"w zakresie czytelnym {100*w['ton_w_zakresie']:.0f}% czasu",
    ]
    if np.isfinite(w["wpm"]):
        linie += [
            f"tempo:         {w['wpm']:.1f} WPM "
            f"(najkrotszy element {e[0]:.0f} ms)",
            f"elementy [ms]: 10pct {e[0]:.0f}, mediana {e[1]:.0f}, "
            f"90pct {e[2]:.0f}   stosunek {w['stosunek']:.2f}",
            f"elementow:     {w['elementow']}, "
            f"klucz {100*w['klucz_udzial']:.0f}% czasu",
        ]
    linie += ["", "ocena:"]
    linie += [f"  - {u}" for u in ocena(w)]
    return "\n".join(linie) + "\n"


def zapisz_opis(wav: Path, w: dict, szablon: dict | None = None) -> Path:
    """Zapisuje nazwa.txt, ZACHOWUJĄC część ręczną, jeśli już istnieje."""
    txt = wav.with_suffix(".txt")
    if txt.exists():
        stare = txt.read_text(encoding="utf-8")
        reczne = stare.split(MARKER)[0].rstrip() + "\n\n"
    else:
        d = {"nadane": "", "nadajnik": "", "klucz": "", "tempo": "",
             "pasmo": "", "filtr": "", "tor": "", "uwagi": ""}
        d.update(szablon or {})
        reczne = SZABLON.format(**d)
    txt.write_text(reczne + blok_mierzony(w), encoding="utf-8")
    return txt


def main(argv=None):
    ap = argparse.ArgumentParser(description="Biblioteka próbek dźwiękowych")
    ap.add_argument("--dir", type=Path, default=C.ROOT / "probki")
    ap.add_argument("--wav", type=Path, help="tylko ta jedna próbka")
    ap.add_argument("--list", action="store_true",
                    help="zbiorcza tabela zamiast opisów")
    args = ap.parse_args(argv)

    wavy = ([args.wav] if args.wav else
            sorted(p for p in args.dir.glob("*")
                   if p.suffix.lower() in (".wav", ".flac", ".ogg")))
    if not wavy:
        raise SystemExit(f"brak plików audio w {args.dir}")

    if args.list:
        print(f"{'plik':34s} {'s':>6s} {'ton':>7s} {'WPM':>6s} "
              f"{'szer':>6s} {'szczyt':>7s} {'kontr':>6s}  ocena")
        print("-" * 116)
        for p in wavy:
            w = zmierz(p)
            print(f"{p.name[:34]:34s} "
                  f"{w['probek']/C.SR:6.0f} "
                  f"{w['ton_mediana']:6.0f}H "
                  f"{w['wpm']:6.1f} "
                  f"{w['szerokosc_hz']:5.0f}H "
                  f"{w['szczyt_db']:6.1f}d "
                  f"{w['kontrast_db']:5.0f}d  {ocena(w)[0][:44]}")
        return

    for p in wavy:
        w = zmierz(p)
        txt = zapisz_opis(p, w)
        print(f"{p.name}")
        for u in ocena(w):
            print(f"    {u}")
        print(f"    -> {txt.name}")


if __name__ == "__main__":
    main()
