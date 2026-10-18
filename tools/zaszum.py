"""ZASZUMIANIE NATURALNYM SZUMEM PASMA (18.10) — próbki testowe o znanej treści
i znanym S/N.

    python -m tools.zaszum czysty.wav tekst.txt                 # nagranie podsłuchu
    python -m tools.zaszum --syntetyk 18 probki/teksty/qso2_relacja.txt
    python -m tools.zaszum ... --sn 20 10 6 3 0 --szum 7025kHz_20261009_2135_500Hz

Pomysł operatora: czysty sygnał z kluczownika IC-7300MK2 (podsłuch przez
USB, bez wychodzenia w eter) + prawdziwy szum pasma z banku `probki/szum`.
Treść i S/N znane co do znaku i decybela — miara bez zgadywania, czym jest
„dobry odczyt". `--syntetyk WPM`: zamiast nagrania sygnał kluczownika
policzony z tekstu (morse.synth_cw, rytm maszynowy, narastanie 5 ms) —
zastępczo, dopóki nie ma nagrań z radia.

Ton sygnału przesuwany na środek pasa -6 dB szumu (pole „pas -6 dB:" w opisie
szumu) — sygnał leży tam, gdzie filtr odbiornika go przepuszcza. S/N w jednej
mierze projektu (dsp/sn: nośna do szumu w 500 Hz wokół tonu); po zmieszaniu
mierzone jeszcze raz i zapisane w opisie obok zadanego.

Wynik: probki/zaszumione/<nazwa>_sn<S/N>.wav + .txt (pole `nadane:`).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import soundfile as sf

from dsp import config as C, frontend, morse, sn, tune

KATALOG_SZUMU = Path("probki/szum")
WYJSCIE = Path("probki/zaszumione")


def srodek_pasa(opis: Path) -> float:
    m = re.search(r"pas -6 dB:\s*(\d+)-(\d+) Hz", opis.read_text(encoding="utf-8"))
    if not m:
        raise SystemExit(f"{opis}: brak pola „pas -6 dB:” — nie wiem, gdzie leży filtr")
    return 0.5 * (float(m.group(1)) + float(m.group(2)))


def szum_dlugosci(szum: np.ndarray, n: int) -> np.ndarray:
    """Szum przedłużony odbiciem w czasie (jak bank w generatorze)."""
    while szum.size < n:
        szum = np.concatenate([szum, szum[::-1]])
    return szum[:n]


def amplituda_nosnej(a: np.ndarray, ton: float) -> float:
    """Amplituda tonu przy wciśniętym kluczu: obwiednia w paśmie ±100 Hz,
    95. percentyl (czysty sygnał — szczyt obwiedni to nośna)."""
    from scipy.signal import hilbert
    x = tune.bandpass(a, ton, 100.0, taps=255)
    return float(np.percentile(np.abs(hilbert(x.astype(np.float64))), 95))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("zrodlo", nargs="+",
                    help="czysty.wav tekst.txt  albo (z --syntetyk) sam tekst.txt")
    ap.add_argument("--syntetyk", type=float, metavar="WPM",
                    help="sygnał kluczownika policzony z tekstu w tym tempie")
    ap.add_argument("--sn", type=float, nargs="+", default=[20, 10, 6, 3, 0])
    ap.add_argument("--szum", default="7025kHz_20261009_2135_500Hz",
                    help="fragment nazwy pliku w probki/szum")
    a = ap.parse_args()

    if a.syntetyk:
        tekst_plik = Path(a.zrodlo[0])
        tekst = tekst_plik.read_text(encoding="utf-8").strip()
        czysty = morse.synth_cw(tekst, wpm=a.syntetyk, tone=C.TONE_CENTER,
                                amp=0.3, rng=np.random.default_rng(0))
        czysty = np.r_[np.zeros(C.SR), czysty, np.zeros(C.SR)].astype(np.float32)
        nazwa = f"{tekst_plik.stem}_syntetyk{a.syntetyk:.0f}wpm"
        tor = f"syntetyk: morse.synth_cw {a.syntetyk:.0f} WPM, rytm maszynowy, narastanie {C.KEY_RAMP_MS:.0f} ms"
    else:
        wav, tekst_plik = Path(a.zrodlo[0]), Path(a.zrodlo[1])
        tekst = tekst_plik.read_text(encoding="utf-8").strip()
        czysty = frontend.load_audio(wav)
        nazwa = wav.stem
        tor = f"nagranie {wav.name}"

    pliki = sorted(KATALOG_SZUMU.glob(f"*{a.szum}*.wav"))
    if len(pliki) != 1:
        raise SystemExit(f"--szum {a.szum!r}: pasuje {len(pliki)} plików w {KATALOG_SZUMU}")
    szum_plik = pliki[0]
    srodek = srodek_pasa(szum_plik.with_suffix(".txt"))

    t, f = tune.track_tone(czysty)
    czysty = tune.retune(czysty, t, f, target=srodek)
    szum = szum_dlugosci(frontend.load_audio(szum_plik), czysty.size).astype(np.float64)
    nosna = amplituda_nosnej(czysty, srodek)

    WYJSCIE.mkdir(parents=True, exist_ok=True)
    for cel in a.sn:
        amp = sn.amp_dla_sn(cel, szum, srodek)
        x = czysty.astype(np.float64) * (amp / nosna) + szum
        x = (x * (0.5 / np.max(np.abs(x)))).astype(np.float32)
        zm = sn.zmierz_sn(x, srodek)["sn_db"]
        wyj = WYJSCIE / f"{nazwa}_{szum_plik.stem.split('_', 3)[3]}_sn{cel:+.0f}.wav"
        sf.write(wyj, x, C.SR)
        wyj.with_suffix(".txt").write_text(
            "# PRÓBKA ZASZUMIONA — tools/zaszum.py (18.10)\n\n"
            f"nadane:        {tekst}\n"
            f"tor:           {tor}\n"
            f"szum:          {szum_plik.name} (pas -6 dB, środek {srodek:.0f} Hz)\n"
            f"ton:           {srodek:.0f} Hz (przesunięty na środek pasa szumu)\n"
            f"S/N:           zadane {cel:+.1f} dB, zmierzone {zm:+.1f} dB (dsp/sn, 500 Hz)\n",
            encoding="utf-8")
        print(f"{wyj.name}: S/N zadane {cel:+.0f} dB, zmierzone {zm:+.1f} dB")


if __name__ == "__main__":
    main()
