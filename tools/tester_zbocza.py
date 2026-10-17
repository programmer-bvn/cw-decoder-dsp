"""TESTER SIECI ZBOCZY NA NAGRANIACH (17.10). Prawdy o zboczach w nagraniu nie
znamy, więc sprawdzian pośredni: grzebień na przebiegu z SIECI kontra grzebień
na progu KLASYCZNYM (grzebien.odcinki_przebiegu) — te same okna 8 s co 4 s,
ta sama bramka S/N >= 3 dB, odczyt tabelą Morse'a z nut. Miary: pewność
grzebienia (jak dobrze krawędzie leżą na siatce) i słowa z pliku .txt.
Zapisuje regenerat wzorcowy każdej drogi do odsłuchu.

    python -m tools.tester_zbocza runs/zbocza_1/best.keras out/zbocza/regenerat [S/N]

S/N (opcja): do nagrania dosypany szum pasmowy do tego S/N [dB] — DIAGNOSTYKA
zakresu treningu: czy sieć gubi się na obrazach czystszych niż w treningu.
"""
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import soundfile as sf

from dsp import config as C, frontend, grzebien, sn, tempo, tune, zbocza as Z
from tools import nagrania, wav2net


def slowa(tekst, chce):
    pod = kol = 0
    for tok, kr in Counter(chce).items():
        pod += min(kr, nagrania.ile_podciagow(tok, tekst))
        kol += min(kr, nagrania.ile_w_kolejnosci(tok, tekst))
    return pod, kol


def droga(a, oo_okna, starty, n8):
    """Nuty z kolejnych okien (jak tools/tester_grzebien) -> (tekst, nuty, pewn, wpm)."""
    wszystkie, pewn, wpm, granica = [], [], [], 0.0
    for i, s in enumerate(starty):
        u0 = tempo.zmierz_tempo(a[s:s + n8])["u_s"]
        r = grzebien.dopasuj_grzebien(oo_okna[i],
                                      u_start=u0 if np.isfinite(u0) else None)
        ostatnie = i == len(starty) - 1
        od, do = granica, (a.size / C.SR if ostatnie else s / C.SR + 6.0)
        granica = do
        if not np.isfinite(r["u_s"]):
            continue
        s_n = sn.zmierz_sn(a[s:s + n8], C.TONE_CENTER)["sn_db"]
        if not (np.isfinite(s_n) and s_n >= 3.0 and r["pewnosc"] >= 0.85):
            continue
        pewn.append(r["pewnosc"]); wpm.append(r["wpm"])
        for st, el in grzebien.nuty(oo_okna[i], r["u_s"], r["waga"] * r["u_s"]):
            ts = s / C.SR + st
            if od <= ts < do:
                wszystkie.append((ts, el, r["u_s"]))
    wszystkie.sort(key=lambda x: x[0])
    nuty_, ost = [], -1.0
    for ts, el, u in wszystkie:
        if ts > ost + 1e-3:
            nuty_.append((ts, el, u)); ost = ts
    tekst = grzebien.odczyt_klasyczny([(ts, el) for ts, el, _ in nuty_])
    return tekst, nuty_, pewn, wpm


def main():
    from tensorflow import keras
    model = keras.models.load_model(sys.argv[1] if len(sys.argv) > 1
                                    else "runs/zbocza_1/best.keras")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "out/zbocza/regenerat")
    out.mkdir(parents=True, exist_ok=True)
    dosyp = float(sys.argv[3]) if len(sys.argv) > 3 else None
    n8, k4 = 8 * C.SR, 4 * C.SR
    # Drogi: próg klasyczny, sieć z jednym progiem, sieć z histerezą,
    # sieć z histerezą i odrzucaniem krótkich impulsów (17.10).
    DROGI = ("klasyczna", "siec", "siec_hist", "siec_hk")
    sumy = np.zeros((len(DROGI), 2), int); razem = 0
    print("grzebień + tabela Morse'a z nut; drogi: " + ", ".join(DROGI))
    print(f"{'nagranie':42s} " + " ".join(f"{d:>14s}" for d in DROGI)
          + "   (pewność/okien, słowa w całości)")
    for p in sorted(Path("probki").glob("*.wav")):
        chce = nagrania.oczekiwane(p.with_suffix(".txt"))
        a = frontend.load_audio(p)
        t, f = tune.track_tone(a)
        a = tune.retune(a, t, f)
        g = wav2net.auto_gain_db(a)
        a = np.clip(a * 10 ** (g / 20), -1, 1).astype(np.float32)
        if dosyp is not None:
            from dsp import radio
            szum = radio.band_noise(a.size, np.random.default_rng(1), rms=0.015,
                                    tilt=-0.5).astype(np.float64)
            amp_ton = float(np.percentile(np.abs(a), 99.9))
            k = amp_ton / sn.amp_dla_sn(dosyp, szum, C.TONE_CENTER)
            a = (a + k * szum).astype(np.float32)
            a = np.clip(a * 10 ** (min(0.0, wav2net.auto_gain_db(a)) / 20), -1, 1).astype(np.float32)
        starty = list(range(0, max(1, a.size - n8 + 1), k4))
        if starty[-1] < a.size - n8:
            starty.append(a.size - n8)
        pz = Z.przebieg_sieci(a, model)
        r5 = n8 // Z.HOP_ZBOCZA
        oo = {d: [] for d in DROGI}
        for s in starty:
            u0 = tempo.zmierz_tempo(a[s:s + n8])["u_s"]
            wyg = 0.5 * u0 if np.isfinite(u0) else 0.015
            oo["klasyczna"].append(grzebien.odcinki_przebiegu(a[s:s + n8], wygladzenie_s=wyg))
            j = s // Z.HOP_ZBOCZA
            pp = pz[j:j + r5]
            hist = (Z.PROG_WL, Z.PROG_WYL)
            oo["siec"].append(Z.odcinki_z_przebiegu(pp))
            oo["siec_hist"].append(Z.odcinki_z_przebiegu(pp, histereza=hist))
            oo["siec_hk"].append(Z.odcinki_z_przebiegu(
                pp, histereza=hist,
                min_s=Z.MIN_ODCINEK_U * u0 if np.isfinite(u0) else None))
        wyniki = [droga(a, oo[d], starty, n8) for d in DROGI]
        for nazwa, (tekst, nuty_, _, _) in zip(DROGI, wyniki):
            sf.write(out / f"{p.stem}_{nazwa}_wzorcowo.wav",
                     grzebien.regeneruj_wzorcowo(nuty_), C.SR)
        kol = []
        sl = [slowa(x[0], chce) for x in wyniki] if chce else None
        for k, w_ in enumerate(wyniki):
            pe = np.median(w_[2]) if w_[2] else float("nan")
            sw = f" {sl[k][0]}/{len(chce)}" if chce else "    "
            kol.append(f"{pe:4.2f}/{len(w_[2]):3d}{sw:>5s}")
        if chce:
            razem += len(chce)
            for k in range(len(DROGI)):
                sumy[k] += sl[k]
        print(f"{p.name[:42]:42s} " + " ".join(f"{x:>14s}" for x in kol))
        if not chce:
            for d, w_ in zip(DROGI, wyniki):
                print(f"    {d:10s}: {w_[0][:70]}")
    print(f"{'RAZEM słowa w całości / w kolejności':42s} "
          + " ".join(f"{f'{x}/{razem} {y}/{razem}':>14s}" for x, y in sumy))


if __name__ == "__main__":
    main()
