"""KALIBRACJA SKALI dB: mierzy, gdzie naprawdę leżą szum i sygnał.

    python -m tools.calibrate
    python -m tools.calibrate --n 300 --noise 0.02

DLACZEGO TO ISTNIEJE
--------------------
DB_MIN i DB_MAX w config.py decydują o tym, co na obrazie jest czarne, a co
białe. W starym projekcie te liczby były zgadywane i za każdym podejściem
inne (-30/0, -35/+5, -50/-10) — stąd raz "różowe ściany", raz obraz prawie
czarny.

Nie da się ich wyprowadzić na piechotę, bo zależą od kilku rzeczy naraz:
librosa NIE normalizuje STFT przez sumę okna (więc poziom skaluje się
z n_fft), a filtrbank mel z norm="slaney" dzieli każde pasmo przez jego
szerokość w hercach (przy 32 pasmach na 400-1200 Hz to dzielenie przez ~48).
Łącznie daje to przesunięcie kilkudziesięciu decybeli względem "naiwnego"
rachunku z amplitudy sygnału.

Więc się to MIERZY. Narzędzie generuje klipy dokładnie takie, jakie robi
generator, i podaje rozkład decybeli osobno dla tła i dla sygnału.

CO Z TYM ZROBIĆ
---------------
Przepisz zaproponowane wartości do config.py i wygeneruj zbiór od nowa.
Odcisk front-endu (config.fingerprint) pilnuje, żeby stary zbiór nie
wmieszał się do treningu po zmianie tych stałych.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsp import config as C
from dsp import frontend, morse


def measure(n: int, noise_rms: float, amp_lo: float, amp_hi: float,
            seed: int = 0):
    """Zwraca (db_tła, db_sygnału) — dwa zbiory pomiarów w decybelach.

    Tło mierzymy na klipach BEZ sygnału (czysty szum), żeby nie zaniżyć go
    ramkami, w których akurat nadawano. Sygnał mierzymy jako 99,9 percentyl
    klipu z tonem — czyli szczyt elementu klucza w pasmie tonu.
    """
    rng = np.random.default_rng(seed)
    bg, sig = [], []

    for i in range(n):
        # --- klip tylko z szumem ---
        noise = rng.normal(0.0, noise_rms, C.CLIP_SAMPLES).astype(np.float32)
        db_n = frontend.power_to_db(frontend.melspec_power(noise))
        bg.append([float(np.percentile(db_n, 5)),
                   float(np.percentile(db_n, 50)),
                   float(np.percentile(db_n, 95))])

        # --- klip z tonem ---
        audio = rng.normal(0.0, noise_rms, C.CLIP_SAMPLES).astype(np.float32)
        amp = float(rng.uniform(amp_lo, amp_hi))
        tone = C.TONE_CENTER + rng.uniform(-C.TONE_SPREAD, C.TONE_SPREAD)
        wave = morse.synth_cw("PARIS", wpm=C.WPM, tone=tone, amp=amp)
        s = max(0, (audio.size - wave.size) // 2)
        m = min(wave.size, audio.size - s)
        audio[s:s + m] += wave[:m]

        db_s = frontend.power_to_db(frontend.melspec_power(audio))
        sig.append([float(np.percentile(db_s, 99.0)),
                    float(np.percentile(db_s, 99.9)),
                    float(np.max(db_s))])

    return np.array(bg), np.array(sig)


def report(bg: np.ndarray, sig: np.ndarray, noise_rms: float,
           amp_lo: float, amp_hi: float) -> tuple[float, float]:
    print(f"\nSZUM (RMS={noise_rms}), {len(bg)} klipów — rozkład dB w obrazie:")
    print(f"   5 percentyl: {bg[:,0].mean():8.1f} dB  "
          f"(odchylenie {bg[:,0].std():.2f})")
    print(f"  50 percentyl: {bg[:,1].mean():8.1f} dB  "
          f"(odchylenie {bg[:,1].std():.2f})")
    print(f"  95 percentyl: {bg[:,2].mean():8.1f} dB  "
          f"(odchylenie {bg[:,2].std():.2f})")

    print(f"\nSYGNAŁ (amplituda {amp_lo}-{amp_hi}), szczyty w obrazie:")
    print(f"  99   percentyl: {sig[:,0].mean():8.1f} dB")
    print(f"  99.9 percentyl: {sig[:,1].mean():8.1f} dB")
    print(f"  maksimum:       {sig[:,2].mean():8.1f} dB  "
          f"(najwyższe {sig[:,2].max():.1f})")

    # DB_MIN: nieco POWYŻEJ mediany szumu, żeby tło poszło do zera i było
    # naprawdę czarne, ale bez ucinania słabszych sygnałów — bierzemy
    # 95 percentyl szumu, czyli poziom, którego zwykły szum prawie nie
    # przekracza.
    db_min = float(np.floor(bg[:, 2].mean()))
    # DB_MAX: szczyt najsilniejszego sygnału, zaokrąglony w górę. Powyżej
    # tego wszystko jest białe — obcięcie szczytu nie boli, bo informacja
    # w Morse'ie jest w KSZTAŁCIE obwiedni, nie w poziomie.
    db_max = float(np.ceil(sig[:, 2].max()))

    span = db_max - db_min
    print("\n" + "=" * 70)
    print("PROPOZYCJA DO config.py")
    print("=" * 70)
    print(f"DB_MIN = {db_min:.1f}")
    print(f"DB_MAX = {db_max:.1f}")
    print(f"        rozpiętość {span:.0f} dB  "
          f"({span/255:.3f} dB na krok przy zapisie uint8)")

    print(f"\nObecnie w config.py: DB_MIN={C.DB_MIN}  DB_MAX={C.DB_MAX} "
          f"(rozpiętość {C.db_span():.0f} dB)")

    lo_clip = float(np.mean(sig[:, 1] < C.DB_MIN))
    hi_clip = float(np.mean(sig[:, 2] > C.DB_MAX))
    print(f"  sygnałów całkowicie pod DB_MIN (obraz czarny): "
          f"{lo_clip*100:.1f}%")
    print(f"  sygnałów obciętych u góry przy DB_MAX:         "
          f"{hi_clip*100:.1f}%")

    # O tym, czy tło jest "szare", decyduje jego poziom PO NORMALIZACJI,
    # a nie samo przekroczenie DB_MIN. Szum w dolnych kilku procentach
    # skali jest ciemny i nieszkodliwy — i tak ma być, bo zostawia miejsce
    # na sygnał słabszy od tego z generatora.
    bg_norm_med = (bg[:, 1].mean() - C.DB_MIN) / C.db_span()
    bg_norm_p95 = (bg[:, 2].mean() - C.DB_MIN) / C.db_span()
    print(f"  poziom tła po normalizacji: mediana {bg_norm_med:.3f}, "
          f"95 pct {bg_norm_p95:.3f}  (skala 0-1)")

    if bg_norm_p95 > 0.35:
        print("    -> TŁO ZA JASNE. Szum zajmuje ponad trzecią część skali "
              "i model będzie się uczył jego tekstury.\n"
              "       To objaw, który w v5.5 dawał 'różowe ściany'. "
              "Podnieś DB_MIN.")
    elif bg_norm_p95 < 0.01:
        print("    -> tło całkowicie przycięte do zera. Czarne i czyste, "
              "ale sygnał słabszy\n"
              "       od tła też zniknie — na pasmie to strata. "
              "Rozważ niższe DB_MIN.")
    else:
        print("    -> tło ciemne, z zapasem na słabszy sygnał. "
              "Tak ma być.")

    if hi_clip > 0.5:
        print("    -> szczyty obcięte. Dla Morse'a to akceptowalne "
              "(kształt obwiedni zostaje), ale różnice poziomu "
              "między stacjami znikają.")

    kontrast = sig[:, 1].mean() - bg[:, 1].mean()
    print(f"\n  kontrast sygnał-tło: {kontrast:.1f} dB "
          f"= {kontrast/C.db_span():.2f} skali")
    return db_min, db_max


def main(argv=None):
    ap = argparse.ArgumentParser(description="Kalibracja skali dB")
    ap.add_argument("--n", type=int, default=200,
                    help="liczba klipów pomiarowych")
    ap.add_argument("--noise", type=float, default=C.NOISE_RMS)
    ap.add_argument("--amp-lo", type=float, default=C.SIGNAL_AMP_MIN)
    ap.add_argument("--amp-hi", type=float, default=C.SIGNAL_AMP_MAX)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    print("=" * 70)
    print("KALIBRACJA SKALI dB")
    print("=" * 70)
    print(C.summary())
    print("-" * 70)
    print(f"mierzę {args.n} klipów szumu i {args.n} klipów z sygnałem...")

    bg, sig = measure(args.n, args.noise, args.amp_lo, args.amp_hi, args.seed)
    report(bg, sig, args.noise, args.amp_lo, args.amp_hi)


if __name__ == "__main__":
    main()
