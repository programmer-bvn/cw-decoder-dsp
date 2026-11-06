"""Zbocza czy wnętrza: sieć zboczy kontra etap A elementów + grzebień (06.11).

Rozmowa z operatorem 06.11: obie drogi odtwarzają prostokąt z keyera. Sieć
zboczy wyznacza każde zbocze z 34 ms wokół niego; grzebień ustawiony w fazie
największego prawdopodobieństwa kładzie zbocza na zębach — położenie wynika
z całego okna, nie z jednej krawędzi. Pierwsze porównanie w TYCH SAMYCH
jednostkach, jeszcze bez etapu B (grupowania):

  syntetyk  — te same klipy (zakresy zboczy_6), błąd położenia zboczy
              (zbocza.blad_zboczy, jak tools.zbocza_trening ocen), zgubione
              i fałszywe zbocza, trafne ramki 5 ms; w przedziałach S/N;
  negatywy  — elementy (zbocza narastające) na minutę, minuty spoza
              treningu (TRENING.txt), siatka 25 WPM, ton 750 Hz.

Etap A: wodospad z siatką co u/4, wszystkie przegródki (każda kolumna
początkiem przegródki), faza = ta z 4, w której suma log max P po
przegródkach największa; przegródki „ton” (P > 0,5) -> prostokąt.
Jednostka siatki „zmierzona” z błędem +/-U_BLAD (cepstrum na kluczowniku
myli się o 1-2%, HISTORIA 18.10), ton +/-15 Hz jak w treningu.
Porównanie tylko w części klipu pokrytej przegródkami (brzegi bez okna FFT).

    python -m tools.porownaj_przegrodki --zbocza runs/zbocza_6/best.keras \
        --elementy runs/elementy_1/best.keras [--n 150] [--bank ...] [--out plik]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsp import config as C, elementy as E, radio, zbocza as Z

PRZEDZIALY_SN = ((-3, 3), (3, 10), (10, 20), (20, 31), (31, 46))
U_BLAD = 0.03
DT = Z.HOP_ZBOCZA / C.SR


def p_elementow(model, audio, u, ton):
    """P [przegródki od każdej kolumny, 3] i początki przegródek [s]."""
    from tools.elementy_trening import _przewiduj
    t = E.kolumny_czasu(audio.size, u)
    db = E.wodospad(audio, ton, u, t)
    X, _, _ = E.wycinki(db, np.zeros(t.size, np.uint8), np.zeros(t.size, np.float32), krok=1)
    if not len(X):
        return np.zeros((0, 3)), np.zeros(0)
    P = _przewiduj(model, X)
    start = t[E.K:E.K + len(P)] - u / E.K / 2
    return P, start


KARA_KOTWICY = 1.0       # log-prawd. za przesunięcie przegródki o u/K względem poprzedniej


def sciezka_kotwicy(P):
    """Kotwica (operator 06.11): kolejna przegródka zaczyna się K kolumn po
    poprzedniej, +/-1 kolumna (u/K) z karą — faza grzebienia idzie za
    sygnałem, błąd jednostki i waga nie rozjeżdżają siatki. Programowanie
    dynamiczne po kolumnach: najlepsza suma log max P minus kary."""
    n = len(P)
    w = np.log(P.max(axis=1) + 1e-9)
    best = np.full(n, -np.inf); skad = np.full(n, -1)
    best[:E.K + 1] = w[:E.K + 1]
    for j in range(E.K + 1, n):
        for d in (-1, 0, 1):
            i = j - E.K - d
            if 0 <= i and best[i] - KARA_KOTWICY * abs(d) + w[j] > best[j]:
                best[j] = best[i] - KARA_KOTWICY * abs(d) + w[j]; skad[j] = i
    j = int(np.argmax(np.where(np.arange(n) >= n - E.K - 1, best, -np.inf)))
    sc = [j]
    while skad[j] >= 0:
        j = int(skad[j]); sc.append(j)
    return np.array(sc[::-1])


def przebieg_przegrodek(P, start, u, n_ramek):
    """P(ton) przegródki = stan klucza w jej ŚRODKU. Wycinek idzie co u/K,
    więc P co u/K w środkach przegródek to wprost odtworzony prostokąt —
    bez wyboru fazy. Interpolacja na ramki 5 ms; zwraca też zakres ramek
    pokryty środkami."""
    tf_ = np.arange(n_ramek) * DT
    if not len(P):
        return np.zeros(n_ramek, np.float32), (0, 0)
    c = start + u / 2
    p = np.interp(tf_, c, P[:, E.TON]).astype(np.float32)
    return p, (int(np.ceil(c[0] / DT)) + 1, int(c[-1] // DT) - 1)


def klucz_z_przegrodek(P, start, u, n_probek, sr=C.SR, kotwica=True):
    """Przegródki -> (klucz w próbkach, zakres pokryty [a, b) w próbkach).
    kotwica=False: jedna faza na całe okno (największa średnia log max P)."""
    if not len(P):
        return np.zeros(n_probek, np.float32), (0, 0)
    if kotwica:
        idx = sciezka_kotwicy(P)
    else:
        wyn = [np.log(P[f::E.K].max(axis=1) + 1e-9).mean() for f in range(min(E.K, len(P)))]
        idx = np.arange(int(np.argmax(wyn)), len(P), E.K)
    Pf, sf = P[idx], start[idx]
    k = np.zeros(n_probek, np.float32)
    L = int(round(u * sr))
    for p, s in zip(Pf, sf):
        if p[E.TON] > 0.5:
            a = int(round(s * sr))
            k[max(a, 0):max(a + L, 0)] = 1.0
    a = int(round(sf[0] * sr))
    return k, (max(a, 0), min(int(round(sf[-1] * sr)) + L, n_probek))


def _elementy(p):
    d = np.diff(np.r_[0, (p > 0.5).astype(np.int8), 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def elementy_zgodnosc(p, prawda, uf):
    """[prawdziwych, zgubionych, fałszywych, zły typ]: element = odcinek P > 0,5;
    prawdziwy trafiony, gdy przewidziany zaczyna się < 0,5 u od niego (każdy
    przewidziany raz); typ: kropka < 2 u <= kreska. Elementy przecięte brzegiem
    okna liczą się jak inne — brzegi są te same dla obu dróg."""
    (ps, pe), (ts, te) = _elementy(p), _elementy(prawda)
    uzyte = np.zeros(ps.size, bool)
    zg = zly = 0
    for a, b in zip(ts, te):
        d = np.abs(ps - a).astype(float)
        d[uzyte] = np.inf
        j = int(np.argmin(d)) if d.size else -1
        if j < 0 or d[j] >= 0.5 * uf:
            zg += 1
            continue
        uzyte[j] = True
        zly += int(((pe[j] - ps[j]) >= 2 * uf) != ((b - a) >= 2 * uf))
    return np.array([ts.size, zg, int((~uzyte).sum()), zly])


def syntetyk(mz, me, n, seed):
    rng = np.random.default_rng([seed, 6])
    linie = [f"SYNTETYK — {n} klipów na przedział S/N, siatka: u +/-{U_BLAD:.0%}, ton +/-15 Hz",
             "  elementy (odcinki P>0,5) zgubione / fałszywe / zły typ (kropka-kreska) [% prawdziwych];",
             "  zbocze: |mediana| błędu [ms], w nawiasie klipy z wagą keyera < 0,1 u; ramki 5 ms trafnie"]
    for lo, hi in PRZEDZIALY_SN:
        wyniki = {"zbocza": [], "przegródki": [], "kotwica": []}
        zebrane = 0
        while zebrane < n:
            tekst = "".join(rng.choice(list(C.ALPHABET[1:]), E.ZNAKOW))
            wpm = float(rng.uniform(*C.TEMPO_ZAKRES))
            audio, m = radio.receive(tekst, rng, n_samples=C.CLIP_SAMPLES, wpm=wpm, realism=True,
                                     tempo_staly=True, z_kluczem=True, zakresy=Z.ZAKRESY_ZBOCZY_6)
            if not lo <= m["snr_db"] < hi:
                continue
            zebrane += 1
            u = E.jednostka(wpm) * (1 + rng.uniform(-U_BLAD, U_BLAD))
            P, st = p_elementow(me, audio, u, m["tone"] + rng.uniform(-15, 15))
            nr = 1 + audio.size // Z.HOP_ZBOCZA
            pe, (fa, fb) = przebieg_przegrodek(P, st, u, nr)
            ke, (a, b) = klucz_z_przegrodek(P, st, u, audio.size)
            prawda = Z.klucz_ramek(m["klucz"], nr)[fa:fb]
            pz = Z.przebieg_sieci(audio, mz)[:nr][fa:fb]
            uf = E.jednostka(wpm) / DT
            for nazwa, p in (("zbocza", pz), ("przegródki", pe[fa:fb]),
                             ("kotwica", Z.klucz_ramek(ke, nr)[fa:fb])):
                r = Z.blad_zboczy(p, prawda, tol_ramek=0.5 * uf)
                r["ramki"] = float(np.mean((p > 0.5) == (prawda > 0.5)))
                r["el"] = elementy_zgodnosc(p, prawda, uf)
                r["waga"] = abs(m["waga"])
                wyniki[nazwa].append(r)
        for nazwa, rs in wyniki.items():
            b_ = np.abs(np.concatenate([r["bledy"] for r in rs] or [[np.nan]])) * DT * 1000
            lek = [r for r in rs if r["waga"] < 0.1]
            b0 = np.abs(np.concatenate([r["bledy"] for r in lek] or [[np.nan]])) * DT * 1000
            pr = max(sum(r["prawdziwych"] for r in rs), 1)
            el = np.sum([r["el"] for r in rs], axis=0)
            linie.append(f"  S/N {lo:3d}..{hi:<3d} dB  {nazwa:10s}: elementy zgubione {100 * el[1] / el[0]:5.1f}%,"
                         f" fałszywe {100 * el[2] / el[0]:5.1f}%, zły typ {100 * el[3] / el[0]:4.1f}%;"
                         f"  zbocze {np.nanmedian(b_):4.1f} ms (waga<0,1: {np.nanmedian(b0):4.1f});"
                         f"  ramki {np.mean([r['ramki'] for r in rs]):.1%}")
        print("\n".join(linie[-3:]), flush=True)
    return linie


def negatywy(mz, me, bank, wpm=25.0):
    from dsp import frontend
    from tools.odczyt import przygotuj
    tr = bank / "TRENING.txt"
    trening = set(tr.read_text(encoding="utf-8").split()) if tr.exists() else set()
    u = E.jednostka(wpm)
    linie = [f"NEGATYWY — elementy (zbocza narastające) na minutę; dźwięk jak w odczycie (przygotuj), siatka {wpm:.0f} WPM"]
    for kat in sorted(p for p in bank.iterdir() if p.is_dir() and p.name not in ("skrypty", "surowe_kiwi")):
        pliki = [p for p in sorted(kat.glob("*.wav")) if f"{kat.name}/{p.name}" not in trening]
        if not pliki:
            continue
        ez, ee, minuty = [], [], 0.0
        for p in pliki:
            # Jak odczyt (tools.odczyt.przygotuj): najsilniejszy ton na 750 Hz,
            # wycięcie obcych, wzmocnienie — tak powstają halucynacje w aplikacji.
            # Pierwszy bieg 06.11 na surowym dźwięku: sieć zboczy 0 elementów/min
            # w pokoju, a przez odczyt ta sama sieć 12 znaków/min.
            x, _ = przygotuj(frontend.load_audio(p))
            mi = x.size / C.SR / 60
            ez.append(len(Z.zbocza(Z.przebieg_sieci(x, mz))[0]) / mi)
            n4, ile = C.CLIP_SAMPLES, 0
            for s in range(0, x.size - n4 + 1, n4):
                P, st = p_elementow(me, x[s:s + n4], u, C.TONE_CENTER)
                ke, _ = klucz_z_przegrodek(P, st, u, n4)
                ile += int(np.sum(np.diff(np.r_[0.0, ke]) > 0))
            ee.append(ile / mi)
            minuty += mi
        linie.append(f"  {kat.name:15s} ({minuty:5.1f} min): zbocza {np.mean(ez):6.1f} /min (mediana {np.median(ez):5.1f}),"
                     f"  przegródki {np.mean(ee):6.1f} /min (mediana {np.median(ee):5.1f})")
        print(linie[-1], flush=True)
    return linie


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--zbocza", type=Path, default=Path("runs/zbocza_6/best.keras"))
    ap.add_argument("--elementy", type=Path, default=Path("runs/elementy_1/best.keras"))
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--u-blad", type=float, default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--bank", type=Path, default=Path(os.environ.get("CW_NEGATYWY", "/mnt/w/nagrania_cw/negatywy")))
    ap.add_argument("--bez-negatywow", action="store_true")
    ap.add_argument("--bez-syntetyku", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    if a.u_blad is not None:
        globals()["U_BLAD"] = a.u_blad
    from tensorflow import keras
    mz, me = keras.models.load_model(a.zbocza), keras.models.load_model(a.elementy)
    linie = [f"zbocza: {a.zbocza}   elementy: {a.elementy}"]
    if not a.bez_syntetyku:
        linie += syntetyk(mz, me, a.n, a.seed)
    if not a.bez_negatywow:
        linie += negatywy(mz, me, a.bank)
    if a.out:
        a.out.write_text("\n".join(linie) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
