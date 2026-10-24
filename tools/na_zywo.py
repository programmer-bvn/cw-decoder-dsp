"""ODCZYT NA ŻYWO (21.10; strumieniowy v3 23.10): dźwięk kawałkami -> tekst.

    python -m tools.na_zywo symuluj nagranie.wav [...]     # nagranie w tempie rzeczywistym
    python -m tools.na_zywo okno  [--device NR|NAZWA] [--civ]   # okienko: tekst, suwak blokady szumów
    python -m tools.na_zywo karta [--device NR|NAZWA] [--civ]   # to samo w konsoli (karta: „USB Audio”)

Z kartą graficzną (karta dźwiękowa i COM są w Windows, WSL2 ich nie widzi):
    WSL:      python -m tools.na_zywo serwer               # dekoder, GPU, tekst
    Windows:  python -m tools.na_zywo most --civ
    (oba naraz: start_na_zywo_gpu.bat; próba bez radia: most --wav nagranie.wav)

Wersje (tagi): na_zywo_v1 — bufor 20 s liczony od nowa co 2,5 s, znak po
6-7 s; na_zywo_v2 — bufor 12 s co >= 0,5 s, tekst wstępny po 1-2 s
(operator 23.10: „szybciej, ale jeszcze nie wicher"). v1 i v2 liczyły każdą
próbkę ~8-24 razy; v3 liczy ją raz-dwa:

- ANALIZA co ANALIZA_CO_S na ostatnich ANALIZA_S (w tle, osobny wątek):
  pętla tonu -> ton naszej; obce do skasowania (te same progi co
  przeciwsygnał w odczycie całości); wzmocnienie;
- co KROK_S tylko KONCOWKA_S: odejmowanie obcych, przestrojenie, obraz,
  JEDNO okno sieci (2,56 s) kończące się teraz. Ramka P zamarza, gdy
  wyjdzie za środek okna — ma wtedy po 1,28 s kontekstu z obu stron, jak
  w odczycie całości (środkowa połowa okien);
- Viterbi (numba, ~10 ms) na ostatnich VITERBI_S przebiegu P. Znak jest
  ZATWIERDZANY, gdy skończył się co najmniej OPOZNIENIE_S przed końcem;
  młodsze to TEKST WSTĘPNY (szary, przerysowywany co krok).
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsp import blokada as B, config as C, frontend, viterbi as V, zbocza as Z  # noqa: E402
from dsp.grzebien import _ODWROTNY, NIEZNANY  # noqa: E402

KROK_S = 0.25
OPOZNIENIE_S = 4.0
ANALIZA_S = 10.0           # skala przeciwsygnału (okna 10 s) i pętli tonu (6 s)
ANALIZA_CO_S = 2.0
KONCOWKA_S = 4.0           # okno sieci 2,56 s + kontekst odejmowania/przestrojenia
SUROWE_S = 12.0
# Po nowej analizie przeliczane od nowa ostatnie PRZELICZ_S: zatwierdzanie
# 4 s + 1,28 s kontekstu sieci z lewej + zapas.
PRZELICZ_S = 7.0
# Ton naszej = mediana pętli z ostatnich TON_S analizy (dryf, zmiana stacji).
TON_S = 3.0
# Cepstrum co JEDNOSTKA_CO_S dźwięku, grzebień (1,5 s) co GRZEBIEN_CO_S; między
# nimi poprzednia jednostka — tempo stacji zmienia się wolniej niż co 10 s.
JEDNOSTKA_CO_S = 12.5
GRZEBIEN_CO_S = 37.5
HISTORIA_S = 20.0          # przebieg P trzymany do jednostki (cepstrum: okna 8 s)
VITERBI_S = 12.0
# Tekst wstępny: litera pokazywana, gdy za nią minęło DOKONCZONA_U jednostek
# ciszy — bez tego „T -> N -> K" w trakcie nadawania (v2: zmian 174 -> 75 na 105 znaków).
DOKONCZONA_U = 2.0
# Blokada szumów (dsp/blokada.py): domyślny próg [dB nad podłogą]; w okienku suwak.
PROG_BLOKADY_DB = 0.0
# Szczyt w oknie analizy o tyle niżej niż ostatniej stacji = cisza, wzmocnienie stoi.
SPADEK_CISZY_DB = 30.0


def znaki_z_odcinkow(odc: list, t0: float) -> list[tuple[str, float, float]]:
    """Odcinki Viterbiego -> [(znak, początek s, koniec s)] w czasie bezwzględnym.
    Spacja dostaje czas przerwy między słowami."""
    out, kod, a0, b0 = [], "", None, None
    for nazwa, a, b in odc:
        if nazwa in ("K1", "K3"):
            if not kod:
                a0 = a
            kod += "." if nazwa == "K1" else "-"
            b0 = b
        elif nazwa in ("PL", "PS", "CI"):
            if kod:
                out.append((_ODWROTNY.get(kod, NIEZNANY), t0 + a0 * V.DT, t0 + b0 * V.DT))
                kod = ""
            if nazwa in ("PS", "CI") and out and out[-1][0] != " ":
                out.append((" ", t0 + a * V.DT, t0 + b * V.DT))
    if kod:
        out.append((_ODWROTNY.get(kod, NIEZNANY), t0 + a0 * V.DT, t0 + b0 * V.DT))
    return out


def _przygotuj(x: np.ndarray, n0: int, f_n: float, obce: list[float],
               wzm_db: float, sr: int) -> np.ndarray:
    """Jak tools.odczyt.przygotuj, ale z decyzjami z analizy: odejmowanie
    obcych, przestrojenie naszej na TONE_CENTER (faza od próbki n0 — kawałki
    się sklejają), wzmocnienie."""
    from dsp import przeciwsygnal as PS, tune
    y = np.asarray(x, np.float64)
    if np.isfinite(f_n):
        for fo in obce:
            y = PS.odejmij(y, fo, f_n, sr)
    przes = C.TONE_CENTER - f_n if np.isfinite(f_n) else 0.0
    X = np.fft.fft(y)
    fr = np.fft.fftfreq(y.size, 1.0 / sr)
    an = np.fft.ifft(X * np.where(fr >= tune.RETUNE_FMIN_HZ, 2.0, 0.0))
    y = np.real(an * np.exp(2j * np.pi * przes * (n0 + np.arange(y.size)) / sr))
    return np.clip(y * 10 ** (wzm_db / 20), -1, 1).astype(np.float32)


def _analiza(x: np.ndarray, n0: int, ton_cw: float | None, hist: np.ndarray | None,
             grzebien: bool, sr: int) -> tuple[float, list[float], float, list[float]]:
    """-> (ton naszej, obce [Hz], wzmocnienie dB, kandydatki jednostki).
    Jednostka też tu: grzebień to 1,5 s — w kroku dawał co 37,5 s krok
    2,3 s zamiast 0,06 s (v3, 23.10)."""
    from dsp import przeciwsygnal as PS, tune
    from tools import wav2net
    from tools.odczyt import jednostka_grzebienia
    kand = []
    if hist is not None and hist.size:
        kand.append(V.jednostka_cepstrum(hist)[0])
        if grzebien:
            kand.append(jednostka_grzebienia(hist))
    t, f = tune.track_tone(x, ton_cw=ton_cw)
    ok = np.isfinite(f)
    ost = ok & (t >= (x.size / sr - TON_S))
    f_n = float(np.median(f[ost])) if ost.any() else (float(np.median(f[ok])) if ok.any() else float("nan"))
    obce = PS.obce_w_oknie(x, f_n, sr)
    szczyt = wav2net.measure_peak_db(_przygotuj(x, n0, f_n, obce, 0.0, sr))
    return f_n, obce, (wav2net.TARGET_PEAK_DB - szczyt if ok.any() else None, szczyt), kand


class OdczytNaZywo:
    def __init__(self, model, ton_cw: float | None = None, krok_s: float = KROK_S,
                 opoznienie_s: float = OPOZNIENIE_S, sr: int = C.SR, watek: bool = True):
        self.model, self.ton_cw, self.sr = model, ton_cw, sr
        self.krok, self.opozn = krok_s, opoznienie_s
        self.dwa_wejscia = len(model.inputs) == 2
        self.raw = np.zeros(0, np.float32)
        self.raw0 = 0                      # numer próbki raw[0] od początku
        self.n = 0                         # próbek dotąd
        self.od_kroku = 0                  # próbek od ostatniego kroku
        self.analiza: tuple | None = None  # (ton naszej, obce [Hz], wzmocnienie dB)
        self.ost_analiza = -np.inf         # próbka, przy której ruszyła ostatnia analiza
        self._przyszla = None
        self._przelicz = False             # przyszła nowa analiza: przelicz niezatwierdzone
        self._watek = None
        if watek:
            # PROCES, nie wątek: grzebień i pętle przeciwsygnału to w dużej
            # części Python — wątek trzymał blokadę interpretera i krok w tym
            # czasie trwał 0,6-1,8 s zamiast 0,05 s (v3, 23.10)
            # „spawn" także w WSL: fork kopiowałby proces z działającym TensorFlow
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            self._watek = ProcessPoolExecutor(1, mp_context=mp.get_context("spawn"))
        self.hist = np.zeros(0, np.float32)    # P; hist[0] to ramka h0 (co V.DT od początku)
        self.hist_poz = np.zeros(0, np.float32)  # poziom tonu naszej [dB], ta sama siatka (blokada)
        self.h0 = 0
        self.szczyt_stacji: float | None = None   # szczyt obrazu ostatniej analizy ze stacją [dB]
        self.prog_db = PROG_BLOKADY_DB         # blokada szumów: dB nad podłogą, 0 = wyłączona
        self.nad_szumem_db = 0.0               # ton naszej nad podłogą w ostatnim kroku (wskaźnik)
        self.ost = -np.inf                 # początek ostatniego zatwierdzonego znaku
        self.u = float("nan")
        self.od_cepstrum = self.od_grzebienia = 0.0   # s dźwięku od ostatniego liczenia jednostki
        self.kand: list[float] = []        # jednostki z wątku analizy, do sprawdzenia Viterbim
        self.czasy: list[float] = []       # czas liczenia kroków [s]
        self.zatwierdzone: list[tuple[str, float, float, float]] = []   # znak, od, do, chwila pokazania
        # tekst wstępny: jeszcze niezatwierdzone znaki z ostatniego kroku; może się zmienić
        self.wstepne: list[tuple[str, float, float]] = []
        self.pomiar = False                # symulacja: zapamiętuj wstępne z każdego kroku
        self.kroki: list[tuple[float, list]] = []     # (chwila pokazania, wstępne)
        self.zegar = None                  # symulacja w tempie rzeczywistym: chwila = zegar()
        self.poziom_db = -180.0            # wejście w ostatnim kroku [dBFS]
        self.ost_y, self.ost_y0 = np.zeros(0, np.float32), 0   # ostatni sygnał do sieci, od próbki

    @property
    def wstepny(self) -> str:
        return "".join(z[0] for z in self.wstepne)

    @property
    def stan(self) -> str:
        """Do paska tytułu. 23.10 operator nagrał 43 s sesji bez jednego znaku —
        z ekranu nie dało się poznać, czy w ogóle dochodził dźwięk."""
        # krótko: karta w Windows Terminal ucina tytuł po ~25 znakach (23.10)
        ton = f"{self.analiza[0]:.0f} Hz" if self.analiza and np.isfinite(self.analiza[0]) else "— Hz"
        wpm = f"{1.2 / self.u:.0f} WPM" if np.isfinite(self.u) and self.u > 0 else "— WPM"
        return (f"{self.poziom_db:.0f} dB {ton} {wpm} | "
                f"krok {self.czasy[-1] if self.czasy else 0:.2f} s")

    def dodaj(self, x: np.ndarray) -> str:
        """Kawałek dźwięku (SR) -> nowo zatwierdzony tekst (może być pusty);
        tekst wstępny w self.wstepny."""
        x = np.asarray(x, np.float32)
        self.raw = np.concatenate([self.raw, x])
        self.n += x.size
        self.od_kroku += x.size
        # gdy krok liczy się dłużej niż KROK_S, przez ten czas przyszło tyle dźwięku
        if self.od_kroku < max(self.krok, self.czasy[-1] if self.czasy else 0.0) * self.sr:
            return ""
        return self._krok(koniec=False)

    def koniec(self) -> str:
        """Koniec dźwięku: zatwierdź resztę."""
        if self._przyszla is not None:
            self._odbierz(self._przyszla.result())
            self._przyszla = None
        if self._watek is not None:
            self._watek.shutdown(wait=False)
            self._watek = None
        return self._krok(koniec=True) if self.raw.size else ""

    # ---------------------------------------------------------------- analiza
    def _odbierz(self, wynik: tuple) -> None:
        f_n, obce, (wzm, szczyt) = wynik[:3]
        # Cisza: wzmocnienie zostaje z ostatniej analizy ze stacją (jak ARW
        # w radiu). Sesja 214326 (23.10): w 10 s ciszy wzmocnienie z samego
        # szumu podnosiło szum do poziomu elementów — na żywo 621 znaków („E EI
        # EE ESIE”), odczyt całości tego nagrania 83 znaki, czysto. Samo „brak
        # kluczowanej stacji w oknie” nie wystarczyło (322 znaki): w prawie
        # cyfrowej ciszy (-59 dB, stacja +38 dB) pętla tonu i tak coś łapie.
        cisza = wzm is None or (self.szczyt_stacji is not None and szczyt < self.szczyt_stacji - SPADEK_CISZY_DB)
        if cisza and self.analiza:
            wzm = self.analiza[2]
        else:
            wzm = 0.0 if wzm is None else wzm
            self.szczyt_stacji = szczyt
        self.analiza = (f_n, obce, wzm)
        self.kand += wynik[3]
        self._przelicz = True

    def _analizuj(self) -> None:
        if self._przyszla is not None and self._przyszla.done():
            self._odbierz(self._przyszla.result())
            self._przyszla = None
        if self._przyszla is not None or self.raw.size < 2 * self.sr \
                or self.n - self.ost_analiza < ANALIZA_CO_S * self.sr:
            return
        x = self.raw[-int(ANALIZA_S * self.sr):].copy()
        n0 = self.raw0 + self.raw.size - x.size
        dzwiek = (self.n - self.ost_analiza) / self.sr if np.isfinite(self.ost_analiza) else 0.0
        self.ost_analiza = self.n
        self.od_cepstrum += dzwiek
        self.od_grzebienia += dzwiek
        # póki przebieg krótszy niż okno cepstrum (8 s) — jednostka przy każdej
        # analizie: z pierwszych 2 s wychodziła za krótka i trzymała się 12,5 s
        # („MMM MTM" na początku 40 m, v2 22.10)
        grzebien = self.hist.size * V.DT < 8.0 or self.od_grzebienia >= GRZEBIEN_CO_S
        hist = None
        if grzebien or self.od_cepstrum >= JEDNOSTKA_CO_S:
            hist = self.hist.copy()
            self.od_cepstrum = 0.0
            if grzebien:
                self.od_grzebienia = 0.0
        arg = (x, n0, self.ton_cw, hist, grzebien, self.sr)
        if self._watek is None or self.analiza is None:      # pierwsza: od razu
            self._odbierz(_analiza(*arg))
        else:
            self._przyszla = self._watek.submit(_analiza, *arg)

    # ------------------------------------------------------------------ sieć
    def _siec(self, y: np.ndarray) -> np.ndarray:
        """Jedno okno sieci kończące się na końcu y -> P jego ramek (<= 512)."""
        R = Z.RAMKI_ZBOCZA
        obrazy = [Z.obraz_zboczy(y)] + ([Z.obraz_drobny(y)] if self.dwa_wejscia else [])
        n = obrazy[0].shape[0]
        okna = []
        for o in obrazy:
            if n < R:
                o = np.concatenate([np.zeros((R - n, o.shape[1]), np.float32), o])
            okna.append(o[-R:][None, ..., None])
        logit = Z._logit(self.model, okna if self.dwa_wejscia else okna[0])[0]
        return (1.0 / (1.0 + np.exp(-logit)))[R - min(n, R):].astype(np.float32)

    def _do_historii(self, p: np.ndarray, a0: int, poz: np.ndarray) -> None:
        """P okna od ramki a0. Starsze od środka okna zostają zamrożone."""
        self._wpisz(p, a0, a0 + Z.RAMKI_ZBOCZA // 2, poz)

    def _wpisz(self, p: np.ndarray, a0: int, od: int, poz: np.ndarray) -> None:
        """P (i poziom tonu do blokady, ta sama siatka) od ramki a0; w historii
        zastępuje ramki od `od` (nie wcześniej niż a0)."""
        p, poz = p.astype(np.float32), poz.astype(np.float32)
        if not self.hist.size:
            self.hist, self.hist_poz, self.h0 = p, poz, a0
        else:
            koniec_h = self.h0 + self.hist.size
            if a0 > koniec_h:                              # luka (nie powinna się zdarzyć)
                self.hist = np.concatenate([self.hist, np.zeros(a0 - koniec_h, np.float32)])
                self.hist_poz = np.concatenate([self.hist_poz, np.full(a0 - koniec_h, -120, np.float32)])
                koniec_h = a0
            w = max(a0, self.h0, min(od, koniec_h))
            self.hist = np.concatenate([self.hist[:w - self.h0], p[w - a0:]])
            self.hist_poz = np.concatenate([self.hist_poz[:w - self.h0], poz[w - a0:]])
        nadmiar = self.hist.size - int(HISTORIA_S / V.DT)
        if nadmiar > 0:
            self.hist = self.hist[nadmiar:]
            self.hist_poz = self.hist_poz[nadmiar:]
            self.h0 += nadmiar

    # ------------------------------------------------------------------ krok
    def _krok(self, koniec: bool) -> str:
        from tools.odczyt import jednostka_grzebienia
        t_start = time.perf_counter()
        hop = Z.HOP_ZBOCZA
        koniec_buf = self.n / self.sr
        self.od_kroku = 0
        nowe = ""
        self._analizuj()
        if self.analiza is not None:
            f_n, obce, wzm = self.analiza
            s0 = max(self.raw0, self.n - int(KONCOWKA_S * self.sr))
            s0 = -(-s0 // hop) * hop                       # ramki na wspólnej siatce
            if self._przelicz:
                # nowa analiza (ton, obce, wzmocnienie) -> od nowa cały jeszcze
                # niezatwierdzony odcinek. Bez tego ramki zamarzały z decyzjami
                # sprzed 2 s: na początku nagrania wzmocnienie z samego szumu
                # (+12,6 dB zamiast -4,8 na 6 m) i szum czytany jako „E I" (23.10)
                self._przelicz = False
                s0 = max(self.raw0, self.n - int(PRZELICZ_S * self.sr))
                s0 = -(-s0 // hop) * hop
                y = _przygotuj(self.raw[s0 - self.raw0:], s0, f_n, obce, wzm, self.sr)
                p = Z.przebieg_sieci(y, self.model)
                od = s0 // hop + (Z.RAMKI_ZBOCZA // 2 if s0 > self.raw0 else 0)
                # poziom bez wzmocnienia: wzmocnienie zmienia się co analiza, podłoga nie może skakać
                self._wpisz(p, s0 // hop, od, B.poziom_tonu(y)[:p.size] - wzm)
            else:
                y = _przygotuj(self.raw[s0 - self.raw0:], s0, f_n, obce, wzm, self.sr)
                p = self._siec(y)
                n_ramek = (self.n - s0) // hop + 1
                self._do_historii(p, s0 // hop + n_ramek - p.size, B.poziom_tonu(y)[-p.size:] - wzm)
            self.ost_y, self.ost_y0 = y, s0           # do wodospadu „do sieci” (okno)
            kand, self.kand = self.kand + [self.u], []
            if not any(np.isfinite(k) and k > 0 for k in kand):     # początek: jeszcze nic
                kand = [V.jednostka_cepstrum(self.hist)[0], jednostka_grzebienia(self.hist)]
            n_v = int(VITERBI_S / V.DT)
            pv = self.hist[-n_v:]
            v0 = self.h0 + self.hist.size - pv.size
            pod = B.podloga(self.hist_poz)
            self.nad_szumem_db = float(np.max(self.hist_poz[-int(self.krok / V.DT):] - pod[-int(self.krok / V.DT):]))
            if self.prog_db > 0:
                pv = B.zastosuj(pv, B.otwarta(self.hist_poz[-n_v:], self.prog_db, pod[-n_v:]))
            kand = sorted({round(float(k), 4) for k in kand if np.isfinite(k) and k > 0})
            # jak V.dekoduj_najlepiej (wygrywa najlepszy wynik na ramkę), ale
            # odcinki zwycięzcy od razu — bez drugiego przebiegu Viterbiego
            naj = None
            for k in kand:
                _, odc_k, wynik = V.dekoduj(pv, k, z_wynikiem=True)
                if naj is None or wynik > naj[2]:
                    naj = (k, odc_k, wynik)
            if naj is not None:
                u, odc, _ = naj
                self.u = u
                granica = koniec_buf if koniec else koniec_buf - self.opozn
                self.wstepne = []
                for z, a0, b0 in znaki_z_odcinkow(odc, v0 * V.DT):
                    # nowy znak: zaczyna się po ostatnim zatwierdzonym (z zapasem
                    # pół jednostki na drgania położenia między krokami)
                    if a0 <= self.ost + 0.5 * u:
                        continue
                    if z == " " and (not (self.zatwierdzone or self.wstepne)
                                     or (self.wstepne or self.zatwierdzone)[-1][0] == " "):
                        continue
                    if b0 <= granica and not self.wstepne:
                        self.zatwierdzone.append((z, a0, b0, 0.0))
                        self.ost = a0
                        nowe += z
                    else:
                        self.wstepne.append((z, a0, b0))
                # litera bez przerwy międzyliterowej (>= 2 u) za sobą może być
                # niedokończona (T -> N -> K): nie pokazujemy jej jeszcze
                while not koniec and self.wstepne and self.wstepne[-1][0] != " " and \
                        self.wstepne[-1][2] > koniec_buf - DOKONCZONA_U * u:
                    self.wstepne.pop()
        # surowy dźwięk: ostatnie SUROWE_S, przycięty o całe ramki
        nadmiar = (self.raw.size - int(SUROWE_S * self.sr)) // hop * hop
        if nadmiar > 0:
            self.raw = self.raw[nadmiar:]
            self.raw0 += nadmiar
        ost = self.raw[-int(self.krok * self.sr):]
        self.poziom_db = 20 * np.log10(np.sqrt(np.mean(ost.astype(np.float64) ** 2)) + 1e-9) if ost.size else -180.0
        self.czasy.append(time.perf_counter() - t_start)
        chwila = self.zegar() if self.zegar else koniec_buf + self.czasy[-1]
        if nowe:
            self.zatwierdzone[-len(nowe):] = [z[:3] + (chwila,) for z in self.zatwierdzone[-len(nowe):]]
        if self.pomiar:
            self.kroki.append((chwila, list(self.wstepne)))
        return nowe

    def tekst(self) -> str:
        return " ".join("".join(z[0] for z in self.zatwierdzone).split())


def rozgrzej(model) -> None:
    """Kilka kroków na szumie przed odbiorem: pierwszy krok w WSL trwał 14,4 s
    (22.10 — TensorFlow buduje funkcję sieci, numba kompiluje Viterbiego),
    tekst z początku nadania przychodził z tym opóźnieniem."""
    d = OdczytNaZywo(model, watek=False)
    x = (0.01 * np.random.default_rng(0).standard_normal(3 * d.sr)).astype(np.float32)
    for i in range(0, x.size, d.sr // 10):
        d.dodaj(x[i:i + d.sr // 10])


class Ekran:
    """Tekst zatwierdzony + wstępny (szary) w bieżącym wierszu konsoli.
    Wstępny jest przerysowywany co krok ('\\r' i kasowanie do końca wiersza),
    więc wiersz nie może się zawinąć — łamiemy go sami na spacji."""

    def __init__(self):
        import os
        import shutil
        if os.name == "nt":
            os.system("")                  # włącza sekwencje ANSI w konsoli Windows
        self.szer = max(20, shutil.get_terminal_size().columns - 1)
        self.linia = ""

    def pokaz(self, nowe: str, wstepny: str, stan: str = "") -> None:
        if stan:                           # pasek tytułu okna: czy dźwięk w ogóle dochodzi
            sys.stdout.write("\x1b]0;" + stan + "\x07")
        for z in nowe:
            if z == " " and (not self.linia or self.linia.endswith(" ")):
                continue
            self.linia += z
            if len(self.linia) >= self.szer or (z == " " and len(self.linia) >= self.szer - 12):
                sys.stdout.write("\r" + self.linia + "\x1b[K\n")
                self.linia = ""
        ogon = wstepny.lstrip() if not self.linia or self.linia.endswith(" ") else wstepny
        ogon = ogon[:self.szer - len(self.linia)]
        sys.stdout.write("\r" + self.linia + "\x1b[90m" + ogon + "\x1b[0m\x1b[K")
        sys.stdout.flush()


def opoznienia(d: OdczytNaZywo) -> tuple[list[float], list[float], int]:
    """Z symulacji (d.pomiar): (zatwierdzenie [s], tekst wstępny [s], zmian).

    Opóźnienie liczone od końca znaku do chwili pokazania (koniec dźwięku
    w kroku + czas liczenia). Wstępny: od kroku, od którego znak stał we
    wstępnym BEZ PRZERWY aż do zatwierdzenia. Zmiany: znaki wstępne, które
    zniknęły, a nie zostały zatwierdzone (operator widzi, jak tekst skacze)."""
    import bisect
    tol = 0.04
    chwile = [k[0] for k in d.kroki]
    zat, wst = [], []
    for z, a0, b0, ch in d.zatwierdzone:
        if z == " ":
            continue
        zat.append(ch - b0)
        pierwszy, j = ch, bisect.bisect_left(chwile, ch) - 1
        while j >= 0 and any(w[0] == z and abs(w[1] - a0) < tol for w in d.kroki[j][1]):
            pierwszy, j = d.kroki[j][0], j - 1
        wst.append(pierwszy - b0)
    zmiany = 0
    zat_poz = [(z[0], z[1]) for z in d.zatwierdzone]
    for (_, przed), (_, po) in zip(d.kroki, d.kroki[1:]):
        for w in przed:
            if w[0] == " " or any(v[0] == w[0] and abs(v[1] - w[1]) < tol for v in po):
                continue
            if not any(z == w[0] and abs(a0 - w[1]) < tol for z, a0 in zat_poz[-40:]):
                zmiany += 1
    return zat, wst, zmiany


def symuluj(pliki: list[Path], kawalek_s: float = 0.1, krok_s: float = KROK_S,
            rzeczywisty: bool = True, prog_db: float = PROG_BLOKADY_DB) -> None:
    """Nagranie podawane kawałkami. rzeczywisty: w tempie nagrania, a chwila
    pokazania z zegara — analiza idzie w osobnym wątku, bez tempa rzeczywistego
    jej wynik spóźniałby się o wielokrotność czasu dźwięku."""
    from tensorflow import keras
    from tools.klasyczny import cer
    from tools.odczyt import MODEL_ZBOCZY, odczytaj
    from tools.radio import civ
    model = keras.models.load_model(MODEL_ZBOCZY)
    rozgrzej(model)
    for p in pliki:
        a = frontend.load_audio(p)
        ton = civ.ton_z_opisu(p.with_suffix(".txt"))
        d = OdczytNaZywo(model, ton_cw=ton, krok_s=krok_s)
        d.pomiar = True
        d.prog_db = prog_db
        n = int(kawalek_s * C.SR)
        t = time.perf_counter()
        if rzeczywisty:
            d.zegar = lambda: time.perf_counter() - t
        for i in range(0, a.size, n):
            if rzeczywisty:
                time.sleep(max(0.0, t + (i + n) / C.SR - time.perf_counter()))
            d.dodaj(a[i:i + n])
        d.koniec()
        calosc = time.perf_counter() - t
        zywo = d.tekst()
        cal = odczytaj(p, model, ton)["tekst"]
        opis = p.with_suffix(".txt")
        m = re.search(r"^nadane:\s*(.*)$", opis.read_text(encoding="utf-8"), re.M) if opis.exists() else None
        nad = m.group(1).replace(" ", "") if m and "nieznane" not in m.group(1) else None
        opozn, wst, zmiany = opoznienia(d)
        print(f"\n{p.name}  ({a.size / C.SR:.0f} s, ton CW {ton}; krok >= {krok_s:g} s, blokada {prog_db:g} dB{', tempo nagrania' if rzeczywisty else ''})")
        print(f"  na żywo : {zywo[:160]}")
        print(f"  całość  : {cal[:160]}")
        wiersz = f"  różnica na żywo / całość: {cer(zywo.replace(' ', ''), cal.replace(' ', ''))} znaków"
        if nad:
            wiersz += (f"; błędy wobec nadanego: na żywo {cer(zywo.replace(' ', ''), nad)}, "
                       f"całość {cer(cal.replace(' ', ''), nad)} / {len(nad)}")
        print(wiersz)
        print(f"  opóźnienie znaku (koniec znaku -> ekran): wstępny mediana {np.median(wst):.1f} s, "
              f"P90 {np.percentile(wst, 90):.1f} s; zatwierdzony mediana {np.median(opozn):.1f} s, "
              f"maks. {np.max(opozn):.1f} s; zmian wstępnego {zmiany} na {len(opozn)} znaków", flush=True)
        print(f"  krok liczony średnio {np.mean(d.czasy):.2f} s, maks. {np.max(d.czasy):.2f} s, "
              f"kroków {len(d.czasy)}; całość {calosc:.0f} s na {a.size / C.SR:.0f} s dźwięku", flush=True)


SESJE = Path(__file__).resolve().parent.parent / "probki" / "sesje"


class Sesja:
    """Zapis sesji na żywo (23.10, prośba operatora): dźwięk z karty (WAV
    8 kHz), opis jak z mic2wav --civ i dziennik kroków — co i kiedy dekoder
    pokazał. Nagrywarka ekranu Windows nie zapisuje wejścia USB z radia,
    a bez dźwięku nie da się zmierzyć opóźnienia ani użyć sesji jako próbki.
    probki/sesje/sesja_RRRRMMDD_GGMMSS.{wav,txt,log}; WAV poza gitem (.gitignore)."""

    ZRZUT_S = 5.0                     # nagłówek WAV dopisywany co tyle: zamknięte okno nie psuje pliku

    def __init__(self, sr: int, ton: float | None, pasmo: str | None, zrodlo: str,
                 skad_ton: str = "CI-V 0x14 0x09"):
        # pliki dopiero przy pierwszym dźwięku: 23.10 trzy puste sesje (0 s)
        # ze startów, w których karta się nie otworzyła
        self.arg = (sr, ton, pasmo, zrodlo, skad_ton)
        self.wav = None

    def _otworz(self) -> None:
        import soundfile as sf
        sr, ton, pasmo, zrodlo, skad_ton = self.arg
        SESJE.mkdir(parents=True, exist_ok=True)
        stempel = time.strftime("%Y%m%d_%H%M%S")
        self.sciezka = SESJE / f"sesja_{stempel}.wav"
        self.sr = sr
        self.wav = sf.SoundFile(self.sciezka, "w", sr, 1, "PCM_16")
        self.od_zrzutu = 0
        tonu = f"{ton:.0f} Hz ({skad_ton})" if ton else "nieodczytany"
        self.sciezka.with_suffix(".txt").write_text(
            "# SESJA NA ŻYWO — tools/na_zywo.py (dopisz, co było nadane)\n\n"
            "nadane:        nieznane — sesja na żywo\n"
            f"pasmo:         {pasmo or 'nieodczytane'} (CI-V przed sesją)\n"
            f"ton CW:        {tonu}\n"
            f"tor:           {zrodlo}\n"
            f"dekoder:       {_wersja()}\n", encoding="utf-8")
        self.log = open(self.sciezka.with_suffix(".log"), "w", encoding="utf-8", buffering=1)
        self.log.write("# dźwięk [s]\tliczenie [s]\tzatwierdzone\twstępne\n")
        print(f"sesja: {self.sciezka}", flush=True)

    def dzwiek(self, x: np.ndarray) -> None:
        if self.wav is None:
            self._otworz()
        self.wav.write(np.clip(np.asarray(x, np.float32), -1, 1))
        self.od_zrzutu += len(x)
        if self.od_zrzutu >= self.ZRZUT_S * self.sr:
            self.wav.flush()
            self.od_zrzutu = 0

    def krok(self, d: "OdczytNaZywo", nowe: str) -> None:
        self.log.write(f"{d.n / d.sr:.3f}\t{d.czasy[-1]:.3f}\t{nowe}\t{d.wstepny}\n")

    def uwaga(self, d: "OdczytNaZywo", tekst: str) -> None:
        if self.wav is not None:
            self.log.write(f"# {d.n / d.sr:.3f}: {tekst}\n")

    def zamknij(self, d: "OdczytNaZywo") -> None:
        if self.wav is None:
            return
        self.log.write(f"# odczyt: {d.tekst()}\n")
        self.log.close()
        self.wav.close()
        print(f"\nsesja zapisana: {self.sciezka} ({d.n / d.sr:.0f} s)", flush=True)


WERSJA: str | None = None                  # aplikacja (exe): wpisana przy budowie, bez gita


def _wersja() -> str:
    if WERSJA:
        return WERSJA
    import subprocess
    try:
        r = subprocess.run(["git", "describe", "--always", "--dirty", "--tags"], capture_output=True,
                           text=True, cwd=Path(__file__).resolve().parent.parent, timeout=5)
        g = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True,
                           cwd=Path(__file__).resolve().parent.parent, timeout=5)
        return f"{r.stdout.strip()} ({g.stdout.strip()})" if r.returncode == 0 else "nieznana"
    except (OSError, subprocess.SubprocessError):
        return "nieznana"


def _petla(d: "OdczytNaZywo", e: Ekran, s: Sesja | None, x: np.ndarray) -> None:
    if s:
        s.dzwiek(x)
    n_krokow = len(d.czasy)
    nowe = d.dodaj(x)
    if len(d.czasy) > n_krokow:
        e.pokaz(nowe, d.wstepny, d.stan)
        if s:
            s.krok(d, nowe)


def karta(device: int | str, z_civ: bool, zapis: bool = True) -> None:
    """Na żywo z karty dźwiękowej (np. IC-7300 USB audio). Tylko odbiór."""
    from tensorflow import keras
    from tools.odczyt import MODEL_ZBOCZY
    st = _stan_radia() if z_civ else {}
    ton = st.get("ton_cw")
    print(f"ton CW: {ton} Hz; Ctrl+C kończy", flush=True)
    model = keras.models.load_model(MODEL_ZBOCZY)
    rozgrzej(model)
    d = OdczytNaZywo(model, ton_cw=ton)
    e = Ekran()
    s = Sesja(C.SR, ton, st.get("pasmo"), f"karta {device} (Windows)") if zapis else None
    try:
        for x in _strumien_karty(device):
            _petla(d, e, s, x)
    except KeyboardInterrupt:
        pass
    finally:
        e.pokaz(d.koniec(), "")
        if s:
            s.zamknij(d)
        print()


class Wodospad:
    """Kolumny widma co KROK próbek (20 ms) w paśmie [fmin, fmax] — dla dwóch
    sygnałów na TYCH SAMYCH chwilach (wejście i to, co idzie do sieci), żeby
    w okienku przesuwały się razem. Wygląd jak Live-Specgram operatora
    (W:\\PyCharmMiscProject\\Live-Specgram-master3: jet); zakres 40 dB, nie 60."""

    N_FFT, KROK = 512, 160                 # 64 ms, 15,6 Hz; 20 ms

    def __init__(self, sr: int = C.SR):
        self.sr = sr
        self.nast = None                   # środek następnej kolumny [próbka]
        self.okno = np.hanning(self.N_FFT)

    def srodki(self, koniec: int, poczatek: int) -> np.ndarray:
        """Środki nowych kolumn, które mają już pełne okno przed końcem."""
        if self.nast is None:
            self.nast = poczatek + self.N_FFT // 2
        ost = koniec - self.N_FFT // 2
        c = np.arange(self.nast, ost + 1, self.KROK)
        if c.size:
            self.nast = int(c[-1]) + self.KROK
        return c

    def widmo(self, x: np.ndarray, x0: int, srodki: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
        """[kolumny, prążki] dB; kolumna poza x -> -120."""
        f = np.fft.rfftfreq(self.N_FFT, 1 / self.sr)
        pas = (f >= fmin) & (f <= fmax)
        out = np.full((srodki.size, int(pas.sum())), -120.0, np.float32)
        h = self.N_FFT // 2
        for i, c in enumerate(srodki):
            a = c - h - x0
            if a < 0 or a + self.N_FFT > x.size:
                continue
            s = np.abs(np.fft.rfft(x[a:a + self.N_FFT] * self.okno))[pas]
            out[i] = 20 * np.log10(s + 1e-9)
        return out


def _jet(v: np.ndarray) -> np.ndarray:
    """[0,1] -> RGB uint8, paleta „jet” (jak w Live-Specgram operatora)."""
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


PASMO_WEJ = (300.0, 1300.0)
PASMO_SYG = (500.0, 1000.0)
W_WOD, H_WOD = 980, 90                     # px; kolumna = 20 ms -> 19,6 s widać
# zakres barw pod maksimum: przy 60 dB (jak Live-Specgram) szum na 40 m był
# jasny jak sygnał; operator 24.10: 40 dB
ZAKRES_WOD_DB = 40.0
KLATKA_WOD_MS = 40                         # animacja wodospadu: 25 klatek/s
ZAPAS_WOD = 35                             # kolumn w kolejce (0,7 s) przy tempie zwykłym
OPADANIE_DB_S = 0.1


class _ObrazWodospadu:
    """Wodospad w Canvas: nowe kolumny z prawej, obraz przesuwa się w lewo."""

    def __init__(self, cv):
        self.cv = cv
        self.rgb = np.zeros((H_WOD, W_WOD, 3), np.uint8)
        self.vmax = None
        self.img = None
        self.id = cv.create_image(0, 0, anchor="nw")

    def dodaj(self, kol: np.ndarray) -> None:
        import tkinter as tk
        k = kol.shape[0]
        if not k:
            return
        ok = kol[kol > -119]
        if ok.size:
            m = float(ok.max())
            # maksimum opada OPADANIE_DB_S: przy 1 dB/s po kilkunastu s ciszy
            # szum świecił jak sygnał (okno na radiu 24.10) — sam obraz, nie dekoder
            self.vmax = m if self.vmax is None else max(m, self.vmax - OPADANIE_DB_S * k * Wodospad.KROK / C.SR)
        vmax = self.vmax if self.vmax is not None else 0.0
        v = np.clip((kol - (vmax - ZAKRES_WOD_DB)) / ZAKRES_WOD_DB, 0, 1)
        wiersze = np.round(np.linspace(kol.shape[1] - 1, 0, H_WOD)).astype(int)   # niskie na dole
        nowe = _jet(v[:, wiersze].T)                                              # [H, k, 3]
        k = min(k, W_WOD)
        self.rgb = np.concatenate([self.rgb[:, k:], nowe[:, -k:]], axis=1)
        self.img = tk.PhotoImage(data=b"P6 %d %d 255 " % (W_WOD, H_WOD) + self.rgb.tobytes(), format="PPM")
        self.cv.itemconfig(self.id, image=self.img)


def _praca_okna(q, stop, prog, cfg: dict) -> None:
    """Dekoder okienka w OSOBNYM PROCESIE (24.10): w wątku dzielił z oknem
    jeden interpreter — przeliczenie po analizie (co 2 s, ~0,35 s numpy)
    trzymało blokadę interpretera i wodospad stawał (zmierzone: 28 przerw
    > 200 ms na 64 s, każda ~0,35 s). Wszystko tu musi być do przesłania
    (cfg: słownik prostych wartości), bo proces startuje od zera (spawn)."""
    import traceback
    global SESJE, WERSJA
    if cfg.get("dziennik"):                   # aplikacja bez konsoli
        f = open(cfg["dziennik"], "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stderr = f
    if cfg.get("sesje"):
        SESJE = Path(cfg["sesje"])
    WERSJA = cfg.get("wersja") or WERSJA
    s = d = None
    try:
        st: dict = {}
        if cfg["z_civ"]:
            try:
                st = civ_stan(cfg.get("civ_port"), cfg.get("civ_baud"))
            except Exception as e:            # CI-V nie może zatrzymać odbioru
                print(f"CI-V: {e!r} — dalej bez tonu z radia", flush=True)
                st = {"ton_cw": None, "pasmo": f"CI-V nieodczytane ({e.__class__.__name__})"}
        if st.get("ton_cw") is None and cfg.get("ton_reczny"):
            st = {**st, "ton_cw": cfg["ton_reczny"], "skad_ton": "wpisany ręcznie"}
        ton = st.get("ton_cw")
        q.put(("info", "laduje", {"ton": ton}))
        if cfg.get("model_npz"):
            from dsp.siec_numpy import SiecNumpy
            model = SiecNumpy(Path(cfg["model_npz"]))
        else:
            from tensorflow import keras
            from tools.odczyt import MODEL_ZBOCZY
            model = keras.models.load_model(MODEL_ZBOCZY)
        rozgrzej(model)
        d = OdczytNaZywo(model, ton_cw=ton)
        s = Sesja(C.SR, ton, st.get("pasmo"), cfg.get("zrodlo", f"karta {cfg['device']} (Windows, okno)"),
                  st.get("skad_ton", "CI-V 0x14 0x09")) if cfg["zapis"] else None
        gen = _strumien_wav(Path(cfg["wav"])) if cfg.get("wav") else _strumien_karty(cfg["device"])
        wod = Wodospad()
        q.put(("info", "nasluch", {"ton": ton}))
        p_ost = None
        sesja_wyslana = False
        for x in gen:
            if stop.is_set():
                break
            if s:
                s.dzwiek(x)
                if not sesja_wyslana and s.wav is not None:   # okno: ścieżka do raportu
                    q.put(("sesja", str(s.sciezka)))
                    sesja_wyslana = True
            if prog.value != p_ost:
                p_ost = d.prog_db = prog.value
                if s:
                    s.uwaga(d, f"blokada szumów {p_ost:.0f} dB")
            n = len(d.czasy)
            nowe = d.dodaj(x)
            if len(d.czasy) > n:
                if s:
                    s.krok(d, nowe)
                c = wod.srodki(d.n, d.raw0)
                q.put(("krok", [z[:3] for z in d.zatwierdzone[len(d.zatwierdzone) - len(nowe):]] if nowe else [],
                       d.wstepny, d.stan, d.nad_szumem_db, d.u,
                       wod.widmo(d.raw, d.raw0, c, *PASMO_WEJ), wod.widmo(d.ost_y, d.ost_y0, c, *PASMO_SYG)))
        gen.close()
    except Exception:
        print(traceback.format_exc(), flush=True)        # aplikacja: do dziennika
        q.put(("blad", traceback.format_exc()))
    finally:
        if d is not None:
            nowe = d.koniec()
            if s:
                s.zamknij(d)
            if d.czasy:                                  # aplikacja: podsumowanie do dziennika
                print(f"koniec: {d.n / d.sr:.0f} s dźwięku, kroków {len(d.czasy)}, krok śr. "
                      f"{np.mean(d.czasy):.3f} s, maks. {np.max(d.czasy):.2f} s; odczyt: {d.tekst()[:300]}",
                      flush=True)
            q.put(("krok", [z[:3] for z in d.zatwierdzone[len(d.zatwierdzone) - len(nowe):]] if nowe else [],
                   "", d.stan, 0.0, d.u, None, None))
        q.put(("koniec",))


def civ_stan(port: str | None, baud: int | None) -> dict:
    from tools.radio import civ
    return civ.stan(port or "COM5", baud or 115200)


def okno(device: int | str, z_civ: bool, zapis: bool = True, model_npz: str | None = None,
         civ_port: str | None = None, civ_baud: int | None = None, ton_reczny: float | None = None,
         tytul: str = "Dekoder CW — na żywo", wav: str | None = None, dziennik: str | None = None,
         geometria: str | None = None, prog_start: float | None = None, na_zamkniecie=None,
         raport_fn=None, sprawdz_wersje=None, ikona: str | None = None) -> None:
    """Na żywo w okienku (23.10, prośba operatora): tekst zatwierdzony (jasny)
    i wstępny (szary), słowa ze słownika zielone, dwa wodospady, wskaźnik tonu
    nad szumem z kreską progu, suwak blokady szumów. Dekoder w osobnym procesie
    (_praca_okna); zamknięcie okna zamyka sesję porządnie.
    model_npz — sieć w numpy (aplikacja bez TensorFlow); civ_port/baud, ton_reczny —
    radio innego operatora; wav — nagranie zamiast karty (próba).
    Aplikacja (tools/aplikacja.py): geometria/prog_start/na_zamkniecie(geometria,
    prog) — zapamiętanie w rejestrze; raport_fn(sesja, root) — przycisk „Zgłoś
    problem”; sprawdz_wersje() -> (wersja, adres) | None — w tle przy starcie."""
    from tools.teksty import T
    import multiprocessing as mp
    import queue
    import tkinter as tk

    TLO, JASNY, SZARY, ZIELONY = "#101214", "#e8e8e8", "#6c7078", "#5fd068"
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    stop = ctx.Event()
    prog = ctx.Value("d", float(PROG_BLOKADY_DB if prog_start is None else prog_start))
    cfg = dict(device=device, z_civ=z_civ, zapis=zapis, model_npz=model_npz, civ_port=civ_port,
               civ_baud=civ_baud, ton_reczny=ton_reczny, wav=wav, dziennik=dziennik,
               sesje=str(SESJE), wersja=WERSJA)

    root = tk.Tk()
    root.title(tytul)
    root.geometry(geometria or "1000x640")
    if ikona:
        try:
            root.iconbitmap(ikona)
        except tk.TclError:
            pass
    root.configure(bg=TLO)
    gora = tk.Frame(root, bg=TLO)
    gora.pack(fill="x", padx=10, pady=(8, 2))
    stan = tk.Label(gora, text=T("start"), fg=SZARY, bg=TLO, font=("Consolas", 11), anchor="w")
    stan.pack(side="left", fill="x", expand=True)
    sesja = {"sciezka": None}
    if raport_fn is not None:
        tk.Button(gora, text=T("zglos"), command=lambda: raport_fn(sesja["sciezka"], root),
                  bg="#2a3038", fg=JASNY, activebackground="#3a4250", relief="flat", padx=10).pack(side="right")
    wersja_lbl = tk.Label(gora, text="", fg="#e0b050", bg=TLO, font=("Segoe UI", 9, "underline"), cursor="hand2")
    wersja_lbl.pack(side="right", padx=10)
    pas = tk.Frame(root, bg=TLO)
    pas.pack(fill="x", padx=10, pady=2)
    tk.Label(pas, text=T("ton_nad_szumem"), fg=SZARY, bg=TLO, font=("Segoe UI", 9)).pack(side="left")
    W_M, ZAKRES = 320, 40.0
    miernik = tk.Canvas(pas, width=W_M, height=16, bg="#22262b", highlightthickness=0)
    miernik.pack(side="left", padx=6)
    slupek = miernik.create_rectangle(0, 0, 0, 16, fill="#3fa34d", width=0)
    kreska = miernik.create_line(0, 0, 0, 16, fill="#e05050", width=2)
    suwak = tk.Scale(pas, from_=0, to=30, resolution=1, orient="horizontal", length=260,
                     label=T("blokada"), bg=TLO, fg=SZARY,
                     troughcolor="#22262b", highlightthickness=0, font=("Segoe UI", 9),
                     command=lambda v: setattr(prog, "value", float(v)))
    suwak.set(prog.value)
    suwak.pack(side="left", padx=12)
    obrazy = []
    # Płynny wodospad (operator 24.10): kolumny przychodzą skokami, raz na krok
    # dekodera (0,25 s i więcej, po ~12 naraz) — do kolejki, a animacja co
    # KLATKA_WOD_MS dosuwa tyle, ile dźwięku w tym czasie przybywa; gdy kolejka
    # rośnie (krok się wydłużył), dosuwa szybciej. Obraz ~1 krok za dźwiękiem.
    kolejka_wod: list[list] = [[], []]

    # Tempo z zapasu w kolejce: przy ZAPAS_WOD kolumnach — tyle, ile przybywa
    # dźwięku (2 kolumny / 40 ms); mniej w kolejce — wolniej, więcej — szybciej.
    # Pierwsza wersja (stałe tempo, zapas 15 kolumn = 0,3 s): po przeliczeniu
    # (co 2 s nowych kolumn nie ma ~0,6 s) kolejka pustoszała i obraz stawał —
    # 31 przerw > 200 ms na 64 s, także z dekoderem w osobnym procesie.
    anim = {"ulamek": 0.0, "start": False}

    def animuj() -> None:
        n = len(kolejka_wod[0])
        if not anim["start"] and n >= ZAPAS_WOD:
            anim["start"] = True
        if anim["start"] and n:
            na_klatke = KLATKA_WOD_MS / 1000 / (Wodospad.KROK / C.SR)          # 2 kolumny / 40 ms
            anim["ulamek"] += na_klatke * min(3.0, max(0.25, n / ZAPAS_WOD))
            k = min(n, int(anim["ulamek"]))
            anim["ulamek"] -= k
            if k:
                for i in (0, 1):
                    obrazy[i].dodaj(np.array(kolejka_wod[i][:k]))
                    del kolejka_wod[i][:k]
        root.after(KLATKA_WOD_MS, animuj)
    for opis in (T("wod_wej", a=f"{PASMO_WEJ[0]:.0f}", b=f"{PASMO_WEJ[1]:.0f}"),
                 T("wod_syg", t=f"{C.TONE_CENTER:.0f}", a=f"{PASMO_SYG[0]:.0f}", b=f"{PASMO_SYG[1]:.0f}")):
        tk.Label(root, text=opis, fg=SZARY, bg=TLO, font=("Segoe UI", 8), anchor="w").pack(fill="x", padx=10)
        cv = tk.Canvas(root, width=W_WOD, height=H_WOD, bg="#000000", highlightthickness=0)
        cv.pack(padx=10, pady=(0, 4), anchor="w")
        obrazy.append(_ObrazWodospadu(cv))
    tekst = tk.Text(root, bg=TLO, fg=JASNY, font=("Consolas", 18), wrap="word", bd=0,
                    insertbackground=TLO, padx=10, pady=8)
    tekst.pack(fill="both", expand=True)
    tekst.tag_configure("wst", foreground=SZARY)
    tekst.tag_configure("zielone", foreground=ZIELONY)
    # Tekst zatwierdzony przez słownik (dsp/slownik.py): spacje mogą się
    # jeszcze przestawić, więc końcówkę (od `od`) rysujemy co krok od nowa;
    # starsze niż OTWARTE znaków zamrażamy na spacji.
    OTWARTE = 120
    zat = {"znaki": [], "od": 0, "u": float("nan")}
    tekst.mark_set("otwarte", "1.0")
    tekst.mark_gravity("otwarte", "left")

    def rysuj(znaki: list, u: float) -> None:
        from dsp import slownik as SL
        if np.isfinite(u) and u > 0:
            slowa = SL.koloruj(znaki, u)
        else:
            slowa = [(w, False) for w in "".join(z[0] for z in znaki).split()]
        for w, zn in slowa:
            tekst.insert("end", w + " ", "zielone" if zn else ())

    def odswiez() -> None:
        try:
            while True:
                m = q.get_nowait()
                if m[0] == "koniec":
                    stan.config(text=T("koniec_pracy"))
                elif m[0] == "sesja":
                    sesja["sciezka"] = m[1]
                elif m[0] == "wersja":
                    wersja_lbl.config(text=T("nowa_wersja", w=m[1]))
                    wersja_lbl.bind("<Button-1>", lambda e, a=m[2]: __import__("webbrowser").open(a))
                elif m[0] == "info":
                    stan.config(text=T(m[1], **m[2]))
                elif m[0] == "blad":
                    stan.config(text=T("blad_okno"), fg="#e05050")
                    tekst.insert("end", "\n" + m[1], "wst")
                else:
                    _, nowe, wst, st, nad, u, w_wej, w_syg = m
                    if w_wej is not None:
                        kolejka_wod[0].extend(w_wej)
                        kolejka_wod[1].extend(w_syg)
                    zn = zat["znaki"]
                    zn += nowe
                    if np.isfinite(u):
                        zat["u"] = u
                    tekst.delete("otwarte", "end")
                    if len(zn) - zat["od"] > OTWARTE:          # zamrożenie starszej części
                        k = len(zn) - OTWARTE // 2
                        while k < len(zn) and zn[k][0] != " ":
                            k += 1
                        rysuj(zn[zat["od"]:k], zat["u"])
                        tekst.mark_set("otwarte", "end-1c")
                        zat["od"] = k
                    rysuj(zn[zat["od"]:], zat["u"])
                    pusto = tekst.get("end-2c", "end-1c") in ("", " ", "\n")
                    tekst.insert("end", wst.lstrip() if pusto else wst, "wst")
                    tekst.see("end")
                    stan.config(text=st)
                    x = max(0.0, min(1.0, nad / ZAKRES)) * W_M
                    miernik.coords(slupek, 0, 0, x, 16)
                    miernik.itemconfig(slupek, fill="#3fa34d" if prog.value <= 0 or nad >= prog.value else "#5a6068")
                    k = prog.value / ZAKRES * W_M
                    miernik.coords(kreska, k, 0, k, 16)
        except queue.Empty:
            pass
        root.after(50, odswiez)

    def zamknij() -> None:
        stop.set()
        stan.config(text=T("zamykam"))
        if na_zamkniecie is not None:
            try:
                na_zamkniecie(root.geometry(), prog.value)
            except Exception as e:                   # zapamiętanie nie może zablokować zamknięcia
                print(f"zapamiętanie ustawień: {e!r}", flush=True)
        root.update()
        # odbierać dalej: dekoder z pełną kolejką nie skończyłby (sesja niezamknięta)
        koniec = time.monotonic() + 8
        while w.is_alive() and time.monotonic() < koniec:
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass
            w.join(timeout=0.1)
        if w.is_alive():
            w.terminate()
        root.destroy()

    # nie daemon: dekoder sam ma proces analizy (ProcessPoolExecutor)
    w = ctx.Process(target=_praca_okna, args=(q, stop, prog, cfg))
    w.start()
    root.protocol("WM_DELETE_WINDOW", zamknij)
    root.after(50, odswiez)
    root.after(KLATKA_WOD_MS, animuj)
    if sprawdz_wersje is not None:
        import threading

        def w_tle() -> None:
            try:
                r = sprawdz_wersje()
                if r:
                    q.put(("wersja", *r))
            except Exception as e:                   # brak sieci itp. — bez znaczenia dla odbioru
                print(f"sprawdzanie wersji: {e!r}", flush=True)
        threading.Thread(target=w_tle, daemon=True).start()
    root.mainloop()


def _stan_radia() -> dict:
    from tools.radio import civ
    return civ.stan()


class _Decymator:
    """Karta (np. 48 kHz) -> C.SR kawałek po kawałku, ze STANEM filtra.
    Pierwsza wersja (21.10) przepróbkowywała każdy kawałek 0,1 s osobno
    (resample_poly) — filtr startował od zera co 100 ms, na stykach
    stuki. Tu FIR antyaliasingowy z pamięcią (lfilter zi) i co m-ta próbka
    w ciągłej fazie."""

    def __init__(self, sr_dev: int):
        from scipy.signal import firwin
        self.sr_dev = sr_dev
        self.m = sr_dev // C.SR if sr_dev % C.SR == 0 else 0
        if self.m:
            self.b = firwin(255, 0.9 * (C.SR / 2) / (sr_dev / 2))
            self.zi = np.zeros(self.b.size - 1)
            self.g = 0                                   # próbek karty dotąd

    def __call__(self, x: np.ndarray) -> np.ndarray:
        from scipy.signal import lfilter, resample_poly
        if not self.m:                                   # np. 44,1 kHz: bez ciągłości
            return resample_poly(x, C.SR, self.sr_dev).astype(np.float32)
        y, self.zi = lfilter(self.b, [1.0], x, zi=self.zi)
        out = y[(-self.g) % self.m::self.m]
        self.g += x.size
        return out.astype(np.float32)


KARTA = "USB Audio"


def znajdz_karte(spec: str | int) -> int:
    """Numer karty z numeru albo z kawałka nazwy (preferowany WASAPI).

    23.10: po przepięciu USB Windows przenumerował karty — „15" zniknęło,
    IC-7300 wrócił jako „Mikrofon (2 — USB Audio Device)" pod 18 i skrypt
    nie otwierał karty. Nazwa przeżywa przenumerowanie."""
    import sounddevice as sd
    if isinstance(spec, int) or str(spec).strip().isdigit():
        return int(spec)
    api = [a["name"] for a in sd.query_hostapis()]
    kand = [(i, d) for i, d in enumerate(sd.query_devices())
            if d["max_input_channels"] > 0 and str(spec).lower() in d["name"].lower()]
    if not kand:
        raise SystemExit(f"Nie ma karty wejściowej z „{spec}” w nazwie — lista: python -m tools.mic2wav --list")
    kand.sort(key=lambda k: (api[k[1]["hostapi"]] != "Windows WASAPI", k[0]))
    return kand[0][0]


def _strumien_karty(device: int | str):
    """Kawałki po ~0,1 s w C.SR z karty dźwiękowej (tylko odbiór)."""
    import queue
    import sounddevice as sd
    # PortAudio od nowa W TYM wątku: uruchomione w innym (aplikacja: lista kart
    # w oknie startowym, karta w wątku dekodera) nie otwiera WASAPI —
    # „Unanticipated host error ... GLE 0x490” (exe na IC-7300, 24.10).
    # Przy okazji świeża lista kart po przepięciu USB.
    sd._terminate()
    sd._initialize()
    device = znajdz_karte(device)
    print(f"karta [{device}] {sd.query_devices(device)['name']}", flush=True)
    sr_dev = int(sd.query_devices(device, "input")["default_samplerate"])
    if sr_dev % C.SR:
        print(f"UWAGA: karta {sr_dev} Hz nie jest wielokrotnością {C.SR} Hz — "
              "przepróbkowanie kawałkami, bez ciągłości filtra", flush=True)
    dec = _Decymator(sr_dev)
    q: queue.Queue = queue.Queue()
    with sd.InputStream(device=device, channels=1, samplerate=sr_dev, blocksize=sr_dev // 10,
                        callback=lambda x, *_: q.put(x[:, 0].astype(np.float64)), latency="high"):
        while True:
            yield dec(q.get())


# --------------------------------------------------------------------------
# Most Windows -> WSL (22.10): karta i CI-V są w Windows (WSL2 ich nie widzi),
# karta graficzna — w WSL. Windows wysyła próbki przez localhost, WSL liczy.
# Protokół: wiersz JSON {"ton_cw": Hz|null, "sr": 8000}, potem float32 LE.
# --------------------------------------------------------------------------
PORT = 7355


def serwer(port: int, zapis: bool = True) -> None:
    """WSL: czeka na most z Windows, dekoduje (karta graficzna), wypisuje tekst.
    Sesję zapisuje tu (ten sam katalog projektu co Windows)."""
    import json
    import socket
    from tensorflow import keras
    from tools.odczyt import MODEL_ZBOCZY
    model = keras.models.load_model(MODEL_ZBOCZY)
    rozgrzej(model)
    with socket.create_server(("0.0.0.0", port)) as s:
        while True:
            print(f"czekam na most z Windows (port {port})...", flush=True)
            c, adr = s.accept()
            with c:
                f = c.makefile("rb")
                naglowek = json.loads(f.readline().decode())
                ton = naglowek.get("ton_cw")
                print(f"połączono {adr[0]}; ton CW: {ton} Hz", flush=True)
                d = OdczytNaZywo(model, ton_cw=ton)
                e = Ekran()
                ss = Sesja(C.SR, ton, naglowek.get("pasmo"), naglowek.get("zrodlo", "most z Windows")) \
                    if zapis and not naglowek.get("proba") else None
                reszta = b""
                try:
                    while True:
                        b = c.recv(65536)
                        if not b:
                            break
                        b = reszta + b
                        n = len(b) // 4 * 4
                        reszta = b[n:]
                        _petla(d, e, ss, np.frombuffer(b[:n], "<f4"))
                finally:
                    e.pokaz(d.koniec(), "")
                    if ss:
                        ss.zamknij(d)
                print("\n--- most rozłączony ---", flush=True)


def _strumien_wav(sciezka: Path):
    """Nagranie zamiast karty, w tempie rzeczywistym (próba mostu bez radia)."""
    a = frontend.load_audio(sciezka)
    n = C.SR // 10
    for i in range(0, a.size, n):
        time.sleep(0.1)
        yield a[i:i + n]


def most(device: int | str, z_civ: bool, host: str, port: int, wav: Path | None = None) -> None:
    """Windows: karta + CI-V -> próbki do serwera w WSL. Tylko odbiór."""
    import json
    import socket
    st = _stan_radia() if z_civ else {}
    ton = st.get("ton_cw")
    if wav and ton is None:
        from tools.radio import civ
        ton = civ.ton_z_opisu(Path(wav).with_suffix(".txt"))
    print(f"ton CW: {ton} Hz; łączę z {host}:{port} (dekoder w WSL)...", flush=True)
    t0 = time.time()
    while True:
        try:
            c = socket.create_connection((host, port), timeout=5)
            break
        except OSError:
            if time.time() - t0 > 120:
                raise SystemExit("Serwer w WSL nie odpowiada od 2 min — czy okno dekodera wystartowało?")
            time.sleep(2)
    c.settimeout(None)
    print("połączono; tekst w oknie dekodera. Ctrl+C kończy.", flush=True)
    with c:
        # proba: odtwarzane nagranie — serwer go nie zapisuje drugi raz jako sesji
        c.sendall((json.dumps({"ton_cw": ton, "sr": C.SR, "pasmo": st.get("pasmo"),
                               "zrodlo": f"karta {device} (Windows, most do WSL)", "proba": bool(wav)})
                   + "\n").encode())
        try:
            for x in (_strumien_wav(wav) if wav else _strumien_karty(device)):
                c.sendall(np.asarray(x, "<f4").tobytes())
        except KeyboardInterrupt:
            pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("symuluj"); s.add_argument("pliki", type=Path, nargs="+")
    s.add_argument("--krok", type=float, default=KROK_S)
    s.add_argument("--blokada", type=float, default=PROG_BLOKADY_DB, help="próg blokady szumów [dB], 0 = wył.")
    s.add_argument("--szybko", action="store_true", help="bez tempa nagrania (chwile pokazania zaniżone)")
    k = sub.add_parser("karta"); k.add_argument("--device", default=KARTA, help="numer albo kawałek nazwy")
    k.add_argument("--civ", action="store_true")
    k.add_argument("--bez-zapisu", action="store_true", help="bez nagrania sesji do probki/sesje")
    o = sub.add_parser("okno"); o.add_argument("--device", default=KARTA, help="numer albo kawałek nazwy")
    o.add_argument("--civ", action="store_true"); o.add_argument("--bez-zapisu", action="store_true")
    sv = sub.add_parser("serwer"); sv.add_argument("--port", type=int, default=PORT)
    sv.add_argument("--bez-zapisu", action="store_true")
    m = sub.add_parser("most"); m.add_argument("--device", default=KARTA, help="numer albo kawałek nazwy")
    m.add_argument("--civ", action="store_true")
    m.add_argument("--host", default="127.0.0.1"); m.add_argument("--port", type=int, default=PORT)
    m.add_argument("--wav", type=Path, default=None, help="nagranie zamiast karty (próba bez radia)")
    a = ap.parse_args()
    if a.cmd == "symuluj":
        symuluj(a.pliki, krok_s=a.krok, rzeczywisty=not a.szybko, prog_db=a.blokada)
    elif a.cmd == "karta":
        karta(a.device, a.civ, zapis=not a.bez_zapisu)
    elif a.cmd == "okno":
        okno(a.device, a.civ, zapis=not a.bez_zapisu)
    elif a.cmd == "serwer":
        serwer(a.port, zapis=not a.bez_zapisu)
    else:
        most(a.device, a.civ, a.host, a.port, a.wav)


if __name__ == "__main__":
    main()
