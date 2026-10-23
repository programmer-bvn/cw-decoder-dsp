"""ODCZYT NA ŻYWO (21.10; strumieniowy v3 23.10): dźwięk kawałkami -> tekst.

    python -m tools.na_zywo symuluj nagranie.wav [...]     # nagranie w tempie rzeczywistym
    python -m tools.na_zywo karta --device 15 [--civ]      # karta dźwiękowa (IC-7300 USB)

Z kartą graficzną (karta dźwiękowa i COM są w Windows, WSL2 ich nie widzi):
    WSL:      python -m tools.na_zywo serwer               # dekoder, GPU, tekst
    Windows:  python -m tools.na_zywo most --device 15 --civ
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

from dsp import config as C, frontend, viterbi as V, zbocza as Z  # noqa: E402
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
    return f_n, obce, wav2net.auto_gain_db(_przygotuj(x, n0, f_n, obce, 0.0, sr)), kand


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
        self.h0 = 0
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

    @property
    def wstepny(self) -> str:
        return "".join(z[0] for z in self.wstepne)

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
        self.analiza = wynik[:3]
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

    def _do_historii(self, p: np.ndarray, a0: int) -> None:
        """P okna od ramki a0. Starsze od środka okna zostają zamrożone."""
        self._wpisz(p, a0, a0 + Z.RAMKI_ZBOCZA // 2)

    def _wpisz(self, p: np.ndarray, a0: int, od: int) -> None:
        """P od ramki a0; w historii zastępuje ramki od `od` (nie wcześniej niż a0)."""
        if not self.hist.size:
            self.hist, self.h0 = p.astype(np.float32), a0
        else:
            koniec_h = self.h0 + self.hist.size
            if a0 > koniec_h:                              # luka (nie powinna się zdarzyć)
                self.hist = np.concatenate([self.hist, np.zeros(a0 - koniec_h, np.float32)])
                koniec_h = a0
            w = max(a0, self.h0, min(od, koniec_h))
            self.hist = np.concatenate([self.hist[:w - self.h0], p[w - a0:].astype(np.float32)])
        nadmiar = self.hist.size - int(HISTORIA_S / V.DT)
        if nadmiar > 0:
            self.hist = self.hist[nadmiar:]
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
                p = Z.przebieg_sieci(_przygotuj(self.raw[s0 - self.raw0:], s0, f_n, obce, wzm, self.sr),
                                     self.model)
                od = s0 // hop + (Z.RAMKI_ZBOCZA // 2 if s0 > self.raw0 else 0)
                self._wpisz(p, s0 // hop, od)
            else:
                p = self._siec(_przygotuj(self.raw[s0 - self.raw0:], s0, f_n, obce, wzm, self.sr))
                n_ramek = (self.n - s0) // hop + 1
                self._do_historii(p, s0 // hop + n_ramek - p.size)
            kand, self.kand = self.kand + [self.u], []
            if not any(np.isfinite(k) and k > 0 for k in kand):     # początek: jeszcze nic
                kand = [V.jednostka_cepstrum(self.hist)[0], jednostka_grzebienia(self.hist)]
            pv = self.hist[-int(VITERBI_S / V.DT):]
            v0 = self.h0 + self.hist.size - pv.size
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

    def pokaz(self, nowe: str, wstepny: str) -> None:
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
            rzeczywisty: bool = True) -> None:
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
        print(f"\n{p.name}  ({a.size / C.SR:.0f} s, ton CW {ton}; krok >= {krok_s:g} s{', tempo nagrania' if rzeczywisty else ''})")
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


def karta(device: int, z_civ: bool) -> None:
    """Na żywo z karty dźwiękowej (np. IC-7300 USB audio). Tylko odbiór."""
    from tensorflow import keras
    from tools.odczyt import MODEL_ZBOCZY
    ton = _ton_z_radia() if z_civ else None
    print(f"ton CW: {ton} Hz; Ctrl+C kończy", flush=True)
    model = keras.models.load_model(MODEL_ZBOCZY)
    rozgrzej(model)
    d = OdczytNaZywo(model, ton_cw=ton)
    e = Ekran()
    try:
        for x in _strumien_karty(device):
            n_krokow = len(d.czasy)
            nowe = d.dodaj(x)
            if len(d.czasy) > n_krokow:
                e.pokaz(nowe, d.wstepny)
    except KeyboardInterrupt:
        e.pokaz(d.koniec(), "")
        print()


def _ton_z_radia() -> float | None:
    from tools.radio import civ
    return civ.stan()["ton_cw"]


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


def _strumien_karty(device: int):
    """Kawałki po ~0,1 s w C.SR z karty dźwiękowej (tylko odbiór)."""
    import queue
    import sounddevice as sd
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


def serwer(port: int) -> None:
    """WSL: czeka na most z Windows, dekoduje (karta graficzna), wypisuje tekst."""
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
                reszta = b""
                while True:
                    b = c.recv(65536)
                    if not b:
                        break
                    b = reszta + b
                    n = len(b) // 4 * 4
                    reszta = b[n:]
                    n_krokow = len(d.czasy)
                    nowe = d.dodaj(np.frombuffer(b[:n], "<f4"))
                    if len(d.czasy) > n_krokow:
                        e.pokaz(nowe, d.wstepny)
                e.pokaz(d.koniec(), "")
                print("\n--- most rozłączony ---", flush=True)


def _strumien_wav(sciezka: Path):
    """Nagranie zamiast karty, w tempie rzeczywistym (próba mostu bez radia)."""
    a = frontend.load_audio(sciezka)
    n = C.SR // 10
    for i in range(0, a.size, n):
        time.sleep(0.1)
        yield a[i:i + n]


def most(device: int, z_civ: bool, host: str, port: int, wav: Path | None = None) -> None:
    """Windows: karta + CI-V -> próbki do serwera w WSL. Tylko odbiór."""
    import json
    import socket
    ton = _ton_z_radia() if z_civ else None
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
        c.sendall((json.dumps({"ton_cw": ton, "sr": C.SR}) + "\n").encode())
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
    s.add_argument("--szybko", action="store_true", help="bez tempa nagrania (chwile pokazania zaniżone)")
    k = sub.add_parser("karta"); k.add_argument("--device", type=int, default=15)
    k.add_argument("--civ", action="store_true")
    sv = sub.add_parser("serwer"); sv.add_argument("--port", type=int, default=PORT)
    m = sub.add_parser("most"); m.add_argument("--device", type=int, default=15)
    m.add_argument("--civ", action="store_true")
    m.add_argument("--host", default="127.0.0.1"); m.add_argument("--port", type=int, default=PORT)
    m.add_argument("--wav", type=Path, default=None, help="nagranie zamiast karty (próba bez radia)")
    a = ap.parse_args()
    if a.cmd == "symuluj":
        symuluj(a.pliki, krok_s=a.krok, rzeczywisty=not a.szybko)
    elif a.cmd == "karta":
        karta(a.device, a.civ)
    elif a.cmd == "serwer":
        serwer(a.port)
    else:
        most(a.device, a.civ, a.host, a.port, a.wav)


if __name__ == "__main__":
    main()
