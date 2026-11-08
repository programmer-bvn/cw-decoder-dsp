"""ODBIÓR NA ŻYWO W OKNIE Qt — przeniesiony z okienka tkinter (08.11).

Dekoder ten sam co w aplikacji 0.9.x: tools/na_zywo._praca_okna w OSOBNYM
PROCESIE (24.10: w wątku przeliczenia co 2 s zatrzymywały okno), wyniki
przez kolejkę. Z cfg["widok"] proces dokłada kolumny 5 ms dla widoku sygnału
(tools/gui/strumien.py). Tu: start/stop procesu, odbiór kolejki co 50 ms,
tekst (zatwierdzony, słownik na zielono, wstępny szary), miernik tonu nad
szumem z progiem blokady, raport RST, S-metr przez CI-V (tylko odczyt).

Logika tekstu jak w na_zywo.okno: końcówka (ostatnie OTWARTE znaków) co krok
rysowana od nowa — słownik może jeszcze przestawić spacje; starsza część
zamrożona na spacji.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import re
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import QWidget

from dsp import config as C, zbocza as Z

OTWARTE = 120
ZAKRES_TONU_DB = 40.0                # jak miernik w okienku tkinter
ZIELONY, SZARY = QColor("#2f9e4f"), QColor("#8a9099")


class MiernikTonu(QWidget):
    """Ton nad szumem [dB] z kreską progu blokady; szary, gdy pod progiem."""

    def __init__(self):
        super().__init__()
        self.nad, self.prog = 0.0, 0.0

    def paintEvent(self, _):
        q = QPainter(self)
        q.fillRect(self.rect(), QColor("#22262b"))
        w = self.width()
        kol = "#3fa34d" if self.prog <= 0 or self.nad >= self.prog else "#5a6068"
        q.fillRect(0, 0, int(w * np.clip(self.nad / ZAKRES_TONU_DB, 0, 1)), self.height(), QColor(kol))
        x = int(w * self.prog / ZAKRES_TONU_DB)
        q.fillRect(x - 1, 0, 2, self.height(), QColor("#e05050"))


class SMetrWidok(QWidget):
    """Skala jak w radiu: S1–S9 na 55 % szerokości, S9+60 dB na reszcie
    (na_zywo.okno, 03.11). Bez CI-V — szary z podpisem, czego brakuje."""

    def __init__(self):
        super().__init__()
        self.sm, self.opis = None, ""

    def paintEvent(self, _):
        q = QPainter(self)
        w, h = self.width(), self.height()
        q.fillRect(self.rect(), QColor("#22262b"))
        q.setFont(QFont(self.font().family(), 8))
        if self.sm is None:
            q.setPen(QColor("#9aa0a8"))
            q.drawText(self.rect(), 0x84, self.opis)          # AlignCenter
            return
        s9 = -93.0 if self.sm.get("mhz") and self.sm["mhz"] > 30 else -73.0
        pol = 0.55 * w
        p = self.sm["dbm"]
        x = max(0.0, (p - (s9 - 54)) / 54 * pol) if p <= s9 else min(w, pol + (p - s9) / 60 * (w - pol))
        q.fillRect(0, 0, int(min(x, pol)), h, QColor("#3fa34d"))
        if x > pol:
            q.fillRect(int(pol), 0, int(x - pol), h, QColor("#c84a3a"))
        q.setPen(QColor("#e8e8e8"))
        q.drawText(self.rect().adjusted(0, 0, -4, 0), 0x82, self.sm["s"])   # AlignRight|VCenter


class Odbior(QObject):
    def __init__(self, dek):
        super().__init__()
        self.dek = dek
        self.ctx = mp.get_context("spawn")
        self.q = self.ctx.Queue()
        self.prog = self.ctx.Value("d", 0.0)
        self.proc = self.stop_ev = None
        self.smetr = None
        self.zn: list = []                    # zatwierdzone znaki (z, od, do)
        self.od = 0                           # od tego znaku końcówka rysowana od nowa
        self.poz_otwarte = 0                  # pozycja w tekście, od której końcówka
        self.u = float("nan")
        self.sesja = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._odbierz)
        self.timer.start(50)

    # ------------------------------------------------------------ proces
    @property
    def dziala(self) -> bool:
        return self.proc is not None and self.proc.is_alive()

    def start(self, cfg: dict) -> None:
        from tools.na_zywo import _praca_okna
        self.stop()
        self.wyczysc_tekst()
        self.stop_ev = self.ctx.Event()
        # nie daemon: dekoder sam ma proces analizy (ProcessPoolExecutor)
        self.proc = self.ctx.Process(target=_praca_okna, args=(self.q, self.stop_ev, self.prog,
                                                               {**cfg, "widok": True, "t0": time.time()}))
        self.proc.start()
        if cfg.get("z_civ") and not cfg.get("wav"):
            # port jest jeden: start dekodera czyta przez niego ton i pasmo,
            # więc S-metr rusza z opóźnieniem (jak w okienku tkinter)
            from tools.radio.smetr import SMetr
            self.smetr = SMetr(cfg.get("civ_port") or "COM5", cfg.get("civ_baud") or 115200,
                               opoznienie=10.0).start()

    def stop(self) -> None:
        if self.smetr is not None:
            self.smetr.stop()
            self.smetr = None
        p = self.proc
        if p is None:
            return
        self.stop_ev.set()
        # odbierać dalej: dekoder z pełną kolejką nie skończyłby (sesja niezamknięta)
        koniec = time.monotonic() + 8
        while p.is_alive() and time.monotonic() < koniec:
            self._oproznij()
            p.join(timeout=0.1)
        if p.is_alive():
            p.terminate()
        self._oproznij()
        self.proc = None

    def _oproznij(self) -> None:
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass

    # ------------------------------------------------------------ kolejka
    def _odbierz(self) -> None:
        from tools.teksty import T
        ost = None
        try:
            for _ in range(200):                  # nie zawłaszczać okna przy zaległościach
                m = self.q.get_nowait()
                if m[0] == "koniec":
                    self.dek.stan.setText(T("koniec_pracy"))
                    self.dek.koniec_odbioru()
                elif m[0] == "sesja":
                    self.sesja = m[1]
                elif m[0] == "info":
                    self.dek.stan.setText(T(m[1], **m[2]))
                elif m[0] == "blad":
                    self.dek.stan.setText(T("blad_okno"))
                    self.dek.tekst.append(m[1])
                elif m[0] == "widok":
                    self._widok(m[1])
                elif m[0] == "krok":
                    _, nowe, wst, st, nad, u, _w1, _w2, rst = m
                    self.zn += nowe
                    if np.isfinite(u):
                        self.u = u
                    ost = (wst, st, nad, u, rst)
        except queue.Empty:
            pass
        if ost is not None:
            self._pokaz(*ost)
        if self.smetr is not None:
            self.dek.smetr.sm = self.smetr.stan
            self.dek.smetr.opis = T("smetr_cisza") if self.smetr.blad else T("smetr_czeka")
            self.dek.smetr.update()

    def _widok(self, o: dict) -> None:
        w = self.dek.widok
        j0 = o["j0"]
        for klucz, nazwa in (("wej", "wejście"), ("dos", "po obróbce"), ("siec", "oczami sieci")):
            if o.get(klucz) is not None:
                w.ustaw(nazwa, j0, o[klucz])
        if "p" in o:
            w.ustaw("P(klucz)", *o["p"])
        n = o.get("nakladki")
        if n:
            w.zastap_znaczniki(n["t_od"], n["zeby"], n["zbocza"], n["pary"])
            h, r = n["regen"]
            od = max(h, int(round(n["t_od"] / (Z.HOP_ZBOCZA / C.SR))))
            w.ustaw("regenerat", od, r[od - h:])
            if "opis" in n:
                w._po_nazwie["regenerat"].opis = n["opis"]
        if np.isfinite(self.u):
            w.opis = f"u {self.u * 1000:.0f} ms · {1.2 / self.u:.0f} WPM"
        if w.na_zywo:
            w.update()

    # ------------------------------------------------------------ tekst
    def wyczysc_tekst(self) -> None:
        """Tylko ekran: zapis sesji zostaje pełny."""
        self.dek.tekst.clear()
        self.zn, self.od, self.poz_otwarte = [], 0, 0

    def _wstaw(self, c: QTextCursor, znaki: list) -> None:
        from dsp import slownik as SL
        if np.isfinite(self.u) and self.u > 0:
            slowa = SL.koloruj(znaki, self.u)
        else:
            slowa = [(w, False) for w in "".join(z[0] for z in znaki).split()]
        zwykly, zielony = QTextCharFormat(), QTextCharFormat()
        zielony.setForeground(ZIELONY)
        for s, znane in slowa:
            c.insertText(s + " ", zielony if znane else zwykly)

    def _pokaz(self, wst: str, st: str, nad: float, u: float, rst: dict | None) -> None:
        from tools.teksty import T
        t = self.dek.tekst
        sb = t.verticalScrollBar()
        na_dole = sb.value() >= sb.maximum() - 2 and not t.textCursor().hasSelection()
        c = QTextCursor(t.document())
        c.setPosition(self.poz_otwarte)
        c.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
        c.removeSelectedText()
        if len(self.zn) - self.od > OTWARTE:               # zamrożenie starszej części
            k = len(self.zn) - OTWARTE // 2
            while k < len(self.zn) and self.zn[k][0] != " ":
                k += 1
            self._wstaw(c, self.zn[self.od:k])
            self.od = k
            self.poz_otwarte = c.position()
        self._wstaw(c, self.zn[self.od:])
        szary = QTextCharFormat()
        szary.setForeground(SZARY)
        c.insertText(wst, szary)
        if na_dole:
            sb.setValue(sb.maximum())
        self.dek.stan.setText(st)
        m = self.dek.miernik
        m.nad, m.prog = float(nad), float(self.prog.value)
        m.update()
        self._raport(rst or {}, T)

    def _raport(self, rst: dict, T) -> None:
        d = self.dek
        sm = self.smetr.stan if self.smetr is not None else None
        s = None
        if sm:
            g = re.search(r"S(\d)", sm["s"])
            s = max(1, int(g.group(1))) if g else None
        for lit, v in (("R", rst.get("r")), ("S", s), ("T", rst.get("t"))):
            d.okno.findChild(QWidget, f"lbl{lit}").setText("–" if v is None else str(v))
        czesci = []
        if rst.get("pewnosc") is not None:
            czesci.append(T("raport_r", p=f"{100 * rst['pewnosc']:.0f}"))
        if rst.get("tetnienie_proc") is not None:
            czesci.append(T("raport_t", t=f"{rst['tetnienie_proc']:.0f}", s=f"{rst['tlo_proc']:.0f}"))
        d.okno.findChild(QWidget, "lblRaportOpis").setText(", ".join(czesci) if czesci else T("raport_czeka"))


def cfg_karty(nazwa: str, u) -> dict:
    """Ustawienia z zakładek -> cfg procesu dekodera (jak aplikacja 0.9.x)."""
    from tools.gui import analiza as A
    from tools.gui.ustawienia import katalog
    ton = u.value("aplikacja/ton", "")
    try:
        ton = float(ton) if ton not in ("", None, "None") else None
    except (TypeError, ValueError):
        ton = None
    return dict(device=nazwa, z_civ=u.value("aplikacja/civ", False, type=bool),
                zapis=u.value("aplikacja/zapis", True, type=bool),
                model_npz=str(Path(A.SIEC).resolve()),
                civ_port=u.value("aplikacja/port", "") or None,
                civ_baud=int(u.value("aplikacja/baud", 115200)), ton_reczny=ton,
                sesje=str(katalog() / "sesje"), zrodlo=f"karta {nazwa} (okno Qt)")
