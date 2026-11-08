"""DEKODER CW — okno Qt z prawdziwym odczytem pliku WAV (08.11).

Pierwszy krok przenosin z tkintera: „Otwórz plik…" -> cały łańcuch odczytu
(tools/gui/analiza.py) w tle -> widok sygnału (wejście, po obróbce, oczami
sieci, P(klucz), grzebień, zbocza regeneratu) i tekst do zaznaczenia
i skopiowania (dla SN2HQ: WAV z zawodów -> tekst). Radio na żywo — jeszcze
w starej aplikacji (tools/aplikacja.py).

    python -m tools.gui.dekoder [PLIK.wav]
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QFile, QObject, QSettings, Qt, Signal
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import (QAbstractButton, QApplication, QFileDialog, QLabel, QMenu,
                               QStatusBar, QTextEdit, QToolButton)

from dsp import config as C
from tools.gui.analiza import _prazki, hz_mel
from tools.gui.podglad import UI, polacz_lupe, wstaw
from tools.gui.widok_sygnalu import WidokSygnalu


class _Most(QObject):
    """Wynik z wątku liczącego do wątku okna (Qt rysuje tylko w swoim)."""
    gotowe = Signal(object, object, float)


class Dekoder:
    def __init__(self):
        f = QFile(str(UI)); f.open(QFile.ReadOnly)
        self.okno = QUiLoader().load(f); f.close()
        db = (C.DB_MIN, C.DB_MAX)
        hz = _prazki()[2]
        self.widok = wstaw(self.okno, "ramkaWidokSygnalu", WidokSygnalu([
            ("wejście", "obraz", hz.size, 3, {"hz": hz, "db": db}),
            ("po obróbce", "obraz", hz.size, 3, {"hz": hz, "db": db}),
            ("oczami sieci", "obraz", C.N_MELS, 2, {"hz": hz_mel(), "db": db}),
            ("P(klucz)", "przebieg", 0, 1),
            ("regenerat", "przebieg", 0, 1, {"kolor": "#7cf0a0", "odchylki": True})]))
        self.widok._po_nazwie["po obróbce"].opis = "(ton przestrojony na 750 Hz)"
        polacz_lupe(self.okno, self.widok)
        self._menu_pasow()
        self.tekst = self.okno.findChild(QTextEdit, "txtOdczyt")
        self.stan = self.okno.findChild(QLabel, "lblStan")
        self.pasek = self.okno.findChild(QStatusBar, "pasekStanu")
        self.btn = self.okno.findChild(QAbstractButton, "btnOtworz")
        self.btn.clicked.connect(self.wybierz)
        self._pasek_tekstu()
        for n in ("btnRadio", "btnZglos", "btnNaZywo"):     # jeszcze nie w tej wersji
            self.okno.findChild(QAbstractButton, n).setEnabled(False)
        self.most = _Most()
        self.most.gotowe.connect(self.pokaz)
        self.siec = None
        self.zrodlo = None                       # ostatni plik: podpowiedź nazwy przy zapisie
        self.stan.setText("Otwórz plik WAV")

    def _pasek_tekstu(self):
        """Pasek obok tekstu (operator 08.11): czcionka +/-, wyczyść, kopiuj,
        zapisz. Wielkość czcionki zapamiętana jak wybór pasów."""
        ust = QSettings("SQ2BVN", "DekoderCW")
        t = self.tekst

        def czcionka(pt: float):
            f = t.font()
            f.setPointSizeF(float(np.clip(pt, 6, 72)))
            t.setFont(f)
            ust.setValue("tekst/punkty", f.pointSizeF())

        czcionka(ust.value("tekst/punkty", 12.0, type=float))
        b = lambda n: self.okno.findChild(QAbstractButton, n)    # noqa: E731
        b("btnCzcionkaWieksza").clicked.connect(lambda: czcionka(t.font().pointSizeF() * 1.15))
        b("btnCzcionkaMniejsza").clicked.connect(lambda: czcionka(t.font().pointSizeF() / 1.15))
        b("btnWyczysc").clicked.connect(t.clear)
        b("btnKopiuj").clicked.connect(self.kopiuj)
        b("btnZapisz").clicked.connect(self.zapisz)

    def kopiuj(self):
        """Zaznaczenie, a bez zaznaczenia cały tekst; do schowka idzie też
        wersja z kolorami (wklejona do Worda czy Thunderbirda zostaje kolorowa)."""
        from PySide6.QtCore import QMimeData
        c = self.tekst.textCursor()
        m = QMimeData()
        if c.hasSelection():
            m.setText(c.selectedText().replace(" ", "\n"))
            m.setHtml(c.selection().toHtml())
        else:
            m.setText(self.tekst.toPlainText())
            m.setHtml(self.tekst.toHtml())
        QApplication.clipboard().setMimeData(m)
        self.pasek.showMessage("Skopiowane do schowka", 3000)

    def zapisz(self):
        """HTML zamiast RTF: Qt zapisuje go sam, z kolorami, a otwiera go każda
        przeglądarka, Word i LibreOffice — bez żadnych dodatkowych narzędzi."""
        domyslna = str(Path(self.zrodlo).with_suffix(".html")) if self.zrodlo else "odczyt.html"
        p, filtr = QFileDialog.getSaveFileName(self.okno, "Zapisz odczyt", domyslna,
                                               "HTML z kolorami (*.html);;Zwykły tekst (*.txt)")
        if not p:
            return
        p = Path(p)
        if "txt" in filtr:
            p.with_suffix(".txt").write_text(self.tekst.toPlainText(), encoding="utf-8")
        else:
            p.with_suffix(".html").write_text(self.tekst.toHtml(), encoding="utf-8")
        self.pasek.showMessage(f"Zapisane: {p}", 5000)

    def _menu_pasow(self):
        """Pasy do wyboru jak w IC-7300 (operator 08.11); wybór w QSettings
        (rejestr Windows), więc okno otwiera się tak, jak je zostawiono."""
        ust = QSettings("SQ2BVN", "DekoderCW")
        menu = QMenu(self.okno)
        for p in self.widok.pasma:
            a = menu.addAction(p.nazwa)
            a.setCheckable(True)
            tak = ust.value(f"pasy/{p.nazwa}", True, type=bool)
            a.setChecked(tak)
            self.widok.pokaz_pasmo(p.nazwa, tak)
            a.toggled.connect(lambda v, n=p.nazwa: (self.widok.pokaz_pasmo(n, v),
                                                    ust.setValue(f"pasy/{n}", v)))
        btn = self.okno.findChild(QToolButton, "btnPasy")
        btn.setMenu(menu)
        self._menu = menu                        # Python trzyma menu przy życiu

    def wybierz(self):
        p, _ = QFileDialog.getOpenFileName(self.okno, "Plik do odczytu", str(Path("probki").resolve()),
                                           "Dźwięk (*.wav *.flac *.ogg);;Wszystkie (*)")
        if p:
            self.otworz(Path(p))

    def otworz(self, sciezka: Path):
        self.btn.setEnabled(False)
        self.stan.setText(f"Liczę: {sciezka.name}…")
        self.tekst.clear()

        def licz():
            t = time.perf_counter()
            try:
                from dsp.siec_numpy import SiecNumpy
                from tools.gui import analiza
                if self.siec is None:
                    self.siec = SiecNumpy(analiza.SIEC)
                r = analiza.analizuj(sciezka, self.siec)
            except Exception as e:                      # awaria ma sama mówić, co się stało
                r = e
            self.most.gotowe.emit(sciezka, r, time.perf_counter() - t)

        threading.Thread(target=licz, daemon=True).start()

    def pokaz(self, sciezka: Path, r, czas: float):
        self.btn.setEnabled(True)
        if isinstance(r, Exception):
            self.stan.setText(f"Błąd: {type(r).__name__}: {r}")
            return
        w = self.widok
        w.ustaw_pamiec(r["dl_s"] + 1)
        w.dodaj("wejście", r["wej"])
        w.dodaj("po obróbce", r["dos"])
        w.dodaj("oczami sieci", r["siec"])
        w.dodaj("P(klucz)", r["p"])
        w.dodaj("regenerat", r["regen"])
        w.dodaj_zeby(r["zeby"]); w.dodaj_zbocza(r["zbocza"])
        o = r["odchylki"]
        w.ustaw_odchylki(o["pary"])
        reg = w._po_nazwie["regenerat"]
        reg.opis = (f"— zbocza P od regeneratu: rozrzut {o['rozrzut_ms']:.1f} ms = {o['rozrzut_u']:.2f} u, "
                    f"waga {o['waga_ms']:+.0f} ms ({o['n']} zboczy)" if "rozrzut_ms" in o else "")
        wpm = 1.2 / r["u"] if np.isfinite(r["u"]) else float("nan")
        w.opis = f"u {r['u'] * 1000:.0f} ms · {wpm:.0f} WPM · ton {r['ton']:.0f} Hz"
        w.ustaw_lupe(4.0)
        w.ustaw_na_zywo(False)
        w.prawy_s = min(4.0, r["dl_s"])                 # plik czyta się od początku
        w.update()
        self.zrodlo = sciezka
        self.tekst.setPlainText(r["tekst"])
        self.stan.setText(f"{sciezka.name} — {r['dl_s']:.0f} s nagrania, odczyt {czas:.0f} s")
        okna = r["okna"]
        if okna:
            pew = np.mean([o[3] for o in okna])
            self.pasek.showMessage(f"grzebień: {len(okna)} okien po ~8 s, średnia pewność {pew:.2f}; "
                                   "lewy: przesuwanie, kółko: lupa, dwuklik: koniec pliku")


def main() -> None:
    app = QApplication(sys.argv)
    d = Dekoder()
    d.okno.show()
    if len(sys.argv) > 1:
        d.otworz(Path(sys.argv[1]))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
