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
from PySide6.QtCore import QFile, QObject, Qt, Signal
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import (QAbstractButton, QApplication, QFileDialog, QLabel,
                               QStatusBar, QTextEdit)

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
            ("po obróbce (ton przestrojony na 750 Hz)", "obraz", hz.size, 3, {"hz": hz, "db": db}),
            ("oczami sieci", "obraz", C.N_MELS, 2, {"hz": hz_mel(), "db": db}),
            ("P(klucz)", "przebieg", 0, 1)]))
        polacz_lupe(self.okno, self.widok)
        self.tekst = self.okno.findChild(QTextEdit, "txtOdczyt")
        self.stan = self.okno.findChild(QLabel, "lblStan")
        self.pasek = self.okno.findChild(QStatusBar, "pasekStanu")
        self.btn = self.okno.findChild(QAbstractButton, "btnOtworz")
        self.btn.clicked.connect(self.wybierz)
        self.okno.findChild(QAbstractButton, "btnWyczysc").clicked.connect(self.tekst.clear)
        for n in ("btnRadio", "btnZglos", "btnNaZywo"):     # jeszcze nie w tej wersji
            self.okno.findChild(QAbstractButton, n).setEnabled(False)
        self.most = _Most()
        self.most.gotowe.connect(self.pokaz)
        self.siec = None
        self.stan.setText("Otwórz plik WAV")

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
        w.dodaj("po obróbce (ton przestrojony na 750 Hz)", r["dos"])
        w.dodaj("oczami sieci", r["siec"])
        w.dodaj("P(klucz)", r["p"])
        w.dodaj_zeby(r["zeby"]); w.dodaj_zbocza(r["zbocza"])
        wpm = 1.2 / r["u"] if np.isfinite(r["u"]) else float("nan")
        w.opis = f"u {r['u'] * 1000:.0f} ms · {wpm:.0f} WPM · ton {r['ton']:.0f} Hz"
        w.ustaw_lupe(4.0)
        w.ustaw_na_zywo(False)
        w.prawy_s = min(4.0, r["dl_s"])                 # plik czyta się od początku
        w.update()
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
