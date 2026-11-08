"""PODGLĄD OKNA Qt — tools/gui/okno.ui z ruchomymi atrapami (08.11).

Operator układa okno w Qt Designerze (tools/gui/designer.bat), zapisuje,
uruchamia ten podgląd i widzi, jak to wygląda „w ruchu": wodospady
przewijają się, S-metr i miernik tonu chodzą, tekst przybywa. Dane są
sztuczne — to podgląd wyglądu, nie dekoder. Styl (kolory, czcionki) można
też dać w Designerze (styleSheet) — kod go nie nadpisuje.

    python -m tools.gui.podglad [--ui tools/gui/okno.ui]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import QFile, QTimer, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication, QFrame, QVBoxLayout, QWidget

UI = Path(__file__).with_name("okno.ui")


def _jet(x: np.ndarray) -> np.ndarray:
    """0..1 -> RGB jak wodospady obecnej aplikacji (jet)."""
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


class Wodospad(QWidget):
    """Przewijany obraz: nowa kolumna z prawej (czas w poziomie, częstotliwość w pionie)."""

    def __init__(self, wierszy: int):
        super().__init__()
        self.obraz = np.zeros((wierszy, 600, 3), np.uint8)

    def dodaj(self, kolumna: np.ndarray) -> None:
        self.obraz = np.roll(self.obraz, -1, axis=1)
        self.obraz[:, -1] = _jet(np.clip(kolumna, 0, 1))[::-1]
        self.update()

    def paintEvent(self, _):
        h, w, _c = self.obraz.shape
        img = QImage(np.ascontiguousarray(self.obraz).data, w, h, 3 * w, QImage.Format_RGB888)
        QPainter(self).drawImage(self.rect(), img)


class Pasek(QWidget):
    """Poziomy wskaźnik 0..1 (miernik tonu, S-metr) z płynnym dojściem do wartości."""

    def __init__(self, kolor: str):
        super().__init__()
        self.kolor, self.cel, self.v = QColor(kolor), 0.0, 0.0

    def ustaw(self, v: float) -> None:
        self.cel = float(np.clip(v, 0, 1))

    def krok(self) -> None:
        self.v += 0.25 * (self.cel - self.v)       # wskazówka dochodzi, nie skacze
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#22262b"))
        p.fillRect(0, 0, int(self.width() * self.v), self.height(), self.kolor)


class Przebieg(QWidget):
    """P(klucz) co 5 ms — linia nad progiem 0,5."""

    def __init__(self):
        super().__init__()
        self.p = np.zeros(600)

    def dodaj(self, v: float) -> None:
        self.p = np.roll(self.p, -1); self.p[-1] = v; self.update()

    def paintEvent(self, _):
        q = QPainter(self)
        q.fillRect(self.rect(), QColor("#111417"))
        w, h = self.width(), self.height()
        q.setPen(QPen(QColor("#3a4048"), 1, Qt.DashLine)); q.drawLine(0, h // 2, w, h // 2)
        q.setPen(QPen(QColor("#e0b050"), 1.5))
        xs = np.linspace(0, w - 1, self.p.size); ys = (1 - self.p) * (h - 2) + 1
        for i in range(1, self.p.size):
            q.drawLine(int(xs[i - 1]), int(ys[i - 1]), int(xs[i]), int(ys[i]))


def wstaw(okno, nazwa: str, widzet: QWidget) -> QWidget:
    ramka = okno.findChild(QFrame, nazwa)
    if ramka is None:
        raise SystemExit(f"okno.ui: brak ramki '{nazwa}' — nazwa zmieniona w Designerze?")
    if ramka.layout() is None:
        lay = QVBoxLayout(ramka); lay.setContentsMargins(0, 0, 0, 0)
    ramka.layout().addWidget(widzet)
    return widzet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ui", type=Path, default=UI)
    ap.add_argument("--zrzut", type=Path, default=None,
                    help="po 4 s zapisz obraz okna do PNG i zakończ (test bez patrzenia)")
    a = ap.parse_args()
    app = QApplication(sys.argv)
    f = QFile(str(a.ui)); f.open(QFile.ReadOnly)
    okno = QUiLoader().load(f); f.close()
    wej, dos, sie = (wstaw(okno, n, Wodospad(k)) for n, k in
                     (("ramkaWodWejscie", 120), ("ramkaWodDoSieci", 120), ("ramkaWodSiec", 32)))
    ton = wstaw(okno, "ramkaMiernikTonu", Pasek("#4caf50"))
    sm = wstaw(okno, "ramkaSmetr", Pasek("#e0b050"))
    pp = wstaw(okno, "ramkaPrzebiegP", Przebieg())
    tekst = okno.findChild(QWidget, "txtOdczyt")
    rng = np.random.default_rng(0)
    morse = "--.-. -.-.  --.-  -.. .  ... --.- ..--- -... ...- -.  .--. ...- -..- ..--- "  # rytm atrapy
    stan = {"t": 0}
    zdanie = "CQ CQ DE SQ2BVN SQ2BVN PSE K   "

    def tik():
        t = stan["t"] = stan["t"] + 1
        klucz = morse[(t // 6) % len(morse)] != " "
        szum = rng.random(120) * 0.35
        kol = szum.copy(); kol[58:62] += 0.6 if klucz else 0
        wej.dodaj(kol + (rng.random(120) > 0.97) * 0.5)        # obce sygnały
        dos.dodaj(np.where(np.arange(120) // 10 == 5, kol, szum * 0.5))
        m = rng.random(32) * 0.25; m[14:18] += 0.7 if klucz else 0
        sie.dodaj(m)
        pp.dodaj((0.92 if klucz else 0.06) + rng.normal(0, 0.03))
        ton.ustaw(0.55 + 0.25 * np.sin(t / 40)); sm.ustaw(0.5 + 0.2 * np.sin(t / 70))
        ton.krok(); sm.krok()
        if t % 25 == 0:
            tekst.insertPlainText(zdanie[(t // 25) % len(zdanie)])

    timer = QTimer(); timer.timeout.connect(tik); timer.start(20)
    if a.zrzut:
        okno.setAttribute(Qt.WA_DontShowOnScreen)
        QTimer.singleShot(4000, lambda: (okno.grab().save(str(a.zrzut)), app.quit()))
    okno.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
