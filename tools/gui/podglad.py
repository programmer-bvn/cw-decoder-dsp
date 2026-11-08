"""PODGLĄD OKNA Qt — tools/gui/okno.ui z ruchomymi atrapami (08.11).

Operator układa okno w Qt Designerze (tools/gui/designer.bat), zapisuje,
uruchamia ten podgląd i widzi, jak to wygląda „w ruchu": wodospady
przewijają się (jeden widok sygnału z lupą i nakładkami), S-metr i miernik tonu chodzą, tekst przybywa. Dane są
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
from PySide6.QtGui import QColor, QPainter
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication, QFrame, QVBoxLayout, QWidget

from tools.gui.widok_sygnalu import KOLUMNA_S, WidokSygnalu  # noqa: E402

UI = Path(__file__).with_name("okno.ui")


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


def polacz_lupe(okno, widok: WidokSygnalu) -> None:
    """Suwak lupy w skali log (0,25 s ... pamięć), przełączniki nakładek, „na żywo"."""
    from PySide6.QtWidgets import QAbstractButton, QLabel, QSlider
    sld, lbl = okno.findChild(QSlider, "sldLupa"), okno.findChild(QLabel, "lblLupaWartosc")
    zyw = okno.findChild(QAbstractButton, "btnNaZywo")
    lo, hi = np.log(0.25), np.log(widok.pamiec_s)

    def na_s(v): return float(np.exp(lo + (hi - lo) * v / 100))
    def z_s(s): return int(round(100 * (np.log(s) - lo) / (hi - lo)))
    def opis(s): lbl.setText(f"{s * 1000:.0f} ms" if s < 1 else f"{s:.1f} s")

    sld.setValue(z_s(widok.widok_s)); opis(widok.widok_s)
    sld.valueChanged.connect(lambda v: widok.ustaw_lupe(na_s(v)))
    widok.lupa_zmieniona.connect(lambda s: (sld.blockSignals(True), sld.setValue(z_s(s)),
                                            sld.blockSignals(False), opis(s)))
    for nazwa, pole in (("chkZbocza", "pokaz_zbocza"), ("chkGrzebien", "pokaz_grzebien")):
        okno.findChild(QAbstractButton, nazwa).toggled.connect(
            lambda v, pole=pole: (setattr(widok, pole, v), widok.update()))
    okno.findChild(QAbstractButton, "chkKursory").toggled.connect(widok.ustaw_kursory)
    okno.findChild(QAbstractButton, "chkPrzyciagaj").toggled.connect(widok.ustaw_przyciaganie)
    zyw.toggled.connect(widok.ustaw_na_zywo)
    widok.na_zywo_zmienione.connect(lambda v: (zyw.blockSignals(True), zyw.setChecked(v),
                                               zyw.blockSignals(False)))


def wstaw(okno, nazwa: str, widzet: QWidget) -> QWidget:
    ramka = okno.findChild(QFrame, nazwa)
    if ramka is None:
        raise SystemExit(f"okno.ui: brak ramki '{nazwa}' — nazwa zmieniona w Designerze?")
    if ramka.layout() is None:
        lay = QVBoxLayout(ramka); lay.setContentsMargins(0, 0, 0, 0)
    ramka.layout().addWidget(widzet)
    return widzet


def atrapa_kluczowania(tekst: str, u: float):
    """Kolumny 5 ms kluczowania tekstu w idealnym takcie u, zbocza (s) i zęby
    grzebienia (s): od początku każdego znaku co u do jego końca."""
    from dsp import config as C
    klucz, zbocza, zeby, t = [], [], [], 0.0

    def dodaj(stan, d):
        nonlocal t
        klucz.extend([stan] * int(round(d / KOLUMNA_S)))
        t += d

    for slowo in tekst.split():
        for z in slowo:
            start = t
            for i, e in enumerate(C.MORSE_DICT[z]):
                if i:
                    dodaj(False, u)
                zbocza.append(t); dodaj(True, u if e == "." else 3 * u); zbocza.append(t)
            zeby.extend(start + k * u for k in range(int(round((t - start) / u)) + 1))
            dodaj(False, 3 * u)
        dodaj(False, 4 * u)
    return np.array(klucz), np.array(zbocza), np.array(zeby), t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ui", type=Path, default=UI)
    ap.add_argument("--zrzut", type=Path, default=None,
                    help="po 4 s zapisz obraz okna do PNG i zakończ (test bez patrzenia)")
    a = ap.parse_args()
    app = QApplication(sys.argv)
    f = QFile(str(a.ui)); f.open(QFile.ReadOnly)
    okno = QUiLoader().load(f); f.close()
    from dsp import config as C, melref
    db = (C.DB_MIN, C.DB_MAX)              # skala koloru jak wejście sieci
    hz_wej = np.linspace(300, 1200, 120)   # atrapa; prawdziwy wodospad poda swoje
    m = np.linspace(melref.hz_to_mel(C.FMIN), melref.hz_to_mel(C.FMAX), C.N_MELS + 2)
    hz_mel = np.asarray(melref.mel_to_hz(m[1:-1]))
    widok = wstaw(okno, "ramkaWidokSygnalu", WidokSygnalu([
        ("wejście", "obraz", 120, 3, {"hz": hz_wej, "db": db}),
        ("po obróbce", "obraz", 120, 3, {"hz": hz_wej, "db": db}),
        ("oczami sieci", "obraz", 32, 2, {"hz": hz_mel, "db": db}),
        ("P(klucz)", "przebieg", 0, 1)]))
    widok.opis = "atrapa · u 60 ms · 20 WPM"
    ton = wstaw(okno, "ramkaMiernikTonu", Pasek("#4caf50"))
    sm = wstaw(okno, "ramkaSmetr", Pasek("#e0b050"))
    tekst = okno.findChild(QWidget, "txtOdczyt")
    polacz_lupe(okno, widok)
    rng = np.random.default_rng(0)
    zdanie = "CQ CQ DE SQ2BVN SQ2BVN PSE K"
    klucz, zbocza, zeby, okres = atrapa_kluczowania(zdanie, 0.06)
    stan = {"j": 0}
    KOL = 4                                   # 20 ms tik = 4 kolumny po 5 ms

    def tik():
        j0 = stan["j"]; stan["j"] += KOL
        idx = np.arange(j0, j0 + KOL)
        k = klucz[idx % klucz.size].astype(np.float32)
        szum = rng.random((KOL, 120)) * 0.35
        wej = szum.copy(); wej[:, 58:62] += 0.6 * k[:, None]
        wej += (rng.random((KOL, 120)) > 0.97) * 0.5          # obce sygnały
        dos = szum * 0.4; dos[:, 58:62] = wej[:, 58:62]
        m = rng.random((KOL, 32)) * 0.25; m[:, 14:18] += 0.7 * k[:, None]
        widok.dodaj("wejście", wej); widok.dodaj("po obróbce", dos)
        widok.dodaj("oczami sieci", m)
        widok.dodaj("P(klucz)", np.clip(0.06 + 0.86 * k + rng.normal(0, 0.03, KOL), 0, 1))
        t0, t1 = j0 * KOLUMNA_S, (j0 + KOL) * KOLUMNA_S
        n0 = int(t0 // okres)
        for zb, ze in ((zbocza, widok.dodaj_zbocza), (zeby, widok.dodaj_zeby)):
            tt = zb + n0 * okres
            tt = np.r_[tt, tt + okres]
            ze(tt[(tt >= t0) & (tt < t1)])
        t = j0 // KOL
        ton.ustaw(0.55 + 0.25 * np.sin(t / 40)); sm.ustaw(0.5 + 0.2 * np.sin(t / 70))
        ton.krok(); sm.krok()
        if t % 25 == 0:
            tekst.insertPlainText((zdanie + "   ")[(t // 25) % (len(zdanie) + 3)])
        if widok.na_zywo:
            widok.update()

    timer = QTimer(); timer.timeout.connect(tik); timer.start(20)
    if a.zrzut:
        okno.setAttribute(Qt.WA_DontShowOnScreen)
        QTimer.singleShot(4000, lambda: (okno.grab().save(str(a.zrzut)), app.quit()))
    okno.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
