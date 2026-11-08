"""WIDOK SYGNAŁU — wodospady i P(klucz) na jednej osi czasu (08.11).

POMYSŁ OPERATORA (08.11): trzy osobne wodospady to „partyzantka". Jeden
komponent: pasy jeden nad drugim (wejście, po obróbce, oczami sieci,
P(klucz)), wspólny czas, lupa w czasie (ile kropek i kresek mieści się na
ekranie), a na wierzchu analiza: białe cienkie linie — zęby grzebienia
tam, gdzie się zakotwiczył; żółte pionowe przez wszystkie pasy — zbocza
sygnału po regeneracji (z nut); u dołu skala w milisekundach. Dla
użytkownika bajer, dla nas ocena na oko, czy detektory trafiają w zbocza —
operator weryfikuje działanie pomiarem, nie czytaniem kodu.

Obsługa myszą:
    kółko            lupa (oś czasu krótsza / dłuższa)
    lewy, ciągnij    przesuwanie w czasie (zatrzymuje „na żywo")
    prawy, ciągnij   pomiar odcinka: Δt w ms (np. długość kropki)
    dwuklik          z powrotem „na żywo" (prawy brzeg = najnowsze dane)

Wszystkie pasy mają kolumnę co KOLUMNA_S (5 ms, jak sieć zboczy) — wtedy
zbocze z P(klucz) leży dokładnie nad tą samą kolumną wodospadu. Pas, który
przychodzi później (sieć liczy oknami po 2,56 s), po prostu ma krótszy
koniec — nie przesuwa reszty.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

KOLUMNA_S = 0.005
SKALA_PX = 18                 # wysokość paska skali czasu u dołu
GESTO_PX = 8                  # najmniejszy średni odstęp linii nakładki (4 px: przy 30 s wciąż sama żółć)


def _lut_jet() -> np.ndarray:
    x = np.linspace(0, 1, 256)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


LUT = _lut_jet()


class _Pierscien:
    """Bufor kołowy kolumn zapisany dwa razy (i oraz i+N): każdy wycinek
    o długości ≤ N jest ciągły w pamięci — bez kopiowania przy rysowaniu."""

    def __init__(self, n: int, wiersze: int, typ):
        self.n, self.licz = n, 0
        self.d = np.zeros((2 * n, wiersze) if wiersze else (2 * n,), typ)

    def dodaj(self, blok: np.ndarray) -> None:
        k = len(blok)
        if k > self.n:
            blok, self.licz = blok[-self.n:], self.licz + k - self.n
            k = self.n
        i = (self.licz + np.arange(k)) % self.n
        self.d[i] = blok
        self.d[i + self.n] = blok
        self.licz += k

    def wycinek(self, j0: int, j1: int) -> tuple[int, np.ndarray]:
        """Kolumny [j0, j1) przycięte do tego, co jest w pamięci: (j0', dane)."""
        j0, j1 = max(j0, self.licz - self.n, 0), min(j1, self.licz)
        if j1 <= j0:
            return j0, self.d[:0]
        i0 = j0 % self.n
        return j0, self.d[i0:i0 + (j1 - j0)]


class Pasmo:
    def __init__(self, nazwa: str, rodzaj: str, wiersze: int, waga: float, n: int):
        self.nazwa, self.rodzaj, self.waga = nazwa, rodzaj, waga
        self.buf = (_Pierscien(n, wiersze, np.uint8) if rodzaj == "obraz"
                    else _Pierscien(n, 0, np.float32))


def _ladny_krok(s: float) -> float:
    """Krok podziałki 1-2-5 x 10^n nie mniejszy niż s."""
    e = 10 ** np.floor(np.log10(s))
    return float(next(m * e for m in (1, 2, 5, 10) if m * e >= s))


class WidokSygnalu(QWidget):
    lupa_zmieniona = Signal(float)          # widoczny odcinek w s
    na_zywo_zmienione = Signal(bool)

    def __init__(self, pasma: list[tuple[str, str, int, float]], pamiec_s: float = 120.0,
                 parent=None):
        """pasma: [(nazwa, 'obraz'|'przebieg', wierszy, waga wysokości), ...]."""
        super().__init__(parent)
        n = int(pamiec_s / KOLUMNA_S)
        self.pasma = [Pasmo(*p, n) for p in pasma]
        self._po_nazwie = {p.nazwa: p for p in self.pasma}
        self.widok_s, self.pamiec_s = 4.0, pamiec_s
        self.prawy_s = 0.0                  # czas prawego brzegu (gdy nie na żywo)
        self.na_zywo = True
        self.pokaz_zbocza = self.pokaz_grzebien = True
        self.zbocza = np.zeros(0)           # czasy w s od początku
        self.zeby = np.zeros(0)
        self.opis = ""                      # np. „u 60 ms · 20 WPM"
        self._ciag = None                   # (x startu, prawy_s na starcie)
        self._pomiar = None                 # (t0, t1)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(120)
        self.setMouseTracking(False)

    # ---- dane ----
    def dodaj(self, nazwa: str, blok: np.ndarray) -> None:
        """Kolumny do pasa: obraz (k, wierszy) wartości 0..1, przebieg (k,) 0..1."""
        p = self._po_nazwie[nazwa]
        b = np.asarray(blok, np.float32)
        if p.rodzaj == "obraz":
            b = (np.clip(np.atleast_2d(b), 0, 1) * 255).astype(np.uint8)
        else:
            b = np.atleast_1d(b)
        p.buf.dodaj(b)

    def dodaj_zbocza(self, t: np.ndarray) -> None:
        self.zbocza = self._przytnij(np.r_[self.zbocza, np.asarray(t, float)])

    def dodaj_zeby(self, t: np.ndarray) -> None:
        self.zeby = self._przytnij(np.r_[self.zeby, np.asarray(t, float)])

    def wyczysc(self) -> None:
        for p in self.pasma:
            p.buf.licz = 0
        self.zbocza = self.zeby = np.zeros(0)
        self.ustaw_na_zywo(True)

    def _przytnij(self, t: np.ndarray) -> np.ndarray:
        return t[t >= self.koniec_s() - self.pamiec_s]

    def koniec_s(self) -> float:
        return max(p.buf.licz for p in self.pasma) * KOLUMNA_S

    # ---- widok ----
    def ustaw_lupe(self, s: float) -> None:
        s = float(np.clip(s, 0.25, self.pamiec_s))
        if s != self.widok_s:
            self.widok_s = s
            self.lupa_zmieniona.emit(s)
            self.update()

    def ustaw_na_zywo(self, tak: bool) -> None:
        if not tak:
            self.prawy_s = self._prawy()
        if tak != self.na_zywo:
            self.na_zywo = tak
            self.na_zywo_zmienione.emit(tak)
        self.update()

    def _prawy(self) -> float:
        return self.koniec_s() if self.na_zywo else self.prawy_s

    def _obszar(self) -> QRectF:
        return QRectF(0, 0, self.width(), self.height() - SKALA_PX)

    def _x(self, t: float, lewy: float) -> float:
        return (t - lewy) / self.widok_s * self.width()

    def _t(self, x: float) -> float:
        return self._prawy() - self.widok_s + x / max(1, self.width()) * self.widok_s

    # ---- mysz ----
    def wheelEvent(self, e):
        self.ustaw_lupe(self.widok_s * (0.8 if e.angleDelta().y() > 0 else 1.25))

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.ustaw_na_zywo(False)
            self._ciag = (e.position().x(), self.prawy_s)
        elif e.button() == Qt.RightButton:
            t = self._t(e.position().x())
            self._pomiar = (t, t)
            self.update()

    def mouseMoveEvent(self, e):
        if self._ciag is not None:
            x0, p0 = self._ciag
            dt = (e.position().x() - x0) / max(1, self.width()) * self.widok_s
            self.prawy_s = float(np.clip(p0 - dt, min(self.widok_s, self.koniec_s()),
                                         self.koniec_s()))
            self.update()
        elif self._pomiar is not None and e.buttons() & Qt.RightButton:
            self._pomiar = (self._pomiar[0], self._t(e.position().x()))
            self.update()

    def mouseReleaseEvent(self, e):
        self._ciag = None

    def mouseDoubleClickEvent(self, e):
        self._pomiar = None
        self.ustaw_na_zywo(True)

    # ---- rysowanie ----
    def paintEvent(self, _):
        q = QPainter(self)
        q.fillRect(self.rect(), QColor("#0d0f12"))
        ob = self._obszar()
        prawy = self._prawy()
        lewy = prawy - self.widok_s
        j0, j1 = int(np.floor(lewy / KOLUMNA_S)), int(np.ceil(prawy / KOLUMNA_S))
        suma = sum(p.waga for p in self.pasma)
        y = ob.top()
        prostokaty = []
        for p in self.pasma:
            h = ob.height() * p.waga / suma
            r = QRectF(0, y, ob.width(), h - 2)
            prostokaty.append(r)
            self._rysuj_pasmo(q, p, r, j0, j1, lewy)
            y += h
        self._rysuj_znaczniki(q, ob, lewy, prawy)
        for p, r in zip(self.pasma, prostokaty):
            self._etykieta(q, r.topLeft() + QPointF(4, 2), p.nazwa)
        if self.opis:
            q.setFont(QFont(self.font().family(), 9))
            w = q.fontMetrics().horizontalAdvance(self.opis) + 10
            self._etykieta(q, QPointF(ob.right() - w, 2), self.opis)
        self._rysuj_skale(q, ob, lewy, prawy)
        self._rysuj_pomiar(q, ob, lewy)

    def _rysuj_pasmo(self, q: QPainter, p: Pasmo, r: QRectF, j0: int, j1: int, lewy: float):
        q.fillRect(r, QColor("#111417"))
        a, d = p.buf.wycinek(j0, j1)
        if len(d) == 0:
            return
        # Więcej kolumn niż pikseli: ŚREDNIA w grupie. Maksimum (pierwsza
        # wersja 08.11) zawyżało szum — przy widoku 30 s tło wodospadu robiło
        # się jasnoniebieskie, ton ginął. Kropka to ≥ 8 kolumn (30 WPM), więc
        # średnia rozmywa ją dopiero przy widoku liczonym w minutach.
        krok = max(1, int(len(d) / max(1, r.width())))
        if krok > 1:
            m = len(d) // krok * krok
            d = d[:m].reshape(m // krok, krok, *d.shape[1:]).mean(axis=1)
            if p.rodzaj == "obraz":
                d = d.astype(np.uint8)
        x0 = r.left() + self._x(a * KOLUMNA_S, lewy)
        szer = len(d) * krok * KOLUMNA_S / self.widok_s * r.width()
        if p.rodzaj == "obraz":
            rgb = np.ascontiguousarray(LUT[d.T[::-1]])         # niskie pasma na dole
            h, w = rgb.shape[:2]
            img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
            q.drawImage(QRectF(x0, r.top(), szer, r.height()), img)
        else:
            q.setPen(QPen(QColor("#3a4048"), 1, Qt.DashLine))
            q.drawLine(QPointF(r.left(), r.center().y()), QPointF(r.right(), r.center().y()))
            xs = x0 + (np.arange(len(d)) + 0.5) * szer / len(d)
            ys = r.bottom() - 1 - np.clip(d, 0, 1) * (r.height() - 3)
            q.setPen(QPen(QColor("#e0b050"), 1.5))
            q.drawPolyline([QPointF(float(a_), float(b_)) for a_, b_ in zip(xs, ys)])

    def _rysuj_znaczniki(self, q: QPainter, ob: QRectF, lewy: float, prawy: float):
        """Linie gęściej niż co GESTO_PX zalewają obraz (widok 30 s: sama żółć
        — zrzut 08.11) — wtedy pas ich nie rysuje i mówi, żeby przybliżyć."""
        ukryte = []
        for pokaz, t, pioro, nazwa in (
                (self.pokaz_grzebien, self.zeby, QPen(QColor(255, 255, 255, 150), 1), "grzebień"),
                (self.pokaz_zbocza, self.zbocza, QPen(QColor("#ffd400"), 1.2), "zbocza")):
            if not pokaz or not t.size:
                continue
            t = t[(t >= lewy) & (t <= prawy)]
            if t.size > self.width() / GESTO_PX:
                ukryte.append(nazwa)
                continue
            q.setPen(pioro)
            for ti in t:
                x = self._x(ti, lewy)
                q.drawLine(QPointF(x, ob.top()), QPointF(x, ob.bottom()))
        if ukryte:
            napis = f"{' i '.join(ukryte)}: za gęsto — przybliż lupą"
            q.setFont(QFont(self.font().family(), 9))
            self._etykieta(q, QPointF(ob.center().x() - q.fontMetrics().horizontalAdvance(napis) / 2,
                                      ob.top() + 2), napis)

    def _rysuj_skale(self, q: QPainter, ob: QRectF, lewy: float, prawy: float):
        """Podziałka liczona od prawego brzegu (−200 ms, −400 ms...): tak
        czyta się długość kropki/kreski. Na prawym brzegu czas bezwzględny."""
        y = ob.bottom() + 1
        q.fillRect(QRectF(0, y, ob.width(), SKALA_PX), QColor("#0d0f12"))
        q.setFont(QFont(self.font().family(), 8))
        q.setPen(QColor("#9aa4ad"))
        m, s = divmod(prawy, 60)
        napis = f"{int(m)}:{s:06.3f}" + ("" if self.na_zywo else "  stop")
        wn = q.fontMetrics().horizontalAdvance(napis)
        q.drawText(QPointF(ob.width() - wn - 4, y + 13), napis)
        krok = _ladny_krok(self.widok_s / max(2, self.width() / 90))
        k = 1
        while k * krok < self.widok_s:
            x = self._x(prawy - k * krok, lewy)
            ms = k * krok * 1000
            opis = f"−{ms:.0f} ms" if krok < 1 else f"−{ms / 1000:g} s"
            if x + 2 + q.fontMetrics().horizontalAdvance(opis) < ob.width() - wn - 10:
                q.drawLine(QPointF(x, y), QPointF(x, y + 4))
                q.drawText(QPointF(x + 2, y + 13), opis)
            k += 1

    def _rysuj_pomiar(self, q: QPainter, ob: QRectF, lewy: float):
        if self._pomiar is None:
            return
        t0, t1 = sorted(self._pomiar)
        x0, x1 = self._x(t0, lewy), self._x(t1, lewy)
        q.fillRect(QRectF(x0, ob.top(), max(1.0, x1 - x0), ob.height()), QColor(80, 200, 255, 50))
        q.setPen(QPen(QColor("#50c8ff"), 1))
        q.drawLine(QPointF(x0, ob.top()), QPointF(x0, ob.bottom()))
        q.drawLine(QPointF(x1, ob.top()), QPointF(x1, ob.bottom()))
        self._etykieta(q, QPointF(x1 + 4, ob.center().y()), f"Δt {1000 * (t1 - t0):.0f} ms")

    def _etykieta(self, q: QPainter, gdzie: QPointF, tekst: str):
        q.setFont(QFont(self.font().family(), 9))
        fm = q.fontMetrics()
        r = QRectF(gdzie.x(), gdzie.y(), fm.horizontalAdvance(tekst) + 8, fm.height() + 2)
        q.fillRect(r, QColor(0, 0, 0, 150))
        q.setPen(QColor("#d8dee4"))
        q.drawText(r, Qt.AlignCenter, tekst)
