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

KURSORY (operator 08.11, „jak na oscyloskopie"). Każdy pas ma swoją grupę:
dwa pionowe kursory czasu A i B (Δt), na każdym dwie kropki do przesuwania
góra-dół. Kropka odczytuje poziom i częstotliwość swojego wiersza; dwie
kropki na jednym kursorze — szerokość pasma (Δf) i stosunek poziomów
(sygnał do tła), te same kropki na A i B — dryf częstotliwości. Kursor
przyciąga się (gdy włączone) do najbliższej widocznej linii: zęba
grzebienia albo zbocza — wtedy kropki i kreski mierzy się co do jednostki.
Ciągnie się je lewym przyciskiem; poza kursorami lewy dalej przesuwa czas.

POZIOM W PROCENTACH, nie w dB (operator 08.11): dB bezwzględne trzeba by
kalibrować na każdej karcie dźwiękowej. Kolor wodospadu jest w skali dB
(DB_MIN...DB_MAX, jak wejście sieci), więc kropka przelicza go z powrotem
na AMPLITUDĘ LINIOWĄ w % góry skali — procent z samego koloru (80% sygnał,
75% tło) wyglądałby na małą różnicę, a to 0,05 x 54 dB = 2,7 dB, 1,4x
amplitudy. Odczyt to średnia 3 x 3 komórek: jedna komórka szumu skacze.

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
    """opcje: hz — środki wierszy w Hz od dołu (mel: nierówne odstępy),
    db — (min, max) skali koloru; bez db procent = wartość wprost."""

    def __init__(self, nazwa: str, rodzaj: str, wiersze: int, waga: float,
                 opcje: dict | None, n: int):
        self.nazwa, self.rodzaj, self.waga, self.wiersze = nazwa, rodzaj, waga, wiersze
        o = opcje or {}
        self.hz = None if o.get("hz") is None else np.asarray(o["hz"], float)
        self.db = o.get("db")
        self.buf = (_Pierscien(n, wiersze, np.uint8) if rodzaj == "obraz"
                    else _Pierscien(n, 0, np.float32))
        self.kursory: list[dict] = []          # A, B: {"t": s, "y": [kropka1, kropka2] 0..1 od dołu}

    def procent(self, v: float) -> float:
        """Wartość 0..1 z bufora -> amplituda liniowa w % góry skali."""
        if self.db is None:
            return 100.0 * v
        return 100.0 * 10 ** ((v - 1.0) * (self.db[1] - self.db[0]) / 20.0)

    def wiersz(self, y: float) -> int:
        return int(np.clip(y * self.wiersze, 0, self.wiersze - 1))

    def odczyt(self, t: float, y: float = 0.0) -> float:
        """Poziom (%) w czasie t i wysokości y: średnia 3 kolumny x 3 wiersze."""
        j = int(t / KOLUMNA_S)
        _, d = self.buf.wycinek(j - 1, j + 2)
        if len(d) == 0:
            return float("nan")
        if self.rodzaj != "obraz":
            return 100.0 * float(d.mean())
        w = self.wiersz(y)
        return self.procent(float(d[:, max(0, w - 1):w + 2].mean()) / 255.0)

    def czestotliwosc(self, y: float) -> float:
        return float("nan") if self.hz is None else float(self.hz[self.wiersz(y)])


def _ladny_krok(s: float) -> float:
    """Krok podziałki 1-2-5 x 10^n nie mniejszy niż s."""
    e = 10 ** np.floor(np.log10(s))
    return float(next(m * e for m in (1, 2, 5, 10) if m * e >= s))


class WidokSygnalu(QWidget):
    lupa_zmieniona = Signal(float)          # widoczny odcinek w s
    na_zywo_zmienione = Signal(bool)

    def __init__(self, pasma: list[tuple[str, str, int, float]], pamiec_s: float = 120.0,
                 parent=None):
        """pasma: [(nazwa, 'obraz'|'przebieg', wierszy, waga wysokości[, opcje]), ...]
        — opcje jak w Pasmo (hz, db)."""
        super().__init__(parent)
        n = int(pamiec_s / KOLUMNA_S)
        self.pasma = [Pasmo(*p[:4], p[4] if len(p) > 4 else None, n) for p in pasma]
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
        self.kursory_wl = False
        self.przyciaganie = True
        self._chwyt = None                  # (pasmo, nr kursora, nr kropki | None)
        self._prostokaty: list[QRectF] = []
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

    def ustaw_pamiec(self, pamiec_s: float) -> None:
        """Plik dłuższy niż pamięć na żywo (120 s): bufory od nowa, na cały plik."""
        self.pamiec_s = pamiec_s
        n = int(np.ceil(pamiec_s / KOLUMNA_S))
        for p in self.pasma:
            p.buf = (_Pierscien(n, p.wiersze, np.uint8) if p.rodzaj == "obraz"
                     else _Pierscien(n, 0, np.float32))
        self.zbocza = self.zeby = np.zeros(0)
        self.lupa_zmieniona.emit(self.widok_s)

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

    def ustaw_kursory(self, tak: bool) -> None:
        """Włączenie stawia A i B na 1/3 i 2/3 widoku (tam, gdzie się patrzy),
        kropki na połowie i na 1/5 wysokości pasa."""
        self.kursory_wl = tak
        if tak:
            lewy = self._prawy() - self.widok_s
            for p in self.pasma:
                p.kursory = [{"t": lewy + self.widok_s * f, "y": [0.5, 0.2]} for f in (1 / 3, 2 / 3)]
        self.update()

    def ustaw_przyciaganie(self, tak: bool) -> None:
        self.przyciaganie = tak

    def _przyciagnij(self, t: float) -> float:
        if not self.przyciaganie:
            return t
        linie = [x for pokaz, x in ((self.pokaz_grzebien, self.zeby), (self.pokaz_zbocza, self.zbocza))
                 if pokaz and x.size]
        if not linie:
            return t
        x = np.concatenate(linie)
        i = int(np.argmin(np.abs(x - t)))
        return float(x[i]) if abs(x[i] - t) / self.widok_s * self.width() < 12 else t

    def _trafienie(self, pos: QPointF):
        """Kursor / kropka pod myszą: (pasmo, nr kursora, nr kropki | None)."""
        if not self.kursory_wl:
            return None
        lewy = self._prawy() - self.widok_s
        for p, r in zip(self.pasma, self._prostokaty):
            if not r.contains(pos):
                continue
            for k, ku in enumerate(p.kursory):
                x = self._x(ku["t"], lewy)
                if p.rodzaj == "obraz":
                    for i, y in enumerate(ku["y"]):
                        if abs(pos.x() - x) < 8 and abs(pos.y() - (r.bottom() - y * r.height())) < 8:
                            return p, k, i
                if abs(pos.x() - x) < 5:
                    return p, k, None
        return None

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
            self._chwyt = self._trafienie(e.position())
            if self._chwyt is not None:
                return
            self.ustaw_na_zywo(False)
            self._ciag = (e.position().x(), self.prawy_s)
        elif e.button() == Qt.RightButton:
            t = self._t(e.position().x())
            self._pomiar = (t, t)
            self.update()

    def mouseMoveEvent(self, e):
        if self._chwyt is not None:
            p, k, i = self._chwyt
            ku = p.kursory[k]
            if i is None:
                ku["t"] = self._przyciagnij(self._t(e.position().x()))
            else:
                r = self._prostokaty[self.pasma.index(p)]
                ku["y"][i] = float(np.clip((r.bottom() - e.position().y()) / r.height(), 0, 1))
            self.update()
        elif self._ciag is not None:
            x0, p0 = self._ciag
            dt = (e.position().x() - x0) / max(1, self.width()) * self.widok_s
            self.prawy_s = float(np.clip(p0 - dt, min(self.widok_s, self.koniec_s()),
                                         self.koniec_s()))
            self.update()
        elif self._pomiar is not None and e.buttons() & Qt.RightButton:
            self._pomiar = (self._pomiar[0], self._t(e.position().x()))
            self.update()

    def mouseReleaseEvent(self, e):
        self._ciag = self._chwyt = None

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
        self._prostokaty = prostokaty
        if self.kursory_wl:
            for p, r in zip(self.pasma, prostokaty):
                self._rysuj_kursory(q, p, r, lewy)
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

    def _rysuj_kursory(self, q: QPainter, p: Pasmo, r: QRectF, lewy: float):
        kolory = (QColor("#40e0ff"), QColor("#ff60d0"))
        q.save()
        q.setClipRect(r)
        opisy = []
        for k, (ku, kol) in enumerate(zip(p.kursory, kolory)):
            x = self._x(ku["t"], lewy)
            # Ciągła 4 px z ciemnym obrysem: przerywana 1 px ginęła w szumie
            # wodospadu, operator 08.11: „2 razy grubsze, żeby było widać"
            q.setPen(QPen(QColor(0, 0, 0, 170), 6))
            q.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
            q.setPen(QPen(kol, 4))
            q.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
            lit = "AB"[k]
            if p.rodzaj != "obraz":
                opisy.append(f"{lit}: P {p.odczyt(ku['t']):.0f}%")
                continue
            vals = []
            for i, y in enumerate(ku["y"]):
                yy = r.bottom() - y * r.height()
                q.setPen(QPen(QColor("black"), 1))
                q.setBrush(kol)
                q.drawEllipse(QPointF(x, yy), 4.5, 4.5)
                v, f = p.odczyt(ku["t"], y), p.czestotliwosc(y)
                vals.append((v, f))
                napis = f"{lit}{i + 1} {v:.1f}%" + ("" if np.isnan(f) else f" · {f:.0f} Hz")
                self._etykieta(q, QPointF(x + 7, yy - 9), napis)
            q.setBrush(Qt.NoBrush)
            (v1, f1), (v2, f2) = vals
            stos = f"{v1 / v2:.1f}×" if v2 > 0 else "∞"
            opisy.append(f"{lit}1/{lit}2 {stos}" + ("" if np.isnan(f1) else f" Δf {abs(f1 - f2):.0f} Hz"))
        a, b = p.kursory
        linia = f"A→B Δt {1000 * abs(b['t'] - a['t']):.0f} ms   " + "   ".join(opisy)
        if p.rodzaj == "obraz" and p.hz is not None:
            linia += f"   dryf A1→B1 {p.czestotliwosc(b['y'][0]) - p.czestotliwosc(a['y'][0]):+.0f} Hz"
        q.setFont(QFont(self.font().family(), 9))
        w = q.fontMetrics().horizontalAdvance(linia) + 10
        self._etykieta(q, QPointF(r.right() - w, r.bottom() - q.fontMetrics().height() - 4), linia)
        q.restore()

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
