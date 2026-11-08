"""DEKODER CW — okno Qt (08.11): odbiór na żywo i odczyt plików.

Zakładki: „Odbiór" (widok sygnału, tekst), „Źródło" (skąd dekodować — jedno
naraz: wejście dźwiękowe albo nagranie odtwarzane jak z radia), „Ustawienia"
(CI-V, ton, zapis sesji). „Słuchaj" — dekoder na żywo (tools/gui/odbior.py,
ten sam co w aplikacji 0.9.x, w osobnym procesie). „Otwórz plik…" — cały
plik naraz (tools/gui/analiza.py): widok od początku, tekst do skopiowania
(dla SN2HQ: WAV z zawodów -> tekst).

    python -m tools.gui.dekoder [PLIK.wav]
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QFile, QObject, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QIcon
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import (QAbstractButton, QApplication, QComboBox, QFileDialog, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem, QMenu, QSlider,
                               QStatusBar, QTabWidget, QTextEdit, QToolButton)

from dsp import config as C
from tools.gui.analiza import _prazki, hz_mel
from tools.gui.odbior import MiernikTonu, Odbior, SMetrWidok, cfg_karty
from tools.gui.podglad import UI, polacz_lupe, wstaw
from tools.gui.ustawienia import katalog, karty, porty_com, ustawienia
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
        self.miernik = wstaw(self.okno, "ramkaMiernikTonu", MiernikTonu())
        self.smetr = wstaw(self.okno, "ramkaSmetr", SMetrWidok())
        self.odbior = Odbior(self)
        self._pasek_tekstu()
        self._zakladka_ustawien()
        self._zakladka_zrodla()
        self._prog_blokady()
        self._pamiec_okna()
        self.okno.findChild(QAbstractButton, "btnZglos").setEnabled(False)   # jeszcze nie w tej wersji
        self.most = _Most()
        self.most.gotowe.connect(self.pokaz)
        self.siec = None
        self.zrodlo = None                       # ostatni plik: podpowiedź nazwy przy zapisie
        self.stan.setText("Słuchaj — odbiór na żywo; Otwórz plik — cały plik naraz")

    def _pasek_tekstu(self):
        """Pasek obok tekstu (operator 08.11): czcionka +/-, wyczyść, kopiuj,
        zapisz. Wielkość czcionki zapamiętana jak wybór pasów."""
        ust = ustawienia()
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
        b("btnWyczysc").clicked.connect(lambda: self.odbior.wyczysc_tekst())
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

    def _zakladka_ustawien(self):
        """Zakładka „Ustawienia" zamiast okienka startowego aplikacji 0.9.x
        (operator 08.11). Te same pola pod tymi samymi nazwami (aplikacja/...),
        zapis od razu po zmianie. Karta zapamiętana NAZWĄ z API, nie numerem:
        Windows przenumerowuje karty po przepięciu USB (23.10: 15 -> 18 -> 20).
        Radio na żywo jeszcze w starej aplikacji — tu pola już czekają."""
        u = ustawienia()
        o = self.okno
        k = katalog()
        o.findChild(QLineEdit, "edtKatalog").setText(str(k))
        o.findChild(QAbstractButton, "btnKatalogOtworz").clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(k))))
        zap = lambda klucz: (lambda v: u.setValue(f"aplikacja/{klucz}", v))   # noqa: E731
        for nazwa, klucz, dom in (("chkCiv", "civ", False), ("chkZapis", "zapis", True),
                                  ("chkAktualizacje", "aktualizacje", True)):
            c = o.findChild(QAbstractButton, nazwa)
            c.setChecked(u.value(f"aplikacja/{klucz}", dom, type=bool))
            c.toggled.connect(zap(klucz))
        cp = o.findChild(QComboBox, "cmbPort")
        porty = porty_com()
        cp.addItems(porty)
        cp.setCurrentText(u.value("aplikacja/port", porty[0] if porty else ""))
        cp.currentTextChanged.connect(zap("port"))
        cb = o.findChild(QComboBox, "cmbBaud")
        cb.setCurrentText(str(u.value("aplikacja/baud", 115200)))
        cb.currentTextChanged.connect(lambda v: u.setValue("aplikacja/baud", int(v or 115200)))
        et = o.findChild(QLineEdit, "edtTon")
        ton = u.value("aplikacja/ton", None)
        et.setText("" if ton in (None, "", "None") else f"{float(ton):g}")

        def ton_zmiana():
            try:
                v = float(et.text().replace(",", ".")) if et.text().strip() else None
            except ValueError:
                v = None
            u.setValue("aplikacja/ton", v if v else "")
        et.editingFinished.connect(ton_zmiana)

    PLIK_JAK_RADIO = "__plik__"

    def _zakladka_zrodla(self):
        """Źródło do wyboru z listy (operator 08.11) — jedno naraz. Karta
        zapamiętana NAZWĄ z API, nie numerem: Windows przenumerowuje karty
        po przepięciu USB (23.10: 15 -> 18 -> 20)."""
        u, o = ustawienia(), self.okno
        self.lista = o.findChild(QListWidget, "lstZrodla")
        opis = o.findChild(QLabel, "lblZrodloOpis")

        def wypelnij():
            self.lista.clear()
            nazwy = [n for _, n in karty()]
            for n in nazwy:
                it = QListWidgetItem(("📻  " if "USB Audio" in n else "🎤  ") + n)
                it.setData(Qt.UserRole, n)
                self.lista.addItem(it)
            it = QListWidgetItem("📼  Nagranie odtwarzane jak z radia (tempo na zakładce Odbiór)…")
            it.setData(Qt.UserRole, self.PLIK_JAK_RADIO)
            self.lista.addItem(it)
            pam = u.value("aplikacja/karta", "")
            dom = next((n for n in nazwy if "USB Audio" in n and "WASAPI" in n), nazwy[0] if nazwy else "")
            cel = pam if pam in nazwy + [self.PLIK_JAK_RADIO] else dom
            for i in range(self.lista.count()):
                if self.lista.item(i).data(Qt.UserRole) == cel:
                    self.lista.setCurrentRow(i)

        def zmiana():
            it = self.lista.currentItem()
            if it is None:
                return
            n = it.data(Qt.UserRole)
            u.setValue("aplikacja/karta", n)
            if n == self.PLIK_JAK_RADIO:
                opis.setText("Przy „Słuchaj” program zapyta o plik; nagranie idzie przez dekoder "
                             "na żywo jak dźwięk z radia.")
            else:
                opis.setText(f"Wejście: {n}. Radio przez USB to zwykle „USB Audio … WASAPI”. "
                             "Program tylko słucha — nic do radia nie wysyła.")

        self.lista.currentItemChanged.connect(lambda *_: zmiana())
        wypelnij()
        zmiana()
        o.findChild(QAbstractButton, "btnOdswiezZrodla").clicked.connect(wypelnij)
        self.przyciski_sluchaj = [o.findChild(QAbstractButton, n) for n in ("btnSluchaj", "btnRadio")]
        for b in self.przyciski_sluchaj:
            b.clicked.connect(self.sluchaj)
        if not u.contains("okno/geometria"):     # pierwszy start: najpierw wybór źródła
            o.findChild(QTabWidget, "zakladki").setCurrentIndex(1)

    def _prog_blokady(self):
        """Suwak blokady szumów -> wspólna wartość z procesem dekodera; zapamiętany."""
        u, o = ustawienia(), self.okno
        sld, lbl = o.findChild(QSlider, "sldProg"), o.findChild(QLabel, "lblProgWartosc")

        def zmiana(v):
            self.odbior.prog.value = float(v)
            lbl.setText(f"{v} dB")
            u.setValue("okno/prog_blokady", v)
            self.miernik.prog = float(v)
            self.miernik.update()
        sld.valueChanged.connect(zmiana)
        sld.setValue(int(u.value("okno/prog_blokady", 0)))
        zmiana(sld.value())

    def _przyciski(self, tak: bool):
        for b in self.przyciski_sluchaj:
            b.blockSignals(True)
            b.setChecked(tak)
            b.setText("Zatrzymaj" if tak else "Słuchaj")
            b.blockSignals(False)

    def sluchaj(self, plik: str | None = None):
        """Start / stop odbioru na żywo ze źródła z zakładki „Źródło".
        plik — nagranie bez pytania (testy)."""
        if self.odbior.dziala:
            self.odbior.stop()
            self._przyciski(False)
            self.stan.setText("Zatrzymane")
            return
        it = self.lista.currentItem()
        if it is None:
            self.okno.findChild(QTabWidget, "zakladki").setCurrentIndex(1)
            self._przyciski(False)
            return
        n, u = it.data(Qt.UserRole), ustawienia()
        if n == self.PLIK_JAK_RADIO or isinstance(plik, str):
            p = plik if isinstance(plik, str) else QFileDialog.getOpenFileName(
                self.okno, "Nagranie odtwarzane jak z radia",
                u.value("okno/folder_plikow", str(katalog() / "sesje")),
                "Dźwięk (*.wav *.flac);;Wszystkie (*)")[0]
            if not p:
                self._przyciski(False)
                return
            u.setValue("okno/folder_plikow", str(Path(p).parent))
            from tools.radio import civ
            tempo = {"1×": 1.0, "2×": 2.0, "4×": 4.0, "max": 0.0}.get(
                self.okno.findChild(QComboBox, "cmbTempo").currentText(), 1.0)
            baza = cfg_karty("plik", u)
            cfg = {**baza, "wav": p, "tempo": tempo, "zapis": False, "z_civ": False,
                   "ton_reczny": civ.ton_z_opisu(Path(p).with_suffix(".txt")) or baza["ton_reczny"]}
            self.zrodlo = Path(p)
        else:
            cfg = cfg_karty(n, u)
            self.zrodlo = None
        w = self.widok
        w.ustaw_pamiec(120.0)
        w.opis = ""
        for p_ in w.pasma:
            p_.opis = "(ton przestrojony na 750 Hz)" if p_.nazwa == "po obróbce" else ""
        w.ustaw_odchylki(np.zeros((0, 2)))
        w.ustaw_na_zywo(True)
        self.odbior.start(cfg)
        self._przyciski(True)
        self.okno.findChild(QTabWidget, "zakladki").setCurrentIndex(0)

    def koniec_odbioru(self):
        self._przyciski(False)

    def _pamiec_okna(self):
        """Położenie i rozmiar okna, lupa, przełączniki nakładek, zakładka —
        przy zamknięciu do ustawienia.ini, przy starcie z powrotem."""
        u, o = ustawienia(), self.okno
        if u.contains("okno/geometria"):
            o.restoreGeometry(u.value("okno/geometria"))
        for n in ("chkZbocza", "chkGrzebien", "chkPrzyciagaj"):
            if u.contains(f"okno/{n}"):
                o.findChild(QAbstractButton, n).setChecked(u.value(f"okno/{n}", True, type=bool))
        if u.contains("okno/lupa_s"):
            self.widok.ustaw_lupe(u.value("okno/lupa_s", 4.0, type=float))

        class _Zamkniecie(QObject):
            def eventFilter(s, obj, ev):
                if ev.type() == QEvent.Close:
                    self.odbior.stop()
                    u.setValue("okno/geometria", o.saveGeometry())
                    for n in ("chkZbocza", "chkGrzebien", "chkPrzyciagaj"):
                        u.setValue(f"okno/{n}", o.findChild(QAbstractButton, n).isChecked())
                    u.setValue("okno/lupa_s", self.widok.widok_s)
                    u.sync()
                return False
        self._filtr = _Zamkniecie()
        o.installEventFilter(self._filtr)

    def _menu_pasow(self):
        """Pasy do wyboru jak w IC-7300 (operator 08.11); wybór w ustawienia.ini,
        więc okno otwiera się tak, jak je zostawiono."""
        ust = ustawienia()
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
        ust = ustawienia()
        start = ust.value("okno/folder_plikow", str(katalog() / "sesje"))
        p, _ = QFileDialog.getOpenFileName(self.okno, "Plik do odczytu", start,
                                           "Dźwięk (*.wav *.flac *.ogg);;Wszystkie (*)")
        if p:
            ust.setValue("okno/folder_plikow", str(Path(p).parent))
            self.otworz(Path(p))

    def otworz(self, sciezka: Path):
        if self.odbior.dziala:                   # plik zamiast radia: odbiór na żywo stop
            self.odbior.stop()
            self._przyciski(False)
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
    if sys.platform == "win32":
        # bez własnego AppUserModelID Windows grupuje okno pod python.exe
        # i na pasku zadań pokazuje ikonę Pythona zamiast naszej
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("SQ2BVN.DekoderCW.Qt")
    app = QApplication(sys.argv)
    from tools.aplikacja import _ikona          # ta sama ikona co 0.9.x i instalator
    if (ik := _ikona()) is not None:
        app.setWindowIcon(QIcon(ik))
    d = Dekoder()
    d.okno.show()
    if len(sys.argv) > 1:
        d.otworz(Path(sys.argv[1]))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
