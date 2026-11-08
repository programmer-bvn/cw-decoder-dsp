"""USTAWIENIA OKNA Qt — plik INI obok nagrań (08.11).

Operator 08.11: wszystko, co użytkownik ustawi, ma wracać — „żeby 20 razy
tego samego nie ustawiał". Plik INI, nie rejestr: program ma działać też
na Linuksie, a tam rejestru nie ma (QSettings.IniFormat — ten sam kod na
obu systemach). Miejsce: folder z nagraniami (Dokumenty/DekoderCW, jak
tools/aplikacja.KATALOG) — na koncie użytkownika, więc zapis działa także
przy programie zainstalowanym przez administratora, a ustawienia i sesje
leżą obok siebie (łatwo przenieść albo dołączyć do zgłoszenia).

Przy pierwszym starcie (brak pliku) przejmuje to, co jest w rejestrze
HKCU\\Software\\SQ2BVN\\DekoderCW: ustawienia aplikacji 0.9.x (JSON pod
nazwami pól) i pierwsze ustawienia okna Qt z 08.11 (pasy, czcionka).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from PySide6.QtCore import QSettings, QStandardPaths

PLIK = "ustawienia.ini"
KLUCZ_REJESTRU = r"Software\SQ2BVN\DekoderCW"
POLA_APLIKACJI = ("jezyk", "karta", "civ", "port", "baud", "ton", "zapis", "aktualizacje")


def katalog() -> Path:
    """Windows: dokładnie jak aplikacja 0.9.x (USERPROFILE\\Documents) —
    sesje z obu wersji w jednym miejscu. Inne systemy: Dokumenty wg XDG.
    DEKODER_KATALOG — inny folder (testy: nie ruszać prawdziwych ustawień)."""
    if os.environ.get("DEKODER_KATALOG"):
        k = Path(os.environ["DEKODER_KATALOG"])
    elif sys.platform == "win32":
        k = Path(os.environ.get("USERPROFILE", Path.home())) / "Documents" / "DekoderCW"
    else:
        dok = QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation) or str(Path.home())
        k = Path(dok) / "DekoderCW"
    k.mkdir(parents=True, exist_ok=True)
    return k


_UST: QSettings | None = None


def ustawienia() -> QSettings:
    global _UST
    if _UST is None:
        p = katalog() / PLIK
        nowy = not p.exists()
        _UST = QSettings(str(p), QSettings.IniFormat)
        if nowy:
            _z_rejestru(_UST)
    return _UST


def _z_rejestru(u: QSettings) -> None:
    if sys.platform != "win32":
        return
    import winreg
    try:
        k = winreg.OpenKey(winreg.HKEY_CURRENT_USER, KLUCZ_REJESTRU)
    except OSError:
        return
    i = 0
    while True:
        try:
            nazwa, wart, _ = winreg.EnumValue(k, i)
        except OSError:
            break
        i += 1
        if nazwa in POLA_APLIKACJI:
            try:
                u.setValue(f"aplikacja/{nazwa}", json.loads(wart))
            except (TypeError, ValueError):
                pass
    stare = QSettings("SQ2BVN", "DekoderCW")      # okno Qt przed 08.11 (rejestr)
    for klucz in stare.allKeys():
        if klucz.startswith(("pasy/", "tekst/")):
            u.setValue(klucz, stare.value(klucz))
    u.sync()


def porty_com() -> list[str]:
    """Windows: rejestr SERIALCOMM (jak aplikacja 0.9.x); Linux: /dev/ttyUSB*,
    /dev/ttyACM* (Icom z USB to zwykle ttyUSB)."""
    if sys.platform == "win32":
        from tools.aplikacja import _porty_com
        return _porty_com()
    return sorted(str(p) for w in ("ttyUSB*", "ttyACM*") for p in Path("/dev").glob(w))


def karty() -> list[tuple[int, str]]:
    try:
        from tools.aplikacja import _karty
        return _karty()
    except Exception:                              # brak sounddevice / PortAudio
        return []
