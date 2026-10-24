"""DEKODER CW — aplikacja do rozdania (24.10, operator: „exe dla SP2MSA”,
potem instalator dla ludzi bez Pythona i bez IC-7300).

To samo okienko co `tools.na_zywo okno`, ale:
- bez TensorFlow: sieć zboczy w numpy (dsp/siec_numpy.py, wagi siec.npz),
- okno startowe: język, karta dźwiękowa, CI-V (Icom) z portem albo ton CW
  ręcznie — u innego operatora inne radio i inne numery,
- ustawienia w rejestrze (HKCU\\Software\\SQ2BVN\\DekoderCW) razem z położeniem
  okna i progiem blokady — raz ustawione wracają,
- „Zgłoś problem z dekodowaniem”: raport w stałym układzie (raport.json, FLAC,
  odczyt, dziennik) do przetworzenia automatem, mail przez program pocztowy,
- przy starcie (wyłączalne) pytanie GitHuba o nowszą wersję.

Budowa: budowa_exe.bat (PyInstaller + Inno Setup). Tylko odbiór: CI-V
wyłącznie odczytuje (częstotliwość, tryb, ton CW).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

WERSJA_APLIKACJI = "0.9.2"
NAZWA = "Dekoder CW SQ2BVN"
REPO = "programmer-bvn/cw-decoder-dsp"
# Skrzynka projektu na raporty (konto założył operator 25.10; w Thunderbirdzie na
# maszynie treningowej, reguła [CW-DECODER] -> folder, tools/zgloszenia.py).
# Puste = raport tylko do pliku.
KONTAKT_EMAIL = "programmer.bvn@gmail.com"
KLUCZ_REJESTRU = r"Software\SQ2BVN\DekoderCW"
KATALOG = Path(os.environ.get("USERPROFILE", Path.home())) / "Documents" / "DekoderCW"
WAGI = "siec.npz"
RAPORT_MIN = 15                              # tyle ostatnich minut sesji idzie do raportu
WZOR_SLOWNIKA = """# Słownik osobisty — Dekoder CW / Personal dictionary — CW Decoder.
# Twoje słowa (imię, QTH, znajomi), które mają świecić na zielono i mieć
# poprawione spacje. Jedno lub kilka słów w wierszu; po „#" komentarz.
# Your words (name, QTH, friends) to be shown in green with spaces fixed.
# One or more words per line; „#" starts a comment. Callsigns and RST
# reports are recognised automatically. Changes apply after restart.
"""


# ------------------------------------------------------------------ zasoby
def _zasob(nazwa: str) -> Path:
    """Plik dołączony do exe (PyInstaller: sys._MEIPASS) albo z repozytorium."""
    baza = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    for p in (baza / nazwa, baza / "runs" / "zbocza_5" / nazwa, baza / "tools" / "instalator" / nazwa):
        if p.exists():
            return p
    raise FileNotFoundError(nazwa)


def _ikona() -> str | None:
    try:
        return str(_zasob("dekoder_cw.ico"))
    except FileNotFoundError:
        return None


def _wersja_pelna() -> str:
    try:
        return f"{WERSJA_APLIKACJI} ({_zasob('wersja.txt').read_text(encoding='utf-8').strip()})"
    except FileNotFoundError:
        return WERSJA_APLIKACJI


# --------------------------------------------------------------- rejestr
def ust_czytaj() -> dict:
    """Ustawienia z rejestru (wartości JSON pod nazwami pól)."""
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_CURRENT_USER, KLUCZ_REJESTRU)
    except OSError:
        return {}
    ust, i = {}, 0
    while True:
        try:
            nazwa, wart, _ = winreg.EnumValue(k, i)
        except OSError:
            break
        try:
            ust[nazwa] = json.loads(wart)
        except (TypeError, ValueError):
            pass
        i += 1
    return ust


def ust_zapisz(zmiany: dict) -> None:
    try:
        import winreg
        k = winreg.CreateKey(winreg.HKEY_CURRENT_USER, KLUCZ_REJESTRU)
        for nazwa, wart in zmiany.items():
            winreg.SetValueEx(k, nazwa, 0, winreg.REG_SZ, json.dumps(wart, ensure_ascii=False))
    except OSError as e:
        print(f"rejestr: {e!r}", flush=True)


def _jezyk_systemu() -> str:
    try:
        import ctypes
        return "pl" if ctypes.windll.kernel32.GetUserDefaultUILanguage() & 0x3FF == 0x15 else "en"
    except (AttributeError, OSError):
        return "pl"


# ---------------------------------------------------------- okno startowe
def _porty_com() -> list[str]:
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DEVICEMAP\SERIALCOMM")
        porty, i = [], 0
        while True:
            try:
                porty.append(winreg.EnumValue(k, i)[1])
                i += 1
            except OSError:
                break
        return sorted(porty, key=lambda p: int("".join(c for c in p if c.isdigit()) or 0))
    except OSError:
        return []


def _karty() -> list[tuple[int, str]]:
    """[(numer, „nazwa — API”)] — zapamiętujemy nazwę z API, nie numer: Windows
    przenumerowuje karty po przepięciu USB (23.10: 15 -> 18 -> 20)."""
    import sounddevice as sd
    api = [a["name"] for a in sd.query_hostapis()]
    kar = [(i, f"{d['name']} — {api[d['hostapi']]}") for i, d in enumerate(sd.query_devices())
           if d["max_input_channels"] > 0]
    # WASAPI najpierw: ta sama karta jest widoczna kilka razy (MME, DirectSound, WASAPI, WDM-KS)
    return sorted(kar, key=lambda k: ("WASAPI" not in k[1], k[0]))


def okno_startowe() -> dict | None:
    import tkinter as tk
    from tkinter import ttk
    from tools import teksty
    from tools.teksty import T
    ust = ust_czytaj()
    teksty.JEZYK = ust.get("jezyk") or _jezyk_systemu()
    karty, porty = _karty(), _porty_com()
    wynik: dict = {}
    r = tk.Tk()
    r.title(f"{T('ust_tytul')} — {WERSJA_APLIKACJI}")
    r.resizable(False, False)
    try:
        r.iconbitmap(str(_zasob("dekoder_cw.ico")))
    except (FileNotFoundError, tk.TclError):
        pass
    f = ttk.Frame(r, padding=14)
    f.pack()
    ttk.Label(f, text=T("ust_jezyk")).grid(row=0, column=0, sticky="w")
    v_jez = tk.StringVar(value={"pl": "Polski", "en": "English"}[teksty.JEZYK])
    cj = ttk.Combobox(f, textvariable=v_jez, values=["Polski", "English"], width=10, state="readonly")
    cj.grid(row=0, column=1, sticky="w", padx=6, pady=(0, 8))

    def zmiana_jezyka(_e=None) -> None:
        ust_zapisz({"jezyk": "pl" if v_jez.get() == "Polski" else "en"})
        wynik["od_nowa"] = True
        r.destroy()
    cj.bind("<<ComboboxSelected>>", zmiana_jezyka)

    ttk.Label(f, text=T("ust_karta")).grid(row=1, column=0, columnspan=3, sticky="w")
    v_karta = tk.StringVar()
    ttk.Combobox(f, textvariable=v_karta, values=[k[1] for k in karty], width=70, state="readonly").grid(
        row=2, column=0, columnspan=3, sticky="we", pady=(2, 10))
    zapamietana = next((k[1] for k in karty if k[1] == ust.get("karta")), None)
    domyslna = next((k[1] for k in karty if "USB Audio" in k[1] and "WASAPI" in k[1]), karty[0][1] if karty else "")
    v_karta.set(zapamietana or domyslna)

    v_civ = tk.BooleanVar(value=ust.get("civ", False))
    ttk.Checkbutton(f, text=T("ust_civ"), variable=v_civ).grid(row=3, column=0, sticky="w")
    v_port = tk.StringVar(value=ust.get("port", porty[0] if porty else "COM1"))
    ttk.Combobox(f, textvariable=v_port, values=porty, width=8).grid(row=3, column=1, sticky="w", padx=6)
    v_baud = tk.StringVar(value=str(ust.get("baud", 115200)))
    ttk.Combobox(f, textvariable=v_baud, values=["115200", "19200", "9600"], width=8).grid(row=3, column=2, sticky="w")
    ttk.Label(f, text=T("ust_ton")).grid(row=4, column=0, sticky="w", pady=(8, 0))
    v_ton = tk.StringVar(value=str(ust.get("ton") or ""))
    ttk.Entry(f, textvariable=v_ton, width=8).grid(row=4, column=1, sticky="w", padx=6, pady=(8, 0))
    v_zapis = tk.BooleanVar(value=ust.get("zapis", True))
    ttk.Checkbutton(f, text=T("ust_zapis", k=KATALOG / "sesje"), variable=v_zapis).grid(
        row=5, column=0, columnspan=3, sticky="w", pady=(10, 0))
    v_akt = tk.BooleanVar(value=ust.get("aktualizacje", True))
    ttk.Checkbutton(f, text=T("ust_aktualizacje"), variable=v_akt).grid(row=6, column=0, columnspan=3, sticky="w")
    ttk.Label(f, text=T("ust_slownik", p=KATALOG / "slownik_osobisty.txt"), foreground="#666").grid(
        row=7, column=0, columnspan=3, sticky="w", pady=(4, 0))

    def start() -> None:
        k = next((k for k in karty if k[1] == v_karta.get()), None)
        if k is None:
            return
        try:
            ton = float(v_ton.get().replace(",", ".")) if v_ton.get().strip() else None
        except ValueError:
            ton = None
        wynik.update(karta=k[0], karta_nazwa=k[1], civ=v_civ.get(), port=v_port.get(),
                     baud=int(v_baud.get() or 115200), ton=ton, zapis=v_zapis.get(), aktualizacje=v_akt.get())
        ust_zapisz({"karta": k[1], "civ": wynik["civ"], "port": wynik["port"], "baud": wynik["baud"],
                    "ton": ton, "zapis": wynik["zapis"], "aktualizacje": wynik["aktualizacje"]})
        r.destroy()

    ttk.Button(f, text=T("ust_start"), command=start).grid(row=8, column=2, sticky="e", pady=(14, 0))
    r.mainloop()
    if wynik.get("od_nowa"):
        return okno_startowe()
    return wynik or None


# ------------------------------------------------------------- aktualizacje
def sprawdz_wersje() -> tuple[str, str] | None:
    """Najnowsze wydanie na GitHubie (publiczne API, bez logowania) -> (wersja, adres)
    albo None, gdy nie nowsze."""
    import urllib.request
    zad = urllib.request.Request(f"https://api.github.com/repos/{REPO}/releases/latest",
                                 headers={"Accept": "application/vnd.github+json", "User-Agent": NAZWA})
    with urllib.request.urlopen(zad, timeout=10) as o:
        d = json.load(o)
    tag = str(d.get("tag_name", "")).lstrip("vV")

    def liczby(w: str) -> tuple:
        return tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in w.split("."))
    return (tag, d.get("html_url", f"https://github.com/{REPO}/releases")) \
        if tag and liczby(tag) > liczby(WERSJA_APLIKACJI) else None


# ------------------------------------------------------------------ raport
def _srodowisko() -> dict:
    """Komputer — bez pytania użytkownika (operator 24.10: „niech nie wpisuje,
    my powinniśmy zarejestrować, w jakim stanie był program”)."""
    import platform
    d = {"system": platform.platform(), "procesor": platform.processor(), "rdzenie": os.cpu_count(),
         "python": platform.python_version(), "exe": bool(getattr(sys, "frozen", False))}
    try:
        import ctypes

        class Pamiec(ctypes.Structure):
            _fields_ = ([("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)]
                        + [(n, ctypes.c_ulonglong) for n in ("tot", "av", "pf", "apf", "vt", "av2", "ext")])
        m = Pamiec()
        m.dwLength = ctypes.sizeof(m)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            d["ram_gb"] = round(m.tot / 2 ** 30, 1)
    except (AttributeError, OSError):
        pass
    return d


def _z_sesji(sesja: Path) -> dict:
    """Pola z opisu sesji (pasmo, ton CW z CI-V, tor) i statystyka kroków z dziennika sesji."""
    import re
    import numpy as np
    out: dict = {}
    t = sesja.with_suffix(".txt")
    if t.exists():
        for w in t.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"^(pasmo|ton CW|tor|dekoder):\s*(.*)$", w)
            if m:
                out[m.group(1).replace(" ", "_")] = m.group(2).strip()
    lg = sesja.with_suffix(".log")
    if lg.exists():
        cz, prog = [], []
        for w in lg.read_text(encoding="utf-8", errors="replace").splitlines():
            if w.startswith("#"):
                if "blokada" in w:
                    prog.append(w[2:].strip())
                continue
            p = w.split("\t")
            if len(p) > 1:
                try:
                    cz.append(float(p[1]))
                except ValueError:
                    pass
        if cz:
            out["kroki"] = {"liczba": len(cz), "srednio_s": round(float(np.mean(cz)), 3),
                            "maks_s": round(float(np.max(cz)), 2)}
        out["zmiany_blokady"] = prog[-10:]
    return out


def _zbuduj_raport(sesja: Path, znak: str, nadane: str, opis: str, ustawienia: dict,
                   stan_okna: dict | None = None) -> Path:
    """ZIP w stałym układzie (do przetworzenia automatem):
    raport.json — wersja, czas, język, znak, nadane, opis, ustawienia, nazwy plików;
    nagranie.flac — ostatnie RAPORT_MIN min sesji; sesja.txt/.log; dziennik.txt (koniec)."""
    import zipfile
    import soundfile as sf
    from tools import teksty
    stempel = time.strftime("%Y%m%d_%H%M%S")
    znak_pliku = "".join(c for c in znak.upper() if c.isalnum() or c == "/").replace("/", "-") or "ANON"
    katalog = KATALOG / "raporty"
    katalog.mkdir(parents=True, exist_ok=True)
    cel = katalog / f"raport_{stempel}_{znak_pliku}.zip"
    x, sr = sf.read(str(sesja), dtype="int16")
    x = x[-int(RAPORT_MIN * 60 * sr):]
    flac = katalog / f"_nagranie_{stempel}.flac"
    sf.write(str(flac), x, sr, format="FLAC")
    meta = {"format": "cw-raport-1", "projekt": "CW-DECODER", "wersja": _wersja_pelna(), "czas": time.strftime("%Y-%m-%d %H:%M:%S"),
            "jezyk": teksty.JEZYK, "znak": znak.upper(), "nadane": nadane, "opis": opis,
            "sesja": sesja.name, "sesja_s": round(sf.info(str(sesja)).duration, 1),
            "nagranie_s": round(len(x) / sr, 1), "ustawienia": ustawienia,
            "stan_okna": stan_okna or {}, "z_sesji": _z_sesji(sesja), "srodowisko": _srodowisko()}
    with zipfile.ZipFile(cel, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("raport.json", json.dumps(meta, ensure_ascii=False, indent=1))
        z.write(flac, "nagranie.flac")
        for roz in (".txt", ".log"):
            p = sesja.with_suffix(roz)
            if p.exists():
                z.write(p, "sesja" + roz)
        dz = KATALOG / "dziennik.txt"
        if dz.exists():
            z.writestr("dziennik.txt", "".join(dz.read_text(encoding="utf-8", errors="replace").splitlines(True)[-400:]))
    flac.unlink(missing_ok=True)
    return cel


def _mapi_wyslij(adres: str, temat: str, tresc: str, plik: Path) -> bool:
    """Nowa wiadomość w domyślnym programie pocztowym z załącznikiem (Simple MAPI:
    Outlook, Thunderbird...). False, gdy brak takiego programu (poczta w przeglądarce)."""
    import ctypes
    from ctypes import wintypes

    class Odbiorca(ctypes.Structure):
        _fields_ = [("ulReserved", wintypes.ULONG), ("ulRecipClass", wintypes.ULONG), ("lpszName", wintypes.LPWSTR),
                    ("lpszAddress", wintypes.LPWSTR), ("ulEIDSize", wintypes.ULONG), ("lpEntryID", ctypes.c_void_p)]

    class Plik(ctypes.Structure):
        _fields_ = [("ulReserved", wintypes.ULONG), ("flFlags", wintypes.ULONG), ("nPosition", wintypes.ULONG),
                    ("lpszPathName", wintypes.LPWSTR), ("lpszFileName", wintypes.LPWSTR), ("lpFileType", ctypes.c_void_p)]

    class Wiadomosc(ctypes.Structure):
        _fields_ = [("ulReserved", wintypes.ULONG), ("lpszSubject", wintypes.LPWSTR), ("lpszNoteText", wintypes.LPWSTR),
                    ("lpszMessageType", wintypes.LPWSTR), ("lpszDateReceived", wintypes.LPWSTR),
                    ("lpszConversationID", wintypes.LPWSTR), ("flFlags", wintypes.ULONG),
                    ("lpOriginator", ctypes.c_void_p), ("nRecipCount", wintypes.ULONG),
                    ("lpRecips", ctypes.POINTER(Odbiorca)), ("nFileCount", wintypes.ULONG),
                    ("lpFiles", ctypes.POINTER(Plik))]
    try:
        mapi = ctypes.WinDLL("mapi32.dll")
        f = mapi.MAPISendMailW
    except (OSError, AttributeError):
        return False
    od = Odbiorca(0, 1, adres, "SMTP:" + adres, 0, None)          # MAPI_TO
    pl = Plik(0, 0, 0xFFFFFFFF, str(plik), plik.name, None)
    w = Wiadomosc(0, temat, tresc, None, None, None, 0, None, 1 if adres else 0,
                  ctypes.pointer(od) if adres else None, 1, ctypes.pointer(pl))
    f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(Wiadomosc), wintypes.ULONG, wintypes.ULONG]
    wynik = f(None, None, ctypes.byref(w), 0x8 | 0x1, 0)            # MAPI_DIALOG | MAPI_LOGON_UI
    print(f"MAPISendMailW: {wynik}", flush=True)
    return wynik in (0, 1)                                         # 0 OK, 1 użytkownik anulował


def okno_wysylki(rodzic, plik: Path, temat: str, tresc: str) -> None:
    """Po utworzeniu raportu: adres i temat do skopiowania, plik do przeciągnięcia.
    Operator 24.10 (Firefox, poczta w przeglądarce): „mail się nie otwiera, powstaje
    tylko zip” — mailto nie przenosi załącznika, a bez programu pocztowego
    (Outlook, Thunderbird — Simple MAPI) Windows nie ma czego otworzyć. Ta droga
    działa z każdą pocztą: Gmail, Onet, WP w przeglądarce."""
    import subprocess
    import threading
    import tkinter as tk
    from tkinter import ttk
    from tools.teksty import T
    d = tk.Toplevel(rodzic)
    d.title(T("rap_tytul"))
    d.transient(rodzic)
    f = ttk.Frame(d, padding=12)
    f.pack(fill="both", expand=True)

    def pole(etykieta: str, wartosc: str, wiersz: int) -> None:
        ttk.Label(f, text=etykieta).grid(row=wiersz, column=0, sticky="w", pady=2)
        e = ttk.Entry(f, width=60)
        e.insert(0, wartosc)
        e.configure(state="readonly")
        e.grid(row=wiersz, column=1, sticky="we", padx=6)

        def kopiuj() -> None:
            d.clipboard_clear()
            d.clipboard_append(wartosc)
        ttk.Button(f, text=T("wys_kopiuj"), command=kopiuj).grid(row=wiersz, column=2)

    ttk.Label(f, text=T("wys_wstep") if KONTAKT_EMAIL else T("wys_brak_adresu"), justify="left").grid(
        row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
    if KONTAKT_EMAIL:
        pole(T("wys_adres"), KONTAKT_EMAIL, 1)
        pole(T("wys_temat"), temat, 2)
    pole(T("wys_plik"), str(plik), 3)
    p = ttk.Frame(f)
    p.grid(row=4, column=0, columnspan=3, sticky="e", pady=(10, 0))
    ttk.Button(p, text=T("wys_pokaz"), command=lambda: subprocess.Popen(["explorer", "/select,", str(plik)])).pack(
        side="left", padx=4)
    if KONTAKT_EMAIL:
        def program_pocztowy() -> None:
            # jeden raport = jedna wiadomość: operator 25.10 klikał kilka razy i program
            # pocztowy otwierał tyle samo okien z tą samą treścią
            b_poczta.configure(state="disabled")

            def w_tle() -> None:
                if not _mapi_wyslij(KONTAKT_EMAIL, temat, tresc, plik):
                    import urllib.parse
                    import webbrowser
                    webbrowser.open(f"mailto:{KONTAKT_EMAIL}?subject={urllib.parse.quote(temat)}"
                                    f"&body={urllib.parse.quote(tresc)}")
            threading.Thread(target=w_tle, daemon=True).start()
        b_poczta = ttk.Button(p, text=T("wys_program"), command=program_pocztowy)
        b_poczta.pack(side="left", padx=4)
    ttk.Button(p, text=T("wys_zamknij"), command=d.destroy).pack(side="left", padx=4)


def okno_raportu(sesja: str | None, rodzic, ustawienia: dict, stan_okna: dict | None = None) -> None:
    import threading
    import tkinter as tk
    from tkinter import messagebox, ttk
    from tools.teksty import T
    if not sesja:
        messagebox.showinfo(T("rap_tytul"), T("rap_brak_sesji"), parent=rodzic)
        return
    ust = ust_czytaj()
    d = tk.Toplevel(rodzic)
    d.title(T("rap_tytul"))
    d.transient(rodzic)
    f = ttk.Frame(d, padding=12)
    f.pack(fill="both", expand=True)
    ttk.Label(f, text=T("rap_wstep", m=RAPORT_MIN), justify="left").pack(anchor="w")
    ttk.Label(f, text=T("rap_znak")).pack(anchor="w", pady=(10, 0))
    v_znak = tk.StringVar(value=ust.get("znak", ""))
    ttk.Entry(f, textvariable=v_znak, width=20).pack(anchor="w")
    ttk.Label(f, text=T("rap_nadane")).pack(anchor="w", pady=(8, 0))
    v_nad = tk.StringVar()
    ttk.Entry(f, textvariable=v_nad, width=80).pack(anchor="w", fill="x")
    ttk.Label(f, text=T("rap_opis")).pack(anchor="w", pady=(8, 0))
    opis = tk.Text(f, width=80, height=6)
    opis.pack(fill="both", expand=True)

    def utworz() -> None:
        znak, nad, op = v_znak.get().strip(), v_nad.get().strip(), opis.get("1.0", "end").strip()
        ust_zapisz({"znak": znak})
        d.destroy()
        try:
            plik = _zbuduj_raport(Path(sesja), znak, nad, op, ustawienia, stan_okna)
        except Exception as e:
            print(f"raport: {e!r}", flush=True)
            messagebox.showerror(T("rap_tytul"), repr(e), parent=rodzic)
            return
        print(f"raport: {plik}", flush=True)
        temat = T("rap_temat", znak=znak.upper() or "-", wersja=WERSJA_APLIKACJI)
        tresc = T("rap_tresc", plik=plik.name, opis=op)

        okno_wysylki(rodzic, plik, temat, tresc)

    ttk.Button(f, text=T("rap_utworz"), command=utworz).pack(anchor="e", pady=(10, 0))


# -------------------------------------------------------------------- start
def _dziennik() -> None:
    """exe bez konsoli: wyjście i błędy do Dokumenty\\DekoderCW\\dziennik.txt.
    24.10 operator: „uruchomiłem exe na IC-7300, nie działa” — i żadnego
    śladu, bo okno bez konsoli gubiło komunikaty. Awaria ma sama mówić."""
    import threading
    import traceback
    KATALOG.mkdir(parents=True, exist_ok=True)
    f = open(KATALOG / "dziennik.txt", "a", encoding="utf-8", buffering=1)
    f.write(f"\n===== start {time.strftime('%Y-%m-%d %H:%M:%S')} {_wersja_pelna()} argv={sys.argv[1:]}\n")
    sys.stdout = sys.stderr = f

    def hak(typ, wart, tb):
        f.write("BŁĄD:\n" + "".join(traceback.format_exception(typ, wart, tb)))
    sys.excepthook = hak
    threading.excepthook = lambda a: hak(a.exc_type, a.exc_value, a.exc_traceback)


def main() -> None:
    import multiprocessing as mp
    mp.freeze_support()                      # exe: procesy dekodera i analizy (spawn) uruchamiają ten plik
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    global KATALOG
    proba, sekundy = None, None
    if "--proba" in sys.argv:              # test exe: nagranie zamiast karty, bez okna startowego
        i = sys.argv.index("--proba")
        proba, sekundy = Path(sys.argv[i + 1]), float(sys.argv[i + 2])
        KATALOG = Path(sys.argv[i + 3])
    dziennik = getattr(sys, "frozen", False) or "--dziennik" in sys.argv
    if dziennik:
        _dziennik()
    from tools import teksty
    if proba is not None:
        teksty.JEZYK = ust_czytaj().get("jezyk") or _jezyk_systemu()
        u = dict(karta=0, karta_nazwa=f"PRÓBA {proba.name}", civ=False, port="", baud=0, ton=760.0, zapis=True,
                 aktualizacje=False)
    elif "--start" in sys.argv:            # test: ostatnie ustawienia bez okna startowego
        ust = ust_czytaj()
        teksty.JEZYK = ust.get("jezyk") or _jezyk_systemu()
        k = next(k for k in _karty() if k[1] == ust["karta"])
        u = dict(karta=k[0], karta_nazwa=k[1], civ=ust["civ"], port=ust["port"], baud=ust["baud"],
                 ton=ust.get("ton"), zapis=ust["zapis"], aktualizacje=False)
        if "--sekundy" in sys.argv:
            sekundy = float(sys.argv[sys.argv.index("--sekundy") + 1])
    else:
        u = okno_startowe()
    print(f"ustawienia: {u}", flush=True)
    if not u:
        return
    KATALOG.mkdir(parents=True, exist_ok=True)
    slow = KATALOG / "slownik_osobisty.txt"
    if not slow.exists():
        slow.write_text(WZOR_SLOWNIKA, encoding="utf-8")
    from dsp import slownik as SL
    SL.OSOBISTY = slow
    SL.SLOWA = SL._slowa()
    from tools import na_zywo as NZ
    NZ.SESJE = KATALOG / "sesje"
    NZ.WERSJA = _wersja_pelna()
    if sekundy is not None:
        import tkinter as tk
        stara = tk.Tk.mainloop

        def mainloop(self, n=0):
            self.after(int(sekundy * 1000), lambda: self.tk.call(self.protocol("WM_DELETE_WINDOW")))
            stara(self, n)
        tk.Tk.mainloop = mainloop
    ust = ust_czytaj()
    opis_ust = {k: u[k] for k in ("karta_nazwa", "civ", "port", "baud", "ton", "zapis")}
    # dekoder liczy w osobnym procesie (NZ._praca_okna) — dostaje proste wartości
    NZ.okno(u["karta"], u["civ"], zapis=u["zapis"], model_npz=str(_zasob(WAGI)),
            civ_port=u["port"] or None, civ_baud=u["baud"] or None, ton_reczny=u["ton"],
            tytul=f"{NAZWA} {WERSJA_APLIKACJI} — {u['karta_nazwa']}", wav=str(proba) if proba else None,
            dziennik=str(KATALOG / "dziennik.txt") if dziennik else None,
            geometria=ust.get("geometria"), prog_start=ust.get("prog"),
            na_zamkniecie=lambda g, p: ust_zapisz({"geometria": g, "prog": p}),
            raport_fn=lambda s, root, st: okno_raportu(s, root, opis_ust, st),
            sprawdz_wersje=sprawdz_wersje if u.get("aktualizacje") else None, ikona=_ikona())


if __name__ == "__main__":
    main()
