"""DEKODER CW — aplikacja do rozdania (24.10, operator: „exe dla SP2MSA").

To samo okienko co `tools.na_zywo okno`, ale:
- bez TensorFlow: sieć zboczy w numpy (dsp/siec_numpy.py, wagi siec.npz),
- okno startowe: karta dźwiękowa, opcjonalnie CI-V (Icom) i port COM albo
  ton CW wpisany ręcznie — u innego operatora inne radio i inne numery,
- sesje i słownik osobisty w Dokumentach\\DekoderCW, ustawienia w %APPDATA%.

Budowa exe: budowa_exe.bat (PyInstaller, katalog dist\\DekoderCW).
Tylko odbiór: CI-V wyłącznie odczytuje (częstotliwość, tryb, ton CW).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

KATALOG = Path(os.environ.get("USERPROFILE", Path.home())) / "Documents" / "DekoderCW"
USTAWIENIA = Path(os.environ.get("APPDATA", Path.home())) / "DekoderCW" / "ustawienia.json"
WAGI = "siec.npz"
WZOR_SLOWNIKA = """# Słownik osobisty — Dekoder CW. Twoje słowa, które mają świecić na zielono
# i mieć poprawione spacje: imię, QTH, imiona i QTH znajomych. Jedno lub kilka
# słów w wierszu, wielkimi lub małymi literami; po „#" komentarz.
# Znaków wywoławczych i raportów RST nie trzeba wpisywać — rozpoznaje je wzór.
# Zmiany działają od następnego uruchomienia.
"""


def _zasob(nazwa: str) -> Path:
    """Plik dołączony do exe (PyInstaller: sys._MEIPASS) albo z repozytorium."""
    baza = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    for p in (baza / nazwa, baza / "runs" / "zbocza_5" / nazwa):
        if p.exists():
            return p
    raise FileNotFoundError(nazwa)


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
    import sounddevice as sd
    api = [a["name"] for a in sd.query_hostapis()]
    kar = [(i, f"[{i}] {d['name']} — {api[d['hostapi']]}") for i, d in enumerate(sd.query_devices())
           if d["max_input_channels"] > 0]
    # WASAPI najpierw: ta sama karta jest widoczna kilka razy (MME, DirectSound, WASAPI, WDM-KS)
    return sorted(kar, key=lambda k: ("WASAPI" not in k[1], k[0]))


def okno_startowe() -> dict | None:
    import tkinter as tk
    from tkinter import ttk
    try:
        ust = json.loads(USTAWIENIA.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        ust = {}
    karty, porty = _karty(), _porty_com()
    wynik: dict = {}
    r = tk.Tk()
    r.title("Dekoder CW — ustawienia")
    r.resizable(False, False)
    f = ttk.Frame(r, padding=14)
    f.pack()
    ttk.Label(f, text="Karta dźwiękowa (wejście z radia):").grid(row=0, column=0, sticky="w")
    v_karta = tk.StringVar()
    cb = ttk.Combobox(f, textvariable=v_karta, values=[k[1] for k in karty], width=70, state="readonly")
    cb.grid(row=1, column=0, columnspan=3, sticky="we", pady=(2, 10))
    zapamietana = next((k[1] for k in karty if k[1] == ust.get("karta")), None)
    domyslna = next((k[1] for k in karty if "USB Audio" in k[1] and "WASAPI" in k[1]), karty[0][1] if karty else "")
    v_karta.set(zapamietana or domyslna)

    v_civ = tk.BooleanVar(value=ust.get("civ", False))
    ttk.Checkbutton(f, text="Ton CW z radia przez CI-V (Icom; tylko odczyt)", variable=v_civ).grid(
        row=2, column=0, sticky="w")
    v_port = tk.StringVar(value=ust.get("port", porty[0] if porty else "COM1"))
    ttk.Combobox(f, textvariable=v_port, values=porty, width=8).grid(row=2, column=1, sticky="w", padx=6)
    v_baud = tk.StringVar(value=str(ust.get("baud", 115200)))
    ttk.Combobox(f, textvariable=v_baud, values=["115200", "19200", "9600"], width=8).grid(row=2, column=2, sticky="w")
    ttk.Label(f, text="albo ton CW wpisany ręcznie [Hz] (puste = sam szuka):").grid(row=3, column=0, sticky="w", pady=(8, 0))
    v_ton = tk.StringVar(value=str(ust.get("ton", "")))
    ttk.Entry(f, textvariable=v_ton, width=8).grid(row=3, column=1, sticky="w", padx=6, pady=(8, 0))
    v_zapis = tk.BooleanVar(value=ust.get("zapis", True))
    ttk.Checkbutton(f, text=f"Zapisuj sesje (dźwięk + tekst) do {KATALOG / 'sesje'}", variable=v_zapis).grid(
        row=4, column=0, columnspan=3, sticky="w", pady=(10, 0))
    ttk.Label(f, text=f"Słownik osobisty: {KATALOG / 'slownik_osobisty.txt'}", foreground="#666").grid(
        row=5, column=0, columnspan=3, sticky="w", pady=(4, 0))

    def start() -> None:
        k = next((k for k in karty if k[1] == v_karta.get()), None)
        if k is None:
            return
        try:
            ton = float(v_ton.get().replace(",", ".")) if v_ton.get().strip() else None
        except ValueError:
            ton = None
        wynik.update(karta=k[0], karta_nazwa=k[1], civ=v_civ.get(), port=v_port.get(),
                     baud=int(v_baud.get() or 115200), ton=ton, zapis=v_zapis.get())
        try:
            USTAWIENIA.parent.mkdir(parents=True, exist_ok=True)
            USTAWIENIA.write_text(json.dumps({**wynik, "karta": k[1], "ton": v_ton.get()}, ensure_ascii=False),
                                  encoding="utf-8")
        except OSError:
            pass
        r.destroy()

    ttk.Button(f, text="Start", command=start).grid(row=6, column=2, sticky="e", pady=(14, 0))
    r.mainloop()
    return wynik or None


def _dziennik() -> None:
    """exe bez konsoli: wyjście i błędy do Dokumenty\\DekoderCW\\dziennik.txt.
    24.10 operator: „uruchomiłem exe na IC-7300, nie działa” — i żadnego
    śladu, bo okno bez konsoli gubiło komunikaty. Awaria ma sama mówić."""
    import threading
    import time
    import traceback
    KATALOG.mkdir(parents=True, exist_ok=True)
    f = open(KATALOG / "dziennik.txt", "a", encoding="utf-8", buffering=1)
    f.write(f"\n===== start {time.strftime('%Y-%m-%d %H:%M:%S')} argv={sys.argv[1:]}\n")
    sys.stdout = sys.stderr = f

    def hak(typ, wart, tb):
        f.write("BŁĄD:\n" + "".join(traceback.format_exception(typ, wart, tb)))
    sys.excepthook = hak
    threading.excepthook = lambda a: hak(a.exc_type, a.exc_value, a.exc_traceback)


def main() -> None:
    import multiprocessing as mp
    mp.freeze_support()                      # exe: proces analizy (spawn) uruchamia ten sam plik
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    global KATALOG
    proba, sekundy = None, None
    if "--proba" in sys.argv:              # test exe: nagranie zamiast karty, bez okna startowego
        i = sys.argv.index("--proba")
        proba, sekundy = Path(sys.argv[i + 1]), float(sys.argv[i + 2])
        KATALOG = Path(sys.argv[i + 3])
    if getattr(sys, "frozen", False) or "--dziennik" in sys.argv:
        _dziennik()
    if proba is not None:
        u = dict(karta=0, karta_nazwa=f"PRÓBA {proba.name}", civ=False, port="", baud=0, ton=760.0, zapis=True)
    elif "--start" in sys.argv:            # test: ostatnie ustawienia bez okna startowego
        ust = json.loads(USTAWIENIA.read_text(encoding="utf-8"))
        k = next(k for k in _karty() if k[1] == ust["karta"])
        u = dict(karta=k[0], karta_nazwa=k[1], civ=ust["civ"], port=ust["port"], baud=ust["baud"],
                 ton=float(ust["ton"]) if ust.get("ton") else None, zapis=ust["zapis"])
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
    try:
        NZ.WERSJA = _zasob("wersja.txt").read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        pass

    def model():
        from dsp.siec_numpy import SiecNumpy
        return SiecNumpy(_zasob(WAGI))

    def stan() -> dict:
        if u["civ"]:
            from tools.radio import civ
            st = civ.stan(u["port"], u["baud"])
            if st.get("ton_cw") or u["ton"] is None:
                return st
            return {**st, "ton_cw": u["ton"], "skad_ton": "wpisany ręcznie, CI-V bez tonu"}
        return {"ton_cw": u["ton"], "pasmo": None, "skad_ton": "wpisany ręcznie"}

    if proba is not None:
        NZ._strumien_karty = lambda device: NZ._strumien_wav(proba)
    if sekundy is not None:
        import tkinter as tk
        stara = tk.Tk.mainloop

        def mainloop(self, n=0):
            self.after(int(sekundy * 1000), lambda: self.tk.call(self.protocol("WM_DELETE_WINDOW")))
            stara(self, n)
        tk.Tk.mainloop = mainloop
    NZ.okno(u["karta"], True, zapis=u["zapis"], model_fn=model, stan_fn=stan,
            tytul=f"Dekoder CW — {u['karta_nazwa']}")


if __name__ == "__main__":
    main()
