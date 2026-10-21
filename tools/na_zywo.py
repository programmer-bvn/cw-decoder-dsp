"""ODCZYT NA ŻYWO (21.10): dźwięk kawałkami -> tekst z opóźnieniem kilku sekund.

    python -m tools.na_zywo symuluj nagranie.wav [...]     # nagranie podawane kawałkami
    python -m tools.na_zywo karta --device 15 [--civ]      # karta dźwiękowa (IC-7300 USB)

Z kartą graficzną (karta dźwiękowa i COM są w Windows, WSL2 ich nie widzi):
    WSL:      python -m tools.na_zywo serwer               # dekoder, GPU, tekst
    Windows:  python -m tools.na_zywo most --device 15 --civ
    (oba naraz: start_na_zywo_gpu.bat; próba bez radia: most --wav nagranie.wav)

Zasada: bufor ostatnich OKNO_S sekund; co KROK_S cała sprawdzona droga
(tools.odczyt.przygotuj: pętla tonu, przeciwsygnał, przestrojenie; sieć
zboczy; jednostka; Viterbi) liczona na buforze od nowa. Znak jest
ZATWIERDZANY, gdy skończył się co najmniej OPOZNIENIE_S przed końcem
bufora — ma wtedy kontekst po obu stronach, a później nie zmienia się.
Nie przepisujemy żadnego kroku na wersję strumieniową: każdy był mierzony
na całych nagraniach (HISTORIA), a okno 20 s ma tę samą skalę co pętla tonu
(6 s), przeciwsygnał (10 s) i cepstrum (8 s).
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsp import config as C, frontend, viterbi as V, zbocza as Z  # noqa: E402
from dsp.grzebien import _ODWROTNY, NIEZNANY  # noqa: E402

OKNO_S = 20.0
KROK_S = 2.5         # krok pełnego bufora z jedną obcą do skasowania: ~2,0 s (21.10) — zapas
OPOZNIENIE_S = 4.0
# Koszt kroku na buforze 20 s (21.10, zmierzone): przygotuj 0,63 s, sieć
# zboczy 0,35 s, Viterbi 0,6 s na jednostkę, cepstrum ~0 s, GRZEBIEŃ 1,55 s.
# Cepstrum co JEDNOSTKA_CO kroków, grzebień co GRZEBIEN_CO; między nimi
# poprzednia jednostka — tempo stacji zmienia się wolniej niż co 10 s.
JEDNOSTKA_CO = 5
GRZEBIEN_CO = 15
# Viterbi na ostatnich VITERBI_S z bufora: zatwierdzane znaki leżą 4-6 s
# przed końcem, kontekst po obu stronach zostaje; koszt o połowę mniejszy.
VITERBI_S = 12.0


def znaki_z_odcinkow(odc: list, t0: float) -> list[tuple[str, float, float]]:
    """Odcinki Viterbiego -> [(znak, początek s, koniec s)] w czasie bezwzględnym.
    Spacja dostaje czas przerwy między słowami."""
    out, kod, a0, b0 = [], "", None, None
    for nazwa, a, b in odc:
        if nazwa in ("K1", "K3"):
            if not kod:
                a0 = a
            kod += "." if nazwa == "K1" else "-"
            b0 = b
        elif nazwa in ("PL", "PS", "CI"):
            if kod:
                out.append((_ODWROTNY.get(kod, NIEZNANY), t0 + a0 * V.DT, t0 + b0 * V.DT))
                kod = ""
            if nazwa in ("PS", "CI") and out and out[-1][0] != " ":
                out.append((" ", t0 + a * V.DT, t0 + b * V.DT))
    if kod:
        out.append((_ODWROTNY.get(kod, NIEZNANY), t0 + a0 * V.DT, t0 + b0 * V.DT))
    return out


class OdczytNaZywo:
    def __init__(self, model, ton_cw: float | None = None, okno_s: float = OKNO_S,
                 krok_s: float = KROK_S, opoznienie_s: float = OPOZNIENIE_S, sr: int = C.SR):
        self.model, self.ton_cw, self.sr = model, ton_cw, sr
        self.okno, self.krok, self.opozn = okno_s, krok_s, opoznienie_s
        self.buf = np.zeros(0, np.float32)
        self.t0 = 0.0                      # czas bezwzględny buf[0]
        self.od_kroku = 0                  # próbek od ostatniego kroku
        self.ost = -np.inf                 # początek ostatniego zatwierdzonego znaku
        self.u = float("nan")
        self.n_krokow = 0
        self.czasy: list[float] = []       # czas liczenia kroków [s]
        self.zatwierdzone: list[tuple[str, float, float, float]] = []   # znak, od, do, chwila zatwierdzenia

    def dodaj(self, x: np.ndarray) -> str:
        """Kawałek dźwięku (SR) -> nowo zatwierdzony tekst (może być pusty)."""
        self.buf = np.concatenate([self.buf, np.asarray(x, np.float32)])
        self.od_kroku += len(x)
        if self.od_kroku < self.krok * self.sr:
            return ""
        self.od_kroku = 0
        return self._krok(koniec=False)

    def koniec(self) -> str:
        """Koniec dźwięku: zatwierdź resztę."""
        return self._krok(koniec=True) if self.buf.size else ""

    def _krok(self, koniec: bool) -> str:
        from tools.odczyt import jednostka_grzebienia, przygotuj
        t_start = time.perf_counter()
        koniec_buf = self.t0 + self.buf.size / self.sr
        nowe = ""
        if self.buf.size >= int(2.0 * self.sr):
            a, _ = przygotuj(self.buf, ton_cw=self.ton_cw)
            pz = Z.przebieg_sieci(a, self.model)
            self.n_krokow += 1
            if not np.isfinite(self.u) or self.n_krokow % GRZEBIEN_CO == 0:
                kand = [V.jednostka_cepstrum(pz)[0], jednostka_grzebienia(pz), self.u]
            elif self.n_krokow % JEDNOSTKA_CO == 0:
                kand = [V.jednostka_cepstrum(pz)[0], self.u]
            else:
                kand = [self.u]
            v0 = max(0, pz.size - int(VITERBI_S / V.DT))
            pv = pz[v0:]
            kand = sorted({round(float(k), 4) for k in kand if np.isfinite(k) and k > 0})
            # jak V.dekoduj_najlepiej (wygrywa najlepszy wynik na ramkę), ale
            # odcinki zwycięzcy od razu — bez drugiego przebiegu Viterbiego
            naj = None
            for k in kand:
                _, odc_k, wynik = V.dekoduj(pv, k, z_wynikiem=True)
                if naj is None or wynik > naj[2]:
                    naj = (k, odc_k, wynik)
            if naj is not None:
                u, odc, _ = naj
                self.u = u
                granica = koniec_buf if koniec else koniec_buf - self.opozn
                for z, a0, b0 in znaki_z_odcinkow(odc, self.t0 + v0 * V.DT):
                    # nowy znak: zaczyna się po ostatnim zatwierdzonym (z zapasem
                    # pół jednostki na drgania położenia między krokami)
                    if a0 > self.ost + 0.5 * u and b0 <= granica:
                        if z == " " and (not self.zatwierdzone or self.zatwierdzone[-1][0] == " "):
                            continue
                        self.zatwierdzone.append((z, a0, b0, koniec_buf))
                        self.ost = a0
                        nowe += z
        # bufor: ostatnie OKNO_S
        nadmiar = self.buf.size - int(self.okno * self.sr)
        if nadmiar > 0:
            self.buf = self.buf[nadmiar:]
            self.t0 += nadmiar / self.sr
        self.czasy.append(time.perf_counter() - t_start)
        return nowe

    def tekst(self) -> str:
        return " ".join("".join(z[0] for z in self.zatwierdzone).split())


def symuluj(pliki: list[Path], kawalek_s: float = 0.1) -> None:
    from tensorflow import keras
    from tools.klasyczny import cer
    from tools.odczyt import MODEL_ZBOCZY, odczytaj
    from tools.radio import civ
    model = keras.models.load_model(MODEL_ZBOCZY)
    for p in pliki:
        a = frontend.load_audio(p)
        ton = civ.ton_z_opisu(p.with_suffix(".txt"))
        d = OdczytNaZywo(model, ton_cw=ton)
        n = int(kawalek_s * C.SR)
        t = time.perf_counter()
        for i in range(0, a.size, n):
            d.dodaj(a[i:i + n])
        d.koniec()
        calosc = time.perf_counter() - t
        zywo = d.tekst()
        cal = odczytaj(p, model, ton)["tekst"]
        opis = p.with_suffix(".txt")
        m = re.search(r"^nadane:\s*(.*)$", opis.read_text(encoding="utf-8"), re.M) if opis.exists() else None
        nad = m.group(1).replace(" ", "") if m and "nieznane" not in m.group(1) else None
        opozn = [z[3] - z[2] for z in d.zatwierdzone if z[0] != " "]
        print(f"\n{p.name}  ({a.size / C.SR:.0f} s, ton CW {ton})")
        print(f"  na żywo : {zywo[:160]}")
        print(f"  całość  : {cal[:160]}")
        wiersz = f"  różnica na żywo / całość: {cer(zywo.replace(' ', ''), cal.replace(' ', ''))} znaków"
        if nad:
            wiersz += (f"; błędy wobec nadanego: na żywo {cer(zywo.replace(' ', ''), nad)}, "
                       f"całość {cer(cal.replace(' ', ''), nad)} / {len(nad)}")
        print(wiersz)
        print(f"  opóźnienie znaku (koniec znaku -> zatwierdzenie): mediana {np.median(opozn):.1f} s, "
              f"maks. {np.max(opozn):.1f} s; krok liczony średnio {np.mean(d.czasy):.2f} s, "
              f"maks. {np.max(d.czasy):.2f} s (krok co {KROK_S:.1f} s); całość {calosc:.0f} s "
              f"na {a.size / C.SR:.0f} s dźwięku", flush=True)


def karta(device: int, z_civ: bool) -> None:
    """Na żywo z karty dźwiękowej (np. IC-7300 USB audio). Tylko odbiór."""
    from tensorflow import keras
    from tools.odczyt import MODEL_ZBOCZY
    ton = _ton_z_radia() if z_civ else None
    print(f"ton CW: {ton} Hz; Ctrl+C kończy", flush=True)
    d = OdczytNaZywo(keras.models.load_model(MODEL_ZBOCZY), ton_cw=ton)
    try:
        for x in _strumien_karty(device):
            nowe = d.dodaj(x)
            if nowe:
                print(nowe, end="", flush=True)
    except KeyboardInterrupt:
        print(d.koniec())


def _ton_z_radia() -> float | None:
    from tools.radio import civ
    return civ.stan()["ton_cw"]


class _Decymator:
    """Karta (np. 48 kHz) -> C.SR kawałek po kawałku, ze STANEM filtra.
    Pierwsza wersja (21.10) przepróbkowywała każdy kawałek 0,1 s osobno
    (resample_poly) — filtr startował od zera co 100 ms, na stykach
    stuki. Tu FIR antyaliasingowy z pamięcią (lfilter zi) i co m-ta próbka
    w ciągłej fazie."""

    def __init__(self, sr_dev: int):
        from scipy.signal import firwin
        self.sr_dev = sr_dev
        self.m = sr_dev // C.SR if sr_dev % C.SR == 0 else 0
        if self.m:
            self.b = firwin(255, 0.9 * (C.SR / 2) / (sr_dev / 2))
            self.zi = np.zeros(self.b.size - 1)
            self.g = 0                                   # próbek karty dotąd

    def __call__(self, x: np.ndarray) -> np.ndarray:
        from scipy.signal import lfilter, resample_poly
        if not self.m:                                   # np. 44,1 kHz: bez ciągłości
            return resample_poly(x, C.SR, self.sr_dev).astype(np.float32)
        y, self.zi = lfilter(self.b, [1.0], x, zi=self.zi)
        out = y[(-self.g) % self.m::self.m]
        self.g += x.size
        return out.astype(np.float32)


def _strumien_karty(device: int):
    """Kawałki po ~0,1 s w C.SR z karty dźwiękowej (tylko odbiór)."""
    import queue
    import sounddevice as sd
    sr_dev = int(sd.query_devices(device, "input")["default_samplerate"])
    if sr_dev % C.SR:
        print(f"UWAGA: karta {sr_dev} Hz nie jest wielokrotnością {C.SR} Hz — "
              "przepróbkowanie kawałkami, bez ciągłości filtra", flush=True)
    dec = _Decymator(sr_dev)
    q: queue.Queue = queue.Queue()
    with sd.InputStream(device=device, channels=1, samplerate=sr_dev, blocksize=sr_dev // 10,
                        callback=lambda x, *_: q.put(x[:, 0].astype(np.float64)), latency="high"):
        while True:
            yield dec(q.get())


# --------------------------------------------------------------------------
# Most Windows -> WSL (22.10): karta i CI-V są w Windows (WSL2 ich nie widzi),
# karta graficzna — w WSL. Windows wysyła próbki przez localhost, WSL liczy.
# Protokół: wiersz JSON {"ton_cw": Hz|null, "sr": 8000}, potem float32 LE.
# --------------------------------------------------------------------------
PORT = 7355


def serwer(port: int) -> None:
    """WSL: czeka na most z Windows, dekoduje (karta graficzna), wypisuje tekst."""
    import json
    import socket
    from tensorflow import keras
    from tools.odczyt import MODEL_ZBOCZY
    model = keras.models.load_model(MODEL_ZBOCZY)
    with socket.create_server(("0.0.0.0", port)) as s:
        while True:
            print(f"czekam na most z Windows (port {port})...", flush=True)
            c, adr = s.accept()
            with c:
                f = c.makefile("rb")
                naglowek = json.loads(f.readline().decode())
                ton = naglowek.get("ton_cw")
                print(f"połączono {adr[0]}; ton CW: {ton} Hz", flush=True)
                d = OdczytNaZywo(model, ton_cw=ton)
                reszta = b""
                while True:
                    b = c.recv(65536)
                    if not b:
                        break
                    b = reszta + b
                    n = len(b) // 4 * 4
                    reszta = b[n:]
                    nowe = d.dodaj(np.frombuffer(b[:n], "<f4"))
                    if nowe:
                        print(nowe, end="", flush=True)
                print(d.koniec(), flush=True)
                print("\n--- most rozłączony ---", flush=True)


def _strumien_wav(sciezka: Path):
    """Nagranie zamiast karty, w tempie rzeczywistym (próba mostu bez radia)."""
    a = frontend.load_audio(sciezka)
    n = C.SR // 10
    for i in range(0, a.size, n):
        time.sleep(0.1)
        yield a[i:i + n]


def most(device: int, z_civ: bool, host: str, port: int, wav: Path | None = None) -> None:
    """Windows: karta + CI-V -> próbki do serwera w WSL. Tylko odbiór."""
    import json
    import socket
    ton = _ton_z_radia() if z_civ else None
    if wav and ton is None:
        from tools.radio import civ
        ton = civ.ton_z_opisu(Path(wav).with_suffix(".txt"))
    print(f"ton CW: {ton} Hz; łączę z {host}:{port} (dekoder w WSL)...", flush=True)
    t0 = time.time()
    while True:
        try:
            c = socket.create_connection((host, port), timeout=5)
            break
        except OSError:
            if time.time() - t0 > 120:
                raise SystemExit("Serwer w WSL nie odpowiada od 2 min — czy okno dekodera wystartowało?")
            time.sleep(2)
    c.settimeout(None)
    print("połączono; tekst w oknie dekodera. Ctrl+C kończy.", flush=True)
    with c:
        c.sendall((json.dumps({"ton_cw": ton, "sr": C.SR}) + "\n").encode())
        try:
            for x in (_strumien_wav(wav) if wav else _strumien_karty(device)):
                c.sendall(np.asarray(x, "<f4").tobytes())
        except KeyboardInterrupt:
            pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("symuluj"); s.add_argument("pliki", type=Path, nargs="+")
    k = sub.add_parser("karta"); k.add_argument("--device", type=int, default=15)
    k.add_argument("--civ", action="store_true")
    sv = sub.add_parser("serwer"); sv.add_argument("--port", type=int, default=PORT)
    m = sub.add_parser("most"); m.add_argument("--device", type=int, default=15)
    m.add_argument("--civ", action="store_true")
    m.add_argument("--host", default="127.0.0.1"); m.add_argument("--port", type=int, default=PORT)
    m.add_argument("--wav", type=Path, default=None, help="nagranie zamiast karty (próba bez radia)")
    a = ap.parse_args()
    if a.cmd == "symuluj":
        symuluj(a.pliki)
    elif a.cmd == "karta":
        karta(a.device, a.civ)
    elif a.cmd == "serwer":
        serwer(a.port)
    else:
        most(a.device, a.civ, a.host, a.port, a.wav)


if __name__ == "__main__":
    main()
