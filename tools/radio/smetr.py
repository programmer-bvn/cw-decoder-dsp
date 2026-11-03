"""S-metr Icoma przez CI-V — TYLKO odczyt — i przeliczenie na dBm i S.

Polecenia (wszystkie bez danych = odczyt; z danymi byłyby ustawieniem, więc
nigdy z danymi): 15 02 S-metr 0-255, 16 02 przedwzmacniacz (00/01/02), 11 tłumik
(00 albo 20 = 20 dB, BCD), 03 częstotliwość. DTR i RTS wyłączone PRZED otwarciem
portu — jak w civ_odczyt.ps1 (w Icomach mogą być PTT albo kluczowaniem).

Kalibracja 03.11.2026 generatorem National VP-8177A (14,0147 MHz, operator, dane:
kalibracja_smetra_ic7300mk2.csv): bez P.AMP od S9 w górę zgodnie z normą (±2 dB),
poniżej S9 radio zaniża (−85 dBm = S5 na ekranie); AMP1 podnosi wskazanie o ~7 dB,
AMP2 o ~12 dB — stąd osobna tabela dla każdego ustawienia. Tabele są z JEDNEGO
egzemplarza IC-7300MK2 (adres CI-V 0xB6); inne Icomy: wzór z opisu CI-V
(0 = S0, 120 = S9, 241 = S9+60), oznaczany „≈”.

    python -m tools.radio.smetr [COM5]      # 10 odczytów: CI-V, P.AMP, ATT, dBm, S
"""

from __future__ import annotations

import threading
import time

import numpy as np

# (CI-V 15 02, dBm na wejściu) — punkty stabilne z kalibracji 03.11. Pominięte:
# −121 i −109 dBm (zawyżone szumem toru generatora, bez kabla CI-V = 0) i −33 dBm
# przy AMP1 (przyrost 0,7 na dB zamiast ~2 — podejrzany tłumik generatora).
KALIBRACJA_IC7300MK2: dict[int, list[tuple[float, float]]] = {
    1: [(48.4, -97), (68.7, -91), (93.1, -85), (121.1, -78), (132.8, -73), (144.7, -67),
        (161.1, -59), (173.6, -53), (193.4, -43), (221.2, -23), (241.0, -12)],
    0: [(54.4, -85), (111.5, -73), (145.0, -59), (177.7, -43)],
    2: [(116.0, -85), (143.0, -73), (171.0, -59), (203.3, -43)],
}
ADRES_IC7300MK2 = 0xB6
DB_NA_JEDNOSTKE_GORA = 60 / 121     # powyżej tabeli: nachylenie z opisu CI-V (zmierzone ~0,48)


def _interp(tab: list[tuple[float, float]], v: float, ksztalt: list[tuple[float, float]]) -> tuple[float, str]:
    """Liniowo w tabeli; w dół kształtem pełnej serii (AMP1) dosuniętym do pierwszego
    punktu, w górę nachyleniem z opisu CI-V. Poniżej najniższego zmierzonego punktu
    serii — „pod”: wiadomo tylko, że słabiej (03.11 niżej był już szum generatora)."""
    c = np.array([p[0] for p in tab])
    d = np.array([p[1] for p in tab])
    if v >= c[-1]:
        return float(d[-1] + (v - c[-1]) * DB_NA_JEDNOSTKE_GORA), "kal"
    if v >= c[0]:
        return float(np.interp(v, c, d)), "kal"
    ck = np.array([p[0] for p in ksztalt])
    dk = np.array([p[1] for p in ksztalt])
    przes = d[0] - float(np.interp(c[0], ck, dk))
    if v >= ck[0]:
        return float(np.interp(v, ck, dk) + przes), "kal"
    return float(dk[0] + przes), "pod"


def dbm(civ: float, pamp: int | None, att_db: int = 0, adres: int = ADRES_IC7300MK2) -> tuple[float, str]:
    """-> (dBm na gnieździe antenowym, „kal” | „pod” (≤ tej wartości) | „icom” (wzór
    z opisu CI-V, bez kalibracji)). ATT po prostu dodany: tłumik nie był kalibrowany
    (03.11 wyłączony) — hipoteza, że tłumi o nominał."""
    if adres == ADRES_IC7300MK2 and pamp in KALIBRACJA_IC7300MK2:
        p, jak = _interp(KALIBRACJA_IC7300MK2[pamp], civ, KALIBRACJA_IC7300MK2[1])
        return p + att_db, jak
    s = civ / 120 * 9 if civ <= 120 else 9 + (civ - 120) / 121 * 60 / 6
    return -73 + (s - 9) * 6 + att_db, "icom"


def s_z_dbm(p_dbm: float, mhz: float | None = None) -> str:
    """Norma IARU: S9 = −73 dBm (50 µV) do 30 MHz, −93 dBm powyżej; 6 dB na jednostkę."""
    s9 = -93.0 if mhz and mhz > 30 else -73.0
    if p_dbm <= s9 + 0.5:
        return f"S{max(0, min(9, int(round(9 + (p_dbm - s9) / 6))))}"
    return f"S9+{int(round(p_dbm - s9))}"


def opis(civ: float, pamp: int | None, att_db: int = 0, adres: int = ADRES_IC7300MK2,
         mhz: float | None = None) -> str:
    p, jak = dbm(civ, pamp, att_db, adres)
    z = {"kal": "", "pod": "≤", "icom": "≈"}[jak]
    return f"{z}{s_z_dbm(p, mhz)} ({z}{p:.0f} dBm)"


def _bcd(b: bytes) -> int:
    v = 0
    for x in b:
        v = v * 100 + (x >> 4) * 10 + (x & 15)
    return v


class SMetr:
    """Odczyt w wątku co `odstep` s. Ostatni wynik w .stan (dict albo None).
    Błąd portu nie zatrzymuje odbioru: próba ponownie po 5 s, opis w .blad."""

    def __init__(self, port: str = "COM5", baud: int = 115200, odstep: float = 0.5, opoznienie: float = 0.0):
        self.port, self.baud, self.odstep, self.opoznienie = port, baud, odstep, opoznienie
        self.stan: dict | None = None
        self.blad: str | None = None
        self._stop = threading.Event()
        self._w = threading.Thread(target=self._praca, daemon=True)

    def start(self) -> "SMetr":
        self._w.start()
        return self

    def stop(self) -> None:
        """Czeka na zamknięcie portu: zaraz po tym ten sam port otwiera start CI-V dekodera."""
        self._stop.set()
        if self._w.is_alive():
            self._w.join(timeout=1.5)

    def _otworz(self):
        import serial
        sp = serial.Serial()
        sp.port, sp.baudrate, sp.timeout = self.port, self.baud, 0.15
        sp.dtr = False                       # stan ustawiony przed open() — port otwiera się bez DTR/RTS
        sp.rts = False
        sp.open()
        return sp

    @staticmethod
    def _zapytaj(sp, adres: int, cmd: bytes) -> bytes | None:
        """Ramka odpowiedzi radia na `cmd` (bez echa zapytania), dane po poleceniu."""
        sp.reset_input_buffer()
        sp.write(bytes([0xFE, 0xFE, adres, 0xE0]) + cmd + b"\xFD")
        buf, t = b"", time.time()
        while time.time() - t < 0.2:
            buf += sp.read(64)
            i = buf.find(bytes([0xFE, 0xFE, 0xE0, adres]) + cmd)
            if i >= 0:
                k = buf.find(b"\xFD", i)
                if k > 0:
                    return buf[i + 4 + len(cmd):k]
        return None

    def _praca(self) -> None:
        if self._stop.wait(self.opoznienie):     # start CI-V (ton, pasmo) idzie przez ten sam port
            return
        sp, adres, n = None, None, 0
        while not self._stop.is_set():
            try:
                if sp is None:
                    sp = self._otworz()
                    for a in (ADRES_IC7300MK2, 0x94, 0x00):
                        if self._zapytaj(sp, a, b"\x15\x02") is not None:
                            adres = a
                            break
                    else:
                        raise OSError("radio nie odpowiada na 15 02")
                v = self._zapytaj(sp, adres, b"\x15\x02")
                if v is not None and len(v) == 2:
                    if n % 10 == 0:                  # ustawienia i pasmo zmieniają się rzadko
                        pa = self._zapytaj(sp, adres, b"\x16\x02")
                        at = self._zapytaj(sp, adres, b"\x11")
                        fr = self._zapytaj(sp, adres, b"\x03")
                        self._pamp = pa[0] if pa and len(pa) == 1 else None
                        self._att = _bcd(at) if at and len(at) == 1 else 0
                        self._mhz = (sum(((b & 15) + (b >> 4) * 10) * 100 ** i for i, b in enumerate(fr)) / 1e6
                                     if fr and len(fr) == 5 else None)
                    civ = _bcd(v)
                    p, jak = dbm(civ, self._pamp, self._att, adres)
                    self.stan = {"civ": civ, "pamp": self._pamp, "att_db": self._att, "mhz": self._mhz,
                                 "dbm": round(p, 1), "skad": jak,
                                 "s": opis(civ, self._pamp, self._att, adres, self._mhz)}
                    self.blad = None
                    n += 1
            except Exception as e:                   # CI-V nie może zatrzymać odbioru
                self.blad, self.stan = repr(e), None
                if sp is not None:
                    try:
                        sp.close()
                    except Exception:
                        pass
                sp, n = None, 0
                if self._stop.wait(5.0):
                    break
                continue
            self._stop.wait(self.odstep)
        if sp is not None:
            sp.close()


def main() -> None:
    import sys
    m = SMetr(sys.argv[1] if len(sys.argv) > 1 else "COM5").start()
    for _ in range(10):
        time.sleep(0.6)
        print(m.stan or m.blad)
    m.stop()


if __name__ == "__main__":
    main()
