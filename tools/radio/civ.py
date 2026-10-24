"""Stan IC-7300MK2 przez CI-V — TYLKO odczyt (civ_odczyt.ps1: polecenia 03, 04, 14 09).

    python -m tools.radio.civ          # pasmo, tryb, filtr, ton CW

Ton CW („CW pitch") to środek filtra dla tune.track_tone (20.10): operator
dostraja swoją stację tak, żeby słyszeć ją na tym tonie.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

KORZEN = Path(__file__).resolve().parents[2]
SKRYPT = KORZEN / "tools" / "radio" / "civ_odczyt.ps1"


def _bcd(bajty: list[int]) -> int:
    """BCD od najmłodszej pary (częstotliwość, polecenie 03)."""
    v, mn = 0, 1
    for b in bajty:
        v += (b & 15) * mn + (b >> 4) * mn * 10
        mn *= 100
    return v


def stan(port: str = "COM5", baud: int = 115200) -> dict:
    """{'pasmo': opis albo 'nieodczytane', 'ton_cw': Hz albo None, 'surowe': wyjście skryptu}."""
    # pełna ścieżka: PATH u innego operatora (albo w teście exe) może nie mieć PowerShella
    ps = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    r = subprocess.run([str(ps) if ps.exists() else "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                        str(SKRYPT), "-Porty", port, "-Bauds", str(baud)], capture_output=True, cwd=KORZEN,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))   # aplikacja: bez czarnego okna
    t = r.stdout.decode("utf-8", "replace")
    wyn = {"pasmo": "nieodczytane", "ton_cw": None, "surowe": t.strip()}
    m = re.search(r"03 ((?:[0-9A-F]{2} ){5})FD", t)
    if m:
        hz = _bcd([int(b, 16) for b in m.group(1).split()])
        tr = re.search(r"tryb: .*04 ([0-9A-F]{2}) ([0-9A-F]{2}) FD", t)
        tryby = {"03": "CW", "07": "CW-R"}
        wyn["pasmo"] = (f"{hz / 1e6:.6f} MHz".replace(".", ",")
                        + (f", {tryby.get(tr.group(1), tr.group(1))}, FIL{int(tr.group(2), 16)}" if tr else ""))
    # odpowiedź: ... 14 09 HH LL FD, wartość BCD 0000-0255 (big-endian) = 300-900 Hz
    m = re.search(r"ton CW: .*14 09 ([0-9A-F]{2}) ([0-9A-F]{2}) FD", t)
    if m:
        hh, ll = int(m.group(1), 16), int(m.group(2), 16)
        v = ((hh >> 4) * 10 + (hh & 15)) * 100 + (ll >> 4) * 10 + (ll & 15)
        if 0 <= v <= 255:
            # 5 Hz krok w radiu: zaokrąglenie usuwa resztę z dzielenia 600/255
            wyn["ton_cw"] = float(round((300.0 + v * 600.0 / 255.0) / 5.0) * 5.0)
    return wyn


def ton_z_opisu(opis: Path) -> float | None:
    """Pole „ton CW:" z opisu nagrania (zapisane przy nagraniu z CI-V)."""
    if not opis.exists():
        return None
    m = re.search(r"^ton CW:\s*([0-9]+(?:[.,][0-9]+)?)\s*Hz", opis.read_text(encoding="utf-8"), re.M)
    return float(m.group(1).replace(",", ".")) if m else None


if __name__ == "__main__":
    s = stan()
    print(s["surowe"])
    print(f"pasmo: {s['pasmo']}\nton CW: {s['ton_cw']} Hz")
