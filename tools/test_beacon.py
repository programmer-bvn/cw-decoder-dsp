"""TEST NA BEACONACH — znana, powtarzana treść z prawdziwego pasma (10.11).

Skąd: zgłoszenie SQ2MMX z 10.11 — beacon SA6RR nagrany na FTDX10, z wpisem
w polu „nadane". Testy słów (tester_viterbi, 1675 słów) mają mało nagrań
z pasma ze znaną treścią; beacon nadaje w kółko ten sam komunikat, więc
wiadomo DOKŁADNIE, co powinno wyjść, znak po znaku, także przy zanikach.

    python -m tools.test_beacon [plik.flac|wav ...]     # domyślnie W:\\nagrania_cw\\beacony\\*

Opis obok nagrania (ten sam rdzeń, .txt): „komunikat:" (jeden cykl),
„cykle:" (pełne cykle w oknie), „od_s:"/„do_s:" (okno testu). Nagrania
zgłaszających są do użytku wewnętrznego — dlatego poza probki/ (CC0).

Łańcuch jak w programie (tools/gui/analiza): przygotuj -> sieć zboczy numpy
-> bramka taktu (z nią i bez) -> Viterbi; okno wycinane z P po sieci, żeby
pętla tonu i bramka widziały całe nagranie jak na żywo.
Miary: słowa jak w testerach (tester_zbocza.slowa, tekst bez spacji) oraz
CER — odległość edycyjna oczekiwanego tekstu (cykle bez spacji) od
najlepiej pasującego fragmentu odczytu, osobno na każdy cykl.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

KATALOG = Path(r"W:\nagrania_cw\beacony")


def opis(plik: Path) -> dict:
    d = {}
    for linia in plik.with_suffix(".txt").read_text(encoding="utf-8").splitlines():
        if ":" in linia and not linia.startswith(("#", " ")):
            k, v = linia.split(":", 1)
            d[k.strip()] = v.strip()
    return d


def dopasuj(chce: str, jest: str) -> list[int]:
    """Odległość edycyjna z dowolnym początkiem i końcem w `jest` (semi-global).
    -> dla każdego znaku `chce`: 1, gdy zgubiony/zamieniony, plus wstawki
    doliczone do znaku, przed którym stoją."""
    n, m = len(chce), len(jest)
    D = np.zeros((n + 1, m + 1), np.int32)
    D[:, 0] = np.arange(n + 1)
    for i in range(1, n + 1):
        a = chce[i - 1]
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j - 1] + (a != jest[j - 1]), D[i - 1, j] + 1, D[i, j - 1] + 1)
    bl = [0] * n
    i, j = n, int(np.argmin(D[n]))
    while i > 0:
        if j > 0 and D[i, j] == D[i - 1, j - 1] + (chce[i - 1] != jest[j - 1]):
            bl[i - 1] += int(chce[i - 1] != jest[j - 1]); i -= 1; j -= 1
        elif D[i, j] == D[i - 1, j] + 1:
            bl[i - 1] += 1; i -= 1
        else:
            bl[min(i, n) - 1] += 1; j -= 1
    return bl


def testuj(plik: Path, siec) -> None:
    from dsp import frontend, viterbi as V, zbocza as Z
    from tools import nagrania, odczyt
    from tools import tester_zbocza as T
    from tools.gui import analiza as A
    o = opis(plik)
    kom, n = o["komunikat"], int(o.get("cykle", 1))
    a, ton = odczyt.przygotuj(frontend.load_audio(plik), ton_cw=None)
    p = Z.przebieg_sieci(a, siec)
    s0, s1 = int(float(o.get("od_s", 0)) / V.DT), int(float(o.get("do_s", 1e9)) / V.DT)
    chce_slowa = nagrania.TOKEN.findall(" ".join([kom] * n))
    cykl = kom.replace(" ", "")
    print(f"{plik.name}: komunikat {len(cykl)} znaków x {n}, ton {ton:.0f} Hz")
    for nazwa, pz in (("z bramką", A.bramka_taktu(p)), ("bez bramki", p)):
        x = pz[s0:s1]
        u_c, _ = V.jednostka_cepstrum(x)
        tekst, u, _ = V.dekoduj_najlepiej(x, [u_c, odczyt.jednostka_grzebienia(x)])
        bez = tekst.replace(" ", "")
        sl = T.slowa(bez, chce_slowa)[0]
        bl = dopasuj(cykl * n, bez)
        na_cykl = [sum(bl[k * len(cykl):(k + 1) * len(cykl)]) for k in range(n)]
        print(f"  {nazwa:10s} {1.2 / u:4.1f} WPM  słowa {sl}/{len(chce_slowa)}  "
              f"CER {sum(bl) / len(bl):.1%}  na cykl " + " ".join(f"{e / len(cykl):.0%}" for e in na_cykl))
        print(f"    {tekst}")


def main() -> None:
    from dsp.siec_numpy import SiecNumpy
    from tools.gui import analiza as A
    pliki = [Path(x) for x in sys.argv[1:]] or sorted(
        f for f in KATALOG.glob("*") if f.suffix.lower() in (".wav", ".flac") and f.with_suffix(".txt").exists())
    siec = SiecNumpy(A.SIEC)
    for f in pliki:
        testuj(f, siec)


if __name__ == "__main__":
    main()
