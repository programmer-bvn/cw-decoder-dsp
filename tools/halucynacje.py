"""HALUCYNACJE — ile znaków dekoder „czyta” tam, gdzie telegrafii nie ma.

Operator 05.11: na mikrofonie (mowa, hałas w pokoju) dekoder bez przerwy
wypisuje E, T, EE… — „sieć sama z siebie nie powinna rozpoznawać zboczy tam,
gdzie ich nie ma”. Generator zbioru zna jako „brak tonu” tylko szum (puste
radio, tło z probki/szum, QRN): mowy, muzyki, SSB i pływającego tonu sieć nie
widziała. Zanim zmienimy dane albo sieć, potrzebna liczba: fałszywe znaki na
minutę na nagraniach, w których na pewno nie ma CW. Obok CER na probki/ —
sieć, która „w razie wątpliwości milczy”, zgubi słabe stacje.

Bank negatywów: W:\\nagrania_cw\\negatywy\\<rodzaj>\\*.wav (poza gitem — nagrania
z pokoju operatora i z cudzych odbiorników; tylko do pomiaru i treningu).
Rodzaj = podkatalog (pokoj_mikrofon, kiwi_ssb, …), wynik osobno dla każdego.
Pliki z TRENING.txt (część treningowa negatywów, dsp/negatywy) są pomijane.

    python -m tools.halucynacje tnij SESJA.wav pokoj_mikrofon [--min 1]   # kawałki do banku
    python -m tools.halucynacje ocen [--co 5] [--model PLIK] [--out PLIK]  # znaki/min
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BANK = Path(r"W:\nagrania_cw\negatywy")


def tnij(sesja: Path, rodzaj: str, minut: float = 1.0, bank: Path = BANK) -> int:
    import soundfile as sf
    kat = bank / rodzaj
    kat.mkdir(parents=True, exist_ok=True)
    n = 0
    with sf.SoundFile(sesja) as f:
        dl = int(minut * 60 * f.samplerate)
        while True:
            x = f.read(dl, dtype="int16")
            if len(x) < dl // 2:                   # ogon krótszy niż pół kawałka — pomijamy
                break
            sf.write(kat / f"{sesja.stem}_{n:04d}.wav", x, f.samplerate, subtype="PCM_16")
            n += 1
    (kat / "ZRODLO.txt").open("a", encoding="utf-8").write(
        f"{time.strftime('%Y-%m-%d %H:%M')}  {sesja}  -> {n} kawałków po {minut:g} min\n")
    return n


def ocen(model_p: Path | None, co: int = 1, bank: Path = BANK, elementy: Path | None = None,
         laczenie: str | None = None) -> dict:
    """Odczyt całości (tools.odczyt, ta sama ścieżka co „Otwórz nagranie”, bez
    blokady szumów) na co `co`-tym kawałku każdego rodzaju -> znaki na minutę."""
    import soundfile as sf
    from tensorflow import keras
    from tools.odczyt import MODEL_ZBOCZY, odczytaj
    # --elementy (07.11): P z etapu A przegródek zamiast sieci zboczy, Viterbi ten sam
    me = keras.models.load_model(elementy) if elementy else None
    model = None if me is not None and not laczenie else keras.models.load_model(model_p or MODEL_ZBOCZY)
    wynik: dict = {"model": f"{model_p or MODEL_ZBOCZY} + {elementy} ({laczenie})" if laczenie
                   else str(elementy or model_p or MODEL_ZBOCZY), "rodzaje": {}}
    # TRENING.txt (06.11, dsp/negatywy): minuty, które zboczy_6 widziała w treningu
    # — pomiar na nich mierzyłby pamięć sieci, nie halucynacje.
    tr = bank / "TRENING.txt"
    trening = set(tr.read_text(encoding="utf-8").split()) if tr.exists() else set()
    for kat in sorted(p for p in bank.iterdir() if p.is_dir()):
        pliki = [p for p in sorted(kat.glob("*.wav")) if f"{kat.name}/{p.name}" not in trening][::co]
        if not pliki:
            continue
        znaki, minuty, przyklady = [], 0.0, []
        for p in pliki:
            t = odczytaj(p, model, ton_cw=None, model_elementow=me, laczenie=laczenie)["tekst"]
            min_ = sf.info(p).duration / 60
            ile = sum(1 for z in t if z != " ")
            znaki.append(ile / min_)
            minuty += min_
            przyklady.append((ile / min_, p.name, t[:60]))
        z = np.array(znaki)
        wynik["rodzaje"][kat.name] = dict(
            kawalkow=len(pliki), minut=round(minuty, 1), znakow_na_min=round(float(z.mean()), 1),
            mediana=round(float(np.median(z)), 1), p90=round(float(np.percentile(z, 90)), 1),
            najgorsze=[f"{a:.0f}/min {b}: {c}" for a, b, c in sorted(przyklady, reverse=True)[:3]])
    return wynik


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tnij"); t.add_argument("sesja", type=Path); t.add_argument("rodzaj")
    t.add_argument("--min", type=float, default=1.0)
    o = sub.add_parser("ocen"); o.add_argument("--model", type=Path, default=None)
    o.add_argument("--elementy", type=Path, default=None, help="model etapu A (dsp/elementy)")
    o.add_argument("--laczenie", choices=("iloczyn", "kaskada"), default=None,
                   help="z --elementy: zbocza (--model) połączone z przegródkami")
    o.add_argument("--co", type=int, default=1, help="co który kawałek (szybki pomiar)")
    o.add_argument("--out", type=Path, default=None)
    for p in (t, o):
        p.add_argument("--bank", type=Path, default=BANK)
    a = ap.parse_args(argv)
    if a.cmd == "tnij":
        print(f"{tnij(a.sesja, a.rodzaj, a.min, a.bank)} kawałków -> {a.bank / a.rodzaj}")
        return 0
    w = ocen(a.model, a.co, a.bank, a.elementy, a.laczenie)
    linie = [f"HALUCYNACJE — model {w['model']}"]
    for r, d in w["rodzaje"].items():
        linie.append(f"  {r}: {d['znakow_na_min']} znaków/min (mediana {d['mediana']}, p90 {d['p90']}), "
                     f"{d['kawalkow']} kawałków, {d['minut']} min")
        linie += [f"      {x}" for x in d["najgorsze"]]
    print("\n".join(linie))
    if a.out:
        a.out.write_text("\n".join(linie) + "\n", encoding="utf-8")
        a.out.with_suffix(".json").write_text(json.dumps(w, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
