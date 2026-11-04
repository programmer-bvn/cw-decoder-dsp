"""Lista transceiverów i odbiorników do formularza raportu: producent, potem model z listy.

Operator 03.11: „żeby nie musiał wpisywać, bo jak wpisze, to coś pomyli, a jak
wybierze, to będzie wybrane” — jak w CHIRP-ie. Gdy radia nie ma na liście, pola
przyjmują wpis ręczny; nowe radio z raportu (tools.zgloszenia, PODSUMOWANIE.md)
dopisuje się tutaj i w następnej wersji jest już do wybrania.

SPRAWDZONE: tylko radia, na których dekoder naprawdę pracował (nagrania w probki/,
zgłoszenia, sesje operatora) — z tego powstaje lista w instrukcji.
"""

from __future__ import annotations

from pathlib import Path

INNY = "Inny / Other"

RADIA: dict[str, list[str]] = {
    "Icom": ["IC-705", "IC-718", "IC-7100", "IC-7300", "IC-7300MK2", "IC-7410", "IC-746", "IC-746PRO",
             "IC-756PRO III", "IC-7600", "IC-7610", "IC-7700", "IC-7851", "IC-9700", "IC-R75", "IC-R8600"],
    "Yaesu": ["FT-101", "FT-450D", "FT-710", "FT-817ND", "FT-818ND", "FT-857D", "FT-891", "FT-897D",
              "FT-920", "FT-950", "FT-990", "FT-991A", "FT-1000MP", "FT-2000", "FTDX10", "FTDX101D",
              "FTDX101MP", "FTDX1200", "FTDX3000", "FTDX5000"],
    "Kenwood": ["TS-50S", "TS-120S", "TS-130S", "TS-140S", "TS-430S", "TS-440S", "TS-450S", "TS-480",
                "TS-520", "TS-530S", "TS-570D", "TS-590S", "TS-590SG", "TS-690S", "TS-830S", "TS-850S",
                "TS-870S", "TS-890S", "TS-930S", "TS-940S", "TS-950SDX", "TS-990S", "TS-2000"],
    "Elecraft": ["K1", "K2", "K3", "K3S", "K4", "KX2", "KX3"],
    "Xiegu": ["G90", "G106", "X5105", "X6100", "X6200"],
    "FlexRadio": ["FLEX-6400", "FLEX-6600", "FLEX-8400", "FLEX-8600"],
    "Ten-Tec": ["Argonaut VI", "Eagle", "Omni VII", "Orion II"],
    "Lab599": ["Discovery TX-500"],
    "CRT": ["SS 6900"],
    "SDR / WebSDR": ["KiwiSDR", "WebSDR", "SDRplay", "Airspy HF+", "RTL-SDR", "Hermes-Lite 2"],
    INNY: ["własna konstrukcja / homebrew"],
}

SPRAWDZONE: list[tuple[str, str, str, str]] = [
    # (producent, model, jak podłączone — do instrukcji, skąd wiadomo — dla nas)
    ("Icom", "IC-7300MK2", "kabel USB: dźwięk i ton CW przez CI-V", "radio operatora, nagrania w probki/"),
    ("Icom", "IC-746", "dźwięk do karty dźwiękowej", "nagrania w probki/"),
    ("Kenwood", "TS-520", "dźwięk do karty dźwiękowej (bez CAT)", "nagrania w probki/"),
    ("CRT", "SS 6900", "dźwięk z głośnika (kabel albo mikrofon), bez CAT", "SQ2PVB, QSO 01.11.2026"),
    ("SDR / WebSDR", "KiwiSDR", "nagrania z odbiornika w sieci", "beacony NCDXF, rejestrator zawodów"),
]

POCZATEK, KONIEC = "<!-- RADIA: tools/radia.py -->", "<!-- /RADIA -->"


def producenci() -> list[str]:
    return list(RADIA)


def modele(producent: str) -> list[str]:
    return RADIA.get(producent.strip(), [])


def na_liscie(producent: str, model: str) -> bool:
    return model.strip() in modele(producent)


def html() -> str:
    # „SDR / WebSDR KiwiSDR” źle się czyta — przy grupie (z ukośnikiem) sam model
    w = "".join(f"  <tr><td>{m if '/' in p else f'{p} {m}'}</td><td>{jak}</td></tr>\n" for p, m, jak, _ in SPRAWDZONE)
    return f"{POCZATEK}\n<table>\n{w}</table>\n{KONIEC}"


def wstaw_do(plik: Path) -> None:
    """Tabela sprawdzonych transceiverów w instrukcji (budowa_exe.bat przed PDF-em):
    jedno źródło listy — instrukcja nie rozjedzie się z tools/radia.py."""
    t = plik.read_text(encoding="utf-8")
    a, b = t.index(POCZATEK), t.index(KONIEC) + len(KONIEC)
    plik.write_text(t[:a] + html() + t[b:], encoding="utf-8")


def main() -> None:
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "--instrukcja":
        wstaw_do(Path(sys.argv[2]))
        return
    for p, m, jak, skad in SPRAWDZONE:
        print(f"{p} {m}: {jak} ({skad})")


if __name__ == "__main__":
    main()
