"""ZGŁOSZENIA — raporty „Zgłoś problem z dekodowaniem” z aplikacji (25.10).

Obieg (operator 25.10): skrzynka Gmail projektu w Thunderbirdzie na tej
maszynie, reguła „temat zawiera [CW-DECODER]” -> folder CW-DECODER. Thunderbird
trzyma kopie wiadomości z załącznikami w plikach profilu (mbox) — skrypt czyta
je wprost, bez logowania i bez haseł (zostają w Thunderbirdzie, u operatora).

    python -m tools.zgloszenia                      # folder CW-DECODER z profilu Thunderbirda
    python -m tools.zgloszenia ŚCIEŻKA [...]        # plik mbox, katalog z ZIP-ami albo ZIP
    python -m tools.zgloszenia --odpisz [ZNAK]      # okna odpowiedzi w Thunderbirdzie (--pokaz: tylko treść,
                                                    #   --od-nowa: jeszcze raz, gdy okno zamknięte bez wysłania)

Każde nowe zgłoszenie -> zgloszenia/<czas>_<ZNAK>/ (poza gitem: audio, adresy):
raport.zip i rozpakowane pliki, analiza.md (odczyt bieżącą wersją, sygnał,
diagnoza), odpowiedz.txt (szkic w języku zgłaszającego), nadawca.txt.
Zbiorczo: zgloszenia/PODSUMOWANIE.md (bez adresów) i ODPOWIEDZI.html
(linki mailto: klik -> okno nowej wiadomości w Thunderbirdzie, „Wyślij”).
Przetworzone: zgloszenia/przetworzone.json (Message-ID albo skrót ZIP-a).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import mailbox
import os
import re
import sys
import time
import zipfile
from email.header import decode_header, make_header
from email.utils import parseaddr
from pathlib import Path

import numpy as np

KORZEN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KORZEN))

KATALOG = KORZEN / "zgloszenia"
ZNACZNIK = "[CW-DECODER]"
FOLDER_TB = "CW-DECODER"


# ---------------------------------------------------------------- źródła
def _folder_thunderbirda() -> list[Path]:
    """Pliki mbox folderu CW-DECODER we wszystkich profilach Thunderbirda."""
    baza = Path(os.environ.get("APPDATA", "")) / "Thunderbird" / "Profiles"
    foldery = [p for p in baza.glob(f"*/*Mail/**/{FOLDER_TB}") if p.is_file()]
    # bez folderu (konto POP3 jak u operatora 25.10: Mail\pop.gmail.com\Inbox) —
    # skrzynki odbiorcze kont Gmail; raporty i tak wybiera temat [CW-DECODER]
    return foldery or [p for p in baza.glob("*/*Mail/*gmail.com*/Inbox") if p.is_file()]


def _tekst(h: str | None) -> str:
    return str(make_header(decode_header(h))) if h else ""


def wiadomosci(zrodla: list[Path]):
    """-> (klucz, nadawca, temat, data, [(nazwa, bajty ZIP)]) z mbox, katalogów i ZIP-ów."""
    for z in zrodla:
        if z.is_dir():
            for p in sorted(z.glob("*.zip")):
                b = p.read_bytes()
                yield hashlib.sha1(b).hexdigest(), "", p.name, "", [(p.name, b)]
        elif z.suffix.lower() == ".zip":
            b = z.read_bytes()
            yield hashlib.sha1(b).hexdigest(), "", z.name, "", [(z.name, b)]
        else:
            for m in mailbox.mbox(str(z), create=False):
                temat = _tekst(m.get("Subject"))
                if ZNACZNIK not in temat:
                    continue
                zal = [(_tekst(c.get_filename()), c.get_payload(decode=True)) for c in m.walk()
                       if c.get_filename() and _tekst(c.get_filename()).lower().endswith(".zip")]
                klucz = (m.get("Message-ID") or "").strip() or hashlib.sha1(bytes(m)).hexdigest()
                yield klucz, parseaddr(m.get("From", ""))[1], temat, m.get("Date", ""), zal


# ---------------------------------------------------------------- analiza
def _model():
    from dsp.siec_numpy import SiecNumpy
    npz = KORZEN / "runs" / "zbocza_5" / "siec.npz"
    if npz.exists():
        return SiecNumpy(npz)
    from tensorflow import keras
    from tools.odczyt import MODEL_ZBOCZY
    return keras.models.load_model(MODEL_ZBOCZY)


PRZERWA_NADAWANIA_S = 2.0   # dłuższa przerwa między kreskami = koniec nadawania (zmiana stacji)


def dryf_tonu(kreski: list[tuple[float, float, float]]) -> float:
    """Kreski (od, do, ton [Hz]) -> dryf: rozrzut tonu kresek W OBRĘBIE jednego
    nadawania (p90 - p10), mediana po nadawaniach z >= 5 kreskami.
    SQ2MMX 27.10: rozrzut z całego nagrania dał „ton pływa o 107 Hz”, a to dwie
    stacje (SX37VER 719 Hz, SV1ENG 695 Hz, każda równa) i sąsiedzi w paśmie."""
    grupy, ost = [], -np.inf
    for a0, b0, fk in sorted(kreski):
        if a0 - ost > PRZERWA_NADAWANIA_S:
            grupy.append([])
        grupy[-1].append(fk)
        ost = b0
    rozrzut = [float(np.percentile(g, 90) - np.percentile(g, 10)) for g in grupy if len(g) >= 5]
    return float(np.median(rozrzut)) if rozrzut else float("nan")


def analizuj(kat: Path, model) -> dict:
    """Odczyt nagrania bieżącą wersją + miary sygnału + diagnoza (heurystyki)."""
    from dsp import blokada as B, config as C, frontend, przeciwsygnal as PS, tune, viterbi as V
    from tools.klasyczny import cer
    from tools.odczyt import odczytaj, przygotuj
    meta = json.loads((kat / "raport.json").read_text(encoding="utf-8"))
    nagr = kat / "nagranie.flac"
    m = re.search(r"([0-9]+(?:[.,][0-9]+)?)\s*Hz", str(meta.get("z_sesji", {}).get("ton_CW", "")))
    ton_cw = float(m.group(1).replace(",", ".")) if m else None
    r = odczytaj(nagr, model, ton_cw)
    surowe = frontend.load_audio(nagr)
    a, ton = przygotuj(surowe, ton_cw=ton_cw)
    p = r["p"]
    poz = B.poziom_tonu(a)[:p.size]
    nad = poz - B.podloga(poz)
    klucz = p[:nad.size] > 0.5
    snr = float(np.percentile(nad[klucz], 50)) if klucz.sum() > 20 else float("nan")
    t, f = tune.track_tone(surowe, ton_cw=ton_cw)
    # dryf tylko w KRESKACH: zgłoszenie 25.10 (mikrofon, operator mówił w przerwach) —
    # z całego nagrania „ton pływa o 126 Hz” (pętla szła za głosem), przy wciśniętym
    # kluczu 37 Hz (pojedyncze „E” z mowy też są „kluczem”); kreski z mowy prawie nie wychodzą
    kreski = []
    if np.isfinite(r["u"]):
        for n, a0, b0 in V.dekoduj(p, r["u"])[1]:
            if n == "K3":
                fk = f[(t >= a0 * V.DT) & (t <= b0 * V.DT) & np.isfinite(f)]
                if fk.size:
                    kreski.append((a0 * V.DT, b0 * V.DT, float(np.median(fk))))
    dryf = dryf_tonu(kreski)
    obce, nasza_db = [], None
    if np.isfinite(ton):
        x = np.asarray(surowe, np.float64)
        # poziom naszej w tej samej skali co obce: szczyty bez wykluczenia okolicy tonu
        nasza = [g for fo, g in PS.obce_stacje(x, ton + 1000.0, C.SR, PS.pasmo_obcych(ton)) if abs(fo - ton) < PS.ODSTEP_HZ]
        nasza_db = max(nasza) if nasza else None
        # względem naszej; liczą się od POZIOM_MIN_DB (-12 dB, jak przeciwsygnał) — słabsze
        # to zwykle prążki własnego kluczowania (6 m spokój: -24 dB, ręka: -32 dB; 40 m tłok: -3 dB)
        obce = [(fo, g - nasza_db) for fo, g in PS.obce_stacje(x, ton, C.SR, PS.pasmo_obcych(ton))
                if nasza_db is not None and g - nasza_db >= PS.POZIOM_MIN_DB]
    u = r["u"]
    elem = {"K1": [], "K3": [], "PL": [], "PS": []}
    if np.isfinite(u):
        for n, a0, b0 in V.dekoduj(p, u)[1]:
            if n in elem:
                elem[n].append((b0 - a0) * V.DT / u)
    zm = {k: (float(np.std(v) / np.mean(v)) if len(v) > 5 else float("nan")) for k, v in elem.items()}
    wynik = {
        "tekst": r["tekst"], "wpm": round(float(r["wpm"]), 1) if np.isfinite(r["wpm"]) else None,
        "ton_hz": round(float(ton), 1) if np.isfinite(ton) else None,
        "poziom_dbfs": round(float(20 * np.log10(np.sqrt(np.mean(surowe ** 2)) + 1e-12)), 1),
        "snr_db": round(snr, 1) if np.isfinite(snr) else None,
        "dryf_hz": round(dryf, 1) if np.isfinite(dryf) else None,
        "obce": [(round(fo), round(g, 1)) for fo, g in obce[:6]],          # (Hz, dB względem naszej)
        "rozrzut_kropek": round(zm["K1"], 2) if np.isfinite(zm["K1"]) else None,
        "rozrzut_przerw_liter": round(zm["PL"], 2) if np.isfinite(zm["PL"]) else None,
        "czas_s": round(surowe.size / C.SR, 1),
    }
    slowa = r["tekst"].split()
    if slowa:
        # pojedyncze E/T/I = szum albo mowa w przerwach czytane jako kropki
        wynik["pojedyncze_et"] = round(sum(len(x) == 1 and x in "ETI" for x in slowa) / len(slowa), 2)
    nad_txt = meta.get("nadane", "").strip().upper()
    # SQ2MMX 27.10: w „nadane” opis („Dane do uczenia algorytmów”, to samo co w uwagach)
    # -> „743 błędy na 23 znaki”. Liczymy tylko tekst, który da się nadać Morse'em.
    if nad_txt == meta.get("opis", "").strip().upper() or not set(nad_txt) <= set(C.MORSE_DICT) | {" "}:
        nad_txt = ""
    if nad_txt:
        bledy = cer(r["tekst"].replace(" ", ""), nad_txt.replace(" ", ""))
        # SQ2MMX 26.10: „Dane do trenowania modelu” — sam ASCII, inny niż opis, a też opis
        # (971 „błędów” na 22 znaki). Żadne słowo nie odczytane i błędów > 2x znaków -> nie treść.
        slowa = [s for s in nad_txt.split() if len(s) >= 3]
        if bledy <= 2 * len(nad_txt.replace(" ", "")) or any(s in r["tekst"] for s in slowa):
            wynik["bledy_znakow"] = bledy
            wynik["znakow_nadanych"] = len(nad_txt.replace(" ", ""))
    wynik["diagnoza"] = diagnoza(wynik)
    return {"meta": meta, "analiza": wynik}


def diagnoza(w: dict) -> list[str]:
    """Najbardziej prawdopodobne przyczyny — progi z pomiarów projektu
    (HISTORIA): Viterbi czyta do ~0 dB S/N; QRM bliżej niż ~60 Hz +6 dB boli;
    ręka: rozrzut przerw > ~0,3."""
    d = []
    if not w["tekst"].strip():
        d.append("brak odczytu — sygnał nie znaleziony (za słaby, zły ton, inna karta?)")
    if w["snr_db"] is not None and w["snr_db"] < 8:
        d.append(f"słaby sygnał: ton {w['snr_db']} dB nad szumem")
    blisko = [o for o in w["obce"] if w["ton_hz"] and abs(o[0] - w["ton_hz"]) < 120]
    if blisko:
        d.append(f"QRM: obce stacje blisko ({', '.join(f'{o[0]} Hz {o[1]:+.0f} dB' for o in blisko)}; "
                 f"nasza {w['ton_hz']:.0f} Hz)")
    if w["dryf_hz"] is not None and w["dryf_hz"] > 30:
        d.append(f"ton pływa o {w['dryf_hz']} Hz (dryf / chirp?)")
    if w["wpm"] is not None and (w["wpm"] > 35 or w["wpm"] < 8):
        d.append(f"tempo poza zakresem treningu: {w['wpm']} WPM")
    if w["rozrzut_przerw_liter"] is not None and w["rozrzut_przerw_liter"] > 0.3:
        d.append(f"nierówne przerwy (ręka?): rozrzut {w['rozrzut_przerw_liter']}")
    if w.get("pojedyncze_et", 0) > 0.3:
        d.append(f"w przerwach szum lub mowa czytane jako E/T ({w['pojedyncze_et']:.0%} słów) — "
                 "pomoże suwak blokady szumów")
    if w["poziom_dbfs"] < -60:
        d.append(f"bardzo cichy dźwięk na wejściu: {w['poziom_dbfs']} dBFS")
    if w["poziom_dbfs"] > -3:
        d.append(f"przesterowanie wejścia: {w['poziom_dbfs']} dBFS")
    return d or ["brak oczywistej przyczyny — do przejrzenia ręcznie"]


# ---------------------------------------------------------------- odpowiedź
ODPOWIEDZ = {
    "pl": ("Dziękuję za raport z Dekodera CW{znak}! Takie nagrania są dla projektu najcenniejsze — "
           "pokazują sygnały, których nie da się zebrać w jednym miejscu.\n\n"
           "Co widać w nagraniu: {diag}.\n\n"
           "Nagranie trafi do zbioru testowego; gdy kolejna wersja będzie je czytać lepiej, "
           "dam znać. Można też samemu sprawdzić: zainstalować nową wersję obok starej "
           "i otworzyć to samo nagranie przyciskiem „Otwórz nagranie…”.\n\n"
           "Proszę o więcej takich zgłoszeń — każde przybliża dekoder do odczytu wszystkiego, "
           "co słychać na paśmie.\n"),
    "en": ("Thank you for the CW Decoder report{znak}! Recordings like this are the most valuable "
           "thing for the project — they show signals that cannot be collected in one place.\n\n"
           "What the recording shows: {diag}.\n\n"
           "The recording goes into the test set; I will let you know when a new version reads it better. "
           "You can also check yourself: install the new version next to the old one and open the same "
           "recording with \"Open recording…\".\n\n"
           "Please keep sending such reports — each one brings the decoder closer to reading everything "
           "you can hear on the band.\n"),
}
# podpis dokładany przy wysyłce (--odpisz), nie w szkicu (operator 27.10).
# Podpisuje operator (28.10): ludzie odchodzą, gdy uznają, że rozmawiają
# z maszyną — narzędzie pomaga, odpowiada i wysyła człowiek.
PODPIS = {
    "pl": "Dziękuję za zgłoszenie i do usłyszenia — 73!\nSQ2BVN — Dekoder CW\n",
    "en": "Thank you for the report — 73!\nSQ2BVN — CW Decoder\n",
}
WSTEP_SEKCJI = {
    "pl": ("Zmiany w projekcie wykonane na podstawie Twojego zgłoszenia i przesłanego nagrania "
           "(wpisy z historii projektu):",
           "Dalszy ciąg Twojego zgłoszenia — kolejne zmiany wykonane na podstawie przesłanego przez "
           "Ciebie materiału (wpisy z historii projektu):"),
    "en": ("Changes made in the project based on your report and recording "
           "(entries from the project history, in Polish):",
           "A follow-up to your report — further changes made based on the material you sent "
           "(entries from the project history, in Polish):"),
}
DIAG_EN = [("brak odczytu — sygnał nie znaleziony (za słaby, zły ton, inna karta?)",
            "no reading — signal not found (too weak, wrong pitch, other sound card?)"),
           ("słaby sygnał", "weak signal"), ("dB nad szumem", "dB above noise"),
           ("obce stacje blisko", "other stations close"), ("nasza", "ours"),
           ("ton pływa o", "pitch drifts by"), ("dryf / chirp?", "drift / chirp?"),
           ("tempo poza zakresem treningu", "speed outside the training range"),
           ("nierówne przerwy (ręka?): rozrzut", "uneven spacing (hand keying?): spread"),
           ("bardzo cichy dźwięk na wejściu", "very low input level"),
           ("przesterowanie wejścia", "input overload"),
           ("w przerwach szum lub mowa czytane jako E/T", "noise or speech between transmissions read as E/T"),
           ("słów", "of words"), ("pomoże suwak blokady szumów", "the noise squelch slider will help"),
           ("brak oczywistej przyczyny — do przejrzenia ręcznie", "no obvious cause — will be checked manually")]


def odpowiedz(meta: dict, an: dict) -> str:
    jez = meta.get("jezyk", "pl") if meta.get("jezyk") in ODPOWIEDZ else "en"
    diag = "; ".join(an["diagnoza"])
    if jez == "en":
        for a, b in DIAG_EN:
            diag = diag.replace(a, b)
    znak = f", {meta['znak']}" if meta.get("znak") else ""
    return ODPOWIEDZ[jez].format(znak=znak, diag=diag)


# ---------------------------------------------------------------- główne
def przetworz(zrodla: list[Path]) -> int:
    KATALOG.mkdir(exist_ok=True)
    plik_przetw = KATALOG / "przetworzone.json"
    przetw = json.loads(plik_przetw.read_text(encoding="utf-8")) if plik_przetw.exists() else {}
    model = None
    nowe = 0
    for klucz, nadawca, temat, data, zal in wiadomosci(zrodla):
        if klucz in przetw or not zal:
            continue
        model = model or _model()
        for nazwa, b in zal:
            try:
                z = zipfile.ZipFile(io.BytesIO(b))
                meta = json.loads(z.read("raport.json"))
            except (zipfile.BadZipFile, KeyError, ValueError) as e:
                print(f"POMIJAM {nazwa}: nie jest raportem ({e!r})")
                continue
            znak = re.sub(r"[^A-Z0-9-]", "", (meta.get("znak") or "ANON").upper().replace("/", "-")) or "ANON"
            kat = KATALOG / f"{time.strftime('%Y%m%d_%H%M%S')}_{znak}"
            n = 1
            while kat.exists():
                kat = kat.with_name(f"{kat.name}_{n}")
                n += 1
            kat.mkdir()
            (kat / "raport.zip").write_bytes(b)
            z.extractall(kat)
            if nadawca:
                (kat / "nadawca.txt").write_text(f"{nadawca}\n{temat}\n{data}\n", encoding="utf-8")
            try:
                w = analizuj(kat, model)
            except Exception as e:                 # jedno zepsute zgłoszenie nie zatrzymuje reszty
                w = {"meta": meta, "analiza": {"blad": repr(e), "diagnoza": [f"błąd analizy: {e!r}"],
                                                "tekst": ""}}
            an = w["analiza"]
            zapisz_analize(kat, w)
            _dopisz_odpowiedz(kat.name, nadawca, temat, (kat / "odpowiedz.txt").read_text(encoding="utf-8"))
            nowe += 1
            print(f"{kat.name}: {'; '.join(an['diagnoza'])}")
        przetw[klucz] = time.strftime("%Y-%m-%d %H:%M:%S")
        plik_przetw.write_text(json.dumps(przetw, ensure_ascii=False, indent=1), encoding="utf-8")
    podsumowanie()
    print(f"nowych zgłoszeń: {nowe}")
    return nowe


def zapisz_analize(kat: Path, w: dict) -> None:
    meta, an = w["meta"], w["analiza"]
    (kat / "analiza.json").write_text(json.dumps(w, ensure_ascii=False, indent=1), encoding="utf-8")
    (kat / "analiza.md").write_text(
        f"# {kat.name}\n\nwersja: {meta.get('wersja')}  \nznak: {meta.get('znak') or '-'}  \n"
        f"radio: {_radio(meta) or '-'}  \n"
        f"nadane: {meta.get('nadane') or '-'}  \nuwagi: {meta.get('opis') or '-'}  \n"
        f"pasmo: {meta.get('z_sesji', {}).get('pasmo', '-')}\n\n"
        f"**odczyt bieżącą wersją:** {an.get('tekst', '')[:600]}\n\n"
        f"**diagnoza:** {'; '.join(an['diagnoza'])}\n\n```\n"
        + json.dumps({k: v for k, v in an.items() if k not in ('tekst', 'diagnoza')}, ensure_ascii=False, indent=1)
        + "\n```\n", encoding="utf-8")
    # szkic poprawiony ręcznie zostaje (27.10: automatyczna diagnoza dwa razy myliła —
    # „szum”, „dryf” — szkice przepisywane, --ponow nie może ich zetrzeć)
    if not (kat / "odpowiedz.txt").exists():
        (kat / "odpowiedz.txt").write_text(odpowiedz(meta, an), encoding="utf-8")


def _radio(meta: dict) -> str:
    r = meta.get("radio") or {}
    return f"{r.get('producent', '')} {r.get('model', '')}".strip()


def podsumowanie() -> None:
    """PODSUMOWANIE.md od zera ze wszystkich analiza.json (bez adresów nadawców).
    Na końcu radia z raportów: „spoza listy” = do dopisania w tools/radia.py
    (wpis ręczny w formularzu, 03.11), żeby w następnej wersji było do wybrania."""
    from tools import radia
    wiersze, ile = [], {}
    for a in sorted(KATALOG.glob("*/analiza.json")):
        w = json.loads(a.read_text(encoding="utf-8"))
        meta, an = w["meta"], w["analiza"]
        r = meta.get("radio") or {}
        if _radio(meta):
            k = (r.get("producent", "").strip(), r.get("model", "").strip())
            ile[k] = ile.get(k, 0) + 1
        wiersze.append(f"- **{a.parent.name}** {str(meta.get('wersja', '?')).split(' ')[0]}, "
                       f"{_radio(meta) or 'radio ?'}, "
                       f"{an.get('czas_s', '?')} s: {'; '.join(an['diagnoza'])} — odczyt: `{an.get('tekst', '')[:120]}`")
    lr = [f"- {p} {m}: {n}" + ("" if radia.na_liscie(p, m) else " — **spoza listy, dopisać do tools/radia.py**")
          for (p, m), n in sorted(ile.items())]
    (KATALOG / "PODSUMOWANIE.md").write_text(
        f"# Zgłoszenia — Dekoder CW ({len(wiersze)})\n\n" + "\n".join(wiersze) + "\n\n"
        f"## Radia w raportach ({len(lr)})\n\n" + ("\n".join(lr) or "- (raporty sprzed pola „radio”)") + "\n",
        encoding="utf-8")


def ponow() -> None:
    """Analiza od nowa wszystkich przyjętych zgłoszeń (po zmianie dekodera albo diagnozy).
    Szkice odpowiedz.txt i ODPOWIEDZI.html zostają."""
    model = _model()
    for kat in sorted(p.parent for p in KATALOG.glob("*/raport.json")):
        w = analizuj(kat, model)
        zapisz_analize(kat, w)
        print(f"{kat.name}: {'; '.join(w['analiza']['diagnoza'])}")
    podsumowanie()


def _dopisz_odpowiedz(nazwa: str, nadawca: str, temat: str, tresc: str) -> None:
    """ODPOWIEDZI.html: klik w link -> nowa wiadomość w Thunderbirdzie (mailto), „Wyślij”."""
    import html
    import urllib.parse
    plik = KATALOG / "ODPOWIEDZI.html"
    if not plik.exists():
        plik.write_text("<!doctype html><meta charset='utf-8'><title>Odpowiedzi — Dekoder CW</title>"
                        "<h1>Odpowiedzi na zgłoszenia</h1><p>Klik w link otwiera gotową wiadomość "
                        "w programie pocztowym.</p>\n", encoding="utf-8")
    if nadawca:
        link = (f"mailto:{nadawca}?subject={urllib.parse.quote('Re: ' + temat)}"
                f"&body={urllib.parse.quote(tresc)}")
        wiersz = f"<p><b>{html.escape(nazwa)}</b> — <a href='{html.escape(link)}'>odpowiedz {html.escape(nadawca)}</a></p>\n"
    else:
        wiersz = f"<p><b>{html.escape(nazwa)}</b> — bez adresu (raport z pliku), szkic w odpowiedz.txt</p>\n"
    with open(plik, "a", encoding="utf-8") as f:
        f.write(wiersz)


# ---------------------------------------------------------------- odpisywanie
# Operator 27.10: zgłaszający ma coś dostać, „żeby nie czuł, że wysłał pocztę
# w kosmos”. Po każdej zmianie wynikłej ze zgłoszenia wpis w HISTORIA.md
# z „zgłoszenie ZNAK” w tytule; --odpisz otwiera w Thunderbirdzie gotową
# wiadomość (szkic + nowe wpisy + podpis), operator klika „Wyślij”.
HISTORIA = KORZEN / "HISTORIA.md"
WYSLANE = "wyslane.json"     # w zgloszenia/: {"odpowiedzi": {katalog: data}, "sekcje": {ZNAK: [tytuły]}}


REPO_GH = "https://github.com/programmer-bvn/cw-decoder-dsp"


def sekcje_historii(znak: str) -> list[tuple[str, str]]:
    """Wpisy HISTORIA.md z „zgłoszenie ZNAK” albo „nagranie ZNAK” w tytule (zmiana
    na podstawie materiału od tej osoby) -> [(tytuł, treść bez znaczników)]."""
    if not znak or znak == "ANON" or not HISTORIA.exists():
        return []
    wynik, tytul, tresc = [], None, []
    for linia in HISTORIA.read_text(encoding="utf-8").splitlines() + ["# koniec"]:
        if linia.startswith("#"):
            if tytul:
                wynik.append((tytul, "\n".join(tresc).strip()))
            t = linia.lstrip("#").strip()
            tytul = t if re.search(rf"(zgłoszeni|nagrani)\w*\s[^)]*\b{re.escape(znak)}\b", t, re.I) else None
            tresc = []
        elif tytul:
            tresc.append(linia.replace("**", "").replace("`", ""))
    return wynik


def _wydane(wersja: str) -> bool:
    """Czy wydanie v<wersja> jest na GitHubie (tag w origin; wydanie tworzy operator)."""
    import subprocess
    try:
        r = subprocess.run(["git", "ls-remote", "--tags", "origin", f"v{wersja}"], cwd=KORZEN,
                           capture_output=True, text=True, timeout=30)
        return bool(r.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        return False


def _galaz() -> str:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=KORZEN,
                              capture_output=True, text=True, timeout=30).stdout.strip() or "main"
    except (OSError, subprocess.TimeoutExpired):
        return "main"


def gdzie_zmiana(tytul: str, tresc: str, jez: str) -> str:
    """Wersja z linii „Wersja: X” wpisu (operator 27.10: w której wersji, w którym
    wydaniu, gdzie plik historii). „Wersja: —” = zmiana tylko w narzędziach autora."""
    m = re.search(r"^Wersja:\s*(\S+)", tresc, re.M)
    w = m.group(1) if m else None
    pl = jez == "pl"
    if w and re.fullmatch(r"\d+\.\d+\.\d+", w):
        if _wydane(w):
            gdzie = (f"Zmiana jest w wersji {w}: {REPO_GH}/releases/tag/v{w}" if pl else
                     f"The change is in version {w}: {REPO_GH}/releases/tag/v{w}")
            hist = f"{REPO_GH}/blob/v{w}/HISTORIA.md"
        else:
            gdzie = (f"Zmiana wejdzie do wersji {w} (wydanie wkrótce; program sam powiadomi o aktualizacji)."
                     if pl else f"The change goes into version {w} (release soon; the program will notify you).")
            hist = f"{REPO_GH}/blob/{_galaz()}/HISTORIA.md"
    else:
        gdzie = ("Zmiana dotyczy narzędzi po stronie autora — w programie bez zmian." if pl else
                 "This change is in the author's tools — the program itself is unchanged.")
        hist = f"{REPO_GH}/blob/{_galaz()}/HISTORIA.md"
    return (f"{gdzie}\nCała historia projektu: {hist} (wpis „{tytul}”)" if pl else
            f"{gdzie}\nFull project history: {hist} (entry “{tytul}”)")


def _bez_podpisu(szkic: str) -> str:
    """Szkice sprzed 27.10 kończą się „73, SQ2BVN — …” — podpis dokłada --odpisz."""
    akapity = szkic.strip().split("\n\n")
    if akapity and akapity[-1].lstrip().startswith("73"):
        akapity.pop()
    return "\n\n".join(akapity).strip()


def _znak(kat: Path) -> str:
    return kat.name.split("_", 2)[2].split("_")[0] if kat.name.count("_") >= 2 else ""


def do_wyslania(kat: Path, wyslane: dict, z_sekcjami: bool = True) -> tuple[str, str, str, list[str]] | None:
    """-> (adres, temat, treść, nowe tytuły sekcji) albo None, gdy nie ma czego wysłać.
    z_sekcjami=False: zgłoszenie nie jest najnowszym od tego znaku — wpisy z historii
    pójdą przy najnowszym (jeden wpis, jedna wiadomość)."""
    if not (kat / "nadawca.txt").exists():
        return None                                      # raport z pliku — bez adresu
    adres, temat = (kat / "nadawca.txt").read_text(encoding="utf-8").splitlines()[:2]
    meta = json.loads((kat / "raport.json").read_text(encoding="utf-8"))
    jez = meta.get("jezyk") if meta.get("jezyk") in PODPIS else "en"
    znak = _znak(kat)
    pierwsza = kat.name not in wyslane["odpowiedzi"]
    juz = set(wyslane["sekcje"].get(znak, []))
    nowe = [(t, x) for t, x in sekcje_historii(znak) if t not in juz] if z_sekcjami else []
    if not pierwsza and not nowe:
        return None
    czesci = [_bez_podpisu((kat / "odpowiedz.txt").read_text(encoding="utf-8"))] if pierwsza \
        else ["Cześć," if jez == "pl" else "Hello,"]
    if nowe:
        czesci.append(WSTEP_SEKCJI[jez][0 if pierwsza else 1])
        czesci += [f"--- {t} ---\n\n{x}\n\n{gdzie_zmiana(t, x, jez)}" for t, x in nowe]
    czesci.append(PODPIS[jez])
    return adres, ("" if temat.lower().startswith("re:") else "Re: ") + temat, \
        "\n\n".join(czesci), [t for t, _ in nowe]


def _thunderbird() -> Path | None:
    for baza in (os.environ.get("ProgramFiles", r"C:\Program Files"),
                 os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")):
        p = Path(baza) / "Mozilla Thunderbird" / "thunderbird.exe"
        if p.exists():
            return p
    return None


def _tozsamosc_projektu() -> str | None:
    """Id tożsamości Thunderbirda ze skrzynką projektu (adres, na który przychodzą
    raporty). Bez tego okno brało domyślne konto — 27.10 prywatne konto operatora."""
    from tools.aplikacja import KONTAKT_EMAIL
    for prefs in (Path(os.environ.get("APPDATA", "")) / "Thunderbird" / "Profiles").glob("*/prefs.js"):
        m = re.search(rf'"mail\.identity\.(id\d+)\.useremail",\s*"{re.escape(KONTAKT_EMAIL)}"',
                      prefs.read_text(encoding="utf-8", errors="replace"), re.I)
        if m:
            return m.group(1)
    return None


def otworz_wiadomosc(adres: str, temat: str, plik_tresci: Path) -> None:
    """Okno nowej wiadomości w Thunderbirdzie (bez wysyłania). Treść z pliku
    (message=) — w wierszu poleceń przecinki i apostrofy psułyby body=."""
    import subprocess
    tb = _thunderbird()
    if tb is None:
        raise SystemExit("Nie ma Thunderbirda w Program Files — treść jest w " + str(plik_tresci))
    temat = temat.replace("'", "’")
    tozs = _tozsamosc_projektu()
    od = f",preselectid='{tozs}'" if tozs else ""
    if not tozs:
        print("UWAGA: brak tożsamości skrzynki projektu w Thunderbirdzie — sprawdź pole „Nadawca”")
    subprocess.Popen([str(tb), "-compose",
                      f"to='{adres}',subject='{temat}',format='text',message='{plik_tresci}'{od}"])


def odpisz(nazwy: list[str], pokaz: bool, od_nowa: bool = False) -> None:
    plik = KATALOG / WYSLANE
    wyslane = json.loads(plik.read_text(encoding="utf-8")) if plik.exists() else {}
    wyslane.setdefault("odpowiedzi", {})
    wyslane.setdefault("sekcje", {})
    wyslane.setdefault("sekcje_przy", {})          # katalog -> tytuły wysłane w jego wiadomości
    katalogi = sorted(p.parent for p in KATALOG.glob("*/raport.json"))
    if nazwy:
        katalogi = [k for k in katalogi if any(n.upper() in k.name.upper() for n in nazwy)]
    najnowsze = {_znak(k): k for k in sorted(p.parent for p in KATALOG.glob("*/raport.json"))}
    if od_nowa:
        # operator 27.10: okno zamknięte krzyżykiem zamiast „Wyślij” — skrypt i tak
        # zapisał je jako wysłane. Tu zapominamy wybrane zgłoszenia i otwieramy jeszcze raz.
        if not nazwy:
            raise SystemExit("--od-nowa tylko z nazwą: --odpisz ZNAK_LUB_KATALOG --od-nowa")
        # sam znak -> tylko jego najnowsze zgłoszenie (starsze mogły dostać odpowiedź ręcznie)
        katalogi = [k for k in katalogi if najnowsze.get(_znak(k)) == k
                    or any(n.upper() == k.name.upper() for n in nazwy)]
        for kat in katalogi:
            wyslane["odpowiedzi"].pop(kat.name, None)
            tytuly = wyslane["sekcje_przy"].pop(kat.name, None)
            if tytuly is None and najnowsze.get(_znak(kat)) == kat:      # zapis sprzed sekcje_przy
                tytuly = wyslane["sekcje"].get(_znak(kat), [])
            wyslane["sekcje"][_znak(kat)] = [t for t in wyslane["sekcje"].get(_znak(kat), [])
                                             if t not in (tytuly or [])]
    otwarte = 0
    for kat in katalogi:
        w = do_wyslania(kat, wyslane, najnowsze.get(_znak(kat)) == kat)
        if w is None:
            continue
        adres, temat, tresc, tytuly = w
        # zapisane przy OTWARCIU okna — czy operator kliknął „Wyślij”, stąd nie widać
        wyslane["odpowiedzi"].setdefault(kat.name, time.strftime("%Y-%m-%d %H:%M"))
        wyslane["sekcje"].setdefault(_znak(kat), []).extend(tytuly)
        wyslane["sekcje_przy"].setdefault(kat.name, []).extend(tytuly)
        if pokaz:
            print(f"===== {kat.name} -> {temat}\n{tresc}\n")
            continue
        plik_tresci = kat / f"wyslij_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        plik_tresci.write_text(tresc, encoding="utf-8")
        otworz_wiadomosc(adres, temat, plik_tresci)
        plik.write_text(json.dumps(wyslane, ensure_ascii=False, indent=1), encoding="utf-8")
        otwarte += 1
        print(f"{kat.name}: okno wiadomości otwarte ({len(tytuly)} wpis(y) z historii)")
        time.sleep(2)                                # Thunderbird gubi okna otwierane naraz
    if not pokaz:
        print(f"otwartych wiadomości: {otwarte}" if otwarte else "nic nowego do wysłania")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("zrodla", type=Path, nargs="*", help="mbox Thunderbirda, katalog z ZIP-ami, ZIP")
    ap.add_argument("--katalog", type=Path, default=None, help="gdzie zapisywać (domyślnie zgloszenia/)")
    ap.add_argument("--ponow", action="store_true", help="analiza od nowa wszystkich przyjętych zgłoszeń")
    ap.add_argument("--odpisz", nargs="*", metavar="ZNAK_LUB_KATALOG",
                    help="okna odpowiedzi w Thunderbirdzie: szkic + nowe wpisy HISTORIA.md + podpis")
    ap.add_argument("--pokaz", action="store_true", help="z --odpisz: tylko wypisz treść, nic nie otwieraj")
    ap.add_argument("--od-nowa", action="store_true",
                    help="z --odpisz ZNAK: otwórz jeszcze raz (okno zamknięte bez wysłania)")
    a = ap.parse_args()
    global KATALOG
    if a.katalog:
        KATALOG = a.katalog
    if a.ponow:
        ponow()
        return
    if a.odpisz is not None:
        odpisz(a.odpisz, a.pokaz, a.od_nowa)
        return
    zrodla = a.zrodla or _folder_thunderbirda()
    if not zrodla:
        raise SystemExit(f"Brak folderu {FOLDER_TB} w profilu Thunderbirda — podaj ścieżkę do raportów.")
    przetworz(zrodla)


if __name__ == "__main__":
    main()
