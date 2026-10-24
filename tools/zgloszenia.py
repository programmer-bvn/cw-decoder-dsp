"""ZGŁOSZENIA — raporty „Zgłoś problem z dekodowaniem” z aplikacji (25.10).

Obieg (operator 25.10): skrzynka Gmail projektu w Thunderbirdzie na tej
maszynie, reguła „temat zawiera [CW-DECODER]” -> folder CW-DECODER. Thunderbird
trzyma kopie wiadomości z załącznikami w plikach profilu (mbox) — skrypt czyta
je wprost, bez logowania i bez haseł (zostają w Thunderbirdzie, u operatora).

    python -m tools.zgloszenia                      # folder CW-DECODER z profilu Thunderbirda
    python -m tools.zgloszenia ŚCIEŻKA [...]        # plik mbox, katalog z ZIP-ami albo ZIP

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
    return [p for p in baza.glob(f"*/*Mail/**/{FOLDER_TB}") if p.is_file()]


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
    dryf = float(np.nanpercentile(f, 95) - np.nanpercentile(f, 5)) if np.isfinite(f).sum() > 10 else float("nan")
    obce, nasza_db = [], None
    if np.isfinite(ton):
        x = np.asarray(surowe, np.float64)
        # poziom naszej w tej samej skali co obce: szczyty bez wykluczenia okolicy tonu
        nasza = [g for fo, g in PS.obce_stacje(x, ton + 1000.0, C.SR) if abs(fo - ton) < PS.ODSTEP_HZ]
        nasza_db = max(nasza) if nasza else None
        # względem naszej; liczą się od POZIOM_MIN_DB (-12 dB, jak przeciwsygnał) — słabsze
        # to zwykle prążki własnego kluczowania (6 m spokój: -24 dB, ręka: -32 dB; 40 m tłok: -3 dB)
        obce = [(fo, g - nasza_db) for fo, g in PS.obce_stacje(x, ton, C.SR)
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
    nad_txt = meta.get("nadane", "").strip().upper()
    if nad_txt:
        wynik["bledy_znakow"] = cer(r["tekst"].replace(" ", ""), nad_txt.replace(" ", ""))
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
           "co słychać na paśmie.\n\n73, SQ2BVN — Dekoder CW\n"),
    "en": ("Thank you for the CW Decoder report{znak}! Recordings like this are the most valuable "
           "thing for the project — they show signals that cannot be collected in one place.\n\n"
           "What the recording shows: {diag}.\n\n"
           "The recording goes into the test set; I will let you know when a new version reads it better. "
           "You can also check yourself: install the new version next to the old one and open the same "
           "recording with \"Open recording…\".\n\n"
           "Please keep sending such reports — each one brings the decoder closer to reading everything "
           "you can hear on the band.\n\n73, SQ2BVN — CW Decoder\n"),
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
            (kat / "analiza.json").write_text(json.dumps(w, ensure_ascii=False, indent=1), encoding="utf-8")
            (kat / "analiza.md").write_text(
                f"# {kat.name}\n\nwersja: {meta.get('wersja')}  \nznak: {meta.get('znak') or '-'}  \n"
                f"nadane: {meta.get('nadane') or '-'}  \nuwagi: {meta.get('opis') or '-'}  \n"
                f"pasmo: {meta.get('z_sesji', {}).get('pasmo', '-')}\n\n"
                f"**odczyt bieżącą wersją:** {an.get('tekst', '')[:600]}\n\n"
                f"**diagnoza:** {'; '.join(an['diagnoza'])}\n\n```\n"
                + json.dumps({k: v for k, v in an.items() if k not in ('tekst', 'diagnoza')}, ensure_ascii=False, indent=1)
                + "\n```\n", encoding="utf-8")
            tresc = odpowiedz(meta, an)
            (kat / "odpowiedz.txt").write_text(tresc, encoding="utf-8")
            with open(KATALOG / "PODSUMOWANIE.md", "a", encoding="utf-8") as f:
                f.write(f"- **{kat.name}** {meta.get('wersja', '?').split(' ')[0]}, {an.get('czas_s', '?')} s: "
                        f"{'; '.join(an['diagnoza'])} — odczyt: `{an.get('tekst', '')[:120]}`\n")
            _dopisz_odpowiedz(kat.name, nadawca, temat, tresc)
            nowe += 1
            print(f"{kat.name}: {'; '.join(an['diagnoza'])}")
        przetw[klucz] = time.strftime("%Y-%m-%d %H:%M:%S")
        plik_przetw.write_text(json.dumps(przetw, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"nowych zgłoszeń: {nowe}")
    return nowe


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("zrodla", type=Path, nargs="*", help="mbox Thunderbirda, katalog z ZIP-ami, ZIP")
    ap.add_argument("--katalog", type=Path, default=None, help="gdzie zapisywać (domyślnie zgloszenia/)")
    a = ap.parse_args()
    global KATALOG
    if a.katalog:
        KATALOG = a.katalog
    zrodla = a.zrodla or _folder_thunderbirda()
    if not zrodla:
        raise SystemExit(f"Brak folderu {FOLDER_TB} w profilu Thunderbirda — podaj ścieżkę do raportów.")
    przetworz(zrodla)


if __name__ == "__main__":
    main()
