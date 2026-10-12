"""PRZEGLĄDARKA X-RAY do oceny okiem: dane treningowe, partie, nagrania.

PO CO. 12.10 operator obejrzał klipy generatora i stwierdził, że część
„nie występuje w realnym świecie". Telegrafista rozpozna nierealny obraz
szybciej niż jakakolwiek miara — ale żeby z tego wyszła poprawka, ocena
musi trafić do pliku razem z parametrami klipu. I musi dać się porównać
z tym, co przychodzi z radia: dlatego jedna przeglądarka, jedna skala,
jedna droga obrazu dla każdego źródła.

Źródła (rozpoznawane po zawartości):

    partia do oceny     out/ocena/partia2.npz      (tools.xray_ocena generuj)
    zbiór treningowy    czesci_sn/morse_200000_00.npz  (--limit, domyślnie 2000)
    nagranie            probki/nagranie.wav  — tą samą drogą co przy odczycie
                        (przestrojenie + tłumik, wav2net.wav_to_obraz);
                        także wyjście mic2wav

    python -m tools.xray_ocena przegladaj out/ocena/partia2.npz
    python -m tools.xray_ocena przegladaj czesci_rt/morse_200000_00.npz
    python -m tools.xray_ocena przegladaj "probki/Nagrywanie (6).wav"
    # dwie kopie obok siebie, mniejsze siatki:
    python -m tools.xray_ocena przegladaj A.npz --kol 2 --pozycja 0,30
    python -m tools.xray_ocena przegladaj B.wav --kol 2 --pozycja 960,30

Siatka: klik = „zły" tak/nie. Strzałki / PgUp / PgDn / spacja — strony.
I — parametry. E — etykiety (domyślnie widoczne). Prawy klik — powiększenie.
Esc — koniec.

Pod obrazkiem pasek etykiety: każdy znak w swoich ramkach, CEL (znak
etykiety, środkowy) na żółto, znak UCIĘTY krawędzią okna na czerwono.
Ucięty znak to niejednoznaczność, którą wskazał operator 12.10: pół kreski
na początku okna to równie dobrze końcówka „T", „M", „O" czy „0" —
a frame_labels daje mu pełną klasę.

Powiększenie: kursor (mysz, klik i przeciąganie, strzałki; Shift = co 8),
suwaki czasu i pasma, odczyt w dB, profil dB wzdłuż czasu w paśmie
kursora i wzdłuż pasm w ramce kursora, skala barw ze znacznikiem.
Z — „zły". PgUp / PgDn — poprzedni / następny obrazek. Esc / Backspace /
prawy klik — powrót do siatki.

Ocena: `out/ocena/<nazwa źródła>_ocena.json`, po każdej zmianie.

Skala barw stała, jak w tools/xray.py: czarne = DB_MIN, białe = DB_MAX.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsp import config as C

KATALOG_OCEN = C.OUT_DIR / "ocena"
LUT_PLIK = Path(__file__).with_name("xray_magma.npy")


def _j(v):
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else round(float(v), 3)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


# --------------------------------------------------------------------------
# 1. generowanie partii
# --------------------------------------------------------------------------
def generuj(n: int, n_nagr: int, out: Path, seed: int) -> None:
    from dsp import frontend, radio

    rng = np.random.default_rng(seed)
    obrazy, opisy = [], []
    bank = radio.bank_szumu()
    for _ in range(n):
        tekst = ("".join(rng.choice(list(C.ALPHABET[1:]), C.CHARS_PER_CLIP))
                 if rng.random() > C.SILENCE_FRACTION else "")
        audio, m = radio.receive(tekst, rng)
        obrazy.append(frontend.to_net_image(audio))
        o = {k: v for k, v in m.items() if k not in ("spans",)}
        o["zrodlo"] = "generator"
        o["plik_szumu"] = bank[m["szum_real"]][0] if m["szum_real"] >= 0 else ""
        # Granice znaków w ramkach OKNA (mogą wychodzić poza 0..IMG_FRAMES —
        # to właśnie znaki ucięte krawędzią).
        o["znaki"] = [
            [tekst[tag],
             round(frontend.sample_to_window_frame(float(a) + m["offset"],
                                                   audio.size), 2),
             round(frontend.sample_to_window_frame(float(b) + m["offset"],
                                                   audio.size), 2),
             tag == C.LABEL_INDEX]
            for tag, a, b in m["spans"] if 0 <= tag < len(tekst)]
        opisy.append(o)

    if n_nagr > 0:
        from tools import wav2net
        cale = {}
        for p in sorted((C.ROOT / "probki").glob("*.wav")):
            obraz, g, _ = wav2net.wav_to_obraz(p, auto_gain=True, retune=True)
            if obraz.shape[0] > C.IMG_FRAMES:
                cale[p.name] = (obraz, g)
        nazwy = sorted(cale)
        for k in range(n_nagr):
            nazwa = nazwy[k % len(nazwy)]
            obraz, g = cale[nazwa]
            s = int(rng.integers(0, obraz.shape[0] - C.IMG_FRAMES))
            obrazy.append(obraz[s:s + C.IMG_FRAMES])
            opisy.append({"zrodlo": "nagranie", "plik": nazwa,
                          "start_s": round(s * C.HOP_LENGTH / C.SR, 2),
                          "korekta_db": round(g, 1)})

    kolejnosc = rng.permutation(len(obrazy))
    X = np.stack([obrazy[i] for i in kolejnosc])
    X = np.clip(np.round(X * 255.0), 0, 255).astype(np.uint8)
    opisy = [opisy[i] for i in kolejnosc]
    tekst_opisow = json.dumps(
        [{k: (_j(v) if not isinstance(v, list) else v) for k, v in o.items()}
         for o in opisy], ensure_ascii=False, default=_j)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, opisy=np.array(tekst_opisow),
                        db=np.array([C.DB_MIN, C.DB_MAX]))
    print(f"zapisane {out}: {n} z generatora + {n_nagr} okien nagrań, "
          f"wymieszane")


# --------------------------------------------------------------------------
# 2. źródła obrazów — jedna skala, jedna postać
# --------------------------------------------------------------------------
class Zrodlo:
    """obraz(i) -> float [IMG_FRAMES, IMG_BINS] w [0,1]; znaki(i) -> lista
    [znak, ramka_od, ramka_do, czy_cel]; opis(i) -> słownik; t0(i) -> s."""
    nazwa = ""
    naglowek = ""
    n = 0

    def obraz(self, i): raise NotImplementedError
    def znaki(self, i): return None
    def opis(self, i): return {}
    def t0(self, i): return 0.0
    def etykieta(self, i): return ""


class ZrodloPartia(Zrodlo):
    def __init__(self, p: Path):
        d = np.load(p)
        self.X = d["X"]
        self.opisy = json.loads(str(d["opisy"]))
        self.n = len(self.X)
        self.nazwa = p.stem
        self.naglowek = f"partia {p.name}"

    def obraz(self, i): return self.X[i].astype(np.float32) / 255.0
    def opis(self, i): return self.opisy[i]

    def znaki(self, i):
        o = self.opisy[i]
        if o.get("zrodlo") != "generator" or "znaki" not in o:
            return None
        return o["znaki"]

    def t0(self, i):
        return float(self.opisy[i].get("start_s", 0.0))

    def etykieta(self, i):
        o = self.opisy[i]
        if o.get("zrodlo") == "nagranie":
            return ""
        t = o.get("text", "")
        return (f'nadane "{t}"  cel "{t[C.LABEL_INDEX]}"' if t
                else "puste radio (cel: brak znaku)")


def _czytaj_poczatek(npz: Path, nazwa: str, k: int) -> np.ndarray:
    """Pierwsze k wierszy tablicy z .npz bez rozpakowywania całości."""
    with zipfile.ZipFile(npz) as z, z.open(nazwa + ".npy") as f:
        wersja = np.lib.format.read_magic(f)
        if wersja == (1, 0):
            ksztalt, fortran, dt = np.lib.format.read_array_header_1_0(f)
        else:
            ksztalt, fortran, dt = np.lib.format.read_array_header_2_0(f)
        k = min(k, ksztalt[0])
        wiersz = int(np.prod(ksztalt[1:], dtype=np.int64)) * dt.itemsize
        bufor = f.read(k * wiersz)
        return np.frombuffer(bufor, dtype=dt).reshape((k,) + tuple(ksztalt[1:]))


class ZrodloZbior(Zrodlo):
    """Część zbioru treningowego: dokładnie to, co dostaje sieć."""

    def __init__(self, p: Path, limit: int):
        C.check_fingerprint(str(np.load(p)["fingerprint"]), source=p.name)
        self.X = _czytaj_poczatek(p, "X", limit)
        self.n = len(self.X)
        self.yf = _czytaj_poczatek(p, "yf", self.n)
        self.text = _czytaj_poczatek(p, "text", self.n)
        d = np.load(p)
        self.meta = {k: d[k][:self.n] for k in d.files
                     if k not in ("X", "y", "yf", "text", "fingerprint",
                                  "dtype_note", "meta", "kod")}
        self.nazwa = f"{p.parent.name}_{p.stem}"
        self.naglowek = f"zbiór {p.parent.name}/{p.name} (pierwsze {self.n})"

    def obraz(self, i): return self.X[i].astype(np.float32) / 255.0
    def opis(self, i):
        o = {k: _j(v[i]) for k, v in self.meta.items()}
        o["text"] = str(self.text[i])
        return o

    def znaki(self, i):
        """Odcinki z etykiet ramek (yf) — dokładnie to, czego uczy się fcn."""
        y = self.yf[i]
        t = str(self.text[i])
        cel_id = C.CHAR_TO_ID.get(t[C.LABEL_INDEX], -1) if t else -1
        wynik, k = [], 0
        while k < len(y):
            if y[k] == 0:
                k += 1
                continue
            j = k
            while j < len(y) and y[j] == y[k]:
                j += 1
            # yf nie zna granic poza oknem: odcinek przy krawędzi dostaje
            # granicę -0,5 / 128,5, żeby rysował się jako ucięty.
            od = -0.5 if k == 0 else float(k)
            do = len(y) + 0.5 if j == len(y) else float(j)
            wynik.append([C.ALPHABET[y[k]], od, do, int(y[k]) == cel_id])
            k = j
        # Gdy cel się powtarza w tekście (np. „EEE"), żółty tylko najbliższy
        # środka okna.
        cele = [w for w in wynik if w[3]]
        if len(cele) > 1:
            naj = min(cele, key=lambda w: abs(0.5 * (w[1] + w[2])
                                              - C.IMG_FRAMES / 2))
            for w in cele:
                w[3] = w is naj
        return wynik

    def etykieta(self, i):
        t = str(self.text[i])
        return (f'nadane "{t}"  cel "{t[C.LABEL_INDEX]}"' if t
                else "puste radio (cel: brak znaku)")


class ZrodloWav(Zrodlo):
    """Nagranie przez TĘ SAMĄ drogę co przy odczycie (wav2net)."""

    def __init__(self, p: Path, krok: int):
        from tools import wav2net
        obraz, g, lock = wav2net.wav_to_obraz(p, auto_gain=True, retune=True)
        if obraz.shape[0] < C.IMG_FRAMES:
            obraz = np.pad(obraz, ((0, C.IMG_FRAMES - obraz.shape[0]), (0, 0)))
        self.cale = obraz
        self.starty = list(range(0, obraz.shape[0] - C.IMG_FRAMES + 1, krok))
        self.n = len(self.starty)
        self.nazwa = p.stem
        opis_p = p.with_suffix(".txt")
        nadane = ""
        if opis_p.exists():
            for linia in opis_p.read_text(encoding="utf-8").splitlines():
                if linia.startswith("nadane:"):
                    nadane = linia.split(":", 1)[1].strip()
        self.naglowek = (f"nagranie {p.name}  korekta {g:+.1f} dB  "
                         + (f"nadane: {nadane[:60]}" if nadane else ""))
        self.lock = lock.splitlines()[0] if lock else ""

    def obraz(self, i):
        s = self.starty[i]
        return self.cale[s:s + C.IMG_FRAMES].astype(np.float32)

    def t0(self, i):
        return self.starty[i] * C.HOP_LENGTH / C.SR

    def opis(self, i):
        return {"zrodlo": "nagranie", "od_s": round(self.t0(i), 2),
                "ton": self.lock}

    def etykieta(self, i):
        return f"od {self.t0(i):.2f} s"


def otworz(p: Path, limit: int, krok: int) -> Zrodlo:
    if p.suffix.lower() in (".wav", ".flac", ".ogg", ".mp3"):
        return ZrodloWav(p, krok)
    with zipfile.ZipFile(p) as z:
        pliki = set(z.namelist())
    if "opisy.npy" in pliki:
        return ZrodloPartia(p)
    if "yf.npy" in pliki:
        return ZrodloZbior(p, limit)
    raise SystemExit(f"{p}: nie rozpoznaję — ani partia, ani część zbioru "
                     f"(brak opisy.npy / yf.npy)")


# --------------------------------------------------------------------------
# 3. przegląd
# --------------------------------------------------------------------------
KAF_W, KAF_H = 320, 130
PASEK_H = 16                     # pasek etykiety pod obrazkiem
POLE_H = PASEK_H + 92            # pasek + pole „zły" + etykieta + parametry
C_CEL = (250, 205, 60)
C_ZNAK = (150, 150, 165)
C_UCIETY = (235, 60, 60)


def _plik_oceny(nazwa: str) -> Path:
    return KATALOG_OCEN / f"{nazwa}_ocena.json"


def _db(v):
    return C.DB_MIN + v * (C.DB_MAX - C.DB_MIN)


def przegladaj(p: Path, kol: int = 4, wier: int = 4, limit: int = 2000,
               krok: int = 64, pozycja: str | None = None) -> None:
    zr = otworz(p, limit, krok)
    if pozycja:
        os.environ["SDL_VIDEO_WINDOW_POS"] = pozycja
    import pygame
    from dsp import melref

    lut = np.load(LUT_PLIK)
    srodki_hz = melref.mel_frequencies(C.N_MELS, C.FMIN, C.FMAX)[1:-1]
    n = zr.n
    na_strone = kol * wier
    stron = max(1, (n + na_strone - 1) // na_strone)

    plik = _plik_oceny(zr.nazwa)
    zle, obejrzane = set(), set()
    if plik.exists():
        z = json.loads(plik.read_text(encoding="utf-8"))
        zle, obejrzane = set(z.get("zle", [])), set(z.get("obejrzane", []))

    def zapisz():
        KATALOG_OCEN.mkdir(parents=True, exist_ok=True)
        plik.write_text(json.dumps(
            {"zrodlo": str(p), "zle": sorted(zle),
             "obejrzane": sorted(obejrzane), "n": n, "na_strone": na_strone,
             "kiedy": time.strftime("%Y-%m-%d %H:%M:%S")},
            ensure_ascii=False, indent=1), encoding="utf-8")

    zapisz()                      # plik oceny istnieje od startu przeglądu
    pygame.init()
    W = max(kol * (KAF_W + 16) + 16, 940)
    H = 46 + wier * (KAF_H + POLE_H) + 10
    ekran = pygame.display.set_mode((W, H))
    pygame.display.set_caption(f"X-Ray — {zr.naglowek}")
    f_dzy = pygame.font.SysFont("consolas", 17)
    f_sr = pygame.font.SysFont("consolas", 14)
    f_mal = pygame.font.SysFont("consolas", 12)

    def powierzchnia(img, w, h):
        idx = np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)
        rgb = lut[idx][:, ::-1, :]               # niskie pasma na dole
        s = pygame.surfarray.make_surface(np.ascontiguousarray(rgb))
        return pygame.transform.scale(s, (w, h))

    def pasek_etykiety(i, x, y, w, h, czcionka):
        """Znaki w ich ramkach; cel żółty, ucięty krawędzią czerwony."""
        pygame.draw.rect(ekran, (40, 40, 46), (x, y, w, h))
        znaki = zr.znaki(i)
        if znaki is None:
            return []
        uciete = []
        for zn, od, do, cel in znaki:
            if do <= 0 or od >= C.IMG_FRAMES:
                continue                         # cały poza oknem
            uciety = od < 0 or do > C.IMG_FRAMES
            a = max(0.0, od) / C.IMG_FRAMES * w
            b = min(float(C.IMG_FRAMES), do) / C.IMG_FRAMES * w
            kolor = C_UCIETY if uciety else (C_CEL if cel else C_ZNAK)
            pygame.draw.rect(ekran, kolor,
                             (x + int(a), y + 2, max(2, int(b - a)), h - 4))
            t = czcionka.render(zn, True, (0, 0, 0))
            if b - a >= t.get_width() + 2:
                ekran.blit(t, (x + int(0.5 * (a + b)) - t.get_width() // 2,
                               y + (h - t.get_height()) // 2))
            if uciety:
                uciete.append(f"{zn}({'lewa' if od < 0 else 'prawa'})")
        return uciete

    def krotki_opis(o):
        if o.get("zrodlo") == "nagranie":
            return [f"NAGRANIE {str(o.get('plik', ''))[:34]}",
                    f"od {o.get('start_s', o.get('od_s'))} s"]
        sn = o.get("snr_db")
        sr_ = o.get("szum_real")
        return [f"{'BANK' if sr_ is not None and sr_ >= 0 else 'syn'} "
                f"S/N {sn if sn is not None else '-'}  tło {o.get('tlo_db')}",
                f"ARW {o.get('agc_tau')}  qsb {o.get('qsb')}  "
                f"pk {o.get('peak')}",
                f"qrm {o.get('qrm')}  qrn {o.get('qrn')}  wpm {o.get('wpm')}"]

    # --- stan ---
    strona, info, etyk, powieksz = 0, False, True, None
    kur = [C.IMG_FRAMES // 2, C.N_MELS // 2]      # ramka, pasmo
    ciagnij = None                                 # "obraz" | "czas" | "pasmo"
    prostokaty = []
    geom = {}
    zegar = pygame.time.Clock()

    def rysuj_siatke():
        obejrzane.add(strona)
        ucietych_na_stronie = 0
        naglowek = (f"{zr.naglowek[:70]}   strona {strona + 1}/{stron}   "
                    f"złych {len(zle)}/{n}")
        ekran.blit(f_dzy.render(naglowek, True, (230, 230, 230)), (16, 6))
        ekran.blit(f_mal.render(
            "[klik = zły] [prawy = powiększ] [I = parametry] [E = etykiety] "
            "[strzałki/spacja] [Esc]", True, (150, 150, 150)), (16, 27))
        prostokaty.clear()
        for k in range(na_strone):
            i = strona * na_strone + k
            if i >= n:
                break
            x = 16 + (k % kol) * (KAF_W + 16)
            y = 46 + (k // kol) * (KAF_H + POLE_H)
            ekran.blit(powierzchnia(zr.obraz(i), KAF_W, KAF_H), (x, y))
            zly = i in zle
            if zly:
                pygame.draw.rect(ekran, (230, 40, 40),
                                 (x - 3, y - 3, KAF_W + 6, KAF_H + 6), 3)
            yy = y + KAF_H + 2
            uciete = []
            if etyk:
                uciete = pasek_etykiety(i, x, yy, KAF_W, PASEK_H, f_mal)
                ucietych_na_stronie += bool(uciete)
            yy += PASEK_H + 3
            pole = pygame.Rect(x, yy, 16, 16)
            pygame.draw.rect(ekran, (230, 230, 230), pole, 2)
            if zly:
                pygame.draw.line(ekran, (230, 40, 40), pole.topleft,
                                 pole.bottomright, 3)
                pygame.draw.line(ekran, (230, 40, 40), pole.topright,
                                 pole.bottomleft, 3)
            ekran.blit(f_sr.render(f"zły  #{i + 1}", True,
                                   (240, 90, 90) if zly else (200, 200, 200)),
                       (x + 22, yy))
            yy += 19
            if etyk:
                linia = zr.etykieta(i)
                if uciete:
                    linia += "  ucięty: " + ", ".join(uciete)
                ekran.blit(f_mal.render(linia[:52], True,
                                        C_UCIETY if uciete else (210, 210, 170)),
                           (x, yy))
            yy += 15
            if info:
                for r, l in enumerate(krotki_opis(zr.opis(i))):
                    ekran.blit(f_mal.render(l[:52], True, (165, 165, 165)),
                               (x, yy + 14 * r))
            prostokaty.append((pygame.Rect(x, y, KAF_W, KAF_H + PASEK_H + 24),
                               i))
        if etyk and not isinstance(zr, ZrodloWav):
            ekran.blit(f_mal.render(
                f"z uciętym znakiem na tej stronie: {ucietych_na_stronie}",
                True, C_UCIETY), (W - 330, 27))

    def rysuj_powiekszenie(i):
        img = zr.obraz(i)
        db = _db(img)
        iw, ih = W - 200, int((H - 46) * 0.42)
        ix, iy = 16, 44
        ekran.blit(f_dzy.render(
            f"#{i + 1}/{n}  {'ZŁY' if i in zle else '-'}   {zr.etykieta(i)}",
            True, (230, 230, 230)), (16, 8))
        ekran.blit(f_mal.render(
            "[mysz/strzałki = kursor, Shift = co 8] [Z = zły] [PgUp/PgDn = "
            "obrazek] [Esc = powrót]", True, (150, 150, 150)), (16, 26))
        ekran.blit(powierzchnia(img, iw, ih), (ix, iy))
        fw, bh = iw / C.IMG_FRAMES, ih / C.N_MELS
        cx = ix + (kur[0] + 0.5) * fw
        cy = iy + ih - (kur[1] + 0.5) * bh
        pygame.draw.line(ekran, (0, 255, 255), (cx, iy), (cx, iy + ih), 1)
        pygame.draw.line(ekran, (0, 255, 255), (ix, cy), (ix + iw, cy), 1)
        pygame.draw.rect(ekran, (0, 255, 255),
                         (int(cx - fw / 2), int(cy - bh / 2),
                          int(fw) + 1, int(bh) + 1), 2)
        geom["obraz"] = pygame.Rect(ix, iy, iw, ih)
        # suwak pasma (pionowy) obok obrazu
        sx = ix + iw + 10
        geom["pasmo"] = pygame.Rect(sx, iy, 14, ih)
        pygame.draw.rect(ekran, (70, 70, 80), geom["pasmo"])
        pygame.draw.rect(ekran, (0, 255, 255),
                         (sx - 2, int(cy) - 5, 18, 10))
        # skala barw ze znacznikiem
        bx = sx + 30
        for k in range(ih):
            v = 1.0 - k / max(1, ih - 1)
            pygame.draw.line(ekran, tuple(int(c) for c in lut[int(v * 255)]),
                             (bx, iy + k), (bx + 18, iy + k))
        for d in range(int(C.DB_MIN), int(C.DB_MAX) + 1, 6):
            yk = iy + ih - (d - C.DB_MIN) / (C.DB_MAX - C.DB_MIN) * ih
            pygame.draw.line(ekran, (200, 200, 200), (bx + 18, yk), (bx + 24, yk))
            ekran.blit(f_mal.render(f"{d:+d}", True, (200, 200, 200)),
                       (bx + 27, yk - 6))
        v = float(db[kur[0], kur[1]])
        yv = iy + ih - (v - C.DB_MIN) / (C.DB_MAX - C.DB_MIN) * ih
        pygame.draw.polygon(ekran, (0, 255, 255),
                            [(bx - 8, yv - 5), (bx - 8, yv + 5), (bx - 1, yv)])
        # pasek etykiety pod obrazem i suwak czasu
        uciete = pasek_etykiety(i, ix, iy + ih + 4, iw, 22, f_sr)
        ty = iy + ih + 32
        geom["czas"] = pygame.Rect(ix, ty, iw, 14)
        pygame.draw.rect(ekran, (70, 70, 80), geom["czas"])
        pygame.draw.rect(ekran, (0, 255, 255), (int(cx) - 5, ty - 2, 10, 18))
        # odczyt
        t = zr.t0(i) + kur[0] * C.HOP_LENGTH / C.SR
        wiersz = db[:, kur[1]]
        odczyt = (f"t = {t:6.2f} s (ramka {kur[0]:3d})   pasmo {kur[1]:2d}: "
                  f"{srodki_hz[kur[1]]:5.0f} Hz   POZIOM {v:+6.1f} dB")
        ekran.blit(f_dzy.render(odczyt, True, (0, 255, 255)), (16, ty + 22))
        druga = (f"w tym paśmie: mediana {np.median(wiersz):+.1f} dB, "
                 f"maks {wiersz.max():+.1f} dB")
        ekran.blit(f_sr.render(druga, True, (0, 200, 200)), (16, ty + 46))
        if uciete:
            ekran.blit(f_sr.render("   ucięty krawędzią: " + ", ".join(uciete),
                                   True, C_UCIETY),
                       (16 + f_sr.size(druga)[0], ty + 46))
        # profile: czas w paśmie kursora | pasma w ramce kursora
        py, ph = ty + 70, H - (ty + 70) - 86
        def wykres(x0, w, wart, os_x, znacznik, podpis):
            pygame.draw.rect(ekran, (34, 34, 40), (x0, py, w, ph))
            for d in range(int(C.DB_MIN), int(C.DB_MAX) + 1, 6):
                yk = py + ph - (d - C.DB_MIN) / (C.DB_MAX - C.DB_MIN) * ph
                pygame.draw.line(ekran, (60, 60, 70), (x0, yk), (x0 + w, yk))
                ekran.blit(f_mal.render(f"{d:+d}", True, (130, 130, 130)),
                           (x0 + 2, max(py, yk - 12)))
            pkt = [(x0 + (k + 0.5) / len(wart) * w,
                    py + ph - (float(np.clip(d, C.DB_MIN, C.DB_MAX)) - C.DB_MIN)
                    / (C.DB_MAX - C.DB_MIN) * ph) for k, d in enumerate(wart)]
            pygame.draw.lines(ekran, (250, 170, 60), False, pkt, 2)
            xm = x0 + (znacznik + 0.5) / len(wart) * w
            pygame.draw.line(ekran, (0, 255, 255), (xm, py), (xm, py + ph), 1)
            ekran.blit(f_mal.render(podpis, True, (200, 200, 200)),
                       (x0 + 40, py + 2))
            ekran.blit(f_mal.render(os_x, True, (130, 130, 130)),
                       (x0 + 2, py + ph + 2))
        wykres(16, iw - 300, wiersz, "czas →", kur[0],
               f"poziom w czasie, pasmo {srodki_hz[kur[1]]:.0f} Hz [dB]")
        wykres(iw - 270, 270 + 60, db[kur[0], :],
               f"{C.FMIN:.0f} Hz → {C.FMAX:.0f} Hz", kur[1],
               f"widmo w ramce {kur[0]} [dB]")
        # opis
        o = zr.opis(i)
        pary = [f"{k}={v}" for k, v in o.items() if k not in ("text", "znaki")]
        linie, akt = [], ""
        for pz in pary:
            if len(akt) + len(pz) > 180:
                linie.append(akt)
                akt = ""
            akt += pz + "  "
        linie.append(akt)
        for r, l in enumerate(linie[:3]):
            ekran.blit(f_mal.render(l, True, (160, 160, 160)),
                       (16, H - 58 + 16 * r))

    def kursor_z_myszy(pos, co):
        r = geom.get(co)
        if r is None:
            return
        if co in ("obraz", "czas"):
            kur[0] = int(np.clip((pos[0] - r.x) / r.w * C.IMG_FRAMES,
                                 0, C.IMG_FRAMES - 1))
        if co in ("obraz", "pasmo"):
            kur[1] = int(np.clip((r.bottom - pos[1]) / r.h * C.N_MELS,
                                 0, C.N_MELS - 1))

    while True:
        ekran.fill((24, 24, 28))
        if powieksz is None:
            rysuj_siatke()
        else:
            rysuj_powiekszenie(powieksz)
        pygame.display.flip()

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                zapisz()
                pygame.quit()
                return
            if powieksz is not None:
                if ev.type == pygame.KEYDOWN:
                    shift = bool(ev.mod & pygame.KMOD_SHIFT)
                    d = 8 if shift else 1
                    if ev.key in (pygame.K_ESCAPE, pygame.K_BACKSPACE):
                        powieksz = None
                    elif ev.key == pygame.K_LEFT:
                        kur[0] = max(0, kur[0] - d)
                    elif ev.key == pygame.K_RIGHT:
                        kur[0] = min(C.IMG_FRAMES - 1, kur[0] + d)
                    elif ev.key == pygame.K_UP:
                        kur[1] = min(C.N_MELS - 1, kur[1] + d)
                    elif ev.key == pygame.K_DOWN:
                        kur[1] = max(0, kur[1] - d)
                    elif ev.key == pygame.K_z:
                        zle.symmetric_difference_update({powieksz})
                        zapisz()
                    elif ev.key == pygame.K_PAGEDOWN:
                        powieksz = min(n - 1, powieksz + 1)
                    elif ev.key == pygame.K_PAGEUP:
                        powieksz = max(0, powieksz - 1)
                elif ev.type == pygame.MOUSEBUTTONDOWN:
                    if ev.button == 3:
                        powieksz = None
                        continue
                    if ev.button == 1:
                        for co in ("obraz", "czas", "pasmo"):
                            if geom.get(co) and geom[co].inflate(0, 8
                                                                 ).collidepoint(ev.pos):
                                ciagnij = co
                                kursor_z_myszy(ev.pos, co)
                                break
                elif ev.type == pygame.MOUSEBUTTONUP:
                    ciagnij = None
                elif ev.type == pygame.MOUSEMOTION and ciagnij:
                    kursor_z_myszy(ev.pos, ciagnij)
                continue

            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    zapisz()
                    pygame.quit()
                    return
                if ev.key in (pygame.K_RIGHT, pygame.K_PAGEDOWN, pygame.K_SPACE):
                    strona = min(stron - 1, strona + 1)
                elif ev.key in (pygame.K_LEFT, pygame.K_PAGEUP):
                    strona = max(0, strona - 1)
                elif ev.key == pygame.K_i:
                    info = not info
                elif ev.key == pygame.K_e:
                    etyk = not etyk
                zapisz()
            if ev.type == pygame.MOUSEBUTTONDOWN:
                for r, i in prostokaty:
                    if r.collidepoint(ev.pos):
                        if ev.button == 1:
                            zle.symmetric_difference_update({i})
                            zapisz()
                        elif ev.button == 3:
                            powieksz = i
        zegar.tick(30)


# --------------------------------------------------------------------------
# 4. podsumowanie oceny partii
# --------------------------------------------------------------------------
def podsumuj(partia: Path) -> None:
    d = np.load(partia)
    opisy = json.loads(str(d["opisy"]))
    plik = _plik_oceny(partia.stem)
    if not plik.exists():
        print(f"Brak oceny: {plik} nie istnieje — partia nie była jeszcze "
              f"przeglądana. Najpierw:\n"
              f"    python -m tools.xray_ocena przegladaj {partia}")
        sys.exit(1)
    z = json.loads(plik.read_text(encoding="utf-8"))
    zle = set(z["zle"])
    na_strone = z.get("na_strone", 16)
    ob = {i for s in z["obejrzane"] for i in range(s * na_strone,
                                                    (s + 1) * na_strone)}
    ob = {i for i in ob if i < len(opisy)}
    print(f"obejrzanych {len(ob)}/{len(opisy)}, złych {len(zle)}")

    nagr = [i for i in ob if opisy[i]["zrodlo"] == "nagranie"]
    gen = [i for i in ob if opisy[i]["zrodlo"] == "generator"]
    print(f"  nagrania (kontrola oka): złych {sum(i in zle for i in nagr)}"
          f"/{len(nagr)}"
          + "".join(f"\n      #{i + 1} {opisy[i]['plik']}" for i in nagr
                    if i in zle))
    print(f"  generator:               złych {sum(i in zle for i in gen)}"
          f"/{len(gen)}")

    def udzial(warunek):
        a = [i for i in gen if warunek(opisy[i])]
        if not a:
            return "   -"
        return (f"{100 * sum(i in zle for i in a) / len(a):5.0f}% "
                f"({sum(i in zle for i in a)}/{len(a)})")

    def uciety(o):
        return any(od < 0 < do or od < C.IMG_FRAMES < do
                   for _, od, do, _ in o.get("znaki", []))
    print("\n  odsetek złych w grupie:")
    grupy = [
        ("tło z banku", lambda o: o["szum_real"] >= 0),
        ("tło syntetyczne", lambda o: o["szum_real"] < 0),
        ("ARW włączone", lambda o: (o["agc_tau"] or 0) > 0),
        ("ARW wyłączone", lambda o: (o["agc_tau"] or 0) == 0),
        ("z QRM", lambda o: (o["qrm"] or 0) > 0),
        ("bez QRM", lambda o: (o["qrm"] or 0) == 0),
        ("z QRN", lambda o: (o["qrn"] or 0) > 0),
        ("zanik > 0,4", lambda o: (o["qsb"] or 0) > 0.4),
        ("pusty (bez stacji)", lambda o: not o["text"]),
        ("znak ucięty krawędzią okna", uciety),
        ("bez uciętego znaku", lambda o: o["text"] and not uciety(o)),
        ("S/N < 5 dB", lambda o: o["text"] and o["snr_db"] is not None
         and o["snr_db"] < 5),
        ("S/N 5-15 dB", lambda o: o["text"] and o["snr_db"] is not None
         and 5 <= o["snr_db"] < 15),
        ("S/N >= 15 dB", lambda o: o["text"] and o["snr_db"] is not None
         and o["snr_db"] >= 15),
    ]
    for nazwa, w in grupy:
        try:
            print(f"    {nazwa:34s} {udzial(w)}")
        except KeyError:
            pass
    banki = sorted({opisy[i].get("plik_szumu") for i in gen
                    if opisy[i].get("plik_szumu")})
    print("\n  wg pliku banku:")
    for b in banki:
        print(f"    {b[-40:]:40s} "
              f"{udzial(lambda o, b=b: o.get('plik_szumu') == b)}")


# --------------------------------------------------------------------------
# 5. ile okien ma znak ucięty krawędzią — liczba, nie wrażenie
# --------------------------------------------------------------------------
def ucinki(p: Path, limit: int) -> None:
    zr = otworz(p, limit, 64)
    if not isinstance(zr, (ZrodloZbior, ZrodloPartia)):
        raise SystemExit("ucinki: tylko zbiór albo partia")
    ze_znakiem = lewa = prawa = cel_uciety = 0
    for i in range(zr.n):
        zn = zr.znaki(i)
        if not zn:
            continue
        ze_znakiem += 1
        l = any(od < 0 < do for _, od, do, _ in zn)
        r = any(od < C.IMG_FRAMES < do for _, od, do, _ in zn)
        lewa += l
        prawa += r
        cel_uciety += any((od < 0 or do > C.IMG_FRAMES) and c
                          for _, od, do, c in zn)
    print(f"{zr.naglowek}: okien ze znakami {ze_znakiem}")
    print(f"  znak ucięty lewą krawędzią:   {lewa:6d}  "
          f"({100 * lewa / max(1, ze_znakiem):.0f}%)")
    print(f"  znak ucięty prawą krawędzią:  {prawa:6d}  "
          f"({100 * prawa / max(1, ze_znakiem):.0f}%)")
    print(f"  ucięty CEL (znak etykiety):   {cel_uciety:6d}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generuj")
    g.add_argument("--n", type=int, default=320)
    g.add_argument("--nagrania", type=int, default=32)
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--out", type=Path, default=KATALOG_OCEN / "partia1.npz")
    pr = sub.add_parser("przegladaj")
    pr.add_argument("zrodlo", type=Path)
    pr.add_argument("--kol", type=int, default=4)
    pr.add_argument("--wier", type=int, default=4)
    pr.add_argument("--limit", type=int, default=2000,
                    help="zbiór: ile pierwszych obrazków wczytać")
    pr.add_argument("--krok", type=int, default=64,
                    help="nagranie: przesunięcie okien w ramkach (64 = 1,28 s)")
    pr.add_argument("--pozycja", default=None,
                    help="położenie okna x,y — dla dwóch kopii obok siebie")
    po = sub.add_parser("podsumuj")
    po.add_argument("partia", type=Path)
    uc = sub.add_parser("ucinki")
    uc.add_argument("zrodlo", type=Path)
    uc.add_argument("--limit", type=int, default=20000)
    a = ap.parse_args(argv)
    if a.cmd == "generuj":
        generuj(a.n, a.nagrania, a.out, a.seed)
    elif a.cmd == "przegladaj":
        przegladaj(a.zrodlo, a.kol, a.wier, a.limit, a.krok, a.pozycja)
    elif a.cmd == "podsumuj":
        podsumuj(a.partia)
    else:
        ucinki(a.zrodlo, a.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
