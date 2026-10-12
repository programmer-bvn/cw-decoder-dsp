"""OCENA OKIEM: przeglądarka X-Ray z polem „zły" pod każdym obrazkiem.

PO CO. 12.10 operator obejrzał arkusz klipów po zmianie generatora i
stwierdził, że część obrazków „nie występuje w realnym świecie". Telegrafista
rozpozna nierealny obraz szybciej niż jakakolwiek miara — ale żeby z tego
wyszła poprawka, ocena musi trafić do pliku razem z parametrami klipu.

Trzy kroki:

    # 1. partia klipów (WSL — tam jest generator); część to okna
    #    z PRAWDZIWYCH nagrań, wymieszane i nieoznaczone (kontrola oka)
    python -m tools.xray_ocena generuj --n 320 --nagrania 32 \\
        --out out/ocena/partia1.npz

    # 2. przegląd (Windows albo WSL; potrzebne tylko numpy i pygame)
    python -m tools.xray_ocena przegladaj out/ocena/partia1.npz

    # 3. co łączy złe obrazki
    python -m tools.xray_ocena podsumuj out/ocena/partia1.npz

Przegląd: klik na obrazku albo polu = „zły" tak/nie. Strzałki / PgUp /
PgDn / spacja — strony. I — parametry pod obrazkami (domyślnie ukryte,
żeby nie podpowiadały oceny). Prawy klik — powiększenie z pełnym opisem.
Esc — koniec. Ocena zapisuje się po każdym kliknięciu do
`<partia>_ocena.json`; przegląd można przerwać i wznowić.

Skala barw stała, jak w tools/xray.py: czarne = DB_MIN, białe = DB_MAX.
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


# --------------------------------------------------------------------------
# 1. generowanie partii
# --------------------------------------------------------------------------
def generuj(n: int, n_nagr: int, out: Path, seed: int) -> None:
    from dsp import config as C
    from dsp import frontend, radio
    import matplotlib

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
        opisy.append(o)

    if n_nagr > 0:
        from tools import wav2net
        nagrania = sorted((C.ROOT / "probki").glob("*.wav"))
        cale = {}
        for p in nagrania:
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

    cmap = (matplotlib.colormaps["magma"] if hasattr(matplotlib, "colormaps")
            else matplotlib.cm.get_cmap("magma"))
    lut = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

    def _j(v):
        if isinstance(v, (np.floating, float)):
            return None if not np.isfinite(v) else round(float(v), 3)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        return v
    tekst_opisow = json.dumps([{k: _j(v) for k, v in o.items()} for o in opisy],
                              ensure_ascii=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, lut=lut, opisy=np.array(tekst_opisow),
                        db=np.array([C.DB_MIN, C.DB_MAX]))
    print(f"zapisane {out}: {n} z generatora + {n_nagr} okien nagrań, "
          f"wymieszane")


# --------------------------------------------------------------------------
# 2. przegląd
# --------------------------------------------------------------------------
KOL, WIER = 4, 4
KAF_W, KAF_H = 320, 150          # 128 ramek x 32 pasma, rozciągnięte
POLE_H = 70                      # pod obrazkiem: pole „zły" i opis


def _plik_oceny(partia: Path) -> Path:
    return partia.with_name(partia.stem + "_ocena.json")


def przegladaj(partia: Path) -> None:
    import pygame

    d = np.load(partia)
    X, lut = d["X"], d["lut"]
    opisy = json.loads(str(d["opisy"]))
    n = len(X)
    na_strone = KOL * WIER
    stron = (n + na_strone - 1) // na_strone

    plik = _plik_oceny(partia)
    zle, obejrzane = set(), set()
    if plik.exists():
        z = json.loads(plik.read_text(encoding="utf-8"))
        zle, obejrzane = set(z.get("zle", [])), set(z.get("obejrzane", []))

    def zapisz():
        plik.write_text(json.dumps(
            {"partia": partia.name, "zle": sorted(zle),
             "obejrzane": sorted(obejrzane), "n": n,
             "kiedy": time.strftime("%Y-%m-%d %H:%M:%S")},
            ensure_ascii=False, indent=1), encoding="utf-8")

    # Zapis od razu: plik oceny istnieje, odkąd przegląd ruszył.
    zapisz()
    pygame.init()
    W = KOL * (KAF_W + 16) + 16
    H = 46 + WIER * (KAF_H + POLE_H) + 10
    ekran = pygame.display.set_mode((W, H))
    pygame.display.set_caption(f"Ocena X-Ray — {partia.name}")
    f_dzy = pygame.font.SysFont("consolas", 18)
    f_mal = pygame.font.SysFont("consolas", 12)

    def powierzchnia(i, w, h):
        rgb = lut[X[i]]                          # [ramki, pasma, 3]
        rgb = rgb[:, ::-1, :]                    # niskie pasma na dole
        s = pygame.surfarray.make_surface(np.ascontiguousarray(rgb))
        return pygame.transform.scale(s, (w, h))

    def krotki_opis(o):
        if o.get("zrodlo") == "nagranie":
            return [f"NAGRANIE {o['plik'][:34]}", f"od {o['start_s']} s"]
        sn = o.get("snr_db")
        return [f"{'BANK' if o.get('szum_real', -1) >= 0 else 'syn'} "
                f"S/N {sn if sn is not None else '-'}  poz {o.get('poziom_db')}",
                f"ARW {o.get('agc_tau')}  qsb {o.get('qsb')}  "
                f"pk {o.get('peak')}",
                f"qrm {o.get('qrm')}  qrn {o.get('qrn')}"]

    strona, info, powieksz = 0, False, None
    prostokaty = []
    zegar = pygame.time.Clock()
    while True:
        obejrzane.add(strona)
        ekran.fill((24, 24, 28))
        prostokaty.clear()
        if powieksz is None:
            naglowek = (f"strona {strona + 1}/{stron}   złych: {len(zle)}/{n}   "
                        f"[klik = zły]  [prawy = powiększ]  [I = parametry "
                        f"{'wł' if info else 'wył'}]  [strzałki/spacja]  [Esc]")
            ekran.blit(f_dzy.render(naglowek, True, (230, 230, 230)), (16, 12))
            for k in range(na_strone):
                i = strona * na_strone + k
                if i >= n:
                    break
                x = 16 + (k % KOL) * (KAF_W + 16)
                y = 46 + (k // KOL) * (KAF_H + POLE_H)
                ekran.blit(powierzchnia(i, KAF_W, KAF_H), (x, y))
                zly = i in zle
                if zly:
                    pygame.draw.rect(ekran, (230, 40, 40),
                                     (x - 3, y - 3, KAF_W + 6, KAF_H + 6), 3)
                pole = pygame.Rect(x, y + KAF_H + 6, 18, 18)
                pygame.draw.rect(ekran, (230, 230, 230), pole, 2)
                if zly:
                    pygame.draw.line(ekran, (230, 40, 40), pole.topleft,
                                     pole.bottomright, 3)
                    pygame.draw.line(ekran, (230, 40, 40), pole.topright,
                                     pole.bottomleft, 3)
                ekran.blit(f_dzy.render(f"zły   #{i + 1}", True,
                                        (240, 90, 90) if zly else (200, 200, 200)),
                           (x + 26, y + KAF_H + 5))
                if info:
                    for r, linia in enumerate(krotki_opis(opisy[i])):
                        ekran.blit(f_mal.render(linia, True, (170, 170, 170)),
                                   (x, y + KAF_H + 30 + 14 * r))
                prostokaty.append((pygame.Rect(x, y, KAF_W, KAF_H + 26), i))
        else:
            i = powieksz
            ekran.blit(powierzchnia(i, W - 32, int((W - 32) * 0.4)), (16, 16))
            y = 24 + int((W - 32) * 0.4)
            stan = "ZŁY" if i in zle else "-"
            ekran.blit(f_dzy.render(f"#{i + 1}   {stan}   [klik = zły, "
                                    f"inny klawisz = powrót]", True,
                                    (230, 230, 230)), (16, y))
            o = opisy[i]
            pary = [f"{k}={v}" for k, v in o.items() if k != "text"]
            linie, akt = [], ""
            for p in pary:
                if len(akt) + len(p) > 150:
                    linie.append(akt); akt = ""
                akt += p + "   "
            linie.append(akt)
            for r, linia in enumerate(linie):
                ekran.blit(f_mal.render(linia, True, (190, 190, 190)),
                           (16, y + 30 + 16 * r))
        pygame.display.flip()

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                zapisz(); pygame.quit(); return
            if ev.type == pygame.KEYDOWN:
                if powieksz is not None:
                    powieksz = None
                elif ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    zapisz(); pygame.quit(); return
                elif ev.key in (pygame.K_RIGHT, pygame.K_PAGEDOWN, pygame.K_SPACE):
                    strona = min(stron - 1, strona + 1)
                elif ev.key in (pygame.K_LEFT, pygame.K_PAGEUP):
                    strona = max(0, strona - 1)
                elif ev.key == pygame.K_i:
                    info = not info
                zapisz()
            if ev.type == pygame.MOUSEBUTTONDOWN:
                if powieksz is not None:
                    if ev.button == 1:
                        zle.symmetric_difference_update({powieksz}); zapisz()
                    else:
                        powieksz = None
                    continue
                for r, i in prostokaty:
                    if r.collidepoint(ev.pos):
                        if ev.button == 1:
                            zle.symmetric_difference_update({i}); zapisz()
                        elif ev.button == 3:
                            powieksz = i
        zegar.tick(30)


# --------------------------------------------------------------------------
# 3. podsumowanie
# --------------------------------------------------------------------------
def podsumuj(partia: Path) -> None:
    d = np.load(partia)
    opisy = json.loads(str(d["opisy"]))
    plik = _plik_oceny(partia)
    if not plik.exists():
        print(f"Brak oceny: {plik} nie istnieje — partia nie była jeszcze "
              f"przeglądana. Najpierw:\n"
              f"    python -m tools.xray_ocena przegladaj {partia}")
        sys.exit(1)
    z = json.loads(plik.read_text(encoding="utf-8"))
    zle = set(z["zle"])
    na_strone = KOL * WIER
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
    print("\n  odsetek złych w grupie:")
    grupy = [
        ("tło z banku", lambda o: o["szum_real"] >= 0),
        ("tło syntetyczne", lambda o: o["szum_real"] < 0),
        ("ARW włączone", lambda o: (o["agc_tau"] or 0) > 0),
        ("ARW wyłączone", lambda o: (o["agc_tau"] or 0) == 0),
        ("karta przesterowana (peak > 1)", lambda o: (o["peak"] or 0) > 1),
        ("karta bez przesterowania", lambda o: (o["peak"] or 0) <= 1),
        ("z QRM", lambda o: (o["qrm"] or 0) > 0),
        ("bez QRM", lambda o: (o["qrm"] or 0) == 0),
        ("z QRN", lambda o: (o["qrn"] or 0) > 0),
        ("zanik > 0,4", lambda o: (o["qsb"] or 0) > 0.4),
        ("pusty (bez stacji)", lambda o: not o["text"]),
        ("S/N < 5 dB", lambda o: o["text"] and o["snr_db"] is not None
         and o["snr_db"] < 5),
        ("S/N 5-15 dB", lambda o: o["text"] and o["snr_db"] is not None
         and 5 <= o["snr_db"] < 15),
        ("S/N >= 15 dB", lambda o: o["text"] and o["snr_db"] is not None
         and o["snr_db"] >= 15),
        ("poziom stacji < 2 dB (ściszony)", lambda o: o["text"]
         and (o["poziom_db"] or 0) < 2),
    ]
    for nazwa, w in grupy:
        print(f"    {nazwa:34s} {udzial(w)}")
    banki = sorted({opisy[i]["plik_szumu"] for i in gen
                    if opisy[i]["plik_szumu"]})
    print("\n  wg pliku banku:")
    for b in banki:
        print(f"    {b[-40:]:40s} {udzial(lambda o, b=b: o['plik_szumu'] == b)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generuj")
    g.add_argument("--n", type=int, default=320)
    g.add_argument("--nagrania", type=int, default=32)
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--out", type=Path, default=Path("out/ocena/partia1.npz"))
    for nazwa in ("przegladaj", "podsumuj"):
        p = sub.add_parser(nazwa)
        p.add_argument("partia", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "generuj":
        generuj(a.n, a.nagrania, a.out, a.seed)
    elif a.cmd == "przegladaj":
        przegladaj(a.partia)
    else:
        podsumuj(a.partia)
    return 0


if __name__ == "__main__":
    sys.exit(main())
