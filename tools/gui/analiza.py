"""ANALIZA PLIKU DLA WIDOKU SYGNAŁU — prawdziwy łańcuch odczytu (08.11).

To samo co tools/odczyt.odczytaj (przygotuj -> sieć zboczy -> Viterbi),
ale zachowuje wszystko, co pokazuje widok sygnału, kolumna co 5 ms:
    wejście      widmo surowego dźwięku (300-1200 Hz)
    po obróbce   to samo po pętli tonu, kasowaniu obcych i przestrojeniu na 750 Hz
    oczami sieci obraz mel zboczy — dokładnie to, co dostaje sieć
    P(klucz)     wyjście sieci
    regenerat    czysty prostokąt z nut, z odchyłkami zboczy P od niego
oraz nakładki: zęby grzebienia i zbocza regeneratu (nuty z grzebienia).

Sieć: dsp.siec_numpy (runs/zbocza_5/siec.npz) — bez TensorFlow, start okna
w sekundę zamiast kilkunastu (import TF), a wynik ten sam co Keras
(HISTORIA, „KV260: odczyt na samym ARM").
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dsp import config as C, frontend, grzebien, melref, viterbi as V, zbocza as Z

SIEC = Path("runs/zbocza_5/siec.npz")
N_FFT_WEJ = 512                       # 64 ms, prążek 15,6 Hz
HZ_OD, HZ_DO = 300.0, 1200.0
# Ten sam odcień dla tego samego tonu co w obrazie sieci: korekta zmierzona
# zmierz_korekte() 08.11 (ton 750 Hz, amplitudy 0,01-0,5): -20,35 dB
# (z rachunku okien 1024 -> 512 wychodziło -20,4).
KOREKTA_WEJ_DB = -20.35
OKNO_GRZEBIENIA_S = 8.0               # jak odczyt.jednostka_grzebienia


def _prazki():
    df = C.SR / N_FFT_WEJ
    a, b = int(np.ceil(HZ_OD / df)), int(np.floor(HZ_DO / df)) + 1
    return a, b, np.arange(a, b) * df


def zmierz_korekte() -> float:
    """Różnica dB (mel zboczy - widmo 512) dla czystego tonu 750 Hz."""
    t = np.arange(C.SR * 2) / C.SR
    a, b, _ = _prazki()
    roz = []
    for amp in (0.01, 0.05, 0.2, 0.5):
        x = (amp * np.sin(2 * np.pi * 750 * t)).astype(np.float32)
        mel = melref.power_to_db(melref.stft_power(x, Z.N_FFT_ZBOCZA, Z.HOP_ZBOCZA) @ Z._fb().T)
        wid = melref.power_to_db(melref.stft_power(x, N_FFT_WEJ, Z.HOP_ZBOCZA)[:, a:b])
        roz.append(np.median(mel.max(1)) - np.median(wid.max(1)))
    return float(np.mean(roz))


def widmo(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """audio -> ([ramki co 5 ms, prążki 300-1200 Hz] 0..1 w skali sieci, Hz prążków)."""
    a, b, hz = _prazki()
    p = melref.stft_power(np.asarray(audio, np.float32), N_FFT_WEJ, Z.HOP_ZBOCZA)[:, a:b]
    return frontend.normalize_db(melref.power_to_db(p) + KOREKTA_WEJ_DB).astype(np.float32), hz


def hz_mel() -> np.ndarray:
    m = np.linspace(melref.hz_to_mel(C.FMIN), melref.hz_to_mel(C.FMAX), C.N_MELS + 2)
    return np.asarray(melref.mel_to_hz(m[1:-1]))


def _kawalki(oo: list, dl_s: float) -> list[tuple[float, list]]:
    """Odcinki znak/przerwa pocięte na kawałki ~dl_s, cięcie zawsze w przerwie
    (żaden element nie jest przecięty). -> [(czas startu, odcinki), ...]."""
    out, biez, t0, t = [], [], 0.0, 0.0
    for jest, d in oo:
        biez.append((jest, d))
        t += d
        if not jest and t - t0 >= dl_s:
            out.append((t0, biez))
            biez, t0 = [], t
    if biez:
        out.append((t0, biez))
    return out


def nakladki(pz: np.ndarray, u: float) -> dict:
    """Grzebień w oknach ~8 s (tempo ręki płynie) na P z histerezą jak odczyt:
    zęby od początku każdego znaku co u, zbocza regeneratu z nut."""
    oo = Z.odcinki_z_przebiegu(pz, histereza=(Z.PROG_WL, Z.PROG_WYL))
    zeby, zbocza, okna = [], [], []
    for t0, kaw in _kawalki(oo, OKNO_GRZEBIENIA_S):
        r = grzebien.dopasuj_grzebien(kaw, u_start=u if np.isfinite(u) else None)
        if not np.isfinite(r["u_s"]):
            continue
        uk = r["u_s"]
        okna.append((t0, uk, r["waga"], r["pewnosc"]))
        for s, el in grzebien.nuty(kaw, uk, r["waga"] * uk):
            kon = max(k + d for k, d in el)
            zeby.extend(t0 + s + k * uk for k in range(kon + 1))
            for k, d in el:
                zbocza += [t0 + s + k * uk, t0 + s + (k + d) * uk]
    return dict(zeby=np.array(zeby), zbocza=np.array(zbocza), okna=okna)


def regenerat(zbocza: np.ndarray, n: int) -> np.ndarray:
    """Czysty prostokąt z nut (pary narastające/opadające) na kolumny 5 ms."""
    r = np.zeros(n, np.float32)
    d = Z.HOP_ZBOCZA / C.SR
    for a, z in zbocza.reshape(-1, 2):
        r[int(round(a / d)):int(round(z / d))] = 1.0
    return r


def odchylki(pz: np.ndarray, zbocza: np.ndarray, u: float) -> dict:
    """Pomysł operatora 08.11: regenerat przyłożony do sygnału, mierzymy, jak
    daleko od zbocza regeneratu P przecina 0,5 (interpolowane). Narastające
    i opadające osobno: nuty stawiają oba na siatce k·u, a prawdziwe znaki
    są dłuższe o wagę b (pogłos, ARW, keyer) — średnia opadających minus
    średnia narastających to waga, a rozrzut wokół średnich to „pofalowanie"
    (hipoteza do R w raporcie: im większe, tym gorsza czytelność).
    Zbocze bez przecięcia P bliżej niż u/2 nie jest liczone."""
    d = Z.HOP_ZBOCZA / C.SR
    gora, dol = (np.asarray(x) for x in Z.zbocza(pz, 0.5))
    gora, dol = (gora + 0.5) * d, (dol + 0.5) * d       # środek ramki, jak w widoku
    pary, typ = [], []
    for i, t in enumerate(zbocza):
        cel = gora if i % 2 == 0 else dol
        if cel.size == 0:
            continue
        j = int(np.argmin(np.abs(cel - t)))
        if abs(cel[j] - t) < 0.5 * u:
            pary.append((t, cel[j]))
            typ.append(i % 2)
    pary, typ = np.array(pary).reshape(-1, 2), np.array(typ)
    dev = pary[:, 1] - pary[:, 0] if len(pary) else np.zeros(0)
    wynik = dict(pary=pary, n=int(dev.size))
    if dev.size >= 4 and (typ == 0).any() and (typ == 1).any():
        sr_g, sr_d = dev[typ == 0].mean(), dev[typ == 1].mean()
        rozrzut = float(np.sqrt(np.mean((dev - np.where(typ == 0, sr_g, sr_d)) ** 2)))
        wynik.update(srednia_nar_ms=1000 * sr_g, srednia_opad_ms=1000 * sr_d,
                     waga_ms=1000 * (sr_d - sr_g), rozrzut_ms=1000 * rozrzut,
                     rozrzut_u=rozrzut / u if np.isfinite(u) else float("nan"))
    return wynik


def analizuj(sciezka: Path, siec=None, ton_cw: float | None = None) -> dict:
    from dsp.siec_numpy import SiecNumpy
    from tools import odczyt
    if siec is None:
        siec = SiecNumpy(SIEC)
    if ton_cw is None:
        from tools.radio import civ
        ton_cw = civ.ton_z_opisu(Path(sciezka).with_suffix(".txt"))
    surowe = frontend.load_audio(sciezka)
    a, ton = odczyt.przygotuj(surowe, ton_cw=ton_cw)
    pz = Z.przebieg_sieci(a, siec)
    u_c, _ = V.jednostka_cepstrum(pz)
    u_g = odczyt.jednostka_grzebienia(pz)
    tekst, u, _ = V.dekoduj_najlepiej(pz, [u_c, u_g])
    wej, hz = widmo(surowe)
    dos, _ = widmo(a)
    nk = nakladki(pz, u)
    return dict(wej=wej, dos=dos, siec=Z.obraz_zboczy(a), p=pz, hz=hz, hz_mel=hz_mel(),
                tekst=tekst, u=u, ton=ton, dl_s=surowe.size / C.SR, **nk,
                regen=regenerat(nk["zbocza"], pz.size), odchylki=odchylki(pz, nk["zbocza"], u))
