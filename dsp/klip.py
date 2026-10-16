"""JEDNA droga: klip -> obraz sieci -> etykiety ramek (normalizacja tempa).

PO CO (16.10). Od normalizacji tempa obraz klipu nie jest już
frontend.to_net_image(audio): każda ramka obejmuje tyle czasu nagrania,
ile przy tempie wzorcowym (C.WPM) trwałoby HOP_LENGTH. Gdyby generator,
koperta, skróty, trafność i X-Ray liczyły obraz każde po swojemu, pomiar
mierzyłby inny świat niż trening — ten projekt ma już taki przypadek za
sobą (trzy różne normalizacje dB w v5). Stąd jeden moduł, a kopia w
train_rtx.py jest pilnowana bit w bit (diag.py, TEST 12).

Przy C.TEMPO_NORM = False wszystko idzie drogą sprzed 16.10.
"""

from __future__ import annotations

import numpy as np

from . import config as C
from . import frontend, radio, tempo


def tempo_klipu(rng: np.random.Generator) -> tuple[float, float, int]:
    """(tempo prawdziwe, tempo „zmierzone", długość klipu w próbkach).

    Długość dobrana tak, żeby po normalizacji klip miał tyle ramek co klip
    CLIP_SAMPLES bez normalizacji — okno sieci zajmuje ten sam ułamek."""
    w = float(rng.uniform(*C.TEMPO_ZAKRES))
    w_n = w * (1.0 + float(rng.uniform(-C.TEMPO_BLAD, C.TEMPO_BLAD)))
    n = int(round(C.CLIP_SAMPLES * C.WPM / w_n))
    return w, w_n, n


def obraz_klipu(audio: np.ndarray, w_n: float, sr: int = C.SR
                ) -> tuple[np.ndarray, np.ndarray]:
    """audio -> (obraz [IMG_FRAMES, IMG_BINS] w [0,1], środki ramek [s]).

    Tempo stałe w klipie (jedna stacja, „zmierzone" w_n) — granice ramek
    k * d bez sumowania kroków."""
    db, srodki = tempo.obraz_w_tempie_wzorcowym(
        audio, sr, tor=(np.array([0.0]), np.array([float(w_n)])))
    k = db.shape[0]
    s0 = max(0, (k - C.IMG_FRAMES) // 2)
    db, srodki = db[s0:s0 + C.IMG_FRAMES], srodki[s0:s0 + C.IMG_FRAMES]
    if db.shape[0] < C.IMG_FRAMES:                  # czarne tło, jak center_window
        brak = C.IMG_FRAMES - db.shape[0]
        d = C.HOP_LENGTH / sr * C.WPM / float(w_n)
        db = np.vstack([db, np.full((brak, db.shape[1]), C.DB_MIN - 10.0,
                                    dtype=db.dtype)])
        srodki = np.r_[srodki, srodki[-1] + d * np.arange(1, brak + 1)]
    return frontend.normalize_db(db).astype(np.float32), srodki


def etykiety_klipu(spans, offset: float, text: str, srodki_s: np.ndarray,
                   sr: int = C.SR) -> np.ndarray:
    """Granice znaków -> klasa dla każdej ramki okna (jak frame_labels:
    ramka należy do znaku, gdy jej ŚRODEK wypada w jego granicach; znak
    krótszy niż odstęp środków dostaje najbliższą ramkę)."""
    y = np.zeros(C.IMG_FRAMES, dtype=np.uint8)
    c = np.asarray(srodki_s, dtype=np.float64) * sr
    for tag, a, b in spans or []:
        if not (0 <= tag < len(text)):
            continue
        cid = C.CHAR_TO_ID.get(text[tag], 0)
        if cid == 0:
            continue
        a2, b2 = float(a) + offset, float(b) + offset
        m = (c >= a2) & (c <= b2)
        if m.any():
            y[m] = cid
        else:
            mid = 0.5 * (a2 + b2)
            if c[0] <= mid <= c[-1]:
                y[int(np.argmin(np.abs(c - mid)))] = cid
    return y


def ramka_probki(probka: float, srodki_s: np.ndarray, sr: int = C.SR) -> float:
    """Próbka klipu -> numer ramki okna (ułamkowy; X-Ray, lab_x0/x1)."""
    return float(np.interp(probka / sr, srodki_s,
                           np.arange(len(srodki_s), dtype=np.float64),
                           left=-1.0, right=float(len(srodki_s))))


def klip_audio(rng: np.random.Generator, text: str, realism: bool = True,
               wpm: float | None = None) -> tuple[np.ndarray, dict]:
    """Klip audio + opis. Przy normalizacji tempo losowane TU (przed
    receive), bo od niego zależy długość klipu; wpm podane = tempo stałe
    bez błędu pomiaru (koperta: „sygnał idealny" w zadanym tempie)."""
    if not C.TEMPO_NORM:
        return radio.receive(text, rng, n_samples=C.CLIP_SAMPLES, wpm=wpm,
                             realism=realism)
    if wpm is None:
        w, w_n, n = tempo_klipu(rng)
    else:
        w = w_n = float(wpm)
        n = int(round(C.CLIP_SAMPLES * C.WPM / w_n))
    audio, meta = radio.receive(text, rng, n_samples=n, wpm=w,
                                realism=realism, tempo_staly=True)
    meta["wpm_norm"] = w_n
    return audio, meta


def obraz_i_etykiety(audio: np.ndarray, meta: dict, sr: int = C.SR
                     ) -> tuple[np.ndarray, np.ndarray, float, float]:
    """(obraz, etykiety ramek, lab_x0, lab_x1) — obie drogi."""
    text = meta.get("text", "")
    spans = meta.get("spans", [])
    offset = float(meta.get("offset", 0.0))
    w_n = meta.get("wpm_norm", float("nan"))
    if w_n is not None and np.isfinite(w_n):
        img, srodki = obraz_klipu(audio, w_n, sr)
        yf = etykiety_klipu(spans, offset, text, srodki, sr)
        do_ramki = lambda v: ramka_probki(v, srodki, sr)
    else:
        img = frontend.to_net_image(audio, sr)
        yf = frontend.frame_labels(spans, offset, text, audio.size)
        do_ramki = lambda v: frontend.sample_to_window_frame(v, audio.size)
    lab = [float("nan") if not np.isfinite(meta.get(k, np.nan))
           else do_ramki(meta[k]) for k in ("lab_a", "lab_b")]
    return img, yf, lab[0], lab[1]


def klip_sieci(rng: np.random.Generator, text: str, realism: bool = True,
               wpm: float | None = None):
    """Gotowy klip dla sieci: (obraz, etykiety ramek, opis, audio)."""
    audio, meta = klip_audio(rng, text, realism, wpm)
    img, yf, x0, x1 = obraz_i_etykiety(audio, meta)
    meta["lab_x0"], meta["lab_x1"] = x0, x1
    return img, yf, meta, audio
