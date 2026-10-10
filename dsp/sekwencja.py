"""Odczyt wyjścia sieci "fcn": znak na każdy krok czasu -> tekst.

Sieć "fcn" (train_rtx.py, build_model) zwraca dla każdego okna
[FCN_KROKI, N_CLASSES]: rozkład klas co 40 ms. Tu jest wszystko, co
zamienia to w tekst — dla okna i dla całego nagrania.

ODPOWIEDNIK train_rtx.dekoduj_kroki. train_rtx.py jest samodzielny (bez
importów z dsp/), więc funkcja istnieje w dwóch miejscach; diag.py
sprawdza, że dają to samo. Tak samo jest z frame_labels.
"""

from __future__ import annotations

import numpy as np

from . import config as C

# Krok wyjścia "fcn" w ramkach obrazu. Musi się zgadzać z FCN_KROK
# w train_rtx.py — diag.py to sprawdza.
FCN_KROK = 2
FCN_KROKI = C.IMG_FRAMES // FCN_KROK


def dekoduj_kroki(probs: np.ndarray, prog_ciszy: float = 0.5,
                  sklejaj: bool = True,
                  prog_pewnej_ciszy: float = 0.8) -> list:
    """[kroki, N_CLASSES] -> [(krok_od, krok_do, id, pewność)].

    Granicą znaków jest CISZA (klasa 0 >= prog_ciszy), a znak odcinka to
    głos całego odcinka. Uzasadnienie przy train_rtx.dekoduj_kroki.
    """
    jest = probs[:, 0] < prog_ciszy
    wynik, k = [], 0
    n = len(jest)
    while k < n:
        if not jest[k]:
            k += 1
            continue
        k1 = k
        while k1 < n and jest[k1]:
            k1 += 1
        suma = probs[k:k1, 1:].sum(axis=0)
        cid = int(np.argmax(suma)) + 1
        wynik.append((k, k1, cid, float(suma[cid - 1] / (k1 - k))))
        k = k1
    if not sklejaj:
        return wynik

    # SKLEJANIE ROZCIĘTYCH LITER. Zmierzone 10.10 na nagraniach z USB
    # (model fcn z prawdziwym tłem): gdy dwa sąsiednie odcinki dzieli 1 krok
    # ciszy, w 54% par to TEN SAM znak (2 kroki: 36%, 5+: 5-13%) — czyli
    # jedna litera rozcięta na dwie (OK1FPS -> 1FFPS). W rozcięciu cisza
    # jest słaba: p(cisza) 0,51-0,64, a między prawdziwymi znakami
    # 0,98-1,00. W etykietach generatora przerwa 1 krok to ~0,1% par.
    # Sklejamy więc TYLKO ten sam znak, i tylko gdy przerwa ma 1 krok albo
    # nie ma w niej ani jednego kroku pewnej ciszy. Różnych znaków nie
    # skleja nigdy — nie może zgubić litery.
    sklejone = []
    for o in wynik:
        if sklejone:
            p = sklejone[-1]
            przerwa = probs[p[1]:o[0], 0]
            if (o[2] == p[2] and (o[0] - p[1] <= 1
                                  or float(przerwa.max()) < prog_pewnej_ciszy)):
                d1, d2 = p[1] - p[0], o[1] - o[0]
                sklejone[-1] = (p[0], o[1], p[2],
                                (p[3] * d1 + o[3] * d2) / (d1 + d2))
                continue
        sklejone.append(o)
    return sklejone


def kroki_nagrania(obraz: np.ndarray, model, krok_okna: int = 32,
                   batch: int = 256) -> np.ndarray:
    """Cały obraz nagrania [ramki, pasma] -> rozkład klas na krok [kroki, C].

    ZSZYWANIE ŚRODKÓW OKIEN. Okna co `krok_okna` ramek; z każdego bierze
    się tylko środkowe `krok_okna` ramek wyjścia. Krok na brzegu okna widzi
    kontekst tylko z jednej strony, a w środku okna — po 64 ramki (1,28 s)
    w obie. Przy krok_okna = 32 każdy użyty krok ma co najmniej 48 ramek
    (0,96 s) kontekstu z każdej strony. Pierwsze okno oddaje też swój
    początek, ostatnie — koniec, bo nic innego tych kroków nie widzi.

    Obraz jest dopełniany zerami (czarne tło, jak center_window) do
    parzystej liczby ramek i co najmniej jednego okna.
    """
    F, K = C.IMG_FRAMES, FCN_KROK
    if krok_okna % (2 * K) or not (0 < krok_okna <= F):
        raise ValueError(f"krok_okna={krok_okna}: musi być wielokrotnością "
                         f"{2 * K} i <= {F}")
    n = obraz.shape[0]
    n_pel = max(F, n + (n % K))
    if n_pel > n:
        obraz = np.concatenate(
            [obraz, np.zeros((n_pel - n, obraz.shape[1]), obraz.dtype)])
    starty = list(range(0, n_pel - F + 1, krok_okna))
    if starty[-1] != n_pel - F:
        starty.append(n_pel - F)          # parzyste, bo n_pel i F parzyste

    okna = np.stack([obraz[s:s + F] for s in starty])[..., np.newaxis]
    p = np.asarray(model.predict(okna.astype(np.float32), batch_size=batch,
                                 verbose=0))
    if p.ndim != 3:
        raise ValueError(f"to nie jest model fcn: wyjście {p.shape}")
    T = p.shape[1]

    wynik = np.zeros((n_pel // K, p.shape[2]), dtype=np.float32)
    pokryte = 0
    lo_srodek = (T - krok_okna // K) // 2
    for i, s in enumerate(starty):
        g0 = s // K
        lo = 0 if i == 0 else lo_srodek
        hi = T if i == len(starty) - 1 else lo_srodek + krok_okna // K
        a = max(pokryte, g0 + lo)
        b = g0 + hi
        if b > a:
            wynik[a:b] = p[i, a - g0:b - g0]
            pokryte = b
    return wynik


def krok_na_sekundy(krok: float) -> float:
    return krok * FCN_KROK * C.HOP_LENGTH / C.SR
