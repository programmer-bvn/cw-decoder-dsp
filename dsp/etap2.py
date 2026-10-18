"""DRUGI ETAP dekodera dwuetapowego (18.10): przebieg z sieci zboczy -> znaki.

Pierwszy etap (dsp/zbocza.py) oddaje P(klucz wciśnięty) co 5 ms w czasie
rzeczywistym. Ten etap czyta ZNAKI z tego przebiegu:

    P co 5 ms --(jednostka u z grzebienia)--> P co u/K  -> sieć -> klasa co 1 u

Dlaczego osobna sieć, a nie główna sieć znaków na regeneracie: 18.10 na 282
słowach kluczownika główna sieć na regeneracie z nut dała 161, na surowym
nagraniu 256, a sieć zboczy + grzebień + tabela 281 (HISTORIA, „Drugi
etap"). Regenerat jest poza rozkładem głównej sieci. Tabela z kolei nie
rozstrzyga niejednoznacznych przerw ręki (2,5 u: jeszcze w znaku czy już
między znakami?) — sieć uczy się tego z kontekstu i z błędów, jakie
pierwszy etap naprawdę robi (zbiór liczony prawdziwą siecią zboczy).

Tempo: przebieg przeskalowany tak, że jednostka = K próbek (normalizacja
jak w głównej sieci, ale z jednostki grzebienia). W treningu jednostka
„zmierzona" = prawdziwa x (1 ± BLAD_U) — grzebień na kluczowniku myli się
o ~1-2% (16,9 WPM przy 17), zapas na rękę.

Wyjście co 1 u: klasa znaku, którego granice (z generatora) obejmują środek
kroku, inaczej 0 (przerwa). Odczyt: kolejne kroki tej samej klasy = jeden znak.
"""

from __future__ import annotations

import numpy as np

from . import config as C
from . import zbocza as Z

K = 4                       # próbek na jednostkę
OKNO_U = 64                 # jednostek w oknie sieci
OKNO = K * OKNO_U           # 256 próbek wejścia
KROKI = OKNO_U              # wyjście co 1 u
BLAD_U = 0.10               # błąd jednostki „zmierzonej" w treningu (±)
ZNAKOW = 14                 # znaków w klipie treningowym (> okno przy każdym tempie)
CISZA = 0.08                # udział klipów bez nadania

DT = Z.HOP_ZBOCZA / C.SR    # 5 ms


def normalizuj(p: np.ndarray, u_s: float) -> np.ndarray:
    """P co 5 ms -> P co u/K (interpolacja liniowa)."""
    p = np.asarray(p, np.float64)
    t = np.arange(p.size) * DT
    tn = np.arange(0.0, t[-1] + 1e-9, u_s / K)
    return np.interp(tn, t, p).astype(np.float32)


def etykiety_krokow(znaki_s: list, tekst: str, u_s: float, n_krokow: int,
                    start_krok: int = 0) -> np.ndarray:
    """Klasa co 1 u od kroku start_krok: znak, którego [a, b] obejmuje środek
    kroku (czas (j + 0,5) * u), inaczej 0."""
    y = np.zeros(n_krokow, np.int16)
    srodki = (start_krok + np.arange(n_krokow) + 0.5) * u_s
    for tag, a, b in znaki_s:
        cid = C.CHAR_TO_ID.get(tekst[tag], 0) if 0 <= tag < len(tekst) else 0
        y[(srodki >= a) & (srodki <= b)] = cid
    return y


def klip(rng: np.random.Generator):
    """Jeden klip drugiego etapu (część na procesorze): obraz zboczy uint8
    całego klipu, granice znaków [s], tekst, prawdziwa jednostka [s]."""
    from . import radio
    wpm = float(rng.uniform(*C.TEMPO_ZAKRES))
    u = C.dot_seconds(wpm)
    if rng.random() < CISZA:
        tekst = ""
    else:
        tekst = "".join(rng.choice(list(C.ALPHABET[1:]), ZNAKOW))
        # spacje między słowami: przerwa 7 u (keying_envelope) — kilka na klip
        tekst = "".join(z + (" " if rng.random() < 0.2 else "") for z in tekst).strip()
    n = int(np.ceil((1.6 * OKNO_U * u + 1.0) * C.SR))
    audio, meta = radio.receive(tekst, rng, n_samples=n, wpm=wpm, realism=True,
                                tempo_staly=True, z_kluczem=bool(tekst),
                                zakresy=Z.ZAKRESY_ZBOCZY)
    obraz = np.round(np.clip(Z.obraz_zboczy(audio), 0, 1) * 255).astype(np.uint8)
    znaki = [(t_, a_ / C.SR, b_ / C.SR) for t_, a_, b_ in meta.get("znaki", [])]
    # indeksy w meta["znaki"] liczą znaki BEZ spacji (morse: słowa osobno)
    return obraz, znaki, tekst.replace(" ", ""), u


def probka(p: np.ndarray, znaki: list, tekst: str, u: float,
           rng: np.random.Generator):
    """(X [OKNO] uint8, Y [KROKI] int16) z przebiegu P całego klipu."""
    u_m = u * (1.0 + float(rng.uniform(-BLAD_U, BLAD_U)))
    pn = normalizuj(p, u_m)
    # okno losowo, ale tak, żeby środek padał na nadanie (gdy jest)
    if znaki:
        a0 = znaki[0][1] / (u_m / K); b0 = znaki[-1][2] / (u_m / K)
        lo = int(max(0, a0 - OKNO // 2)); hi = int(max(lo + 1, min(pn.size - OKNO, b0 - OKNO // 2)))
        s = int(rng.integers(lo, hi)) if hi > lo else lo
    else:
        s = int(rng.integers(0, max(1, pn.size - OKNO)))
    s -= s % K                                   # okno na granicy kroku
    x = np.zeros(OKNO, np.float32)
    w = pn[s:s + OKNO]; x[:w.size] = w
    y = etykiety_krokow(znaki, tekst, u_m, KROKI, start_krok=s // K)
    return np.round(x * 255).astype(np.uint8), y


def model_etap2(n_klas: int = C.N_CLASSES):
    """Sieć drugiego etapu, DPU-bezpieczna (Conv2D/BN/ReLU/Add/MaxPooling):
    sploty po czasie z dylatacją na wejściu 4 próbki/u, potem pulowanie do
    1 kroku/u i klasyfikacja w każdym kroku (logity -> softmax na ARM)."""
    from tensorflow import keras
    L = keras.layers
    wej = keras.Input((OKNO, 1, 1), name="p_klucza")
    x = L.Conv2D(64, (5, 1), padding="same", use_bias=False)(wej)
    x = L.BatchNormalization()(x); x = L.ReLU()(x)
    for d in (1, 2, 4, 8):
        y = L.Conv2D(64, (5, 1), dilation_rate=(d, 1), padding="same", use_bias=False)(x)
        y = L.BatchNormalization()(y); y = L.ReLU()(y); x = L.Add()([x, y])
    x = L.MaxPooling2D((K, 1))(x)                         # 1 krok = 1 u
    x = L.Conv2D(128, (1, 1), use_bias=False)(x)
    x = L.BatchNormalization()(x); x = L.ReLU()(x)
    for d in (1, 2, 4, 8, 16):
        y = L.Conv2D(128, (3, 1), dilation_rate=(d, 1), padding="same", use_bias=False)(x)
        y = L.BatchNormalization()(y); y = L.ReLU()(y); x = L.Add()([x, y])
    x = L.Conv2D(n_klas, (1, 1), name="logit")(x)
    wyj = L.Reshape((KROKI, n_klas))(x)
    return keras.Model(wej, wyj, name="siec_etap2")


def odczyt(p: np.ndarray, u_s: float, model) -> str:
    """P z sieci zboczy (cały odcinek, co 5 ms) + jednostka -> tekst.
    Okna co pół okna, z każdego środkowa połowa kroków."""
    pn = normalizuj(p, u_s)
    pol = KROKI // 2
    pad = np.zeros((KROKI // 4) * K, np.float32)
    x = np.concatenate([pad, pn, np.zeros(OKNO, np.float32)])
    starty = list(range(0, pn.size, pol * K))
    okna = np.stack([x[s:s + OKNO] for s in starty])[..., None, None]
    logit = model.predict(okna, verbose=0, batch_size=128)
    k = np.argmax(logit, axis=-1)[:, KROKI // 4: KROKI // 4 + pol].reshape(-1)
    k = k[:int(np.ceil(pn.size / K))]
    out, prev = [], 0
    for c in k:
        if c != prev and c != 0:
            out.append(C.ALPHABET[c])
        prev = c
    return "".join(out)
