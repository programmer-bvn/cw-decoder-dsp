"""JEDNA miara S/N dla generatora i dla nagrań.

Definicja jak w katalogu odbiornika dla CW: moc nośnej przy wciśniętym
kluczu do mocy szumu w paśmie C.SN_REF_BW_HZ (500 Hz) wokół tonu.

    S/N [dB] = 10 log10( (A^2 / 2) / N_500 )

Dwie drogi do tej samej liczby:

  sn_ze_skladowych — generator zna osobno amplitudę tonu i sam szum,
                     więc liczy S/N dokładnie (prawda naziemna).
  zmierz_sn        — nagranie ma tylko sumę. Ramki dzielą się na „klucz
                     wciśnięty" i „przerwę" po obwiedni w wąskim paśmie
                     przy tonie, a S i N liczy się w paśmie 500 Hz:
                     N = średnia moc w przerwach, S = (moc w znakach) - N.

Czemu nie po prostu „szczyt minus tło w obrazie": tło w obrazie zależy od
gałki AF i od szerokości pasma mel, a S/N od żadnej z nich. To jest
właśnie rozdzielenie, którego zabrakło do 12.10 (config.py, SN_REF_BW_HZ).
Szum z filtra p.cz. 200 Hz liczy się tak, jak go słychać: w 500 Hz jest
tylko jego 200 Hz — tak samo jak w uchu operatora.
"""

from __future__ import annotations

import numpy as np

from . import config as C


def moc_w_pasmie(x: np.ndarray, srodek_hz: float,
                 szerokosc_hz: float = C.SN_REF_BW_HZ,
                 sr: int = C.SR) -> float:
    """Średnia moc sygnału x w paśmie srodek +/- szerokosc/2 (Parseval)."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    m = np.abs(f - srodek_hz) <= szerokosc_hz / 2.0
    return float(2.0 * np.sum(np.abs(X[m]) ** 2) / (n * n))


def sn_ze_skladowych(amp: float, szum: np.ndarray, ton_hz: float,
                     sr: int = C.SR) -> float:
    """S/N [dB] z amplitudy tonu i samego szumu — prawda generatora."""
    n = moc_w_pasmie(szum, ton_hz, sr=sr)
    return float(10.0 * np.log10((amp * amp / 2.0) / max(n, 1e-30)))


def amp_dla_sn(sn_db: float, szum: np.ndarray, ton_hz: float,
               sr: int = C.SR) -> float:
    """Odwrotność sn_ze_skladowych: amplituda tonu dająca zadane S/N."""
    n = moc_w_pasmie(szum, ton_hz, sr=sr)
    return float(np.sqrt(2.0 * n * 10.0 ** (sn_db / 10.0)))


# --- pomiar z gotowego audio ----------------------------------------------
_NFFT = 256          # 32 ms przy 8 kHz: krócej niż kropka przy 35 WPM
_HOP = 80            # 10 ms
_WASKO_HZ = 80.0     # pasmo do decyzji klucz/przerwa (+/-40 Hz)


def _moce_ramek(x: np.ndarray, ton_hz: float, sr: int):
    """Moc w ramkach: (wąsko przy tonie, w paśmie odniesienia)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < _NFFT:
        return np.zeros(0), np.zeros(0)
    okno = np.hanning(_NFFT)
    k = 1 + (x.size - _NFFT) // _HOP
    idx = np.arange(_NFFT)[None, :] + _HOP * np.arange(k)[:, None]
    X = np.fft.rfft(x[idx] * okno, axis=1)
    f = np.fft.rfftfreq(_NFFT, 1.0 / sr)
    p = np.abs(X) ** 2
    # Normalizacja tak, żeby suma po prążkach = średnia moc w ramce
    # (Parseval z oknem): wtedy N i S wychodzą w tych samych jednostkach
    # co moc_w_pasmie i A^2/2.
    p *= 2.0 / (_NFFT * np.sum(okno ** 2))
    wasko = p[:, np.abs(f - ton_hz) <= _WASKO_HZ / 2.0].sum(axis=1)
    ref = p[:, np.abs(f - ton_hz) <= C.SN_REF_BW_HZ / 2.0].sum(axis=1)
    return wasko, ref


def _prog_otsu(v: np.ndarray) -> float:
    """Próg dzielący wartości na dwie grupy (metoda Otsu na histogramie)."""
    h, e = np.histogram(v, bins=64)
    c = (e[:-1] + e[1:]) / 2.0
    w = np.cumsum(h)
    s = np.cumsum(h * c)
    wt, st = w[-1], s[-1]
    w1 = wt - w
    ok = (w > 0) & (w1 > 0)
    m0 = np.where(ok, s / np.maximum(w, 1), 0.0)
    m1 = np.where(ok, (st - s) / np.maximum(w1, 1), 0.0)
    miedzy = np.where(ok, w * w1 * (m0 - m1) ** 2, -1.0)
    return float(c[int(np.argmax(miedzy))])


def zmierz_sn(x: np.ndarray, ton_hz: float = C.TONE_CENTER,
              sr: int = C.SR) -> dict:
    """S/N [dB] z gotowego audio z jedną stacją przy ton_hz.

    Zwraca słownik: sn_db (nan, gdy nie da się zmierzyć), udzial_klucza,
    n_ramek. Ramki na granicy znak/przerwa są pomijane — w nich moc jest
    pośrednia i zawyżałaby N albo zaniżała S.
    """
    wasko, ref = _moce_ramek(x, ton_hz, sr)
    wynik = {"sn_db": float("nan"), "udzial_klucza": float("nan"),
             "n_ramek": int(wasko.size)}
    if wasko.size < 50:
        return wynik
    db = 10.0 * np.log10(wasko + 1e-30)
    klucz = db > _prog_otsu(db)
    # Ramka 32 ms przy kroku 10 ms zachodzi na 3 sąsiednie — dwie ramki
    # odstępu od zmiany stanu z każdej strony.
    pewne = klucz.copy()
    pewne_p = ~klucz
    for s in (1, 2):
        pewne[s:] &= klucz[:-s]
        pewne[:-s] &= klucz[s:]
        pewne_p[s:] &= ~klucz[:-s]
        pewne_p[:-s] &= ~klucz[s:]
    wynik["udzial_klucza"] = float(klucz.mean())
    if pewne.sum() < 10 or pewne_p.sum() < 10:
        return wynik
    n = float(np.mean(ref[pewne_p]))
    s = float(np.mean(ref[pewne])) - n
    # Koncentracja: jaka część przyrostu mocy w „znakach" siedzi w wąskim
    # paśmie przy tonie. Ton: prawie cała. Trzask, skok ARW, szum z banku
    # bez stacji: przyrost rozlany po 500 Hz, czyli ok. 80/500.
    ds = float(np.mean(wasko[pewne]) - np.mean(wasko[pewne_p]))
    wynik["koncentracja"] = ds / s if s > 0 else float("nan")
    if s <= 0 or n <= 0:
        return wynik
    wynik["sn_db_surowe"] = float(10.0 * np.log10(s / n))
    # ZMIERZONE 12.10 (200 klipów 5 s samego szumu na rodzaj): szum
    # syntetyczny nie dał ani razu odczytu >= 0 dB; szum z banku dał
    # 60/171 (do +12 dB) przy koncentracji 0,26-0,64 — skoki ARW dzielone
    # Otsu na pół. Ton z prawdziwym S/N >= 3 dB: koncentracja 0,73-1,00,
    # od 6 dB >= 0,87.
    if wynik["koncentracja"] >= KONCENTRACJA_MIN:
        wynik["sn_db"] = wynik["sn_db_surowe"]
    return wynik


KONCENTRACJA_MIN = 0.8


# --- poziom tła w OBRAZIE ---------------------------------------------------
# 12.10: gałka AF zaczepiona o STACJĘ podciągała szum przy słabej stacji
# aż „po sam dach" (ocena operatora, out/ocena/partia1). Prawdziwe radio
# trzyma w ryzach SZUM (ARW, gałka AF), a słaba stacja jest po prostu
# ciemna. Ta miara mówi, gdzie leży szum w obrazie sieci: mediana po
# ramkach przerw ze średniej mocy w pasmach mel przy tonie.
TLO_PASMO_HZ = 120.0       # pasma mel o środku w tonie +/- 60 Hz
                           # (kopia w train_rtx.py — TEST 12)


def pasma_przy_tonie(ton_hz: float) -> np.ndarray:
    from . import melref
    srodki = melref.mel_frequencies(C.N_MELS, C.FMIN, C.FMAX)[1:-1]
    m = np.abs(srodki - ton_hz) <= TLO_PASMO_HZ / 2.0
    if not m.any():
        m[int(np.argmin(np.abs(srodki - ton_hz)))] = True
    return m


def tlo_db(mel_db: np.ndarray, przerwy: np.ndarray, ton_hz: float) -> float:
    """Poziom szumu w obrazie [dB skali sieci] z ramek przerw."""
    p = 10.0 ** (mel_db[:, pasma_przy_tonie(ton_hz)] / 10.0)
    na_ramke = p.mean(axis=1)[przerwy]
    if na_ramke.size < 5:
        return float("nan")
    return float(10.0 * np.log10(np.median(na_ramke)))


def przerwy_z_klucza(stacja: np.ndarray) -> np.ndarray:
    """Ramki obrazu, w których okno FFT nie obejmuje ani próbki znaku.

    Generator zna obwiednię stacji osobno — nie musi zgadywać przerw.
    Ramki jak w melref: dopełnienie N_FFT/2 zerami, krok HOP_LENGTH.
    """
    x = np.pad(np.abs(np.asarray(stacja, dtype=np.float64)), C.N_FFT // 2,
               mode="constant")
    n = 1 + (x.size - C.N_FFT) // C.HOP_LENGTH
    ramki = np.lib.stride_tricks.as_strided(
        x, shape=(n, C.N_FFT), strides=(x.strides[0] * C.HOP_LENGTH,
                                        x.strides[0]))
    return ramki.max(axis=1) < 1e-3


def przerwy_z_obrazu(mel_db: np.ndarray, ton_hz: float) -> np.ndarray:
    """Ramki przerw w obrazie bez wiedzy o kluczu (nagrania).

    Próg Otsu na mocy w pasmach przy tonie, potem dwie ramki odstępu od
    każdej zmiany stanu (okno 64 ms przy kroku 20 ms zachodzi na 3 ramki).
    Okno bez stacji: Otsu i tak dzieli szum na pół — wtedy „przerwy" to
    cichsza połowa szumu i poziom wychodzi zaniżony o ok. 1-2 dB.
    """
    p = mel_db[:, pasma_przy_tonie(ton_hz)]
    db = 10.0 * np.log10(np.mean(10.0 ** (p / 10.0), axis=1))
    przerwa = db <= _prog_otsu(db)
    pewne = przerwa.copy()
    for s in (1, 2):
        pewne[s:] &= przerwa[:-s]
        pewne[:-s] &= przerwa[s:]
    return pewne
