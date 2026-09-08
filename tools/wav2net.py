"""KONWERTER WAV -> SIEĆ: plik audio na wejście modelu.

    python -m tools.wav2net --wav nagranie.wav                  # tylko konwersja
    python -m tools.wav2net --wav nagranie.wav --save out/we.npy
    python -m tools.wav2net --wav nagranie.wav --predict        # + dekodowanie
    python -m tools.wav2net --wav nagranie.wav --predict --stride 0.2 --min-conf 0.8

Dla pliku dłuższego niż okno sieci robi przesuwane okna i zwraca stos
[n_okien, IMG_FRAMES, IMG_BINS, 1].

DLACZEGO MEL LICZY SIĘ RAZ DLA CAŁEGO PLIKU
-------------------------------------------
Poprzedni dekoder liczył spektrogram osobno dla każdego bloku 100 ms i
wpychał wynik do przesuwanego obrazu. Blok 800 próbek przy n_fft=512 daje
tylko 2-3 ramki, a każde okno FFT obcięte na granicy bloku traci część
elementu klucza — element wypadający na styku był widziany dwa razy po
połowie. Tutaj spektrogram liczy się jednym przebiegiem po całym pliku,
a okna są WYCINANE Z GOTOWEGO OBRAZU. Nie ma efektów krawędziowych poza
początkiem i końcem pliku.

To działa dla plików. Dla pracy na żywo służy frontend.Waterfall, który
przechowuje ogon audio o długości okna FFT właśnie po to, żeby uzyskać
ten sam wynik przy strumieniu.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsp import config as C
from dsp import frontend


# --------------------------------------------------------------------------
# Konwersja
# --------------------------------------------------------------------------
# Docelowy szczyt obrazu w decybelach przy ustawianiu poziomu wejścia.
# Model uczył się na sygnałach o szczycie ok. +3..+23 dB (amplituda
# 0,03-0,50 w generatorze), więc celujemy w środek tego przedziału.
# ZMIERZONE na Nagrywanie (3).wav ("CQ CQ CQ DE SQ2BVN SQ2BVN"):
#   bez zmiany poziomu (szczyt 24,0 dB): CQCCCLQ2BVSQ2BVQ2BV9SE5
#   przyciszone o 12 dB (szczyt 17,3 dB): CCLQ2BVQ2BVQ2BVS
#   podgłośnione o 12 dB (szczyt 24,0 dB): 5FLSH42HQ2H6F5
TARGET_PEAK_DB = 17.0


def measure_peak_db(audio: np.ndarray) -> float:
    """Szczyt obrazu w decybelach dla całego nagrania (99,9 percentyl).

    Percentyl, nie maksimum — jeden trzask nie może przestawić poziomu
    całego nagrania.
    """
    db = frontend.power_to_db(frontend.melspec_power(audio))
    return float(np.percentile(db, 99.9))


def auto_gain_db(audio: np.ndarray,
                 target_db: float = TARGET_PEAK_DB) -> float:
    """Ile decybeli dodać, żeby szczyt obrazu wypadł przy target_db.

    TO NIE JEST powrót do skali ruchomej. Różnica jest istotna:

      ref=np.max (błąd z v5.5)  — skala liczona OSOBNO dla każdego klipu,
                                  więc ten sam sygnał dawał inne liczby
                                  w zależności od tła. Nieodwracalne.
      poziom wejścia (tutaj)    — JEDNO wzmocnienie na całe nagranie,
                                  zastosowane w dziedzinie czasu PRZED
                                  front-endem. Skala dB pozostaje
                                  bezwzględna, a wszystkie okna nagrania
                                  są mierzone tą samą miarą.

    To odpowiednik ustawienia poziomu AF na odbiorniku — etapu, który
    w naszym torze do tej pory nie istniał, a bez którego nagranie
    przesterowane albo zbyt ciche wypada poza zakres, jaki model widział.

    Moc w pasmach mel skaluje się z kwadratem wzmocnienia, więc w skali
    decybelowej zależność jest liniowa i jedna iteracja wystarcza.
    """
    return float(target_db - measure_peak_db(audio))


def wav_to_windows(path: Path, stride_frames: int = 1,
                   gain_db: float = 0.0, auto_gain: bool = False,
                   retune: bool = True
                   ) -> tuple[np.ndarray, np.ndarray, float, str]:
    """Plik audio -> (okna, czasy_środków_okien, wzmocnienie_dB, opis_pętli).

    okna:  [n, IMG_FRAMES, IMG_BINS, 1] float32 w [0, 1]
    czasy: [n] sekundy — środek okna względem początku pliku

    PRZESTRAJANIE JEST DOMYŚLNIE WŁĄCZONE, bo bez niego model czyta tylko
    ton z zakresu treningu. ZMIERZONE na sygnale idealnym 20 WPM:

        ton  500 Hz   nic          ton  500 Hz po przestrojeniu   CCQ2BVQ2B
        ton  600 Hz   nic          ton  600 Hz po przestrojeniu   CCQ2BVQ2B
        ton  750 Hz   CCQ2BVQ2B    (w zakresie treningu)
        ton  900 Hz   nic          ton  900 Hz po przestrojeniu   CCQ2BVQ2B
        ton 1050 Hz   nic          ton 1050 Hz po przestrojeniu   CCQ2BVQ2B

    Poza 670-830 Hz model nie zwraca ANI JEDNEGO zdarzenia, a po
    przestrojeniu odczyt jest identyczny z natywnym. Wąski zakres tonu
    w treningu (TONE_SPREAD) to nie wada — daje SELEKTYWNOŚĆ, czyli
    czytanie stacji przy 750 Hz i ignorowanie pozostałych. Ceną jest
    konieczność dostrojenia, i to robi właśnie pętla z dsp/tune.py.
    """
    audio = frontend.load_audio(path)

    lock = ""
    if retune:
        from dsp import tune
        t, f = tune.track_tone(audio)
        lock = tune.lock_report(t, f)
        audio = tune.retune(audio, t, f)

    if auto_gain:
        gain_db = auto_gain_db(audio)
    if abs(gain_db) > 1e-9:
        audio = np.clip(audio * (10.0 ** (gain_db / 20.0)),
                        -1.0, 1.0).astype(np.float32)

    # Cały spektrogram jednym przebiegiem, tą samą ścieżką co w treningu.
    full = frontend.normalize_db(
        frontend.power_to_db(frontend.melspec_power(audio)))

    n_frames = full.shape[0]
    if n_frames < C.IMG_FRAMES:
        # Plik krótszy niż okno — dopełniamy zerami (czarne tło), tak jak
        # robi frontend.center_window().
        win = frontend.center_window(full)
        return (win[np.newaxis, ..., np.newaxis],
                np.array([n_frames / 2 * C.HOP_LENGTH / C.SR]),
                gain_db, lock)

    starts = np.arange(0, n_frames - C.IMG_FRAMES + 1, max(1, stride_frames))
    windows = np.stack([full[s:s + C.IMG_FRAMES] for s in starts])
    centers = (starts + C.IMG_FRAMES / 2.0) * C.HOP_LENGTH / C.SR
    return (windows[..., np.newaxis].astype(np.float32), centers,
            gain_db, lock)


# --------------------------------------------------------------------------
# Dekodowanie strumienia
# --------------------------------------------------------------------------
def decode_stream(windows: np.ndarray, centers: np.ndarray, model,
                  min_conf: float, batch: int = 256,
                  min_windows: int = 1):
    """Predykcja na wszystkich oknach + scalenie w ciąg znaków.

    Scalanie: sąsiednie okna widzą TEN SAM znak (okno przesuwa się o mniej
    niż długość znaku), więc surowe wyjście to serie powtórzeń. Bierzemy
    jedną decyzję na serię, i to tę o NAJWYŻSZEJ pewności w serii, a nie
    pierwszą — pewność rośnie, gdy znak dojdzie do środka okna.

    min_windows odsiewa serie krótsze niż zadana liczba okien.

    ZMIERZONE NA PRAWDZIWYM NAGRANIU (mic3.wav, "CQ CQ CQ DE SQ2BVN SQ2BVN",
    44 zdarzenia surowe):

        bez filtrów        5PCGEQQPCQPCEQZL5QQ2BVF61Q2B4V66Q72BVN9PSI15
        pewność >= 95%     CQQ2BVQ2BVQ2BVPS5
        >= 3 okien i 95%   CQQ2BVQ2BVQ2BVPS5

    Czyli z 44-znakowej zupy zostaje czytelny znak wywoławczy, trzykrotnie.
    Pewność jest tu mocniejszym kryterium niż liczba okien; oba są dostępne,
    bo przy innym poziomie sygnału może być odwrotnie.

    CZEGO ŻADEN PRÓG NIE NAPRAWI. W odczycie brakuje S i N z SQ2BVN oraz
    całego DE. Gubione są ZNAKI KRÓTKIE (S ..., N -., D -.., E .), a czytane
    długie (Q --.-, 2 ..---, B -..., V ...-). Tych znaków model wcale nie
    zaproponował, więc filtrem ich nie odzyskasz.

    Przyczyna jest strukturalna: okno ma 2,56 s, czyli przy 20 WPM widzi
    4-5 znaków, a model uczono wybierać ŚRODKOWY Z TRZECH. Przy pięciu
    widocznych "środkowy" przestaje być określony i sieć ciąży do znaku
    najdłuższego. Rozwiązaniem jest dekoder sekwencyjny CTC, który etykietuje
    CAŁY ciąg zamiast wybierać jeden znak z okna — patrz README.
    """
    probs = model.predict(windows, batch_size=batch, verbose=0)
    ids = probs.argmax(axis=1)
    conf = probs.max(axis=1)

    events, dropped_conf, dropped_short = [], 0, 0
    i = 0
    while i < len(ids):
        j = i
        while j + 1 < len(ids) and ids[j + 1] == ids[i]:
            j += 1
        seg = slice(i, j + 1)
        best = i + int(np.argmax(conf[seg]))
        n_win = j - i + 1

        if ids[i] != 0:
            if n_win < min_windows:
                dropped_short += 1
            elif conf[best] < min_conf:
                dropped_conf += 1
            else:
                events.append((float(centers[best]), int(ids[i]),
                               float(conf[best]), n_win))
        i = j + 1

    return events, probs, dropped_conf, dropped_short


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Konwerter wav -> wejście sieci (+ dekodowanie)")
    ap.add_argument("--wav", type=Path, required=True)
    ap.add_argument("--save", type=Path,
                    help="zapisz stos okien jako .npy")
    ap.add_argument("--stride", type=float, default=0.1,
                    help="krok okna w sekundach (domyślnie 0.1)")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--model", type=Path, default=C.MODEL_PATH)
    ap.add_argument("--min-conf", type=float, default=0.95,
                    help="próg pewności przy dekodowaniu; 0.95 wynika "
                         "z pomiaru na prawdziwym nagraniu — patrz "
                         "decode_stream()")
    ap.add_argument("--min-windows", type=int, default=3,
                    help="ile kolejnych okien musi wskazać ten sam znak; "
                         "odsiewa wstawki z okien widzących pół znaku")
    ap.add_argument("--gain", type=float, default=0.0,
                    help="wzmocnienie wejścia w dB (ujemne = przyciszenie)")
    ap.add_argument("--no-retune", action="store_true",
                    help="nie przestrajaj tonu na TONE_CENTER; model czyta wtedy "
                         "tylko ton z zakresu treningu (670-830 Hz)")
    ap.add_argument("--no-auto-gain", action="store_true",
                    help="nie ustawiaj poziomu automatycznie; wtedy liczy "
                         "się --gain")
    args = ap.parse_args(argv)

    if not args.wav.exists():
        raise SystemExit(f"nie ma pliku: {args.wav}")

    print("=" * 70)
    print("WAV -> SIEĆ")
    print("=" * 70)
    print(C.summary())
    print("-" * 70)

    stride_frames = max(1, int(round(args.stride * C.frames_per_second())))
    windows, centers, used_gain, lock = wav_to_windows(
        args.wav, stride_frames, gain_db=args.gain,
        auto_gain=not args.no_auto_gain, retune=not args.no_retune)

    print(f"plik:   {args.wav}")
    if lock:
        print(f"petla:  {lock}")
    print(f"okna:   {windows.shape}  krok={stride_frames} ramek "
          f"({stride_frames * C.HOP_LENGTH / C.SR * 1000:.0f} ms)")
    if abs(used_gain) > 0.05:
        print(f"poziom: wzmocnienie {used_gain:+.1f} dB "
              f"{'(dobrane automatycznie' if not args.no_auto_gain else '(z --gain'}"
              f", cel: szczyt obrazu {TARGET_PEAK_DB:.0f} dB)")
        if used_gain < -3:
            print("        nagranie było PRZESTEROWANE dla skali modelu — "
                  "obraz nasycał się przy DB_MAX")
    print(f"zakres: min={windows.min():.3f}  max={windows.max():.3f}  "
          f"średnia={windows.mean():.3f}")
    frontend.check_image(windows[0], where="okno 0")
    print("kontrakt wejścia: OK")

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save, windows)
        print(f"zapisano: {args.save} "
              f"({args.save.stat().st_size/1024/1024:.1f} MB)")

    if args.predict:
        from dsp.model import load_model
        model = load_model(args.model)
        events, probs, drop_c, drop_s = decode_stream(
            windows, centers, model, args.min_conf,
            min_windows=args.min_windows)

        print("-" * 70)
        print(f"okien z sygnałem (klasa != 0): "
              f"{int(np.sum(probs.argmax(axis=1) != 0))}/{len(probs)}")
        print(f"zdarzeń po scaleniu: {len(events)}   "
              f"odrzucone: {drop_c} przez próg pewności "
              f"{args.min_conf}, {drop_s} przez próg "
              f"{args.min_windows} okien")
        print()
        for t, cid, cf, n in events:
            print(f"  {t:6.2f}s  '{C.ID_TO_CHAR[cid]}'  "
                  f"{cf*100:5.1f}%  ({n} okien)")

        text = "".join(C.ID_TO_CHAR[c] for _, c, _, _ in events)
        print("-" * 70)
        print(f"ODCZYT: {text}")

        # Podpowiedź, gdy wynik wygląda na zaszumiony wstawkami. Zmierzone
        # na mic3.wav: podniesienie progu z 0,75 na 0,95 zamieniło
        # 44-znakową zupę na czytelny znak wywoławczy.
        if len(events) > 12 and args.min_conf < 0.9:
            print()
            print("Dużo zdarzeń przy niskim progu — spróbuj "
                  "--min-conf 0.95 --min-windows 3.")
            print("Na prawdziwym nagraniu dało to różnicę między "
                  "'5PCGEQQPCQPCEQZL5QQ2BVF...' a 'CQQ2BVQ2BVQ2BV...'.")


if __name__ == "__main__":
    main()
