"""KONWERTER MIKROFON -> WAV: nagranie wejścia audio do pliku.

    python -m tools.mic2wav --list                    # wypisz urządzenia
    python -m tools.mic2wav --seconds 10              # nagraj 10 s
    python -m tools.mic2wav --seconds 30 --out out/qso.wav
    python -m tools.mic2wav --device 3 --seconds 10
    python -m tools.mic2wav --monitor                 # tylko pomiar, bez zapisu

Zapisuje mono, 16 bit PCM, z częstotliwością config.SR — dokładnie w takim
formacie, jakiego oczekuje wav2net i xray. Konwersja z częstotliwości
urządzenia jest tutaj, żeby nie trafiła do trzech narzędzi osobno.

TRYB --monitor jest do USTAWIENIA POZIOMU I TONU przed nagraniem. Podaje
trzy liczby, które decydują o tym, czy nagranie będzie się nadawało:

  poziom    RMS i szczyt. Szczyt powyżej 0,95 to obcinanie — obcięty ton
            ma harmoniczne, których nie było w danych treningowych.
  ton       zmierzona częstotliwość najsilniejszego prążka w pasmie
            FMIN..FMAX. Musi wypadać w środku pasma; jeśli leci przy
            krawędzi, przestrój odbiornik albo popraw FMIN/FMAX w config.py.
  SNR       różnica między szczytem a tłem w decybelach. Poniżej 10 dB
            model nie ma z czego czytać.
"""

from __future__ import annotations

import argparse
import queue
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsp import config as C
from dsp import frontend

try:
    import sounddevice as sd
except ImportError as exc:                            # pragma: no cover
    raise SystemExit(
        "Brak sounddevice. Zainstaluj: pip install sounddevice"
    ) from exc

try:
    import soundfile as sf
except ImportError as exc:                            # pragma: no cover
    raise SystemExit(
        "Brak soundfile. Zainstaluj: pip install soundfile"
    ) from exc


# --------------------------------------------------------------------------
# Urządzenia
# --------------------------------------------------------------------------
def list_devices() -> None:
    print("Urządzenia wejściowe:\n")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] < 1:
            continue
        default = " (domyślne)" if i == sd.default.device[0] else ""
        print(f"  [{i:2d}] {d['name']}{default}")
        print(f"       kanały={d['max_input_channels']}  "
              f"częstotliwość={d['default_samplerate']:.0f} Hz")
    print("\nNumer wpisz w config.MIC_DEVICE albo podaj przez --device.")


def device_samplerate(device) -> int:
    """Częstotliwość, z jaką urządzenie potrafi pracować.

    Karty USB często nie obsługują 8000 Hz. Nagrywamy wtedy natywnie i
    decymujemy — próba wymuszenia 8 kHz na sterowniku, który tego nie umie,
    kończy się błędem albo cichym przekłamaniem tempa.
    """
    try:
        sd.check_input_settings(device=device, samplerate=C.SR, channels=1)
        return C.SR
    except Exception:
        info = sd.query_devices(device, "input") if device is not None \
            else sd.query_devices(kind="input")
        return int(info["default_samplerate"])


# --------------------------------------------------------------------------
# Pomiary
# --------------------------------------------------------------------------
# Docelowy szczyt obrazu w decybelach — ta sama wartość, której używa
# tools/wav2net.py przy ustawianiu poziomu. Model uczył się na sygnałach
# o szczycie ok. +3..+23 dB, więc celujemy w środek.
#
# PO CO TO W MONITORZE. Pierwsze nagrania z pasma były przesterowane o 12 dB
# i obraz nasycał się przy DB_MAX; wyszło to dopiero po dekodowaniu, przez
# porównanie odczytów przy różnych wzmocnieniach. Poziom da się ustawić
# PRZED nagraniem, jeśli mierzyć go w tej skali, w której patrzy model,
# a nie tylko jako szczyt próbki audio.
TARGET_PEAK_DB = 17.0


def image_peak_db(audio_8k: np.ndarray) -> float:
    """Szczyt obrazu w decybelach — dokładnie to, co widzi sieć.

    Szczyt PRÓBKI audio o niczym nie mówi: nagranie o szczycie 0,7 może
    dawać obraz nasycony, a inne o szczycie 1,0 mieścić się w skali.
    Decyduje moc w pasmie tonu, nie amplituda w dziedzinie czasu.
    """
    if audio_8k.size < C.N_FFT * 2:
        return float("nan")
    db = frontend.power_to_db(frontend.melspec_power(audio_8k))
    return float(np.percentile(db, 99.9))


def measure(block: np.ndarray, sr: int) -> tuple[float, float, float, float]:
    """(RMS, szczyt, ton [Hz], SNR [dB]) dla bloku audio."""
    rms = float(np.sqrt(np.mean(block ** 2)))
    peak = float(np.max(np.abs(block)))

    n = 1 << int(np.ceil(np.log2(max(256, block.size))))
    win = np.hanning(block.size)
    spec = np.abs(np.fft.rfft(block * win, n=n)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / sr)

    band = (freqs >= C.FMIN) & (freqs <= C.FMAX)
    if not np.any(band) or spec[band].max() <= 0:
        return rms, peak, float("nan"), float("nan")

    k = int(np.flatnonzero(band)[np.argmax(spec[band])])
    tone = float(freqs[k])

    # Tło: mediana mocy w pasmie z wyłączeniem okolic szczytu (+/- 3 prążki).
    mask = band.copy()
    mask[max(0, k - 3):k + 4] = False
    floor = float(np.median(spec[mask])) if np.any(mask) else 0.0
    snr = 10.0 * np.log10(spec[k] / floor) if floor > 0 else float("inf")
    return rms, peak, tone, snr


def _bar(value: float, width: int = 24) -> str:
    fill = int(np.clip(value, 0.0, 1.0) * width)
    return "#" * fill + "-" * (width - fill)


# --------------------------------------------------------------------------
# Nagrywanie
# --------------------------------------------------------------------------
def record(seconds: float, device, out_path: Path | None,
           monitor_only: bool = False) -> Path | None:
    dev_sr = device_samplerate(device)
    resample = dev_sr != C.SR

    print(f"urządzenie: {device if device is not None else 'domyślne'}  "
          f"{dev_sr} Hz" + (f"  -> decymacja do {C.SR} Hz" if resample else ""))
    if monitor_only:
        print("tryb pomiaru — Ctrl+C kończy\n")
    else:
        print(f"nagrywanie {seconds:.0f} s — Ctrl+C przerywa\n")

    q: queue.Queue = queue.Queue()
    clipped = 0

    def cb(indata, frames, tinfo, status):
        if status:
            print(f"  [status audio] {status}", flush=True)
        q.put(indata[:, 0].copy())

    chunks: list[np.ndarray] = []
    # Bufor ~1,5 s w 8 kHz do pomiaru szczytu OBRAZU — pojedynczy blok
    # 100 ms jest za krótki na sensowny spektrogram.
    img_buf = np.zeros(0, dtype=np.float32)
    img_peak = float("nan")

    t0 = time.time()
    try:
        with sd.InputStream(samplerate=dev_sr, channels=1, dtype="float32",
                            blocksize=C.MIC_BLOCK, device=device, callback=cb):
            while True:
                elapsed = time.time() - t0
                if not monitor_only and elapsed >= seconds:
                    break
                try:
                    block = q.get(timeout=0.5)
                except queue.Empty:
                    continue

                if not monitor_only:
                    chunks.append(block)

                rms, peak, tone, snr = measure(block, dev_sr)
                if peak >= 0.99:
                    clipped += 1

                # Szczyt obrazu, liczony na buforze w częstotliwości modelu.
                blk8 = block
                if resample:
                    import librosa
                    blk8 = librosa.resample(block, orig_sr=dev_sr,
                                            target_sr=C.SR)
                img_buf = np.concatenate([img_buf, blk8])[-int(1.5 * C.SR):]
                if img_buf.size >= C.N_FFT * 2:
                    img_peak = image_peak_db(img_buf)

                warn = "  OBCINANIE!" if peak >= 0.95 else ""
                if np.isfinite(img_peak):
                    d = img_peak - TARGET_PEAK_DB
                    if img_peak >= C.DB_MAX - 0.5:
                        lvl = f"obraz={img_peak:5.1f}dB NASYCONY, zmniejsz o {d:+.0f}dB"
                    elif abs(d) <= 4:
                        lvl = f"obraz={img_peak:5.1f}dB  POZIOM OK"
                    else:
                        lvl = (f"obraz={img_peak:5.1f}dB  "
                               f"{'zmniejsz' if d > 0 else 'zwieksz'} "
                               f"o {abs(d):.0f}dB")
                else:
                    lvl = "obraz=  ---"
                tone_s = f"{tone:6.0f} Hz" if np.isfinite(tone) else "   ---  "
                print(f"\r  {elapsed:5.1f}s [{_bar(peak)}] "
                      f"szczyt={peak:.3f}  ton={tone_s}  {lvl}{warn}     ",
                      end="", flush=True)
    except KeyboardInterrupt:
        print("\n  przerwane")

    print()
    if monitor_only or not chunks:
        return None

    audio = np.concatenate(chunks)
    if resample:
        import librosa
        audio = librosa.resample(audio, orig_sr=dev_sr, target_sr=C.SR)

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    print(f"\nnagrane: {audio.size / C.SR:.2f}s  {C.SR} Hz mono  "
          f"szczyt={peak:.3f}")
    if clipped:
        print(f"UWAGA: {clipped} bloków obciętych — zmniejsz poziom wejścia. "
              f"Obcięty ton ma harmoniczne, których nie ma w zbiorze "
              f"treningowym, więc model będzie się mylił.")
    if peak < 0.02:
        print("UWAGA: poziom bardzo niski (szczyt < 0.02). Sygnał wypadnie "
              f"poniżej DB_MIN={C.DB_MIN} dB i obraz będzie czarny.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, audio, C.SR, subtype="PCM_16")
    print(f"zapisano: {out_path} "
          f"({out_path.stat().st_size/1024:.0f} kB)")
    print(f"\nDalej:  python -m tools.xray --wav {out_path}")
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description="Konwerter mikrofon -> wav")
    ap.add_argument("--list", action="store_true",
                    help="wypisz urządzenia wejściowe i zakończ")
    ap.add_argument("--monitor", action="store_true",
                    help="tylko pomiar poziomu i tonu, bez zapisu")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--device", type=int, default=C.MIC_DEVICE)
    ap.add_argument("--out", type=Path, default=C.OUT_DIR / "mic.wav")
    args = ap.parse_args(argv)

    if args.list:
        list_devices()
        return

    print("=" * 70)
    print("MIKROFON -> WAV")
    print("=" * 70)
    print(C.summary())
    print("-" * 70)
    record(args.seconds, args.device, args.out, monitor_only=args.monitor)


if __name__ == "__main__":
    main()
