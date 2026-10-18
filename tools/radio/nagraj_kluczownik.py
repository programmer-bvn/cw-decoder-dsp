"""NAGRANIE KLUCZOWNIKA IC-7300MK2 Z PODSŁUCHU USB (18.10).

    python tools/radio/nagraj_kluczownik.py probki/teksty/qso1_cq.txt
    python tools/radio/nagraj_kluczownik.py probki/teksty/qso2_relacja.txt --device 15

Kolejno: kluczownik.ps1 -Sucho (bezpieczniki: CW, break-in wyłączony,
odbiór; odczyt tempa kluczownika), nagranie z karty USB (tools.mic2wav,
--surowe), po 1,5 s tekst do kluczownika (kluczownik.ps1 -Trwanie: czuwa nad
stanem nadawania przez cały czas gry). Wynik: probki/kluczownik/<tekst>_
kluczownik<WPM>wpm_<stempel>.wav + .txt (pole `nadane:` = treść pliku).

Podsłuch na USB wymaga w radiu: SET > Connectors > USB AF/IF Output >
AF Beep/Speech… Output = ON. Bez tego nagranie jest ciche — skrypt to mówi.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

KORZEN = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(KORZEN))
from dsp import config as C, frontend, morse  # noqa: E402

KLUCZ = KORZEN / "tools" / "radio" / "kluczownik.ps1"


def ps(*arg: str) -> tuple[int, str]:
    r = subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                        "-File", str(KLUCZ), *arg], capture_output=True, cwd=KORZEN)
    tekst = r.stdout.decode("utf-8", "replace") + r.stderr.decode("utf-8", "replace")
    return r.returncode, tekst


def przytnij(wav: Path) -> float:
    """Przycina z przodu szum sprzed kluczownika: zostaje 0,1 s przed
    pierwszym znakiem. Zwraca przycięte sekundy. Oryginał (surowe/) bez zmian.

    Po co (18.10): tune.track_tone zaczepia się na szumie, gdy nagranie
    zaczyna się od samego szumu, i nie przechodzi na stację (HISTORIA, „Pętla
    tonu ... OTWARTE"; na 6 m: 0 okien w qso1). Rejestrator potrzebuje ~3 s
    przed kluczownikiem. Test ma mierzyć odczyt, słabość pętli — osobno."""
    import soundfile as sf
    from scipy.signal import hilbert
    from dsp import tune
    x = frontend.load_audio(wav)
    X = np.abs(np.fft.rfft(x))
    fr = np.fft.rfftfreq(x.size, 1 / C.SR)
    pas = (fr > C.FMIN) & (fr < C.FMAX)
    ton = float(fr[pas][np.argmax(X[pas])])
    e = np.abs(hilbert(tune.bandpass(x, ton, 60.0).astype(np.float64)))
    k = int(0.01 * C.SR)
    e = np.convolve(e, np.ones(k) / k, mode="same")
    nad = np.flatnonzero(e > 0.5 * np.percentile(e, 99))
    if not nad.size:
        return 0.0
    start = max(0, int(nad[0] - 0.1 * C.SR))
    if start:
        sf.write(wav, x[start:], C.SR, subtype="PCM_16")
    return start / C.SR


def civ_pasmo() -> str:
    """Częstotliwość i tryb z CI-V (tylko odczyt: civ_odczyt.ps1, polecenia 03/04)."""
    r = subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                        str(KORZEN / "tools" / "radio" / "civ_odczyt.ps1"), "-Porty", "COM5",
                        "-Bauds", "115200"], capture_output=True, cwd=KORZEN)
    t = r.stdout.decode("utf-8", "replace")
    m = re.search(r"03 ((?:[0-9A-F]{2} ){5})FD", t)
    if not m:
        return "nieodczytane"
    hz, mn = 0, 1
    for b in m.group(1).split():
        v = int(b, 16); hz += (v & 15) * mn + (v >> 4) * mn * 10; mn *= 100
    tr = re.search(r"tryb: .*04 ([0-9A-F]{2}) ([0-9A-F]{2}) FD", t)
    tryby = {"03": "CW", "07": "CW-R"}
    return (f"{hz / 1e6:.6f} MHz".replace(".", ",") + (f", {tryby.get(tr.group(1), tr.group(1))}, FIL{int(tr.group(2), 16)}" if tr else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tekst", type=Path)
    ap.add_argument("--device", type=int, default=15)
    a = ap.parse_args()

    rc, wyj = ps(str(a.tekst.resolve()), "-Sucho")
    print(wyj.strip())
    if rc != 0 or "bezpieczniki przesz" not in wyj:
        raise SystemExit("Bezpieczniki nie przeszły — nic nie nagrano.")
    m = re.search(r"tempo kluczownika: ([0-9]+[,.][0-9])", wyj)
    wpm = float(m.group(1).replace(",", ".")) if m else 20.0
    radio = re.search(r"radio: (.*)", wyj).group(1).strip()
    pasmo = civ_pasmo()
    print("pasmo:", pasmo)

    tresc = a.tekst.read_text(encoding="utf-8").strip()
    trwa = morse.synth_cw(tresc, wpm=wpm, rng=np.random.default_rng(0)).size / C.SR
    nagranie = trwa + 5.0
    stempel = datetime.now().strftime("%Y%m%d_%H%M")
    kat = KORZEN / "probki" / "kluczownik"
    kat.mkdir(parents=True, exist_ok=True)
    wav = kat / f"{a.tekst.stem}_kluczownik{wpm:.0f}wpm_{stempel}.wav"
    print(f"tekst ~{trwa:.0f} s przy {wpm:.1f} WPM; nagrywam {nagranie:.0f} s -> {wav.name}")

    rek = subprocess.Popen([sys.executable, "-m", "tools.mic2wav", "--device", str(a.device),
                            "--seconds", f"{nagranie:.0f}", "--surowe", "--out", str(wav)],
                           cwd=KORZEN, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # Kluczownik rusza dopiero, gdy strumień karty jest otwarty: rejestrator
    # potrzebuje 2-3 s na start, a pierwsza wersja (stałe 1,5 s) ucinała
    # pierwszy znak (18.10: „QCQCQ" zamiast „CQCQCQ").
    poczatek = b""
    while b"NAGRYWANIE TRWA" not in poczatek:
        linia = rek.stdout.readline()
        if not linia:
            raise SystemExit("Rejestrator nie wystartował:\n" + poczatek.decode("utf-8", "replace"))
        poczatek += linia
    # Reszta wyjścia rejestratora w osobnym wątku: postęp co blok zapełniłby
    # potok przy długim tekście i rejestrator stanąłby na zapisie.
    import threading
    reszta: list[bytes] = []
    czytnik = threading.Thread(target=lambda: reszta.append(rek.stdout.read()), daemon=True)
    czytnik.start()
    # 3 s: karta oddaje dźwięk z opóźnieniem względem otwarcia strumienia —
    # przy 1 s początek pierwszej kreski był ucięty o ~150 ms (18.10).
    time.sleep(3.0)
    rc, wyj = ps(str(a.tekst.resolve()), "-Trwanie", f"{trwa + 1:.0f}")
    print(wyj.strip())
    rek.wait()
    czytnik.join(timeout=10)
    log = (poczatek + b"".join(reszta)).decode("utf-8", "replace")
    status = sorted(set(re.findall(r"\[status audio\] ([^\r\n]+)", log)))
    if status:
        print("UWAGA, karta zgłaszała:", "; ".join(status))
    if rc != 0:
        print(log[-500:])
        raise SystemExit("Kluczownik przerwany — nagranie niepełne, opisu nie zapisuję.")
    if not wav.exists():
        print(log[-800:])
        raise SystemExit("Nagranie się nie zapisało.")

    przyciete = przytnij(wav)
    x = frontend.load_audio(wav)
    szczyt = float(np.max(np.abs(x))) if x.size else 0.0
    print(f"nagranie: {x.size / C.SR:.1f} s, szczyt {szczyt:.3f}")
    if szczyt < 0.01:
        print("UWAGA: nagranie prawie ciche — czy AF Beep/Speech… Output = ON?")
    wav.with_suffix(".txt").write_text(
        "# NAGRANIE KLUCZOWNIKA — tools/radio/nagraj_kluczownik.py\n\n"
        f"nadane:        {tresc}\n"
        f"nadajnik:      IC-7300MK2, kluczownik wewnętrzny, tekst przez CI-V 0x17 ({a.tekst.name})\n"
        f"tempo zadane:  {wpm:.1f} WPM (CI-V 0x14 0x0C)\n"
        "klucz:         kluczownik — rytm maszynowy\n"
        "tor:           podsłuch CW przez USB audio (AF Beep/Speech Output ON), break-in OFF, bez nadawania\n"
        f"pasmo:         {pasmo} (CI-V przed nagraniem)\n"
        f"radio:         {radio}; karta: urządzenie {a.device}\n"
        f"czas:          {stempel}\n"
        f"szczyt:        {szczyt:.3f}\n"
        f"przycięte:     {przyciete:.2f} s z przodu (szum sprzed kluczownika; surowe/ pełne)\n"
        f"karta:         {'; '.join(status) if status else 'bez zgłoszeń (przepełnienia itp.)'}\n",
        encoding="utf-8")
    print(f"zapisano {wav} + opis")


if __name__ == "__main__":
    main()
