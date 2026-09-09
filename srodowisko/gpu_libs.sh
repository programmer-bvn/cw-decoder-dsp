#!/bin/bash
# =============================================================================
#  gpu_libs.sh  --  KTÓRA biblioteka CUDA się nie ładuje
#
#  TensorFlow przy braku biblioteki wypisuje "Could not load dynamic library"
#  i cicho przechodzi na procesor. Ten skrypt sprawdza każdą osobno, więc
#  widać nazwę winowajcy zamiast jednego ogólnego ostrzeżenia.
#
#  Uruchamiać PO wysourcowaniu środowiska:
#      source srodowisko/rtx3050_setenv.sh
#      ./srodowisko/gpu_libs.sh
# =============================================================================

if ! command -v python >/dev/null 2>&1; then
    echo "BŁĄD: nie ma 'python' w PATH."
    echo "Najpierw:  source srodowisko/rtx3050_setenv.sh"
    exit 1
fi

python - <<'PY'
import ctypes
import sys

# Wersje .so odpowiadają CUDA 12 / cuDNN 9, czyli temu, co ciągnie
# pip install nvidia-*-cu12 dla TensorFlow 2.21.
BIBLIOTEKI = [
    ("libcuda.so.1",       "sterownik NVIDIA (po stronie Windows, przez WSL)"),
    ("libcudart.so.12",    "CUDA runtime"),
    ("libcublas.so.12",    "cuBLAS — mnożenie macierzy"),
    ("libcudnn.so.9",      "cuDNN — splot; bez tego nie ma sieci splotowej"),
    ("libcusparse.so.12",  "cuSPARSE"),
    ("libnvJitLink.so.12", "nvJitLink — kompilacja JIT"),
    ("libcufft.so.11",     "cuFFT"),
    ("libcurand.so.10",    "cuRAND"),
    ("libcusolver.so.11",  "cuSOLVER"),
]

print("=== biblioteki dynamiczne (ctypes) ===")
brak = []
for nazwa, opis in BIBLIOTEKI:
    try:
        ctypes.CDLL(nazwa)
        print(f"  [ OK ] {nazwa:22s} {opis}")
    except OSError as e:
        brak.append(nazwa)
        print(f"  [BRAK] {nazwa:22s} {opis}")
        print(f"         {e}")

print("\n=== rejestracja karty w TensorFlow ===")
try:
    import tensorflow as tf
except Exception as e:
    print("  nie mogę zaimportować tensorflow:", e)
    sys.exit(1)

karty = tf.config.list_physical_devices("GPU")
print(f"  TensorFlow {tf.__version__}, Python {sys.version.split()[0]}")
print(f"  karty: {[k.name for k in karty] or 'BRAK'}")

print()
if karty and not brak:
    print("Środowisko w porządku.")
    sys.exit(0)

if brak:
    print("Brakuje bibliotek:", " ".join(brak))
    print("Najczęstsza przyczyna: katalogów nvidia/*/lib nie ma na ścieżce")
    print("ładowania. Naprawa:  source srodowisko/rtx3050_setenv.sh")
    print("Jeśli to nie pomoże, biblioteki nie są zainstalowane w tym venv:")
    print("    ./srodowisko/setup_gpu_env.sh")
if not karty:
    print("Karta nie jest zarejestrowana w TensorFlow.")
    print("Sprawdź jeszcze stronę Windows:  nvidia-smi")
sys.exit(2)
PY
