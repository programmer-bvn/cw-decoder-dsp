#!/bin/bash
# =============================================================================
#  setup_gpu_env.sh  --  jednorazowa konfiguracja TensorFlow GPU (WSL2)
#
#      ./srodowisko/setup_gpu_env.sh
#
#  Sprzęt odniesienia: RTX 3050, WSL2 na Windows 11, TensorFlow 2.21, CUDA 12.
#
#  RÓŻNICA WOBEC WERSJI, KTÓRA POWSTAŁA W KONSOLI: interpreter Pythona jest
#  WSKAZANY JAWNIE. Pierwotna wersja robiła `python3 -m venv`, a `python3`
#  w aktualnym WSL to 3.14 — dla którego TensorFlow nie ma koła na PyPI
#  (są cp310-cp313). Instalacja rozsypywała się wtedy na `pip install
#  tensorflow` z komunikatem o braku pasującej wersji, co nie wskazuje
#  na przyczynę.
#
#  Po zakończeniu środowisko ładuje się przez:
#      source srodowisko/rtx3050_setenv.sh
# =============================================================================

set -e

KATALOG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${CW_VENV:-$KATALOG/venv_gpu}"

echo "========================================================================"
echo " KONFIGURACJA TENSORFLOW GPU (WSL2)"
echo "========================================================================"

# --- 0. interpreter: JAWNIE, nie 'python3' -------------------------------
# Kolejność od najlepiej sprawdzonego. 3.14 świadomie nieobsługiwane.
PY=""
for kandydat in python3.12 python3.11 python3.13 python3.10; do
    if command -v "$kandydat" >/dev/null 2>&1; then PY="$kandydat"; break; fi
done

if [ -z "$PY" ]; then
    echo "PRZERWANO: nie znalazłem Pythona w wersji 3.10-3.13."
    echo
    echo "Domyślny python3 to: $(python3 --version 2>&1 || echo brak)"
    echo "TensorFlow ma koła tylko dla cp310-cp313, więc na nowszym nie pojedzie."
    echo
    echo "Instalacja 3.12 obok istniejącego:"
    echo "    sudo add-apt-repository ppa:deadsnakes/ppa"
    echo "    sudo apt update && sudo apt install python3.12 python3.12-venv"
    exit 1
fi
echo "[0/5] interpreter: $PY ($($PY --version 2>&1))"

# --- 1. venv -------------------------------------------------------------
if [ ! -d "$ENV_NAME" ]; then
    echo "[1/5] tworzę środowisko $ENV_NAME"
    "$PY" -m venv --system-site-packages "$ENV_NAME"
else
    echo "[1/5] środowisko $ENV_NAME już istnieje"
fi

source "$ENV_NAME/bin/activate"
pip install --upgrade pip setuptools wheel -q

# --- 2. TensorFlow i CUDA 12 --------------------------------------------
echo "[2/5] instaluję TensorFlow i biblioteki NVIDIA CUDA 12"
pip install tensorflow \
            nvidia-cuda-runtime-cu12 \
            nvidia-cublas-cu12 \
            nvidia-cudnn-cu12 \
            nvidia-cusparse-cu12 \
            nvidia-cufft-cu12 \
            nvidia-curand-cu12 \
            nvidia-cusolver-cu12 \
            nvidia-nvjitlink-cu12 \
            nvidia-cuda-nvcc-cu12 -q

# Wersja Pythona w ścieżce site-packages NIE jest wpisywana na sztywno.
SITE="$(ls -d "$ENV_NAME"/lib/python3.*/site-packages | head -1)"
NVCC_DIR="$SITE/nvidia/cuda_nvcc"

# --- 3. symlinki do /usr/lib/wsl/lib ------------------------------------
#  UWAGA: to katalog WSL-a na jego własne biblioteki sterownika GPU.
#  Rozwiązanie działa i to ono uruchomiło kartę, ale ma dwie wady:
#  aktualizacja WSL może nadpisać ten katalog, a symlinki wskazujące na
#  usunięty venv zostaną wiszące. Mniej inwazyjna droga to samo
#  LD_LIBRARY_PATH — robi to rtx3050_setenv.sh i wystarcza do treningu.
#  Ten krok można pominąć: SKIP_SYMLINKS=1 ./srodowisko/setup_gpu_env.sh
if [ "${SKIP_SYMLINKS:-0}" = "1" ]; then
    echo "[3/5] symlinki pominięte (SKIP_SYMLINKS=1)"
else
    echo "[3/5] symlinki bibliotek .so oraz libdevice do /usr/lib/wsl/lib"
    if [ "$(id -u)" != "0" ]; then
        echo "      brak roota — pomijam; LD_LIBRARY_PATH z rtx3050_setenv.sh wystarczy"
    else
        mkdir -p /usr/lib/wsl/lib
        find "$SITE/nvidia/" -name "*.so*" -exec ln -sf {} /usr/lib/wsl/lib/ \;
        if [ -f "$NVCC_DIR/nvvm/libdevice/libdevice.10.bc" ]; then
            ln -sf "$NVCC_DIR/nvvm/libdevice/libdevice.10.bc" /usr/lib/wsl/lib/libdevice.10.bc
        fi
        ldconfig 2>/dev/null || true
    fi
fi

# --- 4. zmienne przy aktywacji venv -------------------------------------
echo "[4/5] dopisuję LD_LIBRARY_PATH i XLA_FLAGS do aktywacji venv"
if ! grep -q "XLA_FLAGS" "$ENV_NAME/bin/activate"; then
    {
        echo 'export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH'
        echo "export XLA_FLAGS=--xla_gpu_cuda_data_dir=$NVCC_DIR"
    } >> "$ENV_NAME/bin/activate"
fi
source "$ENV_NAME/bin/activate"

# --- 5. weryfikacja ------------------------------------------------------
echo "[5/5] sprawdzam"
"$(dirname "${BASH_SOURCE[0]}")/gpu_libs.sh" || true

echo "========================================================================"
echo " Gotowe. Przy każdym uruchomieniu:"
echo "     source srodowisko/rtx3050_setenv.sh"
echo "========================================================================"
