#!/bin/bash
# =============================================================================
#  rtx3050_setenv.sh  --  środowisko GPU dla WSL2
#
#  URUCHAMIANIE:
#      source srodowisko/rtx3050_setenv.sh
#
#  Nie przez ./ — wykonanie tworzy powłokę potomną, która ustawia zmienne
#  i natychmiast umiera razem z nimi. Skrypt to sprawdza i odmawia.
#
#  Wyjątkiem jest wysourcowanie z WNĘTRZA innego skryptu (tak robi noc.sh):
#  tam środowisko jest potrzebne dokładnie tyle, ile żyje ten skrypt, więc
#  powłoka potomna jest w porządku.
#
#  CO USTAWIA I DLACZEGO
#    - aktywuje venv z Pythonem 3.12, bo WSL domyślnie podaje 3.14, a dla
#      3.14 nie ma koła TensorFlow (PyPI ma cp310-cp313)
#    - dopisuje do LD_LIBRARY_PATH katalogi bibliotek CUDA z pipa; bez tego
#      linker ich nie znajdzie, TF cicho spadnie na CPU i noc jest stracona
#
#  ŚCIEŻKA DO VENV nie jest wpisana na sztywno — kolejno sprawdzane są:
#      1. $CW_VENV, jeśli ustawione
#      2. venv_gpu w katalogu projektu (rodzic tego skryptu)
#      3. .venv w katalogu projektu
# =============================================================================

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "BŁĄD: ten skrypt trzeba WYSOURCOWAĆ, nie wykonać."
    echo
    echo "    source ${BASH_SOURCE[0]}"
    echo
    echo "Wykonanie przez ./ tworzy powłokę potomną: zmienne zostaną"
    echo "ustawione i zginą razem z nią, a Twoja powłoka nic nie zobaczy."
    exit 1
fi

_ten="${BASH_SOURCE[0]}"
_katalog="$(cd "$(dirname "$_ten")/.." && pwd)"

# --- gdzie jest venv -----------------------------------------------------
_venv=""
for _k in "${CW_VENV:-}" "$_katalog/venv_gpu" "$_katalog/.venv"; do
    if [ -n "$_k" ] && [ -f "$_k/bin/activate" ]; then _venv="$_k"; break; fi
done

if [ -z "$_venv" ]; then
    echo "BŁĄD: nie znalazłem środowiska wirtualnego."
    echo "Szukałem w:"
    echo "    \$CW_VENV            = ${CW_VENV:-<nieustawione>}"
    echo "    $_katalog/venv_gpu"
    echo "    $_katalog/.venv"
    echo
    echo "Utworzenie:  ./srodowisko/setup_gpu_env.sh"
    return 1
fi

source "$_venv/bin/activate"

# --- biblioteki CUDA z pipa ---------------------------------------------
# Wersja Pythona NIE jest wpisana na sztywno: katalog site-packages nosi
# nazwę pythonX.Y i przy zmianie interpretera ścieżka wpisana na sztywno
# przestaje istnieć w milczeniu.
_sp="$(ls -d "$_venv"/lib/python3.*/site-packages 2>/dev/null | head -1)"

if [ -z "$_sp" ]; then
    echo "UWAGA: nie znalazłem site-packages w $_venv — LD_LIBRARY_PATH bez zmian."
else
    _nvidia=""
    for _lib in "$_sp"/nvidia/*/lib; do
        [ -d "$_lib" ] && _nvidia="$_nvidia${_nvidia:+:}$_lib"
    done
    if [ -n "$_nvidia" ]; then
        export LD_LIBRARY_PATH="$_nvidia${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    else
        echo "UWAGA: brak katalogów nvidia/*/lib w $_sp"
        echo "       Karta prawdopodobnie NIE będzie widziana."
        echo "       Sprawdzenie: ./srodowisko/gpu_libs.sh"
    fi
fi

# --- libdevice dla XLA ---------------------------------------------------
_nvcc="$_sp/nvidia/cuda_nvcc"
if [ -d "$_nvcc/nvvm/libdevice" ]; then
    export XLA_FLAGS="--xla_gpu_cuda_data_dir=$_nvcc"
fi

# --- meldunek ------------------------------------------------------------
echo "venv:   $_venv"
echo "python: $(command -v python)  ($(python --version 2>&1))"
case "$(python --version 2>&1)" in
    *3.1[0-3]*) : ;;
    *) echo "UWAGA: TensorFlow nie ma kół dla tej wersji Pythona"
       echo "       (PyPI ma cp310-cp313). Karta nie zadziała." ;;
esac
echo "CUDA:   $(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -c nvidia) katalogów w LD_LIBRARY_PATH"

unset _ten _katalog _venv _k _sp _nvidia _lib _nvcc
