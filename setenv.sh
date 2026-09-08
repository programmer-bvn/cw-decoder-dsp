#!bash
# Wejście do środowiska bash jako root:  wsl -u root
#
# UWAGA — TEN PLIK TRZEBA ZAŁADOWAĆ, NIE URUCHOMIĆ:
#
#     source setenv.sh          albo      . setenv.sh
#
# NIE:  ./setenv.sh
#
# `./setenv.sh` odpala PODPROCES. Aktywacja venva i wszystkie export-y
# umierają razem z nim, więc po powrocie do powłoki `python` nie istnieje
# i wychodzi "Command 'python' not found, did you mean python3".
# Zdarzyło się to i kosztowało kilka minut zgadywania, dlatego poniżej
# jest kontrola, która sama to wykryje.

# --- czy jestem załadowany, czy uruchomiony? ---
# W bashu $0 przy `source` to nazwa powłoki, a przy uruchomieniu — nazwa
# pliku. BASH_SOURCE[0] wskazuje na plik w obu przypadkach.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "BŁĄD: ten plik trzeba ZAŁADOWAĆ, nie uruchomić."
    echo
    echo "    source setenv.sh"
    echo
    echo "Uruchomiony przez ./setenv.sh nie zmieni nic w twojej powłoce —"
    echo "venv aktywuje się w podprocesie i zniknie razem z nim."
    exit 1
fi

# --- katalog projektu = katalog tego pliku, nie katalog bieżący ---
# Dzięki temu `source setenv.sh` działa niezależnie od tego, skąd wołane.
PROJEKT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "$PROJEKT/venv_gpu" ]; then
    echo "BŁĄD: brak $PROJEKT/venv_gpu"
    echo "Utwórz środowisko:"
    echo "    python3 -m venv venv_gpu"
    echo "    source venv_gpu/bin/activate"
    echo "    pip install 'tensorflow[and-cuda]'"
    return 1
fi

source "$PROJEKT/venv_gpu/bin/activate"

export VIRTUAL_ENV="$PROJEKT/venv_gpu"
export PATH="$VIRTUAL_ENV/bin:$PATH"

# Biblioteki CUDA i cuDNN pochodzą z KÓŁ PIPA (tensorflow[and-cuda]),
# nie z CUDA Toolkit. Toolkit jest potrzebny tylko do kompilowania
# aplikacji CUDA; do uruchamiania wystarczy sterownik NVIDIA na Windows,
# który WSL widzi jako libcuda.so. Sterownika linuksowego w WSL nie
# wolno instalować, bo nadpisze ten stub.
#
# Wersja Pythona nie jest wpisana na sztywno — venv może być zrobiony
# innym niż 3.12 i wtedy ścieżka by nie istniała.
SITE_PACKAGES="$(ls -d "$VIRTUAL_ENV"/lib/python*/site-packages 2>/dev/null | head -1)"

if [ -z "$SITE_PACKAGES" ]; then
    echo "OSTRZEŻENIE: nie znalazłem site-packages w $VIRTUAL_ENV/lib"
else
    for LIB in nvjitlink cudnn cuda_runtime cublas cufft curand \
               cusolver cusparse nccl cuda_nvrtc; do
        D="$SITE_PACKAGES/nvidia/$LIB/lib"
        [ -d "$D" ] && export LD_LIBRARY_PATH="$D:$LD_LIBRARY_PATH"
    done
fi

echo "Środowisko RTX 3050 załadowane."
echo "  projekt:  $PROJEKT"
echo "  python:   $(command -v python) ($(python --version 2>&1))"
echo "  katalogi CUDA na LD_LIBRARY_PATH: $(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -c nvidia)"
echo
echo "Sprawdzenie karty:"
echo "  python -c \"import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))\""
