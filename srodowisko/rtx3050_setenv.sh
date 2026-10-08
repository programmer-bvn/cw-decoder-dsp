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
#  ŚCIEŻKA DO VENV nie jest wpisana na sztywno — kolejno sprawdzane są,
#  i wygrywa PIERWSZY KOMPLETNY:
#      1. $CW_VENV, jeśli ustawione
#      2. ~/venv_gpu — natywny system plików WSL
#      3. venv_gpu w katalogu projektu (rodzic tego skryptu)
#      4. .venv w katalogu projektu
#  Uzasadnienie przy samym wyborze, niżej.
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
#
#  DLACZEGO ~/venv_gpu PRZED venv_gpu PROJEKTU. Zmierzone 24.09
#  (srodowisko/pomiar_dysku.sh): pod /mnt/... drobne pliki czytają się
#  12,6 razy wolniej niż na natywnym systemie plików WSL, a sam import
#  TensorFlow trwał tam 57–70 s. Noc 06.10 z ~/venv_gpu: 2–3 s.
#
#  DLACZEGO BEZ ZMIENNEJ. noc.bat uruchamia noc.sh przez
#  "wsl.exe -- ./noc.sh", czyli bez interaktywnej powłoki — ~/.bashrc
#  nie jest wtedy czytany W OGÓLE. Rada "wpisz export CW_VENV do .bashrc"
#  działała przy ręcznym ./noc.sh i nigdy przy nocy z noc.bat.
#
#  "KOMPLETNY" = ma bin/activate ORAZ zainstalowany tensorflow. Venv
#  założony do połowy (np. setup_gpu_env.sh przerwał się na pip) jest
#  POMIJANY z wyjaśnieniem, zamiast wygrać i wywrócić noc. Sprawdzenie idzie
#  po katalogu, nie przez import — import przez drvfs to minuta.
#
#  Lista kandydatów jest wypisywana w całości. Gdy na maszynie są dwa
#  venvy, z logu ma być widać oba, ich stan i który wygrał — inaczej nie
#  da się sprawdzić, czy noc zachowała się prawidłowo.
_venv=""
_powod=""
_widziane=" "
echo "kandydaci na venv (wygrywa pierwszy kompletny):"
for _k in "${CW_VENV:-}" "$HOME/venv_gpu" "$_katalog/venv_gpu" "$_katalog/.venv"; do
    [ -n "$_k" ] || continue
    case "$_widziane" in *" $_k "*) continue ;; esac
    _widziane="$_widziane$_k "
    if [ ! -f "$_k/bin/activate" ]; then
        _stan="brak"
    elif ! ls -d "$_k"/lib/python3.*/site-packages/tensorflow >/dev/null 2>&1; then
        _stan="NIEKOMPLETNY — jest venv, nie ma tensorflow"
    elif [ -n "$_venv" ]; then
        _stan="kompletny"
    else
        _venv="$_k"
        if   [ "$_k" = "${CW_VENV:-}" ];   then _powod="wskazany przez CW_VENV"
        elif [ "$_k" = "$HOME/venv_gpu" ]; then _powod="natywny system plików WSL"
        else                                    _powod="obok projektu"
        fi
        _stan="kompletny  <-- WYBRANY ($_powod)"
    fi
    echo "    $_k: $_stan"
done

if [ -n "${CW_VENV:-}" ] && [ -n "$_venv" ] && [ "$_venv" != "$CW_VENV" ]; then
    echo "UWAGA: CW_VENV=$CW_VENV się nie nadaje (stan wyżej) — biorę $_venv."
fi

if [ -z "$_venv" ]; then
    echo
    echo "BŁĄD: nie ma ANI JEDNEGO kompletnego środowiska wirtualnego."
    echo
    echo "Założenie na natywnym systemie plików WSL (szybkie, zalecane):"
    echo "    CW_VENV=\$HOME/venv_gpu ./srodowisko/setup_gpu_env.sh"
    echo "Potem nic nie trzeba ustawiać — ~/venv_gpu jest znajdowany sam."
    return 1
fi

# --- wyłączenie POPRZEDNIEGO venv ----------------------------------------
#  Gdy w powłoce był aktywny INNY venv, jego ścieżki zostają w PATH i —
#  gorzej — w LD_LIBRARY_PATH. Nowy venv stanąłby pierwszy w PATH, ale
#  biblioteki CUDA mogłyby się ładować ze starego: dwie wersje cuDNN
#  w jednym procesie dają błędy, których nikt nie skojarzy z venv.
#  Funkcja deactivate z tamtej powłoki tu nie istnieje (funkcje nie
#  przechodzą do skryptu), więc czyścimy ręcznie, po prefiksie ścieżki.
if [ -n "${VIRTUAL_ENV:-}" ] && [ "$VIRTUAL_ENV" != "$_venv" ]; then
    _stary="$VIRTUAL_ENV"
    echo "wyłączam poprzedni venv: $_stary"
    PATH="$(printf '%s' "$PATH" | tr ':' '\n' \
            | awk -v p="$_stary" 'index($0, p) != 1' | paste -sd: -)"
    LD_LIBRARY_PATH="$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' \
            | awk -v p="$_stary" 'index($0, p) != 1' | paste -sd: -)"
    export PATH LD_LIBRARY_PATH
    unset VIRTUAL_ENV
fi

source "$_venv/bin/activate"

# Dla paszportu w noc.sh: który venv i dlaczego, bez zgadywania z PATH.
export CW_VENV_WYBRANY="$_venv"
export CW_VENV_POWOD="$_powod"

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

unset _ten _katalog _venv _k _sp _nvidia _lib _nvcc _powod _widziane _stan _stary
