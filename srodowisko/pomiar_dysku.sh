#!/bin/bash
# =============================================================================
#  pomiar_dysku.sh  --  drvfs kontra natywny system plików WSL
#
#      ./srodowisko/pomiar_dysku.sh
#
#  PO CO. 23.09 padło pytanie, czy przeniesienie venv_gpu z /mnt/... na
#  natywny system plików WSL coś da. Pierwszy pomiar (dd, 24.09) dał
#  115 MB/s dla kopii i 1,7 GB/s dla odczytu z ext4 — i NIE ODPOWIADA
#  na to pytanie, z dwóch powodów.
#
#  POWÓD PIERWSZY: 1,7 GB/s jest powyżej fizycznego sufitu SATA3 (~550
#  MB/s). `echo 3 > /proc/sys/vm/drop_caches` czyści pamięć podręczną
#  Linuksa W ŚRODKU, ale plik ext4 WSL-a to VHDX leżący na dysku Windows
#  i to host go buforuje. Ten odczyt poszedł z RAM-u, nie z dysku.
#
#  POWÓD DRUGI, WAŻNIEJSZY: import TensorFlow nie jest wolny przez
#  przepustowość. To kilkanaście tysięcy DROBNYCH plików, a drvfs płaci
#  narzut na każdym otwarciu osobno. Wąskim gardłem jest LATENCJA NA
#  PLIK, a sekwencyjny dd jej w ogóle nie dotyka. Mierzenie MB/s w tej
#  sprawie to mierzenie nie tego.
#
#  Ten skrypt mierzy jedno i drugie osobno, żeby było widać, które
#  z nich decyduje.
#
#  NIC NIE INSTALUJE I NIC NIE PRZENOSI. Pisze wyłącznie do swoich
#  katalogów tymczasowych i sprząta po sobie.
# =============================================================================

set -u

ILE=${1:-2000}          # ile drobnych plików w pomiarze latencji
MB=${2:-400}            # ile MB w pomiarze przepustowości

if [ ! -f dsp/config.py ]; then
    echo "BŁĄD: uruchom to z katalogu projektu (nie widzę dsp/config.py)."
    exit 1
fi

# TYLKO W WSL. Ten pomiar zapisuje TYSIĄCE drobnych plików w katalogu
# projektu — o to w nim chodzi. Na maszynie treningowej ląduje to na SSD
# i nic nie szkodzi, ale uruchomiony omyłkowo na pendraku pisałby po
# exFAT, który nie ma wear-levelingu. Cały układ z na_hdd.bat istnieje
# właśnie po to, żeby takich zapisów na pendraku NIE było.
#
# Zabezpieczenie dopisane po tym, jak sam odpaliłem go na pendraku
# przy próbie na sucho. Wyszło 7 plików na sekundę i musiałem przerwać.
if ! grep -qi microsoft /proc/version 2>/dev/null; then
    echo "BŁĄD: to ma sens tylko wewnątrz WSL."
    echo "  Porównuje drvfs z natywnym systemem plików WSL, więc poza WSL"
    echo "  nie ma czego z czym zestawiać. A przy okazji zapisuje tysiące"
    echo "  drobnych plików — na pendraku (exFAT, bez wear-levelingu) jest"
    echo "  to dokładnie ten rodzaj zapisu, którego ten projekt unika."
    exit 1
fi

TU="$(pwd)/_pomiar_drvfs"
TAM="$HOME/_pomiar_ext4"
sprzataj() { rm -rf "$TU" "$TAM"; }
trap sprzataj EXIT

echo "==========================================================="
echo " DRVFS kontra natywny system plików WSL"
echo "==========================================================="
echo "  tutaj (projekt): $(pwd)"
echo "  tam   (domowy):  $HOME"
echo

# Sprawdzenie, czy to w ogóle dwa różne systemy plików. Gdyby projekt
# leżał już na ext4, cały pomiar nie miałby sensu, a wynik wyglądałby
# jak "drvfs jest szybki".
RODZ_TU="$(stat -f -c %T . 2>/dev/null || echo '?')"
RODZ_TAM="$(stat -f -c %T "$HOME" 2>/dev/null || echo '?')"
echo "  system plików tutaj: $RODZ_TU      tam: $RODZ_TAM"
if [ "$RODZ_TU" = "$RODZ_TAM" ]; then
    echo
    echo "  UWAGA: po obu stronach to samo. Projekt już leży na natywnym"
    echo "  systemie plików WSL i nie ma tu czego porównywać."
fi
echo

mkdir -p "$TU" "$TAM"

# --- 1. LATENCJA NA PLIK --------------------------------------------------
#  To jest ta liczba, która rządzi importem TensorFlow.
echo "1. WIELE DROBNYCH PLIKÓW  ($ILE plików po ~2 kB)"
echo "   Tak wygląda import biblioteki: dużo otwarć, mało bajtów."
echo

for para in "drvfs $TU" "ext4 $TAM"; do
    set -- $para
    NAZWA="$1"; KAT="$2"
    # Zapis
    # printf JEST WBUDOWANE w powłokę — żadnego fork+exec na plik.
    # Pierwsza wersja używała tu `head -c 2048`, czyli osobnego procesu
    # na każdy plik, i mierzyła koszt URUCHAMIANIA PROCESÓW (7 plików/s
    # na próbie), a nie system plików. Błąd wyszedł dopiero przy próbie
    # na sucho — dlatego takie próby się robi.
    TRESC="$(printf '%2048s' '')"
    T0=$(date +%s.%N)
    i=0
    while [ "$i" -lt "$ILE" ]; do
        printf '%s' "$TRESC" > "$KAT/p_$i.dat"
        i=$((i + 1))
    done
    T1=$(date +%s.%N)
    sync
    # Odczyt
    T2=$(date +%s.%N)
    cat "$KAT"/p_*.dat > /dev/null
    T3=$(date +%s.%N)

    awk -v n="$ILE" -v tz="$T0" -v tk="$T1" -v oz="$T2" -v ok="$T3" \
        -v nazwa="$NAZWA" 'BEGIN {
        zap = tk - tz; odc = ok - oz;
        printf "   %-6s zapis %6.2f s (%6.0f plik/s)   odczyt %6.2f s (%7.0f plik/s)\n",
               nazwa, zap, n/zap, odc, n/odc
    }'
done
echo

# --- 2. PRZEPUSTOWOŚĆ -----------------------------------------------------
#  Dla porządku, żeby obie liczby stały obok siebie. Cache Linuksa
#  czyścimy, ale host swojego i tak nie odda -- stąd ostrzeżenie niżej.
echo "2. JEDEN DUŻY PLIK  (${MB} MB)"
echo

for para in "drvfs $TU" "ext4 $TAM"; do
    set -- $para
    NAZWA="$1"; KAT="$2"
    dd if=/dev/zero of="$KAT/duzy.bin" bs=1M count="$MB" \
        status=none conv=fsync 2>/dev/null
    sync
    # Przekierowanie musi byc W SRODKU podpowloki, inaczej blad otwarcia
    # pliku leci na ekran, zanim '|| true' zdazy cokolwiek zrobic.
    sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
    WYNIK="$(dd if="$KAT/duzy.bin" of=/dev/null bs=1M 2>&1 | tail -1)"
    printf "   %-6s %s\n" "$NAZWA" "${WYNIK##*, }"
    rm -f "$KAT/duzy.bin"
done
echo
echo "   (ta druga liczba bywa zawyżona: drop_caches czyści pamięć"
echo "    podręczną Linuksa, ale VHDX z ext4 buforuje Windows i tego"
echo "    stąd nie widać. Odczyt powyżej ~550 MB/s na SATA3 to znak,"
echo "    że dane przyszły z RAM-u hosta, nie z dysku.)"
echo

# --- 3. TO, O CO NAPRAWDĘ CHODZI -----------------------------------------
echo "3. IMPORT TENSORFLOW  (kilkanaście tysięcy drobnych plików)"
echo
if command -v python >/dev/null 2>&1; then
    GDZIE="$(python -c 'import sys; print(sys.prefix)' 2>/dev/null)"
    T0=$(date +%s.%N)
    python -c 'import tensorflow' >/dev/null 2>&1
    T1=$(date +%s.%N)
    # Czas przez awk (liczby zmiennoprzecinkowe), ale SCIEZKA osobno:
    # awk traktuje '\U' w -v jako sekwencje ucieczki i psuje kazda sciezke
    # windowsowa, np. C:\Users\...
    CZAS="$(awk -v t0="$T0" -v t1="$T1" 'BEGIN { printf "%.1f", t1 - t0 }')"
    echo "   ${CZAS} s   (venv: ${GDZIE})"
    case "$GDZIE" in
        /mnt/*) echo
                echo "   venv leży za drvfs. Jeśli powyżej jest kilkadziesiąt"
                echo "   sekund, a w punkcie 1 ext4 wygrywa wielokrotnie,"
                echo "   to przeniesienie SAMYCH BIBLIOTEK się opłaci:"
                echo "       export CW_VENV=\$HOME/venv_gpu"
                echo "       ./srodowisko/setup_gpu_env.sh"
                echo "   Dane i kod zostają tam, gdzie są." ;;
        *)      echo "   venv jest już poza /mnt — nie ma co przenosić." ;;
    esac
else
    echo "   nie ma 'python' w PATH — wysourcuj środowisko i powtórz"
fi
echo
echo "==========================================================="
echo "JAK TO CZYTAĆ. Jeśli w punkcie 1 ext4 jest wielokrotnie szybszy,"
echo "a w punkcie 2 różnica jest niewielka — wąskim gardłem jest"
echo "latencja na plik, nie przepustowość. Wtedy przenosi się to, co"
echo "składa się z tysięcy małych plików (venv), a NIE to, co jest"
echo "kilkoma wielkimi (zbiór treningowy)."
echo "==========================================================="
