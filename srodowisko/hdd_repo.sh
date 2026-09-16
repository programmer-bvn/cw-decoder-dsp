#!/bin/bash
# =============================================================================
#  hdd_repo.sh  --  katalog roboczy na HDD staje się repozytorium
#
#      ./srodowisko/hdd_repo.sh
#      ./srodowisko/hdd_repo.sh git@github.com:programmer-bvn/cw-decoder-dsp.git
#
#  PO CO. Etap 7 nocy (commit + paczka + push) pomijał się z komunikatem
#  "to nie jest repozytorium git". Nie dlatego, że brakowało gita, tylko
#  dlatego, że katalog na HDD powstaje przez robocopy z pendraka, a ta
#  kopia nie zawiera .git. Po tym skrypcie wyniki mają DRUGĄ DROGĘ
#  powrotu, niezależną od pendraka.
#
#  Ruch jest znikomy: całe repozytorium to około 250 kB, a nagrania (106 MB)
#  w nim nie leżą. Nocny commit to kilkadziesiąt kB logów i pomiarów.
#
#  NIEDESTRUKCYJNIE. To najważniejsza cecha tego skryptu: NIE dotyka ani
#  jednego pliku w katalogu roboczym. Zamiast `clone` albo `checkout -f`
#  robi `git reset --mixed`, który ustawia tylko HEAD i indeks. Na HDD leżą
#  wytrenowane modele, zbiór 4 GB i venv — nie ma mowy, żeby skrypt
#  ustawiający repozytorium mógł je ruszyć.
#
#  Efekt uboczny jest pożyteczny: zaraz po uruchomieniu `git status` pokaże,
#  CZYM kopia na HDD różni się od repozytorium. Jeśli czymkolwiek poza
#  wynikami — znaczy, że robocopy coś przeoczył albo ktoś poprawiał kod
#  po jednej stronie.
# =============================================================================

set -u

ZDALNE="${1:-https://github.com/programmer-bvn/cw-decoder-dsp.git}"
GALAZ="${2:-main}"

if [ ! -f dsp/config.py ]; then
    echo "BŁĄD: uruchom to z katalogu projektu (nie widzę dsp/config.py)."
    exit 1
fi

if ! command -v git >/dev/null 2>&1; then
    echo "BŁĄD: nie ma gita."
    echo "    sudo apt update && sudo apt install git"
    exit 1
fi

KATALOG="$(pwd)"

if [ -d .git ]; then
    echo "To już jest repozytorium."
    echo "  zdalne: $(git remote get-url origin 2>/dev/null || echo '<brak>')"
    echo "  gałąź:  $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '<brak>')"
    echo
    echo "Odświeżenie kodu z repozytorium:  git pull --ff-only"
    exit 0
fi

echo "katalog: $KATALOG"
echo "zdalne:  $ZDALNE"
echo

# /mnt/c nie zapisuje właściciela w sposób, który git uznaje za wiarygodny,
# a przy uruchomieniu jako root to kończy się odmową "dubious ownership".
git config --global --add safe.directory "$KATALOG" 2>/dev/null || true

git init -q
git config core.fileMode false      # drvfs nie trzyma bitu wykonywalnego
git config core.autocrlf false      # zakończenia linii pilnuje .gitattributes

# Tożsamość na potrzeby nocnych commitów. Adres noreply, bo repozytorium
# jest publiczne — ten sam, który jest ustawiony po stronie pendraka.
git config user.name  "Mariusz Lisowski (SQ2BVN)"
git config user.email "229903418+programmer-bvn@users.noreply.github.com"

git remote add origin "$ZDALNE"

echo "pobieram historię..."
if ! git fetch -q origin "$GALAZ"; then
    echo
    echo "NIE UDAŁO SIĘ pobrać. Najczęstsze powody:"
    echo "  - brak sieci albo 4G nie odpowiada"
    echo "  - przy adresie https potrzebne są poświadczenia; w WSL ich nie ma"
    echo "    -> użyj deploy key i adresu git@github.com:..."
    echo
    echo "Repozytorium zostało utworzone, ale bez historii. Można powtórzyć:"
    echo "    git fetch origin $GALAZ"
    exit 1
fi

# reset --mixed: ustawia HEAD i indeks, NIE RUSZA plików.
git symbolic-ref HEAD "refs/heads/$GALAZ"
git reset --mixed -q "origin/$GALAZ"
git branch --set-upstream-to="origin/$GALAZ" "$GALAZ" >/dev/null 2>&1 || true

echo
echo "gotowe: gałąź $GALAZ śledzi origin/$GALAZ"
echo

ILE=$(git status --porcelain | grep -vc '^??' || true)
echo "Różnice wobec repozytorium (bez plików nieśledzonych): $ILE"
if [ "${ILE:-0}" -gt 0 ]; then
    git status --short | grep -v '^??' | head -15
    echo
    echo "Jeśli to tylko wyniki (runs/, out/) — w porządku, tak ma być."
    echo "Jeśli są tam pliki KODU, kopia na HDD rozjechała się z repozytorium:"
    echo "    git diff -- <plik>          co się różni"
    echo "    git checkout -- <plik>      weź wersję z repozytorium"
fi

echo
echo "Od teraz:"
echo "  git pull --ff-only     odświeżenie kodu (zamiast na_hdd.bat)"
echo "  ./noc.sh               etap 7 zacommituje wyniki i zrobi paczkę"
echo
echo "na_hdd.bat wykryje .git i NIE nadpisze już kodu przez robocopy —"
echo "nadal będzie dowoził nagrania probki/*.wav, bo tych w repo nie ma."
