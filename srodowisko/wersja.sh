#!/bin/bash
# =============================================================================
#  wersja.sh  --  numer wersji kodu i spis jego plików (WERSJA.txt)
#
#      ./srodowisko/wersja.sh nowa       nadaj nowy numer, przelicz spis
#      ./srodowisko/wersja.sh sprawdz    czy pliki na dysku = spis
#      ./srodowisko/wersja.sh pokaz      sam numer (dla innych skryptów)
#
#  PO CO. Do 05.10 nie dało się ustalić, jaki kod wykonała noc. 28.09
#  maszyna liczyła kodem sprzed tygodnia — na_hdd.bat przestał go kopiować,
#  bo na HDD pojawił się katalog .git — i żaden log tego nie pokazał. Noc
#  padła na błędzie, którego poprawka od czterech dni leżała na GitHubie.
#
#  CO JEST W WERSJA.txt:
#      linia 1:      numer, np. 2026.10.08-01
#      linie dalej:  <odcisk> <ścieżka>  dla każdego pliku kodu
#
#  ODCISK = identyfikator gita (blob SHA-1) liczony z treści bez znaków CR.
#  Dlaczego tak:
#    - zgadza się z `git rev-parse HEAD:<plik>`, więc z każdego logu da się
#      wskazać commit
#    - nie zależy od końców linii: .bat ma na dysku CRLF, w gicie LF
#    - nie potrzebuje gita — wystarczy Python
#  Sprawdzone 05.10: odcisk z archiwum GitHuba, z pendraka i z gita są
#  identyczne co do bajtu.
#
#  DLACZEGO PYTHON, A NIE tr/wc/sha1sum. Pierwsza wersja liczyła odcisk
#  narzędziami powłoki: osiem procesów na plik, kilkaset na przebieg.
#  W Git Bashu na komputerze w pracy część z nich ginęła przy tworzeniu
#  ("fork: retry: Resource temporarily unavailable", "cygheap read copy
#  failed") i wychodziły FAŁSZYWE niezgodności — 46 plików sprawdzonych
#  z 54, a do tego "inna treść" plików, których nikt nie ruszał. Narzędzie,
#  które ma mówić prawdę o kodzie, nie może samo zmyślać. Teraz cały
#  przebieg to jeden proces.
#
#  NUMER = data i kolejny numer tego dnia. Porównywalny jako NAPIS, także
#  w cmd.exe — noc.bat wybiera po nim, czy kod z GitHuba nie jest starszy
#  od kodu z pendraka (sprawdzone: "-10" > "-09", październik > wrzesień).
#
#  SPIS OBEJMUJE dokładnie to, co na_hdd.bat dowozi na HDD: *.py *.sh *.bat
#  *.md *.txt *.cfg oraz LICENSE, .gitignore, .gitattributes — bez out/,
#  runs/ i samego WERSJA.txt. Co dowiezione, to sprawdzane; co sprawdzane,
#  to dowiezione.
#
#  KTO WOŁA:
#    nowa     — przed commitem zmian w kodzie; hak pre-commit nie przepuści
#               commitu z nieaktualnym spisem, więc nie da się zapomnieć
#    sprawdz  — noc.sh na samym początku, zanim cokolwiek policzy;
#               hak pre-commit przy każdym commicie
# =============================================================================

set -u

KAT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLIK="$KAT/WERSJA.txt"

# Pierwszy Python, który NAPRAWDĘ działa. Samo `command -v` nie wystarcza:
# w Windows "python3" bywa atrapą ze Sklepu, która tylko wypisuje komunikat.
PY=""
for _k in python3 python py; do
    if command -v "$_k" >/dev/null 2>&1 && "$_k" -c 'import hashlib' >/dev/null 2>&1
    then PY="$_k"; break; fi
done

# JEDYNA definicja odcisku — ta sama przy nadawaniu wersji i przy
# sprawdzaniu, na każdej maszynie. Wypisuje tylko ASCII: konsola Windows
# i WSL różnią się kodowaniem, a te linie czyta dalej powłoka.
#   licz:    ścieżki ze stdin  ->  "<odcisk> <ścieżka>"
#   sprawdz: WERSJA.txt        ->  "ZLE <ścieżka>" / "BRAK <ścieżka>" /
#                                  na końcu "WYNIK <numer> <dobre> <zle> <brak>"
odciski() {
    "$PY" -c '
import hashlib, os, sys
tryb, kat = sys.argv[1], sys.argv[2]

def odcisk(sciezka):
    tresc = open(sciezka, "rb").read().replace(b"\r", b"")
    return hashlib.sha1(b"blob %d\0" % len(tresc) + tresc).hexdigest()

if tryb == "licz":
    for linia in sys.stdin:
        p = linia.strip("\r\n")
        if p and os.path.isfile(os.path.join(kat, p)):
            print(odcisk(os.path.join(kat, p)) + " " + p)
else:
    wiersze = open(os.path.join(kat, "WERSJA.txt"), "rb").read() \
        .decode("utf-8").replace("\r", "").split("\n")
    dobre = zle = brak = 0
    for w in wiersze[1:]:
        if not w.strip():
            continue
        spis, p = w.split(" ", 1)
        f = os.path.join(kat, p)
        if not os.path.isfile(f):
            print("BRAK " + p); brak += 1
        elif odcisk(f) != spis:
            print("ZLE " + p); zle += 1
        else:
            dobre += 1
    print("WYNIK %s %d %d %d" % (wiersze[0].strip(), dobre, zle, brak))
' "$1" "$KAT" | tr -d '\r'
    # tr: Python pod Windows kończy linie "\r\n". Doklejony CR psuł potem
    # arytmetykę na liczbach z linii WYNIK — w WSL tego nie ma, ale nowa
    # i hak pre-commit chodzą właśnie pod Windows.
}

# Pliki kodu: tylko ŚLEDZONE przez gita (doraźne skrypty w korzeniu nie
# wchodzą), z tym samym wzorcem, którym kopiuje na_hdd.bat.
pliki_kodu() {
    git -C "$KAT" ls-files \
        | grep -E '\.(py|sh|bat|md|txt|cfg)$|(^|/)(LICENSE|\.gitignore|\.gitattributes)$' \
        | grep -vE '^(out|runs)/' \
        | grep -vx 'WERSJA.txt' \
        | sort
}

case "${1:-pokaz}" in

pokaz)
    if [ -f "$PLIK" ]; then head -1 "$PLIK" | tr -d '\r'
    else echo "nieznana"; fi
    ;;

nowa)
    command -v git >/dev/null 2>&1 || {
        echo "BŁĄD: 'nowa' potrzebuje gita (lista plików kodu)."; exit 1; }
    [ -n "$PY" ] || { echo "BŁĄD: nie ma działającego Pythona."; exit 1; }

    DZIS="$(date +%Y.%m.%d)"
    STARA=""
    [ -f "$PLIK" ] && STARA="$(head -1 "$PLIK" | tr -d '\r')"
    case "$STARA" in
        "$DZIS"-*) NR=$(( 10#${STARA##*-} + 1 )) ;;
        *)         NR=1 ;;
    esac
    NOWA="$(printf '%s-%02d' "$DZIS" "$NR")"

    LISTA="$(pliki_kodu)"
    SPIS="$(printf '%s\n' "$LISTA" | odciski licz)"
    ILE_LISTA="$(printf '%s\n' "$LISTA" | grep -c .)"
    ILE_SPIS="$(printf '%s\n' "$SPIS" | grep -c .)"
    if [ "$ILE_LISTA" != "$ILE_SPIS" ]; then
        # Brakujący odcisk to brak w spisie, czyli plik, którego noc NIE
        # sprawdzi. Lepiej nie wydać wersji niż wydać dziurawą.
        echo "BŁĄD: plików kodu $ILE_LISTA, odcisków $ILE_SPIS — nie zapisuję."
        exit 1
    fi
    { echo "$NOWA"; printf '%s\n' "$SPIS"; } > "$PLIK"

    echo "wersja: ${STARA:-<brak>}  ->  $NOWA   ($ILE_SPIS plików w spisie)"
    echo "Teraz:  git add WERSJA.txt  i commit."
    ;;

sprawdz)
    if [ ! -f "$PLIK" ]; then
        echo "kod: BRAK WERSJA.txt — nie wiadomo, jaki to kod"
        exit 2
    fi
    [ -n "$PY" ] || { echo "kod: nie ma działającego Pythona — nie sprawdzę"; exit 2; }

    WYNIK=""
    while read -r rodzaj reszta; do
        case "$rodzaj" in
            ZLE)   echo "    INNA TREŚĆ:  $reszta" ;;
            BRAK)  echo "    BRAK PLIKU:  $reszta" ;;
            WYNIK) WYNIK="$reszta" ;;
        esac
    done < <(odciski sprawdz)

    # Brak linii WYNIK = Python padł w połowie. To NIE jest "zgodny".
    if [ -z "$WYNIK" ]; then
        echo "kod: sprawdzenie się nie wykonało — nie wiadomo, jaki to kod"
        exit 2
    fi
    read -r NUMER DOBRE ZLE BRAK <<< "$WYNIK"
    if [ $((ZLE + BRAK)) -eq 0 ]; then
        echo "kod: wersja $NUMER — wszystkie $DOBRE plików zgodne ze spisem"
        exit 0
    fi
    echo "kod: wersja $NUMER — NIEZGODNY: $ZLE z inną treścią, $BRAK brakuje" \
         "(zgodnych: $DOBRE)"
    exit 1
    ;;

*)
    echo "użycie: $0 nowa | sprawdz | pokaz"
    exit 1
    ;;
esac
