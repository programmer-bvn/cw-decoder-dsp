#!/bin/bash
# =============================================================================
#  porzadki.sh  --  co w projekcie NIE JEST ani kodem, ani wynikiem
#
#      ./srodowisko/porzadki.sh              wypisz
#      ./srodowisko/porzadki.sh --przenies   odloz do stare/RRRRMMDD/
#
#  PO CO. W korzeniu projektu odkladaja sie doraźne skrypty pisane w trakcie
#  szukania bledu: GPU_check.sh, libsRTX3050.sh.sh, setup_gpu_env.sh.sh,
#  dwie kopie tego samego setenv. Same w sobie nie przeszkadzaja. Problem
#  jest w tym, co sie z nimi dalej dzieje:
#
#      HDD  --z_hdd.bat-->  pendrak  --wypalenie-->  BD-R M-DISC
#
#  z_hdd.bat wozi z HDD "kod .py .md .bat .sh", wiec kazdy taki plik jedzie
#  na pendraka, a stamtad na plyte, ktorej NIE DA SIE SKASOWAC. Smiec
#  zrobiony na pol godziny zostaje na zawsze.
#
#  Druga szkoda jest bliżej: kopie tego samego skryptu w dwoch miejscach
#  rozjezdzaja sie po cichu. Przy `source` liczy sie, ktora zlapiesz, a nic
#  nie powie, ze zlapales starsza. Ten projekt ma juz taki przypadek --
#  rozjazd META_FIELDS miedzy dwiema sciezkami kosztowal tydzien nocy,
#  w ktorych zbior byl co wieczor uznawany za przestarzaly.
#
#  NIC NIE KASUJE. Najwyzej przenosi do stare/, i to dopiero na zadanie.
#  Pliki sa Twoje, a czesc z nich to jedyne kopie -- git ich nie zna.
# =============================================================================

set -u

PRZENIES=0
[ "${1:-}" = "--przenies" ] && PRZENIES=1

if [ ! -f dsp/config.py ]; then
    echo "BŁĄD: uruchom to z katalogu projektu (nie widzę dsp/config.py)."
    exit 1
fi
if ! git rev-parse --git-dir >/dev/null 2>&1; then
    echo "BŁĄD: to nie jest repozytorium git, więc nie wiem, co jest kodem."
    echo "    srodowisko/hdd_repo.sh"
    exit 1
fi

# Pliki, ktorych git ani nie sledzi, ani nie ignoruje -- czyli takie, co do
# ktorych nikt nigdy nie podjal decyzji. Tylko KORZEN: out/ i runs/ maja
# byc pelne nieznanych plikow, bo to wyniki.
mapfile -t SMIECI < <(git status --porcelain --untracked-files=all . \
    | sed -n 's/^?? //p' | grep -v '/' | sort)

if [ "${#SMIECI[@]}" -eq 0 ]; then
    echo "Korzeń projektu czysty: nie ma plików poza kodem i wynikami."
    exit 0
fi

echo "W korzeniu leży ${#SMIECI[@]} plików, których git ani nie śledzi,"
echo "ani nie ignoruje. Każdy z nich pojedzie z_hdd.bat na pendraka,"
echo "a stamtąd na płytę, której nie da się skasować."
echo

# Czy istnieje odpowiednik w srodowisko/ -- czyli czy to kopia czegos,
# co juz ma swoje miejsce. Nazwy sa rozne (GPU_check.sh / gpu_check.sh),
# wiec porownujemy po nazwie sprowadzonej do malych liter i bez kropek.
klucz() { basename "$1" | tr 'A-Z' 'a-z' | sed 's/\.sh$//; s/\.sh$//; s/[._-]//g'; }

for f in "${SMIECI[@]}"; do
    K="$(klucz "$f")"
    ODP=""
    for kandydat in srodowisko/*.sh; do
        [ -e "$kandydat" ] || continue
        if [ "$(klucz "$kandydat")" = "$K" ]; then ODP="$kandydat"; break; fi
    done

    printf '  %-24s %6s B' "$f" "$(stat -c%s "$f" 2>/dev/null)"
    if [ -n "$ODP" ]; then
        if diff -q "$f" "$ODP" >/dev/null 2>&1; then
            printf '   kopia %s (identyczna)\n' "$ODP"
        else
            printf '   STARSZA WERSJA %s\n' "$ODP"
        fi
    else
        printf '   bez odpowiednika w srodowisko/\n'
    fi
done

# Pliki identycznej tresci pod roznymi nazwami. To najgorszy rodzaj: nie
# wiadomo, ktora nazwa jest ta wlasciwa, wiec obie zostaja "na wszelki
# wypadek" i za miesiac nie wiadomo, po co sa.
echo
DUBLE=0
for a in "${SMIECI[@]}"; do
    for b in "${SMIECI[@]}"; do
        [ "$a" \< "$b" ] || continue
        if diff -q "$a" "$b" >/dev/null 2>&1; then
            echo "  TEN SAM PLIK POD DWIEMA NAZWAMI: $a  ==  $b"
            DUBLE=1
        fi
    done
done
[ "$DUBLE" = "0" ] && echo "  (nie ma duplikatów treści)"

if [ "$PRZENIES" = "0" ]; then
    echo
    echo "Nic nie ruszam. Żeby odłożyć je na bok:"
    echo "    ./srodowisko/porzadki.sh --przenies"
    echo "Pliki trafią do stare/<data>/ — skasować możesz sam, kiedy uznasz."
    exit 0
fi

KAT="stare/$(date +%Y%m%d)"
mkdir -p "$KAT"
for f in "${SMIECI[@]}"; do
    # Bez nadpisywania: gdyby porzadki szly drugi raz tego samego dnia,
    # a plik wrocil z HDD w innej wersji, nadpisanie zjadloby tamta.
    CEL="$KAT/$f"
    n=1
    while [ -e "$CEL" ]; do CEL="$KAT/$f.$n"; n=$((n+1)); done
    mv -- "$f" "$CEL" && echo "  $f -> $CEL"
done
echo
echo "Przeniesione: ${#SMIECI[@]}. Nic nie skasowane."
echo "stare/ jest w .gitignore, więc na pendraka już nie pojedzie."
