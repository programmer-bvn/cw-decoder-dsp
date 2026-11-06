#!/bin/bash
# Etap A klasyfikatora elementów od zbioru do oceny (06.11, gałąź elementy).
#
#   wsl -u root -- bash tools/bieg_elementow.sh 60000 20 1
#   pozycje: klipów treningowych, epoki, nazwa biegu (runs/elementy_<nazwa>)
#   Zbiór w out/elementy (istniejący nie jest generowany drugi raz).
# Wynik: macierz pomyłek na osobnych klipach i przegródki „ton” na minutę
# negatywów spoza treningu (TRENING.txt) — w out/elementy/bieg_<nazwa>.log.

N=${1:-60000}
EPOKI=${2:-20}
NAZWA=${3:-1}
cd "$(dirname "$0")/.." || { echo "BŁĄD: brak katalogu projektu"; exit 1; }
ZBIOR=out/elementy
RUN=runs/elementy_${NAZWA}
LOG=$ZBIOR/bieg_${NAZWA}.log
NEG=${CW_NEGATYWY:-/mnt/w/nagrania_cw/negatywy}
mkdir -p "$ZBIOR"
exec > >(tee -a "$LOG") 2>&1
echo "start $(date)  bieg $NAZWA, $N klipów, $EPOKI epok, kod $(git rev-parse --short HEAD 2>/dev/null)$(git diff --quiet 2>/dev/null || echo ' + zmiany')"
source srodowisko/rtx3050_setenv.sh || { echo "BŁĄD: setenv.sh"; exit 1; }
filtr() { grep -v "^I0000\|^W0000\|bfc_allocator\|^WARNING: All log\|rebuild TensorFlow"; }
krok() { echo; echo "=== $1 ($(date +%H:%M)) ==="; }

if [ ! -f $ZBIOR/trening.npz ]; then
  krok "zbiór treningowy"
  python -m tools.elementy_trening generuj $ZBIOR/trening.npz "$N" --seed 1 || { echo "BŁĄD: generowanie"; exit 1; }
fi
if [ ! -f $ZBIOR/walidacja.npz ]; then
  krok "zbiór walidacyjny (osobne klipy)"
  python -m tools.elementy_trening generuj $ZBIOR/walidacja.npz 3000 --seed 99 || { echo "BŁĄD: generowanie"; exit 1; }
fi
krok "trening $RUN"
python -m tools.elementy_trening trenuj $ZBIOR "$RUN" --epoki "$EPOKI" 2>&1 | filtr
[ -f "$RUN/best.keras" ] || { echo "BŁĄD: brak $RUN/best.keras"; exit 1; }
krok "ocena na syntetyku"
python -m tools.elementy_trening ocen "$RUN/best.keras" $ZBIOR/walidacja.npz 2>&1 | filtr
krok "negatywy spoza treningu"
python -m tools.elementy_trening negatywy "$RUN/best.keras" --bank "$NEG" --out "$ZBIOR/negatywy_${NAZWA}.txt" 2>&1 | filtr
echo; echo "KONIEC $(date)"
