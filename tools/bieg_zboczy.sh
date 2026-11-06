#!/bin/bash
# Przebieg sieci zboczy od zbioru do porównania (06.11, gałąź elementy).
#
#   wsl -u root -- bash tools/bieg_zboczy.sh 6 80000 20        (z katalogu projektu)
#   pozycje: wersja generatora (5/6), klipów na część, epoki, części (2),
#            nazwa biegu (domyślnie = wersja; runs/zbocza_<nazwa>)
#   Części zbioru wspólne dla wersji (out/zbocza/z<wersja>) — istniejące
#   nie są generowane drugi raz, więcej części = dołożenie nowych.
#
# Wcześniejsze biegi (zboczy_1..5, out/zbocza/bieg_*.log) szły ręcznymi
# poleceniami — ten skrypt robi to samo w jednym kawałku i kończy dwiema
# liczbami, które muszą iść razem: test słów (zboczy_5 na zestawie z 06.11: 1520/1659) i fałszywe
# znaki na minutę na negatywach, których sieć nie widziała (TRENING.txt).
# Wszystko do out/zbocza/bieg_<nazwa>.log; wynik na końcu logu.

WERSJA=${1:-6}
N=${2:-80000}
EPOKI=${3:-20}
CZESCI=${4:-2}
NAZWA=${5:-$WERSJA}
cd "$(dirname "$0")/.." || { echo "BŁĄD: brak katalogu projektu"; exit 1; }
LOG=out/zbocza/bieg_${NAZWA}.log
ZBIOR=out/zbocza/z${WERSJA}
RUN=runs/zbocza_${NAZWA}
NEG=${CW_NEGATYWY:-/mnt/w/nagrania_cw/negatywy}
mkdir -p out/zbocza "$ZBIOR"
exec > >(tee -a "$LOG") 2>&1
echo "start $(date)  bieg $NAZWA, wersja $WERSJA, $N klipów x $CZESCI, $EPOKI epok, kod $(git rev-parse --short HEAD 2>/dev/null)$(git diff --quiet 2>/dev/null || echo ' + zmiany')"

source srodowisko/rtx3050_setenv.sh || { echo "BŁĄD: setenv.sh"; exit 1; }
python -c "import tensorflow as tf; print('karty:', tf.config.list_physical_devices('GPU'))" 2>&1 | grep karty

krok() { echo; echo "=== $1 ($(date +%H:%M)) ==="; }

for i in $(seq 0 $((CZESCI - 1))); do
  if [ -f "$ZBIOR/czesc_$i.npz" ]; then
    echo "część $i jest — pomijam generowanie"
  else
    krok "zbiór, część $i"
    python -m tools.zbocza_trening generuj "$ZBIOR/czesc_$i.npz" "$N" --seed $((i + 1)) --wersja "$WERSJA" \
      || { echo "BŁĄD: generowanie części $i"; exit 1; }
  fi
done

krok "trening $RUN"
python -m tools.zbocza_trening trenuj "$ZBIOR" "$RUN" --epoki "$EPOKI" 2>&1 | grep -v "^I0000\|^W0000\|bfc_allocator\|^  *[0-9]*/[0-9]* \[" \
  || true
[ -f "$RUN/best.keras" ] || { echo "BŁĄD: brak $RUN/best.keras po treningu"; exit 1; }

krok "syntetyk: sieć vs próg klasyczny"
python -m tools.zbocza_trening ocen "$RUN/best.keras" 2>&1 | grep -v "^I0000\|^W0000"

krok "test słów (zboczy_5 na zestawie z 06.11: 1520/1659)"
python -m tools.tester_viterbi --model "$RUN/best.keras" > "out/viterbi_zb${NAZWA}.txt" 2>&1
grep "RAZEM" "out/viterbi_zb${NAZWA}.txt"

krok "halucynacje na negatywach spoza treningu"
for m in runs/zbocza_5 "$RUN"; do
  [ "$m" = runs/zbocza_5 ] && [ -f out/halucynacje_zbocza_5_test.txt ] && { cat out/halucynacje_zbocza_5_test.txt; continue; }
  python -m tools.halucynacje ocen --model "$m/best.keras" --bank "$NEG" \
    --out "out/halucynacje_$(basename "$m")_test.txt" 2>&1 | grep -v "^I0000\|^W0000"
done

echo; echo "KONIEC $(date)"
