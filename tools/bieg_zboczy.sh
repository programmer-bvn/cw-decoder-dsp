#!/bin/bash
# Przebieg sieci zboczy od zbioru do porównania (06.11, gałąź elementy).
#
#   wsl -u root -- bash tools/bieg_zboczy.sh 6 80000 20        (z katalogu projektu)
#   pozycje: wersja generatora (5/6), klipów na część (2 części), epoki
#
# Wcześniejsze biegi (zboczy_1..5, out/zbocza/bieg_*.log) szły ręcznymi
# poleceniami — ten skrypt robi to samo w jednym kawałku i kończy dwiema
# liczbami, które muszą iść razem: test słów (zboczy_5: 1491/1628) i fałszywe
# znaki na minutę na negatywach, których sieć nie widziała (TRENING.txt).
# Wszystko do out/zbocza/bieg_<wersja>.log; wynik na końcu logu.

WERSJA=${1:-6}
N=${2:-80000}
EPOKI=${3:-20}
cd "$(dirname "$0")/.." || { echo "BŁĄD: brak katalogu projektu"; exit 1; }
LOG=out/zbocza/bieg_${WERSJA}.log
ZBIOR=out/zbocza/z${WERSJA}
RUN=runs/zbocza_${WERSJA}
NEG=${CW_NEGATYWY:-/mnt/w/nagrania_cw/negatywy}
mkdir -p out/zbocza "$ZBIOR"
exec > >(tee -a "$LOG") 2>&1
echo "start $(date)  wersja $WERSJA, $N klipów x 2, $EPOKI epok, kod $(git rev-parse --short HEAD 2>/dev/null)$(git diff --quiet 2>/dev/null || echo ' + zmiany')"

source srodowisko/rtx3050_setenv.sh || { echo "BŁĄD: setenv.sh"; exit 1; }
python -c "import tensorflow as tf; print('karty:', tf.config.list_physical_devices('GPU'))" 2>&1 | grep karty

krok() { echo; echo "=== $1 ($(date +%H:%M)) ==="; }

for i in 0 1; do
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

krok "test słów (zboczy_5: 1491/1628)"
python -m tools.tester_viterbi --model "$RUN/best.keras" > "out/viterbi_zb${WERSJA}.txt" 2>&1
grep "RAZEM" "out/viterbi_zb${WERSJA}.txt"

krok "halucynacje na negatywach spoza treningu"
for m in runs/zbocza_5 "$RUN"; do
  python -m tools.halucynacje ocen --model "$m/best.keras" --bank "$NEG" \
    --out "out/halucynacje_$(basename "$m")_test.txt" 2>&1 | grep -v "^I0000\|^W0000"
done

echo; echo "KONIEC $(date)"
