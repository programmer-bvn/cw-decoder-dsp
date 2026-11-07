#!/bin/bash
# Zbocza + przegródki (07.11, pomysł operatora: kaskada filtrów jak w centrum
# badawczym). zbocza_6 + elementy_2, dwa sposoby łączenia (dsp/elementy.polacz_*).
# Te same miary co zbocza_6: test słów (zb6 1500/1659, zb6b 1509) i halucynacje
# znaków/min (zb6: pokój 12,0, AM 10,0, SSB 2,0), plus syntetyk/negatywy porównania.
cd "$(dirname "$0")/.." || { echo "BŁĄD: brak katalogu projektu"; exit 1; }
mkdir -p out/elementy
LOG=out/elementy/laczenie.log
exec > >(tee -a "$LOG") 2>&1
echo "start $(date)  kod $(git rev-parse --short HEAD)$(git diff --quiet 2>/dev/null || echo ' + zmiany')"
source srodowisko/rtx3050_setenv.sh || exit 1
NEG=${CW_NEGATYWY:-/mnt/w/nagrania_cw/negatywy}
MZ=runs/zbocza_6/best.keras; ME=runs/elementy_2/best.keras
filtr() { grep -v "^I0000\|^W0000\|bfc_allocator\|^WARNING: All log\|rebuild TensorFlow\|cpu_feature\|^warning: in the working"; }
for s in kaskada iloczyn; do
  echo; echo "=== halucynacje: $s"
  python -m tools.halucynacje ocen --model $MZ --elementy $ME --laczenie $s --bank "$NEG" \
    --out "out/elementy/halucynacje_$s.txt" 2>&1 | filtr
done
for s in kaskada iloczyn; do
  echo; echo "=== test słów: $s"
  python -m tools.tester_viterbi --model $MZ --elementy $ME --laczenie $s 2>&1 | filtr > "out/elementy/viterbi_$s.txt"
  grep "RAZEM" "out/elementy/viterbi_$s.txt"
done
echo; echo "=== porównanie syntetyk + negatywy (elementy)"
# u znane z generatora — wynik nie zależy od wyboru u w odczycie; raz wystarczy
[ -f out/elementy/porownanie_laczenie.txt ] && cat out/elementy/porownanie_laczenie.txt || python -m tools.porownaj_przegrodki --zbocza $MZ --elementy $ME --n 150 \
  --out out/elementy/porownanie_laczenie.txt 2>&1 | filtr
echo; echo "KONIEC $(date)"
