#!/bin/bash
# Etap A uczony stopnia wypełnienia (elementy_2, 06.11): ten sam zbiór co
# elementy_1, cel miękki. Potem ocena, negatywy i porównanie ze zboczami
# (w porównaniu wiersz „kotwica" — czy faza grzebienia z P teraz trafia).
cd "$(dirname "$0")/.." || { echo "BŁĄD: brak katalogu projektu"; exit 1; }
LOG=out/elementy/bieg_2.log
exec > >(tee -a "$LOG") 2>&1
echo "start $(date)  kod $(git rev-parse --short HEAD)$(git diff --quiet || echo ' + zmiany')"
source srodowisko/rtx3050_setenv.sh || exit 1
filtr() { grep -v "^I0000\|^W0000\|bfc_allocator\|^WARNING: All log\|rebuild TensorFlow\|^warning: in the working"; }
python -m tools.elementy_trening trenuj out/elementy runs/elementy_2 --epoki 20 --miekkie 2>&1 | filtr
[ -f runs/elementy_2/best.keras ] || { echo "BŁĄD: brak modelu"; exit 1; }
python -m tools.elementy_trening ocen runs/elementy_2/best.keras out/elementy/walidacja.npz 2>&1 | filtr
python -m tools.porownaj_przegrodki --elementy runs/elementy_2/best.keras --n 150 --out out/elementy/porownanie_2.txt 2>&1 | filtr
echo "KONIEC $(date)"
