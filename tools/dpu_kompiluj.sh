#!/bin/bash
# Sieć zboczy -> .xmodel pod DPU B4096 płytki KV260 (07.11).
# Uruchamiać w WSL jako root, po `python -m tools.dpu_eksport okna`
# (okna kalibracyjne w out/dpu/okna.npz). Kontener Vitis AI 3.5 CPU:
# kwantyzacja vai_q + kompilacja vai_c_tensorflow2.
#
# arch.json z odciskiem (fingerprint) sprzętu, nie z nazwą celu: gotowy
# KV260/arch.json z Vitis AI 3.5 daje 0x101000056010407, a DPU
# z bitstreamu kv260-benchmark-b4096 (Vitis 2022.1) zgłasza
# 0x101000016010407 — VART odmówił wykonania („fingerprint check failure",
# 07.11). Odczyt z płytki: rejestry 0x1F4:0x1F0 pod 0xA0010000.
# Log: out/dpu/kompilacja.log.
cd "$(dirname "$0")/.." || exit 1
OBRAZ=xilinx/vitis-ai-tensorflow2-cpu:latest
ODCISK=${ODCISK:-0x101000016010407}
[ -f out/dpu/okna.npz ] || { echo "brak out/dpu/okna.npz — najpierw: python -m tools.dpu_eksport okna"; exit 1; }
mkdir -p out/dpu
echo "{\"fingerprint\": \"$ODCISK\"}" > out/dpu/arch.json
docker run --rm -v "$PWD":/workspace -w /workspace "$OBRAZ" bash -lc '
  source /opt/vitis_ai/conda/etc/profile.d/conda.sh && conda activate vitis-ai-tensorflow2 || exit 1
  python tools/dpu_eksport.py kontener || exit 1
  echo "arch: $(cat out/dpu/arch.json)"
  vai_c_tensorflow2 -m out/dpu/zbocza_5_int8.h5 -a out/dpu/arch.json -o out/dpu/xmodel -n zbocza_5
' 2>&1 | tee out/dpu/kompilacja.log
exit "${PIPESTATUS[0]}"
