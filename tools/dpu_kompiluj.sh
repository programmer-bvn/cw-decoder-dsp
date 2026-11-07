#!/bin/bash
# Sieć zboczy -> .xmodel pod DPU B4096 płytki KV260 (07.11).
# Uruchamiać w WSL jako root, po `python -m tools.dpu_eksport okna`
# (okna kalibracyjne w out/dpu/okna.npz). Kontener Vitis AI 3.5 CPU:
# kwantyzacja vai_q + kompilacja vai_c_tensorflow2 dla arch.json KV260.
# Log: out/dpu/kompilacja.log.
cd "$(dirname "$0")/.." || exit 1
OBRAZ=xilinx/vitis-ai-tensorflow2-cpu:latest
[ -f out/dpu/okna.npz ] || { echo "brak out/dpu/okna.npz — najpierw: python -m tools.dpu_eksport okna"; exit 1; }
docker run --rm -v "$PWD":/workspace -w /workspace "$OBRAZ" bash -lc '
  source /opt/vitis_ai/conda/etc/profile.d/conda.sh && conda activate vitis-ai-tensorflow2 || exit 1
  python tools/dpu_eksport.py kontener || exit 1
  ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json
  echo "arch: $(cat $ARCH)"
  vai_c_tensorflow2 -m out/dpu/zbocza_5_int8.h5 -a $ARCH -o out/dpu/xmodel -n zbocza_5
' 2>&1 | tee out/dpu/kompilacja.log
exit "${PIPESTATUS[0]}"
