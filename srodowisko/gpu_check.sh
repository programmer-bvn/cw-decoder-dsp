#!/bin/bash
# =============================================================================
#  gpu_check.sh  --  czy TensorFlow widzi kartę. Jedno pytanie, jedna odpowiedź.
#
#      source srodowisko/rtx3050_setenv.sh
#      ./srodowisko/gpu_check.sh
#
#  Kod wyjścia: 0 = karta widziana, 2 = nie widziana, 1 = nie ma środowiska.
#  Dzięki temu można tym warunkować skrypty (tak robi noc.sh).
#
#  Jeśli odpowiedź jest "BRAK", następny krok to gpu_libs.sh — on powie,
#  KTÓRA biblioteka nie wchodzi, zamiast samego faktu, że coś nie działa.
# =============================================================================

if ! command -v python >/dev/null 2>&1; then
    echo "BŁĄD: nie ma 'python' w PATH."
    echo "Najpierw:  source srodowisko/rtx3050_setenv.sh"
    exit 1
fi

python - <<'PY'
import sys
try:
    import tensorflow as tf
except Exception as e:
    print("BŁĄD importu tensorflow:", e)
    sys.exit(1)

karty = tf.config.list_physical_devices("GPU")
print(f"TensorFlow {tf.__version__} | Python {sys.version.split()[0]}")
print(f"karta: {[k.name for k in karty] or 'BRAK'}")
if karty:
    for k in karty:
        try:
            d = tf.config.experimental.get_device_details(k)
            if d.get("compute_capability"):
                print(f"  {d.get('device_name', '?')}, "
                      f"compute capability {'.'.join(map(str, d['compute_capability']))}")
        except Exception:
            pass
sys.exit(0 if karty else 2)
PY
