#!/bin/bash
# =============================================================================
#  noc.sh  --  cała noc treningu bez nadzoru
#
#  Uruchomienie (w WSL, po `source setenv.sh`):
#
#      ./noc.sh                          # 200 tys., 120 epok, dpu + gru
#      ./noc.sh 200000 120               # jawnie: próbki i epoki
#      ./noc.sh 200000 120 256 moj.npz   # własny plik zbioru
#
#  DOBÓR ROZMIARU DO OKNA CZASOWEGO. Przy 119 ms/krok i batchu 256:
#      200 tys. -> 782 kroki/epokę  -> 1,6 min/epokę
#      400 tys. -> 1563 kroki       -> 3,1 min/epokę
#  Architektura 'gru' jest cięższa, licz około 1,5x. Dwa przebiegi po
#  120 epok na 200 tys. to ok. 8 h; na 400 tys. byłoby 15 h i nie
#  zmieściłoby się w nocy.
#
#  Ten skrypt WOLNO uruchamiać przez ./ — inaczej niż setenv.sh, bo nie
#  zmienia środowiska powłoki, tylko odpala trening. Ale środowisko musi
#  już być załadowane: `source setenv.sh` PRZED nim.
#
#  CO ROBI, W TEJ KOLEJNOŚCI
#    1. Sprawdza kartę i PRZERYWA, jeśli jej nie widzi. Najpierw, zanim
#       cokolwiek policzy — 12 godzin przeliczone na CPU to strata nocy,
#       a wykrycie tego zajmuje sekundę.
#    2. Sprawdza spójność łańcucha (diag.py). Też przed treningiem.
#    3. Generuje zbiór, JEŚLI go nie ma albo jest niekompletny.
#    4. Trenuje architekturę 'dpu' — tę wdrażalną na KV260.
#    5. Trenuje 'gru' — punkt odniesienia, ile kosztuje brak rekurencji.
#    6. Wypisuje podsumowanie obu przebiegów.
#
#  Etapy 3-5 są POMIJANE, jeśli już się udały. Można więc uruchomić
#  ponownie po przerwaniu i nie zaczynać od zera.
#
#  Wszystko idzie do out/noc_*.log, żeby rano było co czytać.
# =============================================================================

set -u

N_PROBEK="${1:-200000}"
EPOK="${2:-120}"
BATCH="${3:-256}"

# Nazwa domyślna zgodna z tym, co generuje train_rtx.py bez --out.
# Dzięki temu istniejący zbiór jest UŻYWANY, a nie generowany od nowa —
# 800 MB i kilka minut do stracenia przy 12-godzinnym oknie na trening.
ZBIOR="${4:-morse_dataset.npz}"
LOGI="out"
mkdir -p "$LOGI"

STEMPEL="$(date +%Y%m%d_%H%M)"
GLOWNY="$LOGI/noc_${STEMPEL}.log"

# Wszystko na ekran I do pliku jednocześnie.
exec > >(tee -a "$GLOWNY") 2>&1

echo "============================================================"
echo " NOC TRENINGOWA  $(date '+%Y-%m-%d %H:%M')"
echo "============================================================"
echo " zbiór:  $ZBIOR  ($N_PROBEK próbek)"
echo " epoki:  $EPOK   batch: $BATCH"
echo " log:    $GLOWNY"
echo

# --- 0. czy środowisko jest załadowane -----------------------------------
if ! command -v python >/dev/null 2>&1; then
    echo "BŁĄD: nie ma 'python' w PATH."
    echo "Najpierw:  source setenv.sh"
    exit 1
fi
echo "python: $(command -v python)  ($(python --version 2>&1))"

# --- 1. KARTA — sprawdzamy PRZED wszystkim -------------------------------
echo
echo "--- karta ---"
if ! python - <<'PY'
import sys
try:
    import tensorflow as tf
except Exception as e:
    print("BLAD importu tensorflow:", e); sys.exit(1)
g = tf.config.list_physical_devices("GPU")
print("TensorFlow", tf.__version__, "| GPU:", [d.name for d in g] or "BRAK")
sys.exit(0 if g else 2)
PY
then
    echo
    echo "PRZERWANO: karta nie jest widziana, a bez niej ta noc nic nie da."
    echo "Diagnostyka:"
    echo "    nvidia-smi"
    echo "    python train_rtx.py train --require-gpu   (wypisze powód)"
    exit 1
fi

# --- 2. SPÓJNOŚĆ ŁAŃCUCHA ------------------------------------------------
echo
echo "--- diag.py ---"
if [ -f diag.py ]; then
    if python diag.py > "$LOGI/noc_${STEMPEL}_diag.log" 2>&1; then
        tail -3 "$LOGI/noc_${STEMPEL}_diag.log"
    else
        echo "BŁĄD: diag.py nie przeszedł. Szczegóły w"
        echo "      $LOGI/noc_${STEMPEL}_diag.log"
        tail -12 "$LOGI/noc_${STEMPEL}_diag.log"
        echo
        echo "PRZERWANO. Trening na niespójnym łańcuchu to zmarnowana noc."
        exit 1
    fi
else
    echo "brak diag.py — pomijam (ale lepiej go mieć)"
fi

# --- 3. ZBIÓR ------------------------------------------------------------
echo
echo "--- zbiór ---"
POTRZEBNE="chirp sag hum agc_tau fist_drift gap_jitter"
GENERUJ=1
if [ -f "$ZBIOR" ]; then
    if python - "$ZBIOR" $POTRZEBNE <<'PY'
import sys
import numpy as np
p, wymagane = sys.argv[1], sys.argv[2:]
try:
    d = np.load(p, allow_pickle=False)
except Exception as e:
    print("nie moge otworzyc:", e); sys.exit(1)
brak = [k for k in wymagane if k not in d.files]
print(f"{p}: {len(d['y'])} probek | meta: {str(d['meta'])}")
if brak:
    print("BRAKUJE kolumn modelu kanalu:", " ".join(brak))
    print("-> zbior powstal PRZED modelem kanalu, generuje od nowa")
    sys.exit(2)
print("kolumny modelu kanalu: wszystkie obecne")
sys.exit(0)
PY
    then
        GENERUJ=0
        echo "zbiór aktualny — nie generuję"
    fi
else
    echo "$ZBIOR nie istnieje"
fi

if [ "$GENERUJ" = "1" ]; then
    echo "generuję $N_PROBEK próbek..."
    if ! python train_rtx.py generate --n "$N_PROBEK" --out "$ZBIOR" \
            > "$LOGI/noc_${STEMPEL}_gen.log" 2>&1; then
        echo "BŁĄD generowania, szczegóły w $LOGI/noc_${STEMPEL}_gen.log"
        tail -15 "$LOGI/noc_${STEMPEL}_gen.log"
        exit 1
    fi
    tail -12 "$LOGI/noc_${STEMPEL}_gen.log"
fi

# --- 4-5. DWA PRZEBIEGI --------------------------------------------------
# Kolejność nie jest przypadkowa: 'dpu' pierwszy, bo to model, który da się
# wdrożyć na KV260. Gdyby noc się urwała, ma się liczyć ten właściwy.
trenuj () {
    local ARCH="$1" RUN="$2"
    echo
    echo "============================================================"
    echo " TRENING  arch=$ARCH  ->  $RUN     $(date '+%H:%M')"
    echo "============================================================"

    if [ -f "$RUN/state.json" ] && python - "$RUN/state.json" "$EPOK" <<'PY'
import json, sys
s = json.load(open(sys.argv[1], encoding="utf-8"))
done, want = int(s.get("epoch", 0)), int(sys.argv[2])
print(f"  w {sys.argv[1]}: epoka {done} z {want}")
sys.exit(0 if done >= want else 1)
PY
    then
        echo "  ten przebieg jest już ukończony — pomijam"
        return 0
    fi

    local LOG="$LOGI/noc_${STEMPEL}_${ARCH}.log"
    # --fresh TYLKO gdy nie ma stanu. Przy wznowieniu po przerwaniu
    # chcemy kontynuować, a nie zaczynać od zera.
    local SWIEZY=""
    [ -f "$RUN/state.json" ] || SWIEZY="--fresh"

    python train_rtx.py train \
        --dataset "$ZBIOR" --run "$RUN" --arch "$ARCH" \
        --epochs "$EPOK" --batch "$BATCH" --mixed --require-gpu $SWIEZY \
        > "$LOG" 2>&1
    local RC=$?

    # Postęp epok bez zaśmiecania: same wiersze z walidacją.
    grep -a "val_accuracy" "$LOG" | tail -3
    echo
    grep -aE "Dokładność na walidacji|klasa 0|znaki:|wzięty za ciszę" "$LOG" || true

    if [ $RC -ne 0 ]; then
        echo "  UWAGA: trening zakończył się kodem $RC — patrz $LOG"
    fi
    return 0
}

trenuj dpu runs/cw2
trenuj gru runs/gru1

# --- 6. PODSUMOWANIE -----------------------------------------------------
echo
echo "============================================================"
echo " PODSUMOWANIE  $(date '+%Y-%m-%d %H:%M')"
echo "============================================================"
python - <<'PY'
import json
from pathlib import Path

for run, opis in (("runs/cw2", "dpu (wdrażalny na KV260)"),
                  ("runs/gru1", "gru (odniesienie, NIE wdrażalny)")):
    p = Path(run) / "state.json"
    if not p.exists():
        print(f"{run:12s} {opis:34s} brak wyniku")
        continue
    s = json.load(open(p, encoding="utf-8"))
    h = s.get("history", {})
    va = h.get("val_accuracy", [])
    ac = h.get("accuracy", [])
    best = s.get("best_val_acc", -1)
    print(f"{run:12s} {opis}")
    print(f"             epok {s.get('epoch', 0)}, "
          f"najlepsza walidacja {best*100:.2f}%"
          + (f" (epoka {va.index(max(va))+1})" if va else ""))
    if va:
        print(f"             val_accuracy koniec {va[-1]*100:.2f}%, "
              f"accuracy treningowa {ac[-1]*100:.2f}%")
        if ac and ac[-1] > 0.99:
            print("             UWAGA: accuracy treningowa 100% = model "
                  "zapamiętał zbiór,\n                    a nie nauczył się "
                  "zadania. Potrzeba więcej danych.")
PY

echo
echo "Do zabrania na pendraka:"
echo "    ./z_hdd.bat  (z Windows)  albo skopiuj runs/ i out/*.log"
echo
echo "Pełne logi: $LOGI/noc_${STEMPEL}*.log"
echo "============================================================"

# --- 7. WYPCHNIĘCIE HISTORII TRENINGU ------------------------------------
#  Po co: rano masz wynik na GitHubie, zanim dotkniesz pendraka. Jeśli
#  dysk w maszynie stęknie w nocy, log.csv i state.json są już poza nią.
#
#  CO trafia do commita: TYLKO historia treningu (runs/**/log.csv,
#  runs/**/state.json, out/noc_*.log). Świadomie NIE "git add -A" —
#  nocny skrypt bez nadzoru nie ma prawa zacommitować przypadkowych
#  zmian w kodzie, które zostały w drzewie roboczym z wieczora.
#
#  KIEDY jest pomijany: gdy nie ma katalogu .git (kopia robocza zrobiona
#  przez na_hdd.bat nie jest repozytorium) albo gdy nie ma zdalnego
#  "origin". W obu przypadkach to NIE błąd — ta maszyna po prostu nie
#  jest ustawiona do wypychania i trening ma się liczyć tak samo.
#
#  BatchMode=yes jest KLUCZOWE. Bez niego ssh przy nieznanym odcisku
#  hosta albo kluczu z hasłem czeka na odpowiedź z terminala, którego
#  w nocy nie ma — i skrypt wisi do rana zamiast wypisać błąd.
#
#  Klucz: deploy key TEGO repozytorium, nie klucz konta. Ścieżkę można
#  nadpisać zmienną CW_DEPLOY_KEY.
# -------------------------------------------------------------------------
echo
echo "--- wypchnięcie na GitHub ---"

if [ ! -d .git ]; then
    echo "to nie jest repozytorium git — pomijam"
elif ! git remote get-url origin >/dev/null 2>&1; then
    echo "brak zdalnego 'origin' — pomijam"
    echo "  ustawienie: git remote add origin git@github.com:<konto>/<repo>.git"
elif ! git rev-parse --verify -q HEAD >/dev/null 2>&1; then
    echo "repozytorium bez ani jednego commita — pomijam"
    echo "  pierwszy commit rób ręcznie, na oczy, nie w nocy"
else
    KLUCZ="${CW_DEPLOY_KEY:-$HOME/.ssh/cw_deploy}"
    if [ -f "$KLUCZ" ]; then
        export GIT_SSH_COMMAND="ssh -i $KLUCZ -o IdentitiesOnly=yes -o BatchMode=yes"
        echo "klucz: $KLUCZ"
    else
        export GIT_SSH_COMMAND="ssh -o BatchMode=yes"
        echo "brak $KLUCZ — próbuję domyślnej tożsamości ssh"
    fi

    git add -- 'runs/**/log.csv' 'runs/**/state.json' "$LOGI"/noc_*.log 2>/dev/null

    if git diff --cached --quiet; then
        echo "nic nowego w historii treningu — nie commituję"
    else
        # core.hooksPath wyłączony: hook, który w nocy zapyta o cokolwiek,
        # zatrzymałby skrypt tak samo jak ssh bez BatchMode.
        if git -c core.hooksPath=/dev/null commit -q \
               -m "trening $STEMPEL: historia przebiegów dpu i gru"; then
            echo "commit: $(git log -1 --format='%h %s')"
        else
            echo "commit NIE przeszedł — patrz wyżej"
        fi
    fi

    GALAZ="$(git rev-parse --abbrev-ref HEAD)"
    if git push origin "$GALAZ" 2>&1 | sed 's/^/  /'; then
        echo "wypchnięte na origin/$GALAZ"
    else
        echo "PUSH NIE PRZESZEDŁ. Commit lokalnie JEST, więc nic nie zginęło."
        echo "Najczęstsze powody, w tej kolejności:"
        echo "  1. klucz nie dodany w repo jako Deploy key z 'Allow write access'"
        echo "  2. brak github.com w ~/.ssh/known_hosts"
        echo "     -> ssh-keyscan github.com >> ~/.ssh/known_hosts"
        echo "  3. WSL ma inny HOME niż Windows — klucza tu po prostu nie ma"
        echo "Sprawdzenie: ssh -i \"\$KLUCZ\" -o IdentitiesOnly=yes -T git@github.com"
    fi
fi
