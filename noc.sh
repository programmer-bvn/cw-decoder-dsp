#!/bin/bash
# =============================================================================
#  noc.sh  --  cała noc treningu bez nadzoru
#
#  Uruchomienie (w WSL) — bez niczego przed tym:
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
#  Ten skrypt WOLNO uruchamiać przez ./ i NIE trzeba nic robić przed nim.
#
#  Dlaczego to działa, choć setenv.sh trzeba sourcować: wykonanie przez ./
#  tworzy powłokę potomną, a zmienne ustawione w niej giną razem z nią.
#  Ale trening dzieje się WEWNĄTRZ tej powłoki, więc środowisko jest
#  potrzebne dokładnie tyle, ile ona żyje. Dlatego noc.sh sourcuje je SAM
#  (etap 0) i tryb awarii "zapomniałem source" przestaje istnieć.
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

# --- 0. ŚRODOWISKO — ładowane przez ten skrypt ---------------------------
#  Kolejność szukania: $CW_SETENV, potem wersja dla maszyny z kartą, potem
#  ogólna. Sourcujemy tylko wtedy, gdy środowisko nie jest już aktywne —
#  żeby dało się je nadpisać z zewnątrz, gdy ktoś wie, co robi.
echo "--- środowisko ---"
_gotowe=0
if command -v python >/dev/null 2>&1 && python -c "import tensorflow" 2>/dev/null; then
    _gotowe=1
    echo "już aktywne — nie ruszam"
fi

if [ "$_gotowe" = "0" ]; then
    for _s in "${CW_SETENV:-}" "srodowisko/rtx3050_setenv.sh" "setenv.sh"; do
        if [ -n "$_s" ] && [ -f "$_s" ]; then
            echo "ładuję $_s"
            # shellcheck disable=SC1090
            source "$_s" || true
            break
        fi
    done
fi

if ! command -v python >/dev/null 2>&1; then
    echo
    echo "PRZERWANO: nie ma 'python' w PATH i nie znalazłem czego wysourcować."
    echo "Szukałem: \$CW_SETENV, srodowisko/rtx3050_setenv.sh, setenv.sh"
    echo "Konfiguracja od zera:  ./srodowisko/setup_gpu_env.sh"
    exit 1
fi

echo "python: $(command -v python)  ($(python --version 2>&1))"

# Wersja Pythona jest tu sprawdzana ODDZIELNIE, bo to najtańsza możliwa
# diagnoza: TensorFlow ma koła tylko dla cp310-cp313, a WSL domyślnie
# podaje 3.14. Bez tego objawem jest "No module named tensorflow" albo
# pip bez pasującej wersji — komunikaty, które nie wskazują przyczyny.
case "$(python --version 2>&1)" in
    *3.1[0-3]*) : ;;
    *) echo
       echo "PRZERWANO: TensorFlow nie ma koła dla tej wersji Pythona."
       echo "Potrzebny 3.10-3.13 (PyPI ma cp310-cp313)."
       echo "To NIE jest kwestia sterowników NVIDIA — te są niezależne"
       echo "od Pythona. Bramką jest wyłącznie dostępność koła."
       exit 1 ;;
esac

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

# --- 7. HISTORIA TRENINGU: commit, paczka, ewentualny push --------------
#  Po co: rano wynik ma być poza maszyną, która go policzyła. Jeśli dysk
#  stęknie w nocy, log.csv i state.json są już gdzie indziej.
#
#  DLACZEGO PACZKA, A NIE SAM PUSH. W WSL nie ma Credential Managera
#  Windows, więc token, którym pcha maszyna z Windows, tutaj nie istnieje.
#  Wkładanie go tu oznaczałoby ~/.git-credentials, czyli sekret czystym
#  tekstem na dysku. Zamiast tego `git bundle` pakuje commity do JEDNEGO
#  pliku, ten wraca na pendraku razem z wynikami, a wypycha go maszyna,
#  która poświadczenia ma. Cała historia tego repo to ~230 kB.
#
#  Push jest próbowany mimo to — jeśli jest deploy key z prawem zapisu,
#  wypchnie się od razu i paczka będzie tylko nadmiarowa. Nie jest to
#  jednak droga krytyczna: niepowodzenie pushu NIE psuje nocy.
#
#  CO trafia do commita: TYLKO historia treningu (runs/**/log.csv,
#  runs/**/state.json, out/noc_*.log). Świadomie NIE "git add -A" —
#  skrypt bez nadzoru nie ma prawa wciągnąć do historii zmian w kodzie
#  zostawionych w drzewie roboczym z wieczora.
# -------------------------------------------------------------------------
echo
echo "--- historia treningu ---"

if [ ! -d .git ]; then
    echo "to nie jest repozytorium git — pomijam"
    echo "  (kopia robocza z na_hdd.bat nie jest repozytorium; wyniki"
    echo "   wracają przez z_hdd.bat i to wystarcza)"
elif ! git rev-parse --verify -q HEAD >/dev/null 2>&1; then
    echo "repozytorium bez ani jednego commita — pomijam"
    echo "  pierwszy commit rób ręcznie, na oczy, nie w nocy"
else
    git add -- 'runs/**/log.csv' 'runs/**/state.json' "$LOGI"/noc_*.log 2>/dev/null

    if git diff --cached --quiet; then
        echo "nic nowego w historii treningu — nie commituję"
    else
        # core.hooksPath wyłączony: hook, który w nocy zapyta o cokolwiek,
        # zatrzymałby skrypt tak samo jak ssh bez BatchMode.
        if git -c core.hooksPath=/dev/null commit -q \
               -m "trening $STEMPEL: historia przebiegów"; then
            echo "commit: $(git log -1 --format='%h %s')"
        else
            echo "commit NIE przeszedł — patrz wyżej"
        fi
    fi

    GALAZ="$(git rev-parse --abbrev-ref HEAD)"
    PACZKA="$LOGI/historia_${STEMPEL}.bundle"

    # --- paczka: zawsze, bo nie wymaga niczego ---------------------------
    if git bundle create "$PACZKA" "$GALAZ" >/dev/null 2>&1; then
        echo "paczka: $PACZKA ($(du -h "$PACZKA" | cut -f1))"
        PACZKA_OK=1
    else
        echo "UWAGA: nie udało się zrobić paczki git bundle"
        PACZKA_OK=0
    fi

    # --- push: próba, nie wymóg -----------------------------------------
    if ! git remote get-url origin >/dev/null 2>&1; then
        echo "push: brak zdalnego 'origin' — pomijam"
    else
        KLUCZ="${CW_DEPLOY_KEY:-$HOME/.ssh/cw_deploy}"
        if [ -f "$KLUCZ" ]; then
            export GIT_SSH_COMMAND="ssh -i $KLUCZ -o IdentitiesOnly=yes -o BatchMode=yes"
        else
            export GIT_SSH_COMMAND="ssh -o BatchMode=yes"
        fi
        # GIT_TERMINAL_PROMPT=0: bez tego git po HTTPS bez poświadczeń
        # czeka na login i hasło z terminala, którego w nocy nie ma.
        if GIT_TERMINAL_PROMPT=0 git push origin "$GALAZ" >/dev/null 2>&1; then
            echo "push: wypchnięte na origin/$GALAZ"
        else
            echo "push: NIE przeszedł — to normalne w WSL i nic nie zginęło"
            if [ "$PACZKA_OK" = "1" ]; then
                echo
                echo "  Rano, z Windows, z katalogu repozytorium na pendraku:"
                echo "      git fetch \"<ścieżka>/$(basename "$PACZKA")\" $GALAZ"
                echo "      git merge --ff-only FETCH_HEAD"
                echo "      git push origin $GALAZ"
                echo
                echo "  Albo raz na zawsze: deploy key z 'Allow write access'"
                echo "  w Settings -> Deploy keys, wtedy push idzie stąd sam."
            fi
        fi
    fi
fi
