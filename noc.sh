#!/bin/bash
# =============================================================================
#  noc.sh  --  kolejka pomiarów bez nadzoru, na całą noc
#
#  Uruchomienie (w WSL) — bez niczego przed tym:
#
#      ./noc.sh
#      ./noc.sh 200000 40                       # próbki i epoki
#      ./noc.sh 200000 40 256 moj.npz           # własny plik zbioru
#      ./noc.sh 200000 40 256 moj.npz 60 5000   # większy pomiar koperty
#
#  CO ROBI, PO KOLEI
#      0. ładuje środowisko SAM (nie trzeba source setenv.sh)
#      1. karta -- przerywa, jeśli jej nie widzi
#      2. diag.py
#      3. zbiór -- generuje, jeśli brak albo niekompletny
#      4. trening dpu  -> runs/cw2      (wdrażalny na KV260)
#      5. trening gru  -> runs/gru1     (odniesienie: koszt braku rekurencji)
#      6. koperta i błędy dla dpu
#      7. koperta i błędy dla gru
#      8. out/RANO.txt -- kilkanaście linii do przeczytania po powrocie
#      9. commit + paczka git bundle + próba pushu
#
#  Etapy 1-3 PRZERYWAJĄ przy awarii, bo bez nich nic nie ma sensu.
#  Od etapu 4 awaria JEDNEGO nie zabija pozostałych: jeśli 'gru' padnie,
#  koperta dla 'dpu' i tak się policzy.
#
#  ZMIERZONE 8/9.09.2026 na RTX 3050, nie szacowane. Poprzednia wersja
#  tego nagłówka mówiła 119 ms/krok i była zmyślona — pomyliłem się
#  czterokrotnie, co doprowadziło do domyślnych 120 epok.
#
#      200 tys., batch 256  ->  719 kroków/epokę, 31 ms/krok
#                           ->  23 s/epokę
#
#  DLACZEGO DOMYŚLNIE 40 EPOK, A NIE 120. Z log.csv przebiegu 'dpu':
#      val_loss ma MINIMUM w epoce 16 (0,1361) i potem tylko rośnie,
#      do 0,2130 w setnej. val_accuracy stoi od 20. epoki.
#      Epoki 17-100 dały +0,3 punktu procentowego i podniosły val_loss
#      o 57%, przy dokładności treningowej 99,98%.
#  Czyli po 20. epoce model już tylko zapamiętuje zbiór. 40 epok daje
#  zapas na wolniejszą zbieżność 'gru' i nic nie traci.
#
#  Wąskim gardłem NIE JEST moc obliczeniowa, a DANE: 200 tys. stałych
#  obrazów to za mało na 624 tys. parametrów przy tej zmienności. Dłuższy
#  trening tego nie naprawia — naprawi to generowanie w locie albo
#  większy zbiór.
#
#  Skoro epoka trwa 23 s, w oknie nocnym mieści się nie jeden trening,
#  a KOLEJKA pomiarów. Dwa przebiegi po 40 epok to około 40 minut.
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
EPOK="${2:-40}"
BATCH="${3:-256}"

# Rozmiar pomiaru koperty. Domyślne 40 próbek na komórkę siatki 9x7 to
# 2520 obrazów -- kilkadziesiąt sekund. Błędy z 3000 próbek pełnego modelu
# kanału dają w kubełkach widoczności po kilkaset sztuk, czyli dość, żeby
# różnica 15 punktów procentowych nie była szumem.
KOPERTA_N="${5:-40}"
KOPERTA_BLEDY="${6:-3000}"

# Nazwa domyślna zgodna z tym, co generuje train_rtx.py bez --out.
# Dzięki temu istniejący zbiór jest UŻYWANY, a nie generowany od nowa —
# 800 MB i kilka minut do stracenia przy 12-godzinnym oknie na trening.
ZBIOR="${4:-morse_dataset.npz}"
LOGI="out"
mkdir -p "$LOGI"

STEMPEL="$(date +%Y%m%d_%H%M)"

# =============================================================================
#  CZARNA SKRZYNKA
#
#  Po co: "sprawdź nvidia-smi" dopisane do logu jest bezużyteczne, kiedy
#  wracasz o 23:00 — wtedy stan maszyny jest już inny. Dowody trzeba
#  zebrać W MOMENCIE awarii, dopóki skrypt jeszcze działa.
#
#  Wywoływana raz na starcie (stan odniesienia) i za każdym razem, gdy
#  etap się nie uda. Wszystko idzie do osobnego pliku, nie na ekran —
#  ekran ma zostać czytelny.
# =============================================================================
SKRZYNKA="$LOGI/noc_${STEMPEL}_srodowisko.log"

czarna_skrzynka () {
    local POWOD="$1"
    {
        echo
        echo "########################################################"
        echo "# $POWOD     $(date '+%Y-%m-%d %H:%M:%S')"
        echo "########################################################"
        echo
        echo "--- powloka i katalog ---"
        echo "pwd:    $(pwd)"
        echo "user:   $(id -un) (uid $(id -u))"
        echo "bash:   $BASH_VERSION"
        echo "argv:   $0 $*"
        echo
        echo "--- python ---"
        echo "which:  $(command -v python 2>&1)"
        echo "wersja: $(python --version 2>&1)"
        echo "VIRTUAL_ENV: ${VIRTUAL_ENV:-<brak>}"
        echo
        echo "--- LD_LIBRARY_PATH (po jednym na wiersz) ---"
        echo "${LD_LIBRARY_PATH:-<pusty>}" | tr ':' '\n' | sed 's/^/  /'
        echo
        echo "--- XLA_FLAGS ---"
        echo "  ${XLA_FLAGS:-<brak>}"
        echo
        echo "--- nvidia-smi ---"
        if command -v nvidia-smi >/dev/null 2>&1; then
            nvidia-smi 2>&1
        else
            echo "  nvidia-smi NIE ISTNIEJE w PATH"
            echo "  (w WSL przychodzi z /usr/lib/wsl/lib — jeśli go nie ma,"
            echo "   sterownik po stronie Windows nie jest przekazany)"
        fi
        echo
        echo "--- biblioteki CUDA po jednej ---"
        if [ -x srodowisko/gpu_libs.sh ]; then
            ./srodowisko/gpu_libs.sh 2>&1
        else
            python - <<'PYLIB' 2>&1
import ctypes
for lib in ("libcuda.so.1", "libcudart.so.12", "libcublas.so.12",
            "libcudnn.so.9", "libnvJitLink.so.12"):
    try:
        ctypes.CDLL(lib)
        print(f"  [ OK ] {lib}")
    except OSError as e:
        print(f"  [BRAK] {lib} -> {e}")
PYLIB
        fi
        echo
        echo "--- pamiec ---"
        free -h 2>&1 || true
        echo
        echo "--- miejsce na dysku ---"
        df -h . 2>&1 || true
        echo
        echo "--- limity ---"
        ulimit -a 2>&1 || true
        echo
        echo "--- najwieksze pliki w katalogu ---"
        ls -lS 2>/dev/null | head -8
    } >> "$SKRZYNKA" 2>&1
}

czarna_skrzynka "STAN ODNIESIENIA (start)"
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
    echo
    # Dowody zbieramy TERAZ, bo rano tego stanu już nie będzie.
    echo "zbieram dowody do $SKRZYNKA ..."
    czarna_skrzynka "AWARIA ETAPU 1: karta nie widziana"
    python train_rtx.py train --require-gpu \
        > "$LOGI/noc_${STEMPEL}_karta.log" 2>&1 || true
    {
        echo
        echo "--- train_rtx.py --require-gpu (pelny powod) ---"
        cat "$LOGI/noc_${STEMPEL}_karta.log"
    } >> "$SKRZYNKA" 2>&1
    echo
    echo "Najczestsze przyczyny, w kolejnosci:"
    echo "  1. brak katalogow nvidia/*/lib w LD_LIBRARY_PATH"
    echo "     -> source srodowisko/rtx3050_setenv.sh"
    echo "  2. Python poza 3.10-3.13 (TensorFlow nie ma kola)"
    echo "  3. sterownik po stronie Windows -> nvidia-smi"
    echo
    echo "--- ostatnie 25 wierszy z train_rtx.py --require-gpu ---"
    tail -25 "$LOGI/noc_${STEMPEL}_karta.log" || true
    echo
    # Wskazanie pliku MUSI byc ostatnie na ekranie. Inaczej ginie pod
    # surowym Tracebackiem i po ciemku nie wiadomo, gdzie zajrzec.
    echo "PELNY STAN MASZYNY W MOMENCIE AWARII:"
    echo "    $SKRZYNKA"
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
        czarna_skrzynka "AWARIA ETAPU 2: diag.py nie przeszedł"
        {
            echo
            echo "--- diag.py: linie z bledami ---"
            grep -anE "BLAD|BŁĄD|FAIL|Traceback|Error" \
                "$LOGI/noc_${STEMPEL}_diag.log" || true
        } >> "$SKRZYNKA" 2>&1
        echo
        echo "PRZERWANO. Trening na niespójnym łańcuchu to zmarnowana noc."
        echo "Środowisko w momencie awarii: $SKRZYNKA"
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
    # Miejsce sprawdzamy PRZED, bo generowanie 800 MB na pelnym dysku
    # konczy sie po kilku minutach i noc jest stracona. Zapas 2 GB:
    # zbior 200 tys. to ~800 MB, plus modele i logi.
    WOLNE_MB=$(df -Pm . | awk 'NR==2 {print $4}')
    echo "wolne miejsce: ${WOLNE_MB} MB"
    if [ "${WOLNE_MB:-0}" -lt 2048 ]; then
        echo
        echo "PRZERWANO: mniej niz 2 GB wolnego, a zbior potrzebuje ~800 MB"
        echo "plus modele i logi."
        czarna_skrzynka "AWARIA ETAPU 3: brak miejsca (${WOLNE_MB} MB)"
        echo "Szczegoly: $SKRZYNKA"
        exit 1
    fi
    echo "generuję $N_PROBEK próbek..."
    if ! python train_rtx.py generate --n "$N_PROBEK" --out "$ZBIOR" \
            > "$LOGI/noc_${STEMPEL}_gen.log" 2>&1; then
        echo "BŁĄD generowania, szczegóły w $LOGI/noc_${STEMPEL}_gen.log"
        tail -15 "$LOGI/noc_${STEMPEL}_gen.log"
        czarna_skrzynka "AWARIA ETAPU 3: generowanie zbioru"
        echo "Środowisko w momencie awarii: $SKRZYNKA"
        exit 1
    fi
    tail -12 "$LOGI/noc_${STEMPEL}_gen.log"
fi

# =============================================================================
#  KOLEJKA POMIARÓW
#
#  Dlaczego kolejka, a nie dwa treningi: epoka trwa 23 s, więc 40 epok to
#  15 minut. Okno nocne ma 12 godzin. Wąskim gardłem nie jest moc, a to,
#  że przy pracy w przerwach między inną robotą jest JEDEN strzał na dobę.
#  Więc jedna noc musi odpowiedzieć na wszystkie otwarte pytania, a nie
#  na jedno.
#
#  ZASADA: od tego miejsca awaria etapu NIE przerywa skryptu. Etapy 1-3
#  (karta, diag, zbiór) przerywają, bo bez nich nic nie ma sensu. Dalej
#  każdy etap jest niezależny — jeśli 'gru' się wywali, koperta dla 'dpu'
#  ma się i tak policzyć.
# =============================================================================

# Stan etapów do podsumowania. Format: "nazwa|wynik|szczegół".
STAN=()

# Log konkretnego etapu ustawia sam etap przez tę zmienną — RANO.txt ma
# wtedy podać ŚCIEŻKĘ, a nie kazać szukać. Bez tego "zbadaj czemu nie
# działa" oznacza przeglądanie katalogu out/ po ciemku.
ETAP_LOG=""

etap () {
    local NAZWA="$1"; shift
    ETAP_LOG=""
    local T0=$SECONDS
    echo
    echo "============================================================"
    echo " $NAZWA     $(date '+%H:%M')"
    echo "============================================================"

    if "$@"; then
        local MIN=$(( (SECONDS - T0) / 60 ))
        echo "  OK, ${MIN} min"
        STAN+=("$NAZWA|OK|${MIN} min")
        return 0
    fi
    local RC=$?
    local MIN=$(( (SECONDS - T0) / 60 ))

    # Dowody TERAZ, nie rano. To jest cała różnica między logiem, który
    # mówi "nie wyszło", i logiem, z którego wiadomo dlaczego.
    czarna_skrzynka "AWARIA: $NAZWA (kod $RC, po ${MIN} min)"
    if [ -n "$ETAP_LOG" ] && [ -f "$ETAP_LOG" ]; then
        {
            echo
            echo "--- $ETAP_LOG: ostatnie 40 wierszy ---"
            tail -40 "$ETAP_LOG"
            echo
            echo "--- $ETAP_LOG: wiersze z bledami ---"
            grep -anE "Error|Traceback|BLAD|BŁĄD|OOM|out of memory|Killed" \
                "$ETAP_LOG" | tail -20 || true
        } >> "$SKRZYNKA" 2>&1
        echo "  NIE UDAŁO SIĘ (kod $RC, ${MIN} min) — dowody w:"
        echo "      $ETAP_LOG"
        echo "      $SKRZYNKA"
        STAN+=("$NAZWA|BŁĄD|kod $RC, log: $ETAP_LOG")
    else
        echo "  NIE UDAŁO SIĘ (kod $RC, ${MIN} min) — dowody w $SKRZYNKA"
        STAN+=("$NAZWA|BŁĄD|kod $RC, log: $SKRZYNKA")
    fi
    echo "  lecę dalej — pozostałe etapy są niezależne"
    return 0
}

# --- TRENING -------------------------------------------------------------
trenuj () {
    local ARCH="$1" RUN="$2"

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
    ETAP_LOG="$LOG"
    # --fresh TYLKO gdy nie ma stanu. Przy wznowieniu po przerwaniu
    # chcemy kontynuować, a nie zaczynać od zera.
    local SWIEZY=""
    [ -f "$RUN/state.json" ] || SWIEZY="--fresh"

    python train_rtx.py train \
        --dataset "$ZBIOR" --run "$RUN" --arch "$ARCH" \
        --epochs "$EPOK" --batch "$BATCH" --mixed --require-gpu $SWIEZY \
        > "$LOG" 2>&1
    local RC=$?

    grep -a "val_accuracy" "$LOG" | tail -3
    echo
    grep -aE "Dokładność na walidacji|klasa 0|znaki:|wzięty za ciszę" "$LOG" || true
    return $RC
}

# --- KOPERTA I BŁĘDY -----------------------------------------------------
#  To odpowiada na dwa pytania, których dokładność walidacyjna nie dotyka:
#  w jakim zakresie tempa i tonu model czyta, oraz czy gubienie elementów
#  bierze się z obcinania okna 2,56 s, czy z niezdolności sieci do
#  zliczania. Drugie rozstrzyga się przez widoczność znaku w oknie —
#  patrz opis w tools/koperta.py.
koperta () {
    local ARCH="$1" RUN="$2"
    if [ ! -f "$RUN/best.keras" ]; then
        echo "  brak $RUN/best.keras — nie ma czego mierzyć"
        return 1
    fi
    local WYNIK="$LOGI/koperta_${ARCH}_${STEMPEL}.txt"
    ETAP_LOG="$LOGI/noc_${STEMPEL}_koperta_${ARCH}.log"
    python -m tools.koperta --model "$RUN/best.keras" \
        --n "$KOPERTA_N" --n-bledy "$KOPERTA_BLEDY" --out "$WYNIK" \
        > "$LOGI/noc_${STEMPEL}_koperta_${ARCH}.log" 2>&1
    local RC=$?
    if [ -f "$WYNIK" ]; then
        sed -n '/^KOPERTA/,/kropka = ZERO/p' "$WYNIK"
        echo
        sed -n '/^WYROK/,/^$/p' "$WYNIK"
    else
        tail -12 "$LOGI/noc_${STEMPEL}_koperta_${ARCH}.log"
    fi
    return $RC
}

etap "TRENING dpu  -> runs/cw2"   trenuj dpu runs/cw2
etap "TRENING gru  -> runs/gru1"  trenuj gru runs/gru1
etap "KOPERTA dpu"                koperta dpu runs/cw2
etap "KOPERTA gru"                koperta gru runs/gru1

# --- PODSUMOWANIE DLA CZŁOWIEKA -----------------------------------------
#  out/RANO.txt: kilkanaście linii do przeczytania po ciemku, po powrocie
#  od innej roboty. Pełne logi zostają obok, ale nie po to, żeby ich
#  szukać o 23:00.
RANO="$LOGI/RANO.txt"

{
    echo "============================================================"
    echo " NOC $(date '+%Y-%m-%d')  --  co wyszło"
    echo "============================================================"
    echo
    echo "ETAPY"
    for w in "${STAN[@]}"; do
        NAZWA="${w%%|*}"; RESZTA="${w#*|}"
        WYNIK="${RESZTA%%|*}"; SZCZ="${RESZTA#*|}"
        printf "  %-28s %-6s %s\n" "$NAZWA" "$WYNIK" "$SZCZ"
    done
    echo
} > "$RANO"

python - "$RANO" runs/cw2 runs/gru1 <<'PY' || true
import json
import sys
from pathlib import Path

rano = Path(sys.argv[1])
w = ["WYNIKI TRENINGU", ""]

for run, opis in ((sys.argv[2], "dpu (wdrażalny na KV260)"),
                  (sys.argv[3], "gru (odniesienie, NIE wdrażalny)")):
    p = Path(run) / "state.json"
    if not p.exists():
        w.append(f"  {run:12s} {opis:34s} brak wyniku")
        continue
    s = json.load(open(p, encoding="utf-8"))
    h = s.get("history", {})
    va, ac, vl = (h.get("val_accuracy", []), h.get("accuracy", []),
                  h.get("val_loss", []))
    w.append(f"  {run}  {opis}")
    w.append(f"      epok {s.get('epoch', 0)}, "
             f"najlepsza walidacja {s.get('best_val_acc', -1) * 100:.2f}%"
             + (f" (epoka {va.index(max(va)) + 1})" if va else ""))
    if vl:
        i = vl.index(min(vl)) + 1
        w.append(f"      val_loss: minimum {min(vl):.4f} w epoce {i}, "
                 f"koniec {vl[-1]:.4f}")
        # Ten warunek jest tu, bo to najczęstszy sposób zmarnowania nocy:
        # trening biegnie dalej, a od pewnej epoki tylko zapamiętuje zbiór.
        if len(vl) > i + 5 and vl[-1] > min(vl) * 1.2:
            w.append(f"      ZAPAMIĘTYWANIE: val_loss rośnie od epoki {i}."
                     f" Epoki po {i + 5} nic nie wniosły.")
    if ac and ac[-1] > 0.995:
        w.append("      accuracy treningowa ~100% = model zapamiętał zbiór."
                 " Potrzeba WIĘCEJ DANYCH, nie epok.")

w.append("")
with open(rano, "a", encoding="utf-8") as f:
    f.write("\n".join(w) + "\n")
PY

# Wyroki z koperty — po jednej linii sedem, bez powtarzania całych tabel.
{
    echo "KOPERTA I PRZYCZYNA BŁĘDÓW"
    echo
    for A in dpu gru; do
        F="$LOGI/koperta_${A}_${STEMPEL}.txt"
        if [ -f "$F" ]; then
            echo "  --- $A ---"
            grep -aE "^  czyta \(" "$F" | sed 's/^/  /' || true
            sed -n '/^WYROK/,/^$/p' "$F" | sed '1,2d;/^$/d' | sed 's/^/  /'
            echo
        else
            echo "  --- $A --- brak pomiaru"
            echo
        fi
    done
    # Sekcja tylko wtedy, gdy jest o czym mówić — pusty nagłówek
    # "co się nie udało" o 23:00 tylko dezorientuje.
    if printf '%s\n' "${STAN[@]}" | grep -q "|BŁĄD|"; then
        echo "CO SIĘ NIE UDAŁO — I GDZIE TEGO SZUKAĆ"
        echo
        for w in "${STAN[@]}"; do
            case "$w" in
                *"|BŁĄD|"*)
                    echo "  ${w%%|*}"
                    echo "      ${w##*|}"
                    ;;
            esac
        done
        echo
        echo "  Stan maszyny w momencie każdej awarii (nvidia-smi,"
        echo "  LD_LIBRARY_PATH, biblioteki CUDA, pamięć, dysk):"
        echo "      $SKRZYNKA"
        echo
    fi

    echo "PEŁNE LOGI"
    echo "  $GLOWNY"
    for F in "$LOGI"/noc_${STEMPEL}_*.log "$LOGI"/koperta_*_${STEMPEL}.txt; do
        [ -f "$F" ] && printf "  %-46s %s\n" "$F" "$(du -h "$F" | cut -f1)"
    done
    echo
    echo "Na pendraka:  bvn_z.bat  (z Windows)"
} >> "$RANO"

echo
echo "============================================================"
cat "$RANO"
echo "============================================================"

# --- HISTORIA TRENINGU: commit, paczka, ewentualny push -----------------
#  Po co: rano wynik ma być poza maszyną, która go policzyła. Jeśli dysk
#  stęknie w nocy, log.csv i state.json są już gdzie indziej.
#
#  DLACZEGO PACZKA, A NIE SAM PUSH. W WSL nie ma Credential Managera
#  Windows, więc token, którym pcha maszyna z Windows, tutaj nie istnieje.
#  Wkładanie go tu oznaczałoby ~/.git-credentials, czyli sekret czystym
#  tekstem na dysku. Zamiast tego `git bundle` pakuje commity do JEDNEGO
#  pliku, ten wraca na pendraku razem z wynikami, a wypycha go maszyna,
#  która poświadczenia ma. Cała historia tego repo to ~212 kB.
#
#  Push jest próbowany mimo to — jeśli jest deploy key z prawem zapisu,
#  wypchnie się od razu. Nie jest to droga krytyczna: niepowodzenie
#  pushu NIE psuje nocy.
#
#  CO trafia do commita: TYLKO historia treningu i pomiary (runs/**/log.csv,
#  runs/**/state.json, out/*.log, out/koperta_*.txt, out/RANO.txt).
#  Świadomie NIE "git add -A" — skrypt bez nadzoru nie ma prawa wciągnąć
#  do historii zmian w kodzie zostawionych w drzewie roboczym z wieczora.
# -------------------------------------------------------------------------
echo
echo "--- historia treningu ---"

if [ ! -d .git ]; then
    echo "to nie jest repozytorium git — pomijam"
    echo "  (kopia robocza z na_hdd.bat nie jest repozytorium; wyniki"
    echo "   wracają przez bvn_z.bat i to wystarcza)"
elif ! git rev-parse --verify -q HEAD >/dev/null 2>&1; then
    echo "repozytorium bez ani jednego commita — pomijam"
    echo "  pierwszy commit rób ręcznie, na oczy, nie w nocy"
else
    git add -- 'runs/**/log.csv' 'runs/**/state.json' \
               "$LOGI"/noc_*.log "$LOGI"/koperta_*.txt "$RANO" 2>/dev/null

    if git diff --cached --quiet; then
        echo "nic nowego w historii treningu — nie commituję"
    else
        # core.hooksPath wyłączony: hook, który w nocy zapyta o cokolwiek,
        # zatrzymałby skrypt tak samo jak ssh bez BatchMode.
        if git -c core.hooksPath=/dev/null commit -q \
               -m "trening $STEMPEL: historia przebiegów i pomiary"; then
            echo "commit: $(git log -1 --format='%h %s')"
        else
            echo "commit NIE przeszedł — patrz wyżej"
        fi
    fi

    GALAZ="$(git rev-parse --abbrev-ref HEAD)"
    PACZKA="$LOGI/historia_${STEMPEL}.bundle"

    if git bundle create "$PACZKA" "$GALAZ" >/dev/null 2>&1; then
        echo "paczka: $PACZKA ($(du -h "$PACZKA" | cut -f1))"
        PACZKA_OK=1
    else
        echo "UWAGA: nie udało się zrobić paczki git bundle"
        PACZKA_OK=0
    fi

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
        # czeka na login z terminala, którego w nocy nie ma.
        if GIT_TERMINAL_PROMPT=0 git push origin "$GALAZ" >/dev/null 2>&1; then
            echo "push: wypchnięte na origin/$GALAZ"
        else
            echo "push: NIE przeszedł — to normalne w WSL i nic nie zginęło"
            if [ "$PACZKA_OK" = "1" ]; then
                echo
                echo "  Rano, z Windows, z katalogu repozytorium na pendraku:"
                echo "      git fetch \"out/$(basename "$PACZKA")\" $GALAZ"
                echo "      git merge --ff-only FETCH_HEAD"
                echo "      git push origin $GALAZ"
                echo
                echo "  Albo raz na zawsze: deploy key z 'Allow write access'"
                echo "  w Settings -> Deploy keys, wtedy push idzie stąd sam."
            fi
        fi
    fi
fi

echo
echo "Koniec: $(date '+%Y-%m-%d %H:%M')"
echo "Do przeczytania: $RANO"
