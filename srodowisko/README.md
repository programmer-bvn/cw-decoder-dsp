# Środowisko GPU w WSL2 — co trzeba zrobić ZA KAŻDYM RAZEM

Ten katalog istnieje, bo konfiguracja karty w WSL2 zabrała trzy nieudane
podejścia jednej nocy, a przyczyny były trzy różne i żadna nie wynikała
z drugiej. Wszystkie trzy trzeba obsłużyć przy **każdym** uruchomieniu —
to nie jest jednorazowa instalacja.

Sprzęt odniesienia: RTX 3050, WSL2 na Windows 11, TensorFlow 2.21,
Python 3.12.

---

## 1. Zmienne środowiskowe giną razem z powłoką

Nie ma czego „ustawić na stałe". `LD_LIBRARY_PATH` i aktywacja venv żyją
w jednej instancji powłoki i umierają z nią.

Skutek, o który najłatwiej się potknąć: **wykonanie skryptu przez `./`
tworzy powłokę potomną.** Cokolwiek skrypt w sobie wysourcuje, obowiązuje
tylko do końca jego działania i nie wraca do powłoki wołającej.

```bash
source srodowisko/rtx3050_setenv.sh   # tak — zmienia BIEŻĄCĄ powłokę
./srodowisko/rtx3050_setenv.sh        # bez sensu — ustawia i natychmiast traci
```

Ale ta sama zasada działa na naszą korzyść: **`noc.sh` ładuje środowisko
sam**, bo potrzebuje go dokładnie tyle, ile trwa trening, czyli tyle, ile
żyje jego własna powłoka. Dlatego `./noc.sh` wystarcza i nie wymaga
niczego przed sobą.

## 2. WSL domyślnie podaje Pythona, na którym to nie pojedzie

Aktualne WSL ładuje **Python 3.14**, a TensorFlow nie ma dla niego koła —
na PyPI są `cp310`–`cp313` i nic wyżej. Bez aktywacji venv z 3.12 dostaniesz
albo „No module named tensorflow", albo pip, który nie znajduje wersji.

Uwaga na `setup_gpu_env.sh`: wywołanie `python3 -m venv` bierze **domyślnego**
Pythona, czyli dziś 3.14, i cała instalacja się rozsypuje. Dlatego w wersji
w tym katalogu interpreter jest **wskazany jawnie** i skrypt przerywa, jeśli
go nie ma.

To nie ma nic wspólnego ze sterownikami NVIDIA — te są niezależne od
Pythona. Bramką jest wyłącznie dostępność koła TensorFlow.

## 3. Nośniki podłączone PO starcie WSL nie montują się same

WSL montuje dyski obecne w momencie startu. Pendrak wetknięty później —
a tak jest zawsze z D888 — nie pojawia się w `/mnt/`, i `wsl` odpowiada:

```
wsl: Failed to translate 'd:\AI_DSP'
```

`[automount] enabled=true` w `/etc/wsl.conf` tego nie załatwia, bo dotyczy
startu, nie podłączenia w trakcie. Trzeba zamontować z ręki —
`srodowisko/mount_pendrak.sh` to robi.

**Litera dysku zmienia się między maszynami.** Na jednej D888 jest pod `D:`,
na innej pod `I:`. Skrypt przyjmuje literę jako argument i nie zgaduje.

---

## Kolejność

```bash
# raz na maszynę
./srodowisko/setup_gpu_env.sh

# przy każdym uruchomieniu, jeśli pracujesz z pendraka
sudo ./srodowisko/mount_pendrak.sh d

# trening — ładuje środowisko sam
./noc.sh
```

Diagnostyka, gdy karta nie jest widziana, w tej kolejności:

```bash
./srodowisko/gpu_check.sh    # czy TF ją widzi
./srodowisko/gpu_libs.sh     # KTÓRA biblioteka CUDA się nie ładuje
nvidia-smi                   # czy sterownik po stronie Windows żyje
```

`gpu_libs.sh` jest tym, który mówi coś konkretnego: wypisuje każdą
bibliotekę osobno, więc widać, czy brakuje cuDNN, cuBLAS, czy nvJitLink —
zamiast jednego „Could not load dynamic library" bez nazwy.

## Dlaczego LD_LIBRARY_PATH jest w ogóle potrzebny

Biblioteki CUDA instalowane pipem lądują w
`venv/lib/python3.12/site-packages/nvidia/*/lib`, a tych katalogów **nie ma
na ścieżce ładowania** dynamicznego linkera. TensorFlow ich nie znajduje i
cicho spada na CPU, wypisując tylko ostrzeżenie na poziomie logów, który
łatwo wyciszyć.

Właśnie dlatego `TF_CPP_MIN_LOG_LEVEL` w `train_rtx.py` jest ustawiony na
`1`, a nie `2`: poziom 2 ukrywa te ostrzeżenia i trening przez dwanaście
godzin liczy na procesorze bez śladu w logu, dlaczego.

Są dwa sposoby, oba działają:

- **`rtx3050_setenv.sh`** — dopisuje te katalogi do `LD_LIBRARY_PATH`.
  Nie wymaga roota, nic nie zmienia w systemie. Preferowany.
- **`setup_gpu_env.sh`** — robi symlinki do `/usr/lib/wsl/lib`. Skuteczne,
  ale to katalog WSL-a na jego własne biblioteki sterownika GPU;
  aktualizacja WSL może je nadpisać, a symlinki wskazujące na usunięty
  venv zostaną wiszące.
