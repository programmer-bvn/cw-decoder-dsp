# AI_DSP — dekoder CW z jednym wspólnym front-endem

Przetwarzanie sygnałów audio (Morse / CW) + TensorFlow. Projekt zbudowany
wokół jednej zasady:

> **Żaden plik nie ma własnych stałych.** Wszystko idzie z `dsp/config.py`.

## Po co ta zasada

W poprzedniej wersji projektu każde narzędzie miało własną kopię parametrów
front-endu, a normalizacja dB była zapisana trzy razy inaczej:

| plik | odniesienie dB | normalizacja |
|---|---|---|
| `Generator_v5_6.py` | `ref=0.5` | `(db + 30) / 30` |
| `Algorytmv6.1.py` | `ref=1.0` | `(db + 35) / 40` |
| `LiveDecoderav5.5.py` | `ref=1.0` | `(db + 50) / 40` |

Model uczony na pierwszym wariancie, karmiony trzecim, widzi obraz przesunięty
o 20 dB i przeskalowany. **Nie ma z tego żadnego błędu ani ostrzeżenia** —
`val_accuracy` w treningu jest wysokie, a na żywym sygnale model się myli.
Taka usterka jest praktycznie niewykrywalna metodą prób.

Dodatkowo `librosa.power_to_db()` ma domyślnie `top_db=80`, co obcina wynik
**względem maksimum klipu** — czyli po cichu przywraca skalę ruchomą nawet
tam, gdzie podano stałe `ref=1.0`. To jest źródło „różowych ścian”: sam szum,
rozciągnięty na pełny kontrast, wygląda na obrazie jak sygnał.

## Struktura

```
dsp/config.py      JEDNO ŹRÓDŁO PRAWDY — wszystkie stałe
dsp/frontend.py    audio -> obraz dla sieci; jedna implementacja
dsp/morse.py       kodowanie i kluczowanie (co nadaje operator)
dsp/radio.py       kanał radiowy (co z tego dochodzi): QSB, QRM, QRN, dryf
dsp/model.py       architektura sieci

tools/generator.py generator treningowy    audio  -> npz
tools/xray.py      przeglądarka X-Ray      npz/wav -> png (wybrane próbki)
tools/xray_all.py  X-Ray KOMPLETU danych   npz + wav -> strony png + html
tools/wav2net.py   konwerter wav -> sieć   wav    -> wejście modelu
tools/mic2wav.py   konwerter mic -> wav    mikrofon -> wav
tools/calibrate.py pomiar skali dB
tools/trainer.py   trening lokalny         npz    -> .keras

train_rtx.py       PLIK SAMODZIELNY do treningu na maszynie z GPU
diag.py            kontrola liczbowa całego łańcucha, bez treningu
```

Wszystkie cztery narzędzia wołają **tę samą** `frontend.to_net_image()`.
Nie ma „wersji dla live” i „wersji dla treningu”.

Rozdzielenie `morse.py` / `radio.py` jest celowe: pierwszy plik wie, jak
wygląda alfabet Morse'a i timing, drugi — co pasmo robi z sygnałem. Dzięki
temu da się wygenerować zbiór z jednym zjawiskiem włączonym i zmierzyć,
które z nich model przenosi, a które go przewraca.

## Zabezpieczenia przed rozjazdem

1. **Odcisk front-endu.** Generator zapisuje w pliku `.npz` odcisk parametrów
   (`config.fingerprint_str()`). Trener i X-Ray porównują go ze swoim i przy
   różnicy **odmawiają pracy**, wypisując które wartości się różnią:

   ```
   zbiór powstał na innych parametrach front-endu niż obecny config.py:
     DB_MIN: zbiór=-30.0  config.py=-35.0
   ```

2. **`frontend.check_image()`** sprawdza kształt i zakres `[0, 1]`. Obraz
   poza zakresem znaczy, że dane powstały inną normalizacją.

3. **`diag.py`** — 20 kontroli liczbowych całego łańcucha bez treningu.
   Najważniejszy jest TEST 4: ten sam ton o tej samej amplitudzie musi dawać
   ten sam piksel niezależnie od tego, co jeszcze jest w klipie. Przy
   `ref=np.max` albo `top_db=80` ten test nie przechodzi.

## Model kanału radiowego

Sygnał ze stałą amplitudą, białym szumem, tonem co do herca stałym i
maszynowym timingiem **nie występuje na pasmie**. Model uczony wyłącznie na
takim materiale nie ma pojęcia o istnieniu zjawisk, które na falach krótkich
są regułą. `dsp/radio.py` dokłada je kolejno, od najważniejszego:

| zjawisko | co robi | jak wygląda w obrazie | znacznik w X-Ray |
|---|---|---|---|
| **QSB** | zanik, amplituda pływa w rytmie sekund | sygnał jaśnieje i ciemnieje | `~` |
| **QRM** | inna stacja w pasmie | drugi poziomy pas na innej wysokości | `M` |
| **QRN** | trzaski atmosferyczne | **pionowa** kreska przez wszystkie pasma | `N` |
| **fist** | rozjazd ręcznego klucza ±0-25% | nierówne kropki i przerwy | `F` |
| **dryf** | niestabilny VFO, ±8 Hz na klip | pas lekko skośny | — |
| **szum** | nachylenie widma i pływający poziom | tło od −30 do −22 dB | — |

Trzask jest szerokopasmowy, więc daje kreskę **pionową**, a CW jest
wąskopasmowe i daje pas **poziomy**. Dla splotu to łatwe rozróżnienie — pod
warunkiem że model kiedykolwiek widział trzask.

Klipy klasy 0 („puste radio") **mogą zawierać QRM**. To zamierzone: model ma
się uczyć, że sygnał poza spodziewanym tonem nie jest znakiem do odczytu.

Wyłączenie całości — zbiór jak w v5.6:
```bash
python -m tools.generator --no-realism
```

## X-Ray kompletu danych — przegląd wzrokowy

Odniesieniem w tym projekcie jest telegrafista. Jeśli operator nie potrafi
odczytać z obrazu, co sieć dostaje na wejściu, to nikt tego nie odczyta —
a wtedy nie da się rozdzielić, czy model myli się z powodu złych danych,
czy złej architektury.

```bash
python -m tools.xray_all --dataset out/ds30k.npz --cols 16 --zoom 2
```

Rysuje **każdą** próbkę zbioru, pogrupowaną po klasach, na **jednej skali
decybelowej wspólnej dla wszystkich źródeł** — zbioru syntetycznego, plików
wav i nagrań z mikrofonu. Otwiera się `out/xray_all/index.html`.

Na każdym kafelku (128×32 px, powiększenie całkowitą krotnością, bez
interpolacji — te same liczby, jakie dostaje sieć, tylko większe):

```
    oś pozioma   czas, 128 ramek = 2,56 s   (1 piksel = 20 ms)
    oś pionowa   32 pasma mel, 400-1200 Hz  (u dołu niskie)
    jasność      moc od DB_MIN (czarne) do DB_MAX (białe)
```

Pod kafelkiem dwa paski:

| pasek | znaczenie |
|---|---|
| **biały odcinek** | DOKŁADNE granice znaku, który opisuje etykieta |
| czerwone kreski na brzegu | znak wychodzi za kadr okna |
| niebieski | QSB, zanik głębszy niż 0,3 |
| czerwony | QRM, inna stacja w pasmie |
| żółty | QRN, trzaski atmosferyczne |
| zielony | rozjazd ręcznego klucza powyżej 15% |

**Biały wskaźnik jest tu najważniejszy.** Bez niego kafelek pokazuje trzy
nadane znaki i nie da się stwierdzić, który z nich jest opisany — a to nie
jest środkowy piksel obrazu. Zmierzone na zbiorze 30 tys.: środek znaku
z etykiety błądzi między ramką **39,8 i 87,8** (ideał 64), odchylenie
standardowe 8,1 ramki = 0,16 s. Powód: sąsiedzi są losowi i mają różną
długość (`E` to 1 jednostka, `0` to 19), a rozjazd klucza dokłada swoje.
Dlatego model musi znak **znaleźć**, nie tylko odczytać ze środka — i
dlatego oś czasu idzie do warstwy rekurencyjnej, a nie do uśrednienia.

Wskaźnik jest też kontrolą samego generatora: gdyby biały odcinek nie
pokrywał się z grupą elementów widoczną na kafelku, timing byłby policzony
błędnie.

Domyślne sortowanie to `--sort contrast`, od najsłabszych. Najtrudniejsze
próbki zbierają się w lewym górnym narożniku, więc od nich zaczyna się
przegląd — jeśli w zbiorze są próbki nieczytelne dla operatora, model uczy
się na etykietach, których nie da się potwierdzić.

Nagrania z pasma na tej samej skali, do porównania z syntetykiem:
```bash
python -m tools.xray_all --dataset out/ds30k.npz --wav-dir nagrania/
```

## GPU w WSL2 — bez CUDA Toolkit

Do TensorFlow z GPU w WSL2 **nie trzeba instalować CUDA Toolkit**:

- na Windows wystarczy sterownik NVIDIA — wystawia się w WSL jako
  `libcuda.so`;
- **sterownika linuksowego w WSL nie wolno instalować** (nadpisze ten stub);
- biblioteki CUDA ciągnie `pip install tensorflow[and-cuda]` jako koła.

Toolkit jest potrzebny wyłącznie do *kompilowania* aplikacji CUDA.

Jedno zastrzeżenie: `[and-cuda]` istnieje od TF ~2.14. **TF 2.10 nie ma tego
wariantu** i wymaga ręcznie CUDA 11.2 + cuDNN 8.1. Czyli:

| droga | co instalować | Keras | Vitis AI |
|---|---|---|---|
| TF 2.10 na Windows natywnie | CUDA 11.2 + cuDNN 8.1 | 2.10 | zgodne bez konwersji |
| TF 2.10 w WSL2 | CUDA 11.2 + cuDNN 8.1 w WSL | 2.10 | zgodne bez konwersji |
| nowy TF w WSL2 | tylko `pip install tensorflow[and-cuda]` | 3.x | wymaga konwersji modelu |

Najprostsza w instalacji jest trzecia, ale kosztuje krok konwersji przed
Vitis AI. `train_rtx.py` obsługuje wszystkie trzy.

Sources: [CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html), [TensorFlow pip install](https://www.tensorflow.org/install/pip)

## Sprzęt docelowy: AMD Kria KV260

KV260 to Zynq UltraScale+ z DPU **DPUCZDX8G** i Vitis AI. To narzuca dwa
ograniczenia, które zmieniają projekt, a nie tylko sposób wdrożenia.

### 1. DPU nie obsługuje rekurencji

DPUCZDX8G jest akceleratorem **splotowym**. RNN w Vitis AI ma osobne
nakładki — DPURADR16L (Alveo U25) i DPURAHR16L (U50LV) — których na Zynq
nie ma. Model z `GRU`/`LSTM` albo się nie skompiluje, albo kompilator
rozetnie graf i część policzy ARM, co przy strumieniu audio zabija
przepustowość.

Dlatego domyślna architektura to `--arch dpu`: **sam splot**, bez
rekurencji, 624 tys. parametrów. Wersja `--arch gru` została jako punkt
odniesienia — żeby wiedzieć, ile dokładności kosztuje rezygnacja
z rekurencji — ale **nie jest wdrażalna**.

Pilnuje tego **TEST 13** w `diag.py`: sprawdza typy warstw wobec listy
obsługiwanych i osobno potwierdza, że wersja z GRU jest wykrywana jako
niewdrażalna. Kontrola statyczna nie zastępuje kompilatora Vitis AI — on
sprawdzi jeszcze rozmiary jąder, liczbę kanałów i kolejność warstw.

Zasięg widzenia po osi czasu wynosi **110 ramek = 2,20 s** i jest liczony
przez test ze stosu warstw, nie przepisany z komentarza. Musi pokryć
najdłuższy znak z przerwami międzyznakowymi (1,50 s), bo inaczej neuron
w ostatniej warstwie nie widzi obu granic znaku naraz i nie ma z czego
odczytać jego długości.

### 2. Front-end trzeba przenieść bez librosy

Na PetaLinuksie w produkcji librosy nie będzie. `dsp/melref.py` to ta sama
ścieżka na samym numpy, z jawnymi wzorami — **wzorzec do przepisania na C**.
Zgodność zmierzona (**TEST 14**):

| co | różnica względem librosy |
|---|---|
| krawędzie pasm mel | **0,0 Hz** |
| macierz filtrbanku | 2,4·10⁻⁹ |
| STFT (`pad_mode="constant"`) | względna 3,9·10⁻¹⁶ |
| cały obraz, 8 ziaren | 1,2·10⁻⁷ = **6·10⁻⁶ dB** |

Trzy konwencje, które musiały się zgodzić co do cyfry:

- **skala mel to SLANEY, nie HTK.** librosa domyślnie `htk=False`: skala
  jest liniowa poniżej 1000 Hz (`mel = f/(200/3)`). HTK jest logarytmiczna
  wszędzie. Filtrbank `tf.signal.linear_to_mel_weight_matrix` jest w HTK
  i **nie jest wymienny**.
- **normalizacja pasm `norm="slaney"`**: trójkąty mają jednakowe **pole**,
  nie jednakową wysokość (szczyt wychodzi 0,0395, nie 1,0). `tf.signal`
  tego nie robi.
- **dopełnienie STFT zerami, nie odbiciem.** `constant` zgadza się do
  4·10⁻¹⁶, `reflect` rozjeżdża się o 7,6·10⁻³. Domyślna wartość zmieniała
  się między wersjami librosy, więc test ją sprawdza, a nie zakłada.

### 3. Kwantyzacja do int8

Vitis AI wymaga kwantyzacji. Tu jest jedna dobra wiadomość: obrazy są już
**uint8 o stałej, bezwzględnej skali decybelowej**, więc kalibracja
kwantyzatora nie musi zgadywać zakresu wejścia. To bezpośredni skutek
pomiaru `DB_MIN`/`DB_MAX` narzędziem `calibrate`, nie przypadek — przy
skali ruchomej (`ref=np.max`) zakres wejścia zależałby od klipu.

Zbiór kalibracyjny bierze się wprost ze zbioru treningowego.

### Do sprawdzenia przed wdrożeniem

Nie mam tu dostępu do KV260, więc te punkty zostają otwarte:

- **wejście audio.** KV260 nie ma kodeka audio. Wejście przez USB (IC-7300
  wystawia się jako UAC) albo przez I2S w PL.
- **`Flatten` + `Dense(128)`** po 8 krokach czasu — obsługiwane, ale warto
  potwierdzić rozmiar u kompilatora.
- **wersja TF a kwantyzator** — rozwiązane przez trening na TF 2.10
  (patrz sekcja o treningu), ale warto potwierdzić, której wersji Vitis AI
  będziesz używać.

### Czego NIE udało się tu wykonać

TF 2.10 wymaga Pythona najwyżej 3.10, a na tej maszynie jest 3.12 —
**nie mogłem uruchomić `train_rtx.py` na docelowym TF 2.10.** Zgodność
została wyprowadzona z różnic API, nie zmierzona. Konkretnie zmienione:

| co | dlaczego |
|---|---|
| `import keras` → `tf.keras` | w TF 2.10 kanoniczną drogą jest `tf.keras`; w 2.16+ `tf.keras` to Keras 3, więc działa w obu |
| `.keras` → `.h5` (dobierane) | format `.keras` istnieje od Kerasa 3; na 2.10 zapis pod tą nazwą zawiódłby albo cicho zrobił SavedModel |
| librosa → numpy | tamto środowisko nie ma librosy |
| `Path` → `str()` przy zapisie i `CSVLogger` | starsze wersje nie wszędzie przyjmują `Path` |

Co jeszcze może zaskoczyć na TF 2.10, a czego nie sprawdziłem:
`tf.data` z trójelementową krotką `(x, y, sample_weight)` i funkcją `map`
o sygnaturze `(img, label, *rest)`; `initial_epoch` razem z
`CSVLogger(append=True)`. Oba są udokumentowane jako obsługiwane w 2.10,
ale gdyby coś się wysypało, to najpierw tam.

Sources: [DPUCZDX8G PG338 — Core Overview](https://docs.amd.com/r/en-US/pg338-dpu/Core-Overview), [Vitis AI 3.5 — Developing a Model](https://xilinx.github.io/Vitis-AI/3.5/html/docs/workflow-model-development.html), [Vitis AI RNN UG1563 — Supported Targets and Operators](https://docs.amd.com/r/3.0-English/ug1563-vitis-ai-rnn/Supported-Targets-and-Operators)

## Trening na maszynie z GPU

`train_rtx.py` jest **samodzielny** — jeden plik, bez importów z projektu.
Kopiujesz go na maszynę z kartą i uruchamiasz; dane potrafi wygenerować sam.

**Zależności: tylko `tensorflow` i `numpy`.** Front-end (mel, STFT, skala dB)
jest wliczony w plik, na samym numpy — bez librosy, bo nie ma jej ani
w środowisku z TF 2.10, ani na KV260.

Docelowe środowisko treningowe:

```
tensorflow==2.10.0   tensorflow-gpu==2.10.0   keras==2.10.0
numpy==1.21.6        scipy==1.4.1
```

TF 2.10 to ostatnia wersja z **natywnym GPU na Windows** i zarazem ta, którą
przyjmuje Vitis AI — więc nie jest to kompromis, tylko właściwy wybór.
Model zapisany z Kerasa 2.10 w `.h5` załaduje się w Kerasie 2.12 używanym
przez Vitis AI. Gdyby trening szedł na Kerasie 3, trzeba by konwertować.

Plik obsługuje oba zestawy: format zapisu modelu dobiera się do wersji
(`.h5` dla Kerasa 2, `.keras` dla 3), a wznowienie szuka obu.

```bash
python train_rtx.py generate --n 200000
```
```bash
python train_rtx.py train --epochs 200 --batch 256 --mixed
```

Na starcie wypisuje wersje i wykryte GPU — warto zerknąć, czy karta jest
widziana, zanim zostawisz to na noc.

### 200 tys. próbek to liczba DLA GPU

Zmierzone na CPU (12 rdzeni, batch 64, obecny model 30,3 mln mnożeń
na próbkę, 0,255 s na krok — `python -m tools.profil`):

| próbek | kroków/epoka | epoka | 40 epok | 100 epok |
|---:|---:|---:|---:|---:|
| 10 000 | 156 | 0,7 min | 0,4 h | 1,1 h |
| 30 000 | 469 | 2,0 min | **1,3 h** | 3,3 h |
| 60 000 | 938 | 4,0 min | 2,7 h | 6,6 h |
| 200 000 | 3 125 | 13,3 min | 8,8 h | **22,1 h** |

Na CPU pierwszy przebieg rób na **30 tys. próbek i 40 epok** — 1,3 godziny
zamiast nocy. 200 tys. ma sens dopiero z kartą.

```bash
python train_rtx.py generate --n 30000
```
```bash
python train_rtx.py train --epochs 40 --batch 64
```

### Gdyby CPU nadal był wąskim gardłem

Rozkład kosztu (z `tools.profil`) pokazuje, gdzie siedzi czas:

| warstwa | mapa wyjścia | mln MAC | udział |
|---|---|---:|---:|
| conv2d | 128×32×32 | 1,18 | 3,9% |
| **conv2d_1** | **64×16×48** | **14,16** | **46,6%** |
| conv2d_2 | 32×8×64 | 7,08 | 23,3% |
| conv2d_3 | 16×4×96 | 3,54 | 11,7% |
| conv2d_4 | 8×2×128 | 1,77 | 5,8% |
| conv2d_5 | 8×2×128 | 2,36 | 7,8% |

Koszt jest w warstwach o **dużej mapie**, nie o wielu kanałach — obcinanie
kanałów w głębi sieci prawie nic nie daje. Zmierzone warianty:

| wariant | mln MAC | s/krok | przyspieszenie |
|---|---:|---:|---:|
| obecny 128×32 | 30,3 | 0,255 | 1,00× |
| 16 pasm mel | 17,4 | 0,124 | 2,05× |
| 8 pasm mel | 11,8 | 0,060 | 4,26× |
| 16 pasm + mniej kanałów | 7,7 | 0,061 | 4,18× |
| 8 pasm + mniej kanałów | 5,5 | 0,027 | 9,39× |

Zmniejszenie liczby pasm jest uzasadnione pomiarem z TEST 10 (silny ton
podnosi 21 z 32 pasm, oś częstotliwości jest częściowo redundantna), **ale
ma koszt**: przy 8 pasmach na 400-1200 Hz jedno pasmo to 100 Hz, więc obca
stacja 150 Hz od naszego tonu wypada 1-2 pasma dalej i przestaje być
odróżnialna. Przy 16 pasmach to 50 Hz na pasmo. **Najpierw zmniejsz zbiór,
liczbę pasm ruszaj dopiero, jeśli to nie wystarczy** — i pamiętaj, że
zmienia ona odcisk front-endu, czyli wymaga wygenerowania zbioru od nowa.

### OpenBLAS i liczba wątków

Na tej maszynie (12 rdzeni, 12 GB RAM) OpenBLAS przy domyślnej liczbie
wątków przewracał się na `Memory allocation still failed after 10 retries`.
Pomaga ograniczenie:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -m tools.profil
```

Po każdej **ukończonej epoce** zapisuje `runs/cw1/last.keras`,
`runs/cw1/best.keras`, `runs/cw1/state.json` i dopisuje `runs/cw1/log.csv`.
Ponowne uruchomienie **tego samego polecenia** wznawia od następnej epoki —
nic nie trzeba podawać ręcznie. Wyłączenie komputera w środku epoki kosztuje
najwyżej tę jedną epokę. Świeży start: `--fresh`.

Kolejność zapisu jest istotna: najpierw model, potem `state.json`. Gdyby
`state.json` trafił na dysk pierwszy, a prąd padł przed zapisem modelu,
wznowienie ruszyłoby z niezgodnym numerem epoki.

Stałe front-endu w `train_rtx.py` są kopią `dsp/config.py`. Kopia jest tam
świadomie — plik ma być samodzielny — a przed rozjechaniem się chroni
**TEST 12** w `diag.py`, który porównuje nie tylko stałe, ale i obrazy
generowane z tego samego ziarna, i wymaga zgodności co do bitu.

## Kolejność pracy

```bash
python diag.py
```
```bash
python -m tools.xray --text SOS --wpm 20
```
```bash
python -m tools.generator --n 5000
```
```bash
python -m tools.xray_all --dataset out/morse_dataset.npz --zoom 2
```
```bash
python -m tools.trainer --epochs 30 --batch 64
```
```bash
python -m tools.mic2wav --monitor
```
```bash
python -m tools.wav2net --wav out/mic.wav --predict
```

Najpierw `diag.py`, potem podgląd syntezy, potem **mały** zbiór (5000) i
obejrzenie go w X-Ray. Dopiero gdy obrazy wyglądają poprawnie — pełne 40000.
Wygenerowanie 40000 próbek na złych parametrach to godzina pracy do kosza.

## Co zostało poprawione względem v5.6

| rzecz | było | jest |
|---|---|---|
| normalizacja | trzy różne kopie | jedna, w `frontend.py` |
| `top_db` | domyślne 80 (skala ruchoma!) | jawnie `None` |
| rozmiar zbioru | 1,3 GB float32 | 164 MB uint8 (0,16 dB/krok) |
| wczytywanie do treningu | cała tablica w RAM | `tf.data`, konwersja w partii |
| podział na walidację | losowy | warstwowy po klasach |
| rozmiar modelu | 51 MB (`Flatten`+`Dense(512)`) | ~1,5 MB (CNN + GRU) |
| kluczowanie | narastanie liniowe | cosinus podniesiony |
| timing | sklejane `linspace` | granice na siatce jednostek |
| tryb na żywo | bufor obrazu | bufor audio (wynik = ścieżka plikowa) |
| format modelu | `.h5` | `.keras` |

### Rozmiar modelu

`Flatten` + `Dense(512)` po trzech blokach splotowych to 8192 → 512, czyli
4,2 mln wag. Te wagi uczą się **osobno dla każdej pozycji w czasie**, więc
znak przesunięty o dwie ramki trafia w inny zestaw wag — model musi nauczyć
się każdego znaku w każdym przesunięciu i dlatego wymaga wycentrowania.

Tutaj splot redukuje głównie oś **częstotliwości** (ton jest wąskopasmowy,
32 pasma to nadmiar), a oś czasu przechodzi do dwukierunkowego GRU, który
rozpoznaje wzór niezależnie od jego położenia w oknie.

Uwaga: `GlobalAveragePooling` po osi czasu byłoby tu **błędem**. `..-` i `-..`
mają identyczny zestaw elementów i identyczną średnią energię — różnią się
wyłącznie kolejnością. Uśrednienie po czasie zrównuje U z D.

## ZMIERZONA granica kompetencji modelu

Model po 200 epokach, **98,68% dokładności walidacyjnej na syntetyku**,
sprawdzony na sygnale IDEALNYM (bez zaniku, chirpu i rozjazdu klucza):

| | zakres, w którym czyta | poza nim |
|---|---|---|
| tempo | **18–22 WPM** | **zero zdarzeń** |
| ton | **670–830 Hz** | **zero zdarzeń** |

Nie „gorzej" — zero. To najważniejsza liczba w tym projekcie i wyjaśnia,
dlaczego wysoka dokładność walidacyjna nic nie mówiła o pracy na pasmie.

### Trzy przyczyny, każda zmierzona osobno

**1. Tempo.** `WPM_JITTER = 0` dawał zbiór o DOKŁADNIE jednym tempie.
Pierwsze nagranie z automatu IC-7300 (`probki/mic3_pamiec_15wpm.wav`,
timing idealny) idzie 15 WPM — kropka 80 ms zamiast 60 ms, czyli każdy
element o trzecią dłuższy niż cokolwiek, co model widział. Przy 15 WPM
i idealnym tonie 750 Hz model nie zwrócił **ani jednego znaku**.
Naprawione: `WPM_JITTER = 7.0`, czyli 13–27 WPM. Wymaga treningu od nowa.

**2. Ton.** Poza 670–830 Hz model milczy. Po przestrojeniu przez
`dsp/tune.py` odczyt jest **identyczny** z natywnym:

| ton | bez przestrojenia | z `tune.py` |
|---|---|---|
| 500 Hz | nic | `CCQ2BVQ2B` |
| 750 Hz | `CCQ2BVQ2B` | — |
| 1050 Hz | nic | `CCQ2BVQ2B` |

Wąski `TONE_SPREAD` **nie jest wadą do naprawienia** — daje
SELEKTYWNOŚĆ, czyli ignorowanie obcej stacji. Poszerzenie treningu na
całe 400–1200 Hz dałoby odporność na ton kosztem tej selektywności.
Dlatego: wąski trening plus dostrajanie front-endem. Przestrajanie jest
w `wav2net` **domyślnie włączone**.

**3. Poziom.** Nagrania mikrofonem PC były przesterowane o 12 dB i obraz
nasycał się przy `DB_MAX`. `wav2net` ustawia poziom sam (`auto_gain_db`),
celując w szczyt 17 dB. Na `mic3` dało to różnicę:

```
bez korekty (szczyt 24,0 dB):  CQCCCLQ2BVSQ2BVQ2BV9SE5
z korektą  (szczyt 17,3 dB):   CCLQ2BVQ2BVQ2BVS
```

To NIE jest powrót do skali ruchomej: jedno wzmocnienie na całe nagranie,
w dziedzinie czasu, przed front-endem. Skala dB pozostaje bezwzględna.

### Co zostaje po naprawieniu wszystkich trzech

`probki/mic2_manipulator_20wpm.wav` — 20 WPM, 751 Hz, elementy poprawne,
czyli najlepsze możliwe dopasowanie do treningu:

```
CMCQCQBVQ2BVSHCQCQCQCQQDQ2BVQ2LVP2KP2KP2KP2KBV73HHEE5HYN1AIUARUAUARUARRIUAUCCCQ2BVQ2BVQ2BV9S
```

Widać `CQCQ`, `Q2BV` cztery razy, `P2KP2KP2KP2K`, `UARUAUAR` i `73`.
Rozpoznawalne fragmenty, nie czysty tekst.

Zostaje więc **jedno** ograniczenie: model klasyfikuje jeden znak z okna
2,56 s, a przy 20 WPM w oknie widać 4–5 znaków, podczas gdy uczono go
wybierać środkowy z **trzech**. Przy pięciu widocznych „środkowy" nie jest
określony i sieć ciąży do znaku najdłuższego. Skutek: **gubione są znaki
krótkie** (`S ...`, `N -.`, `D -..`, `E .`), a czytane długie
(`Q --.-`, `2 ..---`, `B -...`, `V ...-`).

(Wcześniejsza wersja tej sekcji podawała jako główne ograniczenie scalanie
powtórzeń `EE`/`OO`. Na prawdziwym sygnale dominuje usuwanie znaków
krótkich. Poprawione po pomiarze.)

Rozwiązaniem jest dekoder sekwencyjny **CTC**: ta sama sieć splotowo-
rekurencyjna, ale zwracająca wyjście dla każdego kroku czasowego, uczona
funkcją straty CTC na całych ciągach znaków. Nie wymaga centrowania ani
etykietowania pojedynczych znaków, radzi sobie z powtórzeniami i zwraca
cały tekst. Front-end, generator i X-Ray z tego projektu przechodzą do
wersji CTC bez zmian — zmienia się `dsp/model.py`, sposób etykietowania
w generatorze (cały tekst zamiast jednego znaku) i pętla treningowa.

Dla samego Morse'a warto też znać drogę bez sieci: Goertzel na tonie →
obwiednia → próg adaptacyjny → automat czasowy. Dla czystego sygnału
działa lepiej niż jakikolwiek model i nie wymaga treningu; sieć wygrywa
przy silnym szumie, zaniku i nakładających się stacjach.

## Środowisko

```bash
pip install -r requirements.txt
```

TensorFlow na Windows natywnie **nie widzi GPU** (od 2.11) — potwierdza to
sam przy starcie. Dlatego trening idzie na osobnej maszynie przez
`train_rtx.py`.

### Ograniczenia maszyny lokalnej (12 GB RAM)

Zmierzone, nie wywnioskowane:

- **`--batch 128` zabija proces** — bez komunikatu, z kodem wyjścia 0.
  To brak pamięci, nie błąd TensorFlow. Lokalnie działa `--batch 64`.
- Jednoczesne uruchomienie `diag.py` i treningu daje
  `OpenBLAS: Memory allocation failed`. Jedno naraz.
- Trening 30 tys. próbek zajmuje ok. 118 s na epokę na CPU, czyli
  sensowna liczba epok to godziny — stąd `train_rtx.py`.
- Generowanie: 30 tys. próbek w 125 s, plik 119 MB w uint8.
  W `train_rtx.py` generowanie idzie na wszystkich rdzeniach.
