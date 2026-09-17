# Historia problemów i rozwiązań

Ten plik istnieje, bo **rozwiązanie zostało raz znalezione i zgubione**.
17.09.2026 dekoder dawał odczyty w rodzaju `CCQQQCCCCQQQQBBBVVV` i
zabraliśmy się za wymyślanie progu czasowego — a ten sam problem był już
rozwiązany w lutym, lepiej, i leżał na tym samym pendraku.

Zasada jest więc jedna: **każdy rozwiązany problem dostaje tu wpis.**
Nie opis kodu — kod opisuje się sam. Wpis ma mówić: jaki był objaw, co
było przyczyną, co pomogło i gdzie to teraz mieszka.

Co się tu NIE nadaje: historia zmian w plikach (od tego jest `git log`),
opis działania (od tego są komentarze w kodzie), plany (od tego README).

---

## Dlaczego lutowy kod niczego nie wyjaśnia

W `CW_decoder_luty_2026` leży 88 plików z 21–23 lutego 2026: modele od
`morse_brain_v2` do `v5_6_ULTRA`, zbiory do 1,3 GB, kilkanaście wersji
generatora, trenera, dekodera na żywo i X-Raya.

Próbowałem odtworzyć z nich historię i **się nie da**. Nagłówki plików
zawierają tylko `# --- KONFIGURACJA ---`, a porównanie stałych między
wersjami generatora daje jedną różnicę (`DRIFT_MAX`), bo reszta siedzi
wewnątrz funkcji. Numer wersji mówi, że coś się zmieniło, ale nie mówi
**co ani po co**.

To nie jest zarzut wobec tamtej pracy — osiem dni intensywnego szukania
wygląda dokładnie tak. To jest powód, dla którego ten plik powstał.

---

## Rozwiązane

### Powtórzenia tego samego znaku w odczycie

**Objaw.** Odczyt wygląda jak `CCQQQCCCCQQQQBBBVVV` albo
`QCQQCQZZDZDDEEEQ2B`. Informacja jest w środku — miary z
`tools/nagrania.py` pokazują 6/9 „w kolejności" przy 4/9 „w całości" —
ale dla człowieka to nieczytelne.

**Przyczyna.** Okno sieci przesuwa się co 20 ms, a znak przy 20 WPM trwa
około pół sekundy. Ten sam znak widzi więc kilkanaście kolejnych okien
i każde z nich zgłasza go osobno.

**Rozwiązanie, luty 2026** — `LiveDecoder_V5_3_FINAL.py`:

```python
last_char = ""
if amp > 0.02:                                  # jest sygnał
    if pewnosc > 0.85:
        curr = id_to_char.get(char_id, " ")
        if curr != " " and curr != last_char:   # nie powtarzaj
            print(curr)
            last_char = curr
else:
    last_char = ""                              # CISZA resetuje blokadę
```

Dopracowane w `LiveDecoderav5.5.py` — reset nie następuje przy pierwszym
cichym bloku, tylko po ośmiu:

```python
silence_counter += 1
if silence_counter > 8:
    last_char = ""
```

**Dlaczego to jest dobre.** Separatorem jest **cisza w sygnale**, a nie
zgadnięta stała czasowa. Dzięki temu dwie takie same litery obok siebie
(`LL` w `HELLO`) wychodzą poprawnie — dzieli je przerwa międzyznakowa,
więc blokada sama się zwalnia. Licznik ośmiu bloków jest odszumieniem:
chwilowy spadek amplitudy wewnątrz znaku nie kasuje blokady.

**Czego NIE da się przenieść wprost.** Tamten dekoder decydował o każdym
krótkim bloku audio i miał próg amplitudy pod ręką. Tutaj decyzja zapada
dla okna 2,56 s, które **zawsze** zawiera kilka znaków — przerwy
międzyznakowej na tym poziomie nie widać. Klasa 0 („puste radio") też nie
zadziała jako zamiennik: model uczył się jej na klipach bez stacji
w ogóle, a nie na trzyjednostkowych przerwach wewnątrz nadania.

**Co z tym zrobiono.** 17.09.2026 w `tools/nagrania.py` wszedł próg
czasowy 0,35 s — obejście, nie naprawa, i celowo tylko w narzędziu
POMIAROWYM. Próg jest zmierzony: od góry ogranicza go odstęp dwóch
takich samych liter (480 ms przy 30 WPM), od dołu długość powtórzeń
(0,25 s nie sklejało, 0,35 s sklejało).

**Właściwa naprawa** to odtworzenie lutowej logiki na obwiedni
kluczowania liczonej z audio przy tej samej rozdzielczości co ramki
(20 ms) — wtedy „cisza" znowu staje się dostępna. Albo dekoder CTC,
który znosi całą tę klasę problemu, bo etykietuje ciąg zamiast wybierać
znak z okna.

---

### Sieć sypie losowymi odpowiedziami przy odczycie z mikrofonu

**Objaw.** Model wytrenowany poprawnie daje na żywo ciągi typu `YYYYY`.

**Przyczyna.** Format tensora wejściowego przy inferencji różnił się od
treningowego: zakres wartości (0–255 kontra 0,0–1,0), orientacja osi,
kształt obwiedni.

**Zapisane w** `CW/Mapa Projektu CW-AI (Architektura Systemu).md`,
luty 2026. Cytat, który okazał się najtrwalszą myślą tamtego etapu:
jakość i format danych przy inferencji muszą być tożsame z treningowymi.

**Gdzie to mieszka teraz.** To jest powód istnienia `dsp/config.py`
i odcisku front-endu (`fingerprint_str`, `check_fingerprint`). Wszystkie
cztery wejścia — generator, X-Ray, wav2net, mikrofon — przechodzą przez
tę samą funkcję `frontend.to_net_image()`, a zbiór zapisuje odcisk
parametrów, więc rozjazd jest wykrywany, a nie przemilczany.

---

### Skalowanie względem klipu — tu stanął etap lutowy

**Skąd się wziął obecny front-end.** Punktem wyjścia był wodospad:
mel-spektrogram rysowany jako wąski pasek 128×32 z czasem na osi X.
Operator opisuje te obrazy jako „wyglądające jak laska niewidomego" —
długi pasek z segmentami, czytany wzrokiem. Cała interpretacja szła
graficznie i to była dobra intuicja, bo telegrafista rozpoznaje wzór,
a nie liczby.

Etap stanął na **skalowaniu danych i sygnałów**. Przyczyna jest w dwóch
linijkach `CW_decoder_luty_2026/Audio2Waterfall.py`:

```python
spec_db   = librosa.power_to_db(spec, ref=np.max)
spec_norm = (spec_db - np.min(spec_db)) / (np.max(spec_db) - np.min(spec_db))
```

**Obie normalizacje są WZGLĘDEM SAMEGO KLIPU.** `ref=np.max` odnosi
decybele do najgłośniejszego punktu w tym klipie, a min-max rozciąga
kontrast do pełnej skali — również w obrębie klipu.

Skutki, wszystkie zabójcze dla uczenia:

- sygnał mocny i ledwo słyszalny dają **identyczny obrazek**, więc sieć
  nie ma jak nauczyć się, co jest sygnałem, a co szumem tła;
- ten sam sygnał fizyczny wygląda inaczej zależnie od tego, co jeszcze
  jest w klipie — wystarczy jeden trzask, żeby przeskalować całą resztę;
- „cisza" po rozciągnięciu kontrastu przestaje być cicha, bo sam szum
  zostaje rozciągnięty na pełny zakres.

**Gdzie to mieszka teraz.** Obecny front-end jest bezpośrednim potomkiem
tamtego wodospadu — `n_mels=32`, `hop_length=160`, obraz 128×32, czas na
osi X, wszystko to samo. Naprawiona jest wyłącznie skala:

| | luty 2026 | teraz |
|---|---|---|
| odniesienie dB | `ref=np.max` (klip) | `DB_REF = 1.0` (bezwzględne) |
| zakres | `min..max` klipu | `DB_MIN=-30`, `DB_MAX=24` (zmierzone) |
| obcinanie | `top_db` domyślny | `top_db` nie istnieje |

Dlatego w `dsp/config.py` przy `DB_MIN`/`DB_MAX` stoi adnotacja, że są
ZMIERZONE, nie dobrane. I dlatego `diag.py` ma osobny TEST 4 na skalę
bezwzględną — bo to jest ta jedna rzecz, której zepsucie nie daje błędu,
tylko cicho psuje uczenie.

**Wniosek ogólny.** Normalizacja „na obrazek" jest odruchem z widzenia
komputerowego, gdzie jasność sceny naprawdę nie ma znaczenia. W radiu
poziom sygnału JEST informacją. To samo nieporozumienie wróciło później
w innej postaci: nagrania mikrofonowe przesterowane o 12 dB nasycały
obraz przy `DB_MAX` — stąd `auto_gain_db()` w `wav2net`, które ustawia
poziom PRZED front-endem, zamiast pozwolić front-endowi się dopasować.

---

### Model czytał tylko 18–22 WPM

**Objaw.** 98,68% na walidacji, a na paśmie fragmenty albo nic.
Nagranie z pamięci IC-7300 przy 15 WPM: **ani jednego znaku**.

**Przyczyna.** `WPM_JITTER = 0` w konfiguracji generatora. Zbiór miał
dokładnie jedno tempo, więc wysoka dokładność walidacyjna mierzyła
zdolność do czytania tego jednego tempa.

**Rozwiązanie.** `WPM_JITTER = 7.0` (13–27 WPM) i trening od nowa.
Zmierzony skutek: koperta tempa 10–30 WPM, przy 750 Hz od 98 do 100%.

**Wniosek ogólny**, wart więcej niż sama poprawka: dokładność walidacyjna
mierzy tylko to, co jest w rozkładzie zbioru. Stąd `tools/koperta.py` —
przemiata tempo i ton po siatce i pokazuje, GDZIE model milknie.

---

### Odjazd nośnej kończył dekodowanie — i nadal by kończył

**Objaw, zaobserwowany na modelu AG1LE.** Dekodowanie idzie 1:1, po czym
ton odjeżdża odrobinę w bok i odczyt się urywa. Nie pogarsza się — ustaje.

**Dlaczego tak się dzieje.** Sieć uczona na wąskim rozkładzie tonu widzi
przestrojony sygnał jako coś, czego nigdy nie widziała. Nie ma powodu
zgadywać „to pewnie to samo, tylko wyżej" — dla niej to inny obraz.

**To NIE jest wada, którą naprawia się treningiem.** Poszerzenie rozkładu
tonu na całe 400–1200 Hz dałoby odporność, ale kosztem SELEKTYWNOŚCI:
model przestałby ignorować obcą stację kilkadziesiąt herców obok, a na
paśmie to jest częstszy przypadek niż rozstrojenie. Wybór jest świadomy.

**Zabezpieczenie: `dsp/tune.py`.** Pętla śledząca ton przestraja sygnał
PRZED front-endem, więc do sieci zawsze trafia ton nominalny. Zakresy:

| | |
|---|---|
| zaczep | całe pasmo `FMIN`–`FMAX`, czyli 400–1200 Hz |
| nadążanie | `CAPTURE_HZ = 12` Hz na krok 50 ms, czyli do **240 Hz/s** |
| wybieg bez sygnału | `HOLD_S = 3` s |

**Zmierzone na prawdziwych nagraniach** (16.09.2026, `tools/nagrania.py`):
zaczep w **100% czasu na wszystkich czterech**, mediany tonu 700, 750,
750 i 839 Hz. Czyli pętla robi dokładnie to, do czego jest.

**DWIE RÓŻNE KOPERTY — i łatwo je pomylić.** `tools/koperta.py` podaje
obraz sieci WPROST, bez przestrajania, więc jego 670–830 Hz to koperta
**samej sieci**. Tolerancja CAŁEGO UKŁADU to 400–1200 Hz i wyznacza ją
`tune.py`, nie model. Mieszanie tych dwóch liczb prowadzi albo do paniki
(„model czyta tylko 160 Hz pasma"), albo do złudzenia („mamy 800 Hz
zapasu"), zależnie od tego, którą się weźmie.

**Pułapka, która właśnie się zacieśniła.** Zwiększenie zbioru z 200 tys.
do 1 mln ZWĘZIŁO kopertę sieci: przy 650 Hz trafienia spadły z 40–82% do
0–20%. Lepszy model to ostrzejsze milczenie poza rozkładem. Znaczy to,
że **udział `tune.py` w działaniu układu rośnie z każdą poprawą modelu**.
Gdy pętla zgubi zaczep, odczyt nie pogorszy się — zniknie, dokładnie tak
jak w AG1LE. Stąd `lock_report()` podaje procent czasu z zaczepem i stąd
ta liczba jest w raporcie z nagrań: to jest wskaźnik, który trzeba
obserwować, a nie założyć.

---

### PLL na nośną uciekał na szum poniżej pasma

**Objaw, luty 2026.** Operator: „miałem coś w rodzaju PLL na nośną, ale
nie zawsze okno ustawiało się na sygnale — szczególnie jak szum na
f{0..300} był silny, to przeskakiwało na szum zamiast być w oknie
f{500–1200}".

**Dlaczego to jest gorsze, niż wygląda.** Fałszywy zaczep nie daje gorszego
odczytu, tylko przestraja cały front-end w bok — czyli działa jak celowe
rozstrojenie odbiornika. Sieć dostaje wtedy pasmo, w którym nadania nie
ma wcale. Objawem jest cisza, a przyczyna siedzi dwa moduły wcześniej.

**Dlaczego nasza pętla tego nie robi.** `dsp/tune.py`, wybór prążka:

```python
band = (f >= fmin) & (f <= fmax)                    # 400-1200 Hz
k = int(np.flatnonzero(band)[np.argmax(sp[band])])  # argmax TYLKO w paśmie
```

Szum poniżej 400 Hz **nie bierze udziału w wyborze**. Do tego trzy
rzeczy, które się składają:

- tło to **mediana mocy w paśmie**, więc silny szum podnosi także próg
  i zamiast fałszywego zaczepu daje BRAK zaczepu — a milczenie jest tu
  poprawnym zachowaniem, bo zmyślona częstotliwość przesunęłaby front-end;
- okno analizy to Hann (listki boczne −31 dB, opadające 18 dB/oktawę),
  więc przeciek z 50–100 Hz w okolice 400 Hz jest pomijalny;
- po zaczepieniu wybór zawęża się do `CAPTURE_HZ = 12` Hz wokół bieżącej
  estymaty, więc pojedynczy trzask nie porywa pętli.

**To było twierdzenie o kodzie, nie pomiar** — a takie twierdzenia
przestają być prawdziwe po refaktoryzacji, cicho. Dlatego 17.09 doszedł
`diag.py` **TEST 16**: nadanie 750 Hz plus dudnienie i przydźwięk
50/100/150/250 Hz o 20 dB silniejsze od niego. Sprawdza, że zaczep
zostaje w paśmie, że nadal wskazuje ton nadania, i że sam szum bez
sygnału nie produkuje zaczepu byle gdzie.

**Czego to NIE chroni.** Silne zakłócenie WEWNĄTRZ 400–1200 Hz — obca
stacja, nośna — może przejąć pętlę. To nie jest błąd, tylko ta sama
sytuacja, w której jest człowiek. Widać to na `radio1`: dwie stacje
55 Hz od siebie (790 i 845 Hz), mediana zaczepu 839 Hz — pętla wybrała
mocniejszą, nie tę, której chcemy.

**Odpowiedzią NIE jest zawężanie filtra.** Napisałem tak najpierw i było
to błędne: przy odstępie 55 Hz filtr 250 Hz przepuszcza obie stacje,
a zawężenie na tyle, żeby odciąć jedną, ucięłoby też drugą razem z jej
wstęgami kluczowania. Operator w tej sytuacji używa **filtra notch** —
wycina jedną konkretną częstotliwość i zostawia resztę pasma.

To są dwa różne narzędzia do dwóch różnych przypadków:

| | wąski filtr (250 Hz) | notch |
|---|---|---|
| co robi | przepuszcza okno wokół tonu | wycina jedną częstotliwość |
| kiedy | zakłócenie DALEKO w paśmie | zakłócenie BLISKO sygnału |
| na `radio1` | nie pomoże | to jest to |

---

### Zbiór trafiał do pamięci karty zamiast do RAM

**Objaw.** Trening padał po minucie na zbiorach większych niż 200 tys.
próbek. `FailedPreconditionError: Failed to allocate scratch buffer for
device 0`. Kosztowało dwie noce, bo szukałem w dwóch złych miejscach:
w OOM killerze systemu i w limicie 2 GB na protobuf.

**Przyczyna.** `tf.data.Dataset.from_tensor_slices` tworzy stałą na
urządzeniu DOMYŚLNYM, a gdy widziana jest karta, domyślnym jest GPU.
RTX 3050 ma 6144 MiB; 200 tys. próbek to 790 MB i mieściło się,
400 tys. to 1,6 GB i już nie.

**Rozwiązanie.** `with tf.device("/cpu:0")` wokół tworzenia zbioru.

**Zabezpieczenie na przyszłość.** `diag.py` TEST 15 uruchamia całą ścieżkę
danych treningowych na 64 próbkach, jako etap 2 nocy — czyli PRZED
godzinnym generowaniem zbioru. Wszystkie trzy błędy, które zabrały noce
10, 11 i 14 września, zostałyby przez niego złapane w kilka sekund.

---

## Sprawdzone i odrzucone

### Rekurencja (GRU) zamiast czystej sieci splotowej

Noc 9/10.09.2026, ten sam zbiór, te same 40 epok:

| | dokładność | czas | koperta tonu |
|---|---|---|---|
| `dpu` (splotowa) | 97,51% | 16 min | 650–850 Hz |
| `gru` (rekurencyjna) | 97,51% | 60 min | 650–750 Hz |

Identyczny wynik, czterokrotnie dłuższy trening, **węższa** koperta.

**ZAKRES TEGO WNIOSKU — dopisane 17.09.** Pomiar dotyczy głowicy
„jeden znak z okna" i **tylko jej**. W dekoderze sekwencyjnym (CTC)
rekurencja ma inne zadanie: modeluje ciąg, a nie pojedynczy znak, więc
z tego wyniku nie wolno wnioskować, że tam też nic nie da. Referencyjny
dekoder AG1LE używa dwóch warstw LSTM właśnie w tej roli.

Wniosek sprzętowy zostaje więc warunkowy: **przy obecnej architekturze
KV260 wystarcza**. Gdyby projekt poszedł w CTC z rekurencją, pytanie
o DPUCZDX8G wraca — choć CTC daje się zbudować także na samych splotach,
bez żadnej warstwy rekurencyjnej, więc i to nie jest przesądzone.

### Generowanie danych w locie

Odrzucone przed napisaniem kodu, na podstawie pomiaru: generowanie idzie
238 próbek/s na 11 procesach, a karta konsumuje 8000/s. Strumień
zagłodziłby GPU 34-krotnie — epoka trwałaby 13 minut zamiast 23 sekund.
Zamiast tego zbiór powstaje zawczasu w kilku częściach.

### Model AG1LE na TF1

Uruchomiony jako pierwszy punkt odniesienia tego projektu i **działał**.
Operator: „dekodowało 1:1, ale delikatny odjazd czegoś w bok, np. nośnej,
i po dekodowaniu". Odrzucony, bo był nieodporny na szum, zmianę tempa
i zmianę tonu. Buildy i materiały w `AG1LE/` na D888.

To „1:1 aż do odjazdu nośnej" jest ważniejsze niż samo odrzucenie —
opisuje tryb awarii, który **mamy do dziś**, tylko obudowany
zabezpieczeniem. Patrz niżej.

---

## Otwarte

### Znaki krótkie są gubione, długie czytane

Na `mic3` (nadane `CQ CQ CQ DE SQ2BVN SQ2BVN`) odczyt zawiera `Q2B`,
`Q2BB`, `Q2BNN` — brakuje `S` (`...`) i rozjeżdża się `N` (`-.`), a
czytane są `Q` (`--.-`), `2` (`..---`), `B` (`-...`).

Hipoteza o obcinaniu okna została **sprawdzona i odrzucona**: obcięcie
występuje w 5 przypadkach na 3000 (0,17%), bo etykietą jest środkowy
z trzech znaków, a nadanie wstawiane jest wyśrodkowane. Przyczyna leży
gdzie indziej i nie jest jeszcze ustalona.

### Dekoder sekwencyjny (CTC) — jest gotowy kod odniesienia

Wraca przy każdym problemie z powtórzeniami i przy gubieniu znaków, więc
warto mieć zebrane, co o nim wiadomo.

**Dlaczego to jest właściwa droga, a nie kolejne obejście.** CTC uczy
sieć etykietować CAŁY ciąg zamiast wybierać jeden znak z okna. Scalanie
powtórzeń jest jego wbudowaną własnością, nie dodatkiem — w TF1 widać to
wprost w parametrze `ctc_merge_repeated=True`. Czyli `CCQQQCCCC`, lutowe
`last_char` i wrześniowy próg 0,35 s to trzy objawy jednego braku.

**Kod odniesienia leży na D888:** `AG1LE/LSTM_morse-master.zip`,
plik `MorseDecoder.py`. Działający CTC na TF1:

```python
tf.compat.v1.nn.ctc_loss(labels=..., inputs=self.ctcIn3dTBC,
                         sequence_length=..., ctc_merge_repeated=True)
tf.nn.ctc_greedy_decoder(inputs=self.ctcIn3dTBC, sequence_length=...)
```

**Architektura (klasyczny CRNN, ten sam co w rozpoznawaniu pisma):**

```
5 warstw splotowych   jadra 5,5,3,3,3   kanaly 1->32->64->128->128->256
pooling               (2,2)(2,2)(1,2)(1,2)(1,2)
                      czas /4, czestotliwosc /32
squeeze osi czestotliwosci  ->  sekwencja w czasie
2 warstwy LSTM po 256 jednostek
CTC
```

Warto zauważyć, jak dobrany jest pooling: częstotliwość jest zgniatana
32-krotnie, a czas tylko 4-krotnie. Oś czasu musi zostać, bo to po niej
biegnie sekwencja — i to jest różnica wobec naszej obecnej sieci, która
spłaszcza obie osie do jednego wektora.

**Czego ten kod NIE rozwiązuje.** Model AG1LE był uruchomiony w tym
projekcie jako pierwszy punkt odniesienia i działał, ale był nieodporny
na szum, zmianę tempa i zmianę tonu — czyli dokładnie na to, czemu służy
nasz model kanału w `dsp/radio.py`. Do wzięcia jest struktura wyjścia,
nie cały pomysł.

### Brak notcha — układ nie ma narzędzia, po które sięga operator

**Luka.** QRM jest w modelu kanału (`QRM_PROB = 0.35`, amplituda do 0,35,
czyli druga stacja bywa mocniejsza od naszej), więc sieć jest UCZONA to
znosić. Ale w całym łańcuchu DSP nie ma ani jednego notcha — sprawdzone,
słowo nie występuje w kodzie. Operator przy dwóch stacjach obok siebie
sięga po notch; nasz układ nie ma czym.

**Gdzie to by mieszkało.** W `dsp/tune.py`, obok przestrajania, bo tam
już liczone są widma bloków i tam jest znany ton chciany. Algorytm
oczywisty: po zaczepieniu na `f0` znaleźć pozostałe wąskie prążki
w paśmie o wystarczającym SNR i wytłumić je pasmowo-zaporowo
(`scipy.signal.iirnotch`; scipy jest już zależnością, `tune.retune`
używa `hilbert`).

**Trudność jest realna i warto ją nazwać przed pisaniem kodu.** Notch ma
skończoną szerokość, a kluczowanie CW rozmywa nośną: przy 20 WPM kropka
trwa 60 ms, więc wstęgi sięgają rzędu ±25 Hz. Przy odstępie 55 Hz notch
musi być węższy niż jakieś 30 Hz, żeby nie zjadać wstęg sygnału
chcianego. To jest do zmierzenia, nie do zgadnięcia — i mamy na czym:
`radio1` jako przypadek prawdziwy oraz `radio.add_qrm` jako źródło
przypadków syntetycznych o znanym odstępie.

**Czego to NIE zastąpi.** Notch usuwa nośną zakłócającą, ale nie pomoże,
gdy obie stacje są w tym samym miejscu widma albo gdy zakłócenie jest
szerokopasmowe (QRN, trzaski). Tam zostaje sieć i model kanału.

### Model zapamiętuje zbiór

`val_loss` ma minimum w epoce 13 i potem rośnie, przy dokładności
treningowej bliskiej 100%. Pięciokrotne zwiększenie zbioru (200 tys. ->
1 mln) zmniejszyło błąd o połowę, więc dźwignia nie jest wyczerpana.

### Więcej danych zwęża kopertę tonu

Nieoczekiwane i zmierzone: przy 650 Hz model na 200 tys. trafiał 40–82%,
model na 1 mln trafia 0–20%. Lepsze dopasowanie do rozkładu treningowego
(750 ± 80 Hz) oznacza ostrzejsze milczenie poza nim. To nie jest regres,
tylko wybrany kompromis selektywność–odporność — ale **zależność od
`dsp/tune.py` przez to wzrosła**.
