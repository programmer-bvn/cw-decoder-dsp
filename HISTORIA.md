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
Odrzucony, bo był nieodporny na szum, zmianę tempa i zmianę tonu.
Materiały w `AG1LE/` na D888.

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
