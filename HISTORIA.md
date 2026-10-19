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

**ZMIERZONY SKUTEK, 17.09.2026.** Scalanie działa i to wyraźnie:

| | w całości | w kolejności |
|---|---|---|
| odczyt surowy | 5/9 | 8/9 |
| po scaleniu 0,35 s | **7/9** | 8/9 |

Na `mic3` (nadane `CQ CQ CQ DE SQ2BVN SQ2BVN`) odczyt po scaleniu to:

```
REQCQZCQZQZDDE I Q2BVVN SQ2BVVN I SQ2BVN PSE
                        ^^^^^^^   ^^^^^^^
```

**Znak wywoławczy wychodzi w całości, dwa razy.** Dla porównania model
sprzed poprawek dawał `CCLQ2BVQ2BVQ2BVS` — bez `S` i bez `N`. To jest
pierwszy raz, kiedy ten projekt przeczytał znak z prawdziwego radia.

**Właściwa naprawa** to nadal odtworzenie lutowej logiki na obwiedni
kluczowania liczonej z audio przy tej samej rozdzielczości co ramki
(20 ms) — wtedy „cisza" znowu staje się dostępna. Albo dekoder CTC,
który znosi całą tę klasę problemu, bo etykietuje ciąg zamiast wybierać
znak z okna. Scalanie po czasie zostaje obejściem — ale obejściem
o zmierzonej wartości, więc jest czym uzasadnić przeniesienie go do
`wav2net`.

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

### „Znaki krótkie są gubione" — OBALONE pomiarem

**Obserwacja, od której się zaczęło.** Na `mic3` odczyt zawierał `Q2B`,
`Q2BB`, `Q2BNN` — brakowało `S` (`...`) i `N` (`-.`), a czytane były
`Q` (`--.-`), `2` (`..---`), `B` (`-...`). Wniosek narzucał się sam:
sieć gubi znaki krótkie.

**Dlaczego to była słaba przesłanka.** Obserwacja pochodziła z zupy
znaków na czterech nagraniach, z których tylko jedno ma pełną treść.
Nie było wiadomo, ile razy krótki znak w ogóle wystąpił.

**Pomiar, 17.09.2026** — `tools/koperta.py`, 3000 próbek Z ETYKIETĄ,
pogrupowane po długości kodu:

| elementów | próbek | trafień | znaki |
|---|---|---|---|
| 1 | 154 | 98,1% | E T |
| 2 | 321 | **99,7%** | A I M N |
| 3 | 678 | 99,1% | D G K O R S U W |
| 4 | 1028 | 98,6% | B C F H J L P Q V X Y Z |
| 5 | 819 | 98,5% | cyfry |

Krótkie (1–2 elementy) **99,2%**, długie (4–5) **98,6%**. Różnica
0,6 punktu procentowego — i to na korzyść KRÓTKICH.

**Hipoteza jest obalona.** Sieć nie ma z krótkimi znakami żadnego
problemu; jeśli już, czyta je odrobinę lepiej.

**Gdzie więc naprawdę giną.** Nie w sieci, tylko DALEJ — w dekodowaniu
przesuwanym oknem. Długi znak trwa dłużej, więc widzi go więcej okien,
zbiera więcej zdarzeń i wygrywa przy scalaniu. Krótki znak, otoczony
dłuższymi sąsiadami w oknie 2,56 s, przegrywa konkurencję o zdarzenie —
choć sam w sobie jest rozpoznawany bez trudu.

**Dlaczego to ważne.** Przekierowuje pracę. Gdyby hipoteza była prawdziwa,
trzeba by zmieniać architekturę albo rozkład treningowy. Skoro jest
fałszywa, cała poprawa leży w ETAPIE DEKODOWANIA — tam gdzie działa
scalanie powtórzeń i gdzie działałby CTC.

---

### Więcej danych: prawo potęgowe i jego koniec

**Zmierzone na trzech punktach:**

| zbiór | dokładność | błąd | `val_loss` min |
|---|---|---|---|
| 200 tys. | 97,51% | 2,49% | 0,1224 |
| 1 mln | 98,73% | 1,27% | 0,0637 |
| 1,6 mln | 98,95% | **1,05%** | 0,0512 |
| 4 mln (rotacja) | 99,34% | **0,66%** | 0,0288 |
| 10 mln (rotacja, 20 epok) | 99,46% | **0,54%** | 0,0211 |

Błąd spada jak `n^-0,41`, i wykładnik jest zgodny na obu krokach
(0,418 przy pięciokrotnym zwiększeniu, 0,405 przy 1,6-krotnym). Trzy
punkty to mało na prawo, ale zgodność jest uderzająca.

**Czwarty punkt, noc 06.10** (`out/RANO.txt`, `runs/dpu_4000000`, pierwszy
trening na raty). Prawo przewidywało 0,72%, wyszło **0,66%**. Wykładnik
między 1,6 a 4 mln to ≈ −0,51, więc krzywa nie wypłaszcza się, a nawet
lekko przyspiesza. Walidacja przy rotacji pochodzi z jednej części
(80 tys. próbek), stąd niepewność ±0,03 pkt. Na nagraniach: w kolejności
8/9 (było 6/9 przy 1 mln), w całości 5/9 (było 4/9).

**Piąty punkt, noc 08/09.10** (`runs/dpu_10000000`, 20 epok zamiast 40,
402 min treningu). Prawo przewidywało 0,41–0,45%, wyszło **0,54%**.
Wykładnik między 4 a 10 mln to ≈ −0,22 — krzywa się WYPŁASZCZA, po raz
pierwszy wyraźnie. Uczciwie porównując tę samą liczbę epok (20): 4 mln
99,32%, 10 mln 99,44%, czyli wykładnik ≈ −0,25 — wniosek ten sam.
Na nagraniach bez poprawy: w całości 5/9 (bez zmian), w kolejności 6/9
(było 8/9 — różnica to jedno nagranie, `mic3`, gdzie zgubione zostały S
w SQ2BVN; przy 9 słowach to w granicach szumu, ale poprawy na pewno nie
ma). Po scaleniu 0,35 s: w całości 6/9.

**Wniosek.** Dźwignia „więcej danych" słabnie na syntetyku i nie dała nic
na antenie. To potwierdza wpis o krótkich znakach: straty są w etapie
dekodowania, nie w sieci. Kolejne 15 mln próbek (~1 doba liczenia) za
przewidywane ~0,1 pkt na syntetyku nie jest teraz warte nocy.

**Co z tego wynika dla kolejnych kroków:**

| cel | potrzeba danych | w pamięci (szczyt 2,5x) |
|---|---|---|
| błąd 0,75% | 3,6 mln | 37 GB |
| błąd 0,50% | 9,8 mln | 100 GB |
| błąd 0,25% | 53 mln | 543 GB |

~~**Dźwignia jest wyczerpana na tej maszynie.**~~ Było tak do 23.09:
przy limicie WSL 28 GB mieściło się najwyżej 2,1 mln próbek, czyli
około 0,94%. **Sufit pamięci został zdjęty** — patrz wpis niżej.
Zostaje drugi, nieusuwalny koszt: generowanie 9,8 mln próbek to
jedenaście godzin, a trening kolejnych siedem.

**I nie o to chodzi.** Błąd na syntetyku nie jest wąskim gardłem pracy
na antenie — dowodzi tego wpis wyżej o znakach krótkich. Tam, gdzie
projekt naprawdę traci, nie pomoże ani jedna próbka więcej.


---

### Sufit wielkości zbioru siedział w RAM-ie, nie w dysku

**Objaw.** Przy zbiorze powyżej ~2,1 mln próbek trening nie ruszał:
strażnik w `load_dataset` przerywał z rachunkiem pamięci, a bez strażnika
proces ginął od OOM killera bez żadnego komunikatu. Prawo potęgowe
(wpis wyżej) mówiło jednocześnie, że właśnie tam trzeba iść.

**Przyczyna.** Cały zbiór szedł do pamięci naraz, a w drodze do
`model.fit()` obrazy istnieją w około 2,5 kopiach. Nie było w tym błędu —
było założenie, że zbiór musi leżeć w całości. Założenie z czasów, gdy
dane mieściły się bez trudu.

**Dlaczego nie naprawiono tego wcześniej.** Bo nie było wiadomo, czy to
się opłaca. Wymiana części w trakcie epoki ma sens tylko wtedy, gdy
czytanie z dysku jest tanie wobec liczenia na karcie, a tego nie da się
zgadnąć — trzeba zmierzyć. Pomiar (22.09, `out/hdd.log`):

    dd if=czesci/morse_200000_00.npz of=/dev/null bs=1M
    862 MB w 2,99 s  =  289 MB/s

Część 818 MB wczytuje się w 2,8 s. Przy dziesięciu częściach to 28 s na
epokę wobec kilku minut samego liczenia — narzut rzędu procenta. **Gdyby
dane leżały na pendraku (~40 MB/s), ta sama rotacja kosztowałaby 3,5
minuty na epokę i nie byłaby warta zachodu.** Pomiar był warunkiem
decyzji, a nie ciekawostką — i to jest tu rzecz do zapamiętania.

**Co pomogło.** `ZbiorNaRaty` w `train_rtx.py`: części wczytywane po
kolei, w pamięci leży jedna (~820 MB) plus zbiór walidacyjny. Sufit
przenosi się z RAM-u na pojemność dysku, czyli z 2,1 mln na rząd
wielkości więcej.

Dwie decyzje w środku, obie nieoczywiste:

- **Walidacja pochodzi z jednej części i zostaje w pamięci na stałe.**
  Części różnią się ziarnem, nie rozkładem, więc jedna jest
  reprezentatywna. Gdyby walidacja zmieniała się razem z częścią, skoki
  krzywej `val_*` mówiłyby o podmianie danych, a nie o uczeniu —
  i `ReduceLROnPlateau` reagowałby na szum.
- **`.repeat()` w potoku jest konieczne**, choć epoka to dokładnie jeden
  przebieg. Keras przy podanym `steps_per_epoch` nie odtwarza iteratora
  na początku kolejnej epoki. Bez `.repeat()` pierwsza epoka policzyłaby
  się normalnie, a druga wywróciła na `ran out of data` — po kilkunastu
  minutach, w nocy, bez nikogo przy maszynie.

**Gdzie mieszka.** `ZbiorNaRaty` w `train_rtx.py`, flaga `--rotacja`.
W `noc.sh` dziewiąta pozycja `ROTACJA` (`auto` domyślnie: włącz, gdy
zbiór się nie mieści — zamiast przerywać noc). Sprawdza to TEST 15f
w `diag.py`, w tym najważniejsze: że próbki walidacyjne nie przeciekają
do treningu przez dwie epoki. Gdyby przeciekały, `val_accuracy` byłaby
zawyżona i cała seria pomiarów prowadziłaby na manowce, nie zgłaszając
niczego.

---

### Skrypt `hdd_repo.sh` nie umiał dokończyć własnej przerwanej roboty

**Objaw.** Na maszynie treningowej `git pull --ff-only` odpowiadał
`There is no tracking information for the current branch`, a `hdd_repo.sh`
uruchomiony po to, żeby to naprawić, mówił `To już jest repozytorium`
i wychodził. Druga droga powrotu wyników — ta niezależna od pendraka —
nie istniała, a komunikat brzmiał uspokajająco.

**Przyczyna.** Pierwsze uruchomienie (22.09) doszło do `git fetch`
i wywróciło się na `Could not resolve host: github.com` — maszyna nie
miała w tym momencie sieci. Zostawiło repozytorium z `origin`, ale bez
historii i bez śledzenia. Skrypt sprawdzał jedną rzecz: czy istnieje
katalog `.git`. Istniał, więc od tej chwili każde uruchomienie wychodziło
bez zrobienia czegokolwiek.

**Co pomogło.** Zamiast pytać „czy jest `.git`", skrypt bada trzy
warunki: jest `origin`, jest choć jeden commit, gałąź kogoś śledzi.
Przy braku któregokolwiek kontynuuje zamiast wychodzić. Przy okazji:
`git rev-parse --abbrev-ref HEAD` na gałęzi bez commitów wypisuje słowo
`HEAD` **i jednocześnie zwraca błąd**, przez co w logu pod `gałąź: HEAD`
stało osierocone `<brak>` z gałęzi awaryjnej.

Doszedł też warunek STOP: gdyby na HDD leżały własne commity spoza
`origin` (np. nocne z etapu 7), `reset --mixed` wypchnąłby je poza
historię — pliki zostałyby, ale zapis o tym, co i kiedy zmierzono,
przepadłby. Teraz skrypt się wtedy zatrzymuje i pokazuje, co to jest.

**Ogólniejsza nauka.** Skrypt, który wywraca się w połowie, musi umieć
rozpoznać własną niedokończoną robotę. Sprawdzanie „czy zrobione"
po jednym śladzie (istnieje katalog) zamiast po **skutku** (działa to,
o co chodziło) daje właśnie taki układ: cicho zepsuty i odporny na
powtórzenie.

**Gdzie mieszka.** Nigdzie: `hdd_repo.sh` usunięty 08.10 razem z całym
przenoszeniem pendrakiem (wpis niżej o nocy 28.09). Nauka zostaje.

### Start środowiska kosztował 70 s, bo venv leżał za drvfs

**Objaw.** Maszyna na 100% procesora, karta bezczynna, wiatraki milczą —
i tak przez minuty, zanim cokolwiek się policzy. Wyglądało to jak trening,
który nie umie użyć karty. 22.09 noc została przerwana Ctrl+C w trakcie
`import wrapt` **wewnątrz** importu TensorFlow.

**Zła droga, w którą poszedłem pierwszy.** Uznałem, że potok wejściowy
głodzi kartę, i zacząłem szukać w `tf.data`. Nie miało to poparcia
w danych: ostatni prawdziwy trening był 17.09, a przez kolejne dni
chodziło wyłącznie generowanie (z założenia procesorowe) i importy.

**Druga zła droga: zmierzyłem nie to.** `dd` dał 115 MB/s dla kopii
i 1,7 GB/s dla odczytu z ext4. Ta druga liczba jest **powyżej fizycznego
sufitu SATA3** (~550 MB/s) — `drop_caches` czyści pamięć podręczną
Linuksa w środku, ale plik ext4 WSL-a to VHDX na dysku Windows i to host
go buforuje. A przede wszystkim: import biblioteki nie jest ograniczony
przepustowością, tylko **latencją na plik**, której sekwencyjny `dd`
w ogóle nie dotyka.

**Pomiar, który rozstrzygnął** (`srodowisko/pomiar_dysku.sh`, 24.09):

| | drvfs (`v9fs`) | ext4 (WSL) | stosunek |
|---|---|---|---|
| zapis drobnych plików | 270 plik/s | 3075 plik/s | 11,4× |
| odczyt drobnych plików | 174 plik/s | 2190 plik/s | **12,6×** |
| jeden duży plik | 257 MB/s | 4,6 GB/s (zawyżone) | — |
| **import TensorFlow** | **69,6 s** | — | — |

Różnica siedzi w latencji na plik, nie w przepustowości. TensorFlow to
kilkanaście tysięcy drobnych plików, a `/mnt/...` płaci narzut na każdym
otwarciu osobno.

**Co z tego wynika.** Przenosi się to, co składa się z tysięcy małych
plików (`venv_gpu`), a **nie** to, co jest kilkoma wielkimi (zbiór
treningowy — 257 MB/s w zupełności wystarcza rotacji części, która
potrzebuje ~3 s na część). Koszt ponosi się przy KAŻDYM etapie nocy,
bo każdy woła pythona od nowa.

**Gdzie mieszka.** `srodowisko/pomiar_dysku.sh`; etap środowiska
w `noc.sh` mierzy czas importu przy każdym uruchomieniu i ostrzega, gdy
venv jest pod `/mnt/` i start przekracza 20 s. Przeniesienie:
`export CW_VENV=$HOME/venv_gpu` i `./srodowisko/setup_gpu_env.sh`.

---

### Noc przepadła na kontroli drogi, której ta noc nie używała

**Objaw.** 24.09 `diag.py` dał 82/83 i noc się zatrzymała przed
treningiem. Jedyna nieudana kontrola dotyczyła zbioru **na raty**, przy
zbiorze 1,6 mln próbek — który mieści się w pamięci, więc rotacja i tak
nie miała być włączona.

**Przyczyna, podwójna.** Sama awaria siedziała w **teście**, nie
w mierzonym kodzie: po `_do_float` obraz ma dodatkowy wymiar kanału, więc
`img[:, 0, 0]` daje kształt `(partia, 1)` zamiast `(partia,)`, numery
próbek robią się listami jednoelementowymi i `set()` wywala
`TypeError: unhashable type: 'list'`. Rzeczy istotne przeszły: walidacja
nie przeciekała, a potok wydawał dane także w drugiej epoce.

Druga połowa przyczyny jest w `noc.sh`: każda nieudana kontrola
przerywała noc, niezależnie od tego, czy dotyczy drogi, którą ta noc
w ogóle pojedzie.

**Co pomogło.** Test poprawiony (`np.squeeze` przed indeksowaniem).
W `noc.sh` decyzja o rotacji przeniesiona PRZED `diag.py`, żeby noc
wiedziała, czego użyje, a nieudana kontrola zbioru na raty przy
wyłączonej rotacji daje ostrzeżenie zamiast przerwania.

**Czego NIE zmieniono i dlaczego.** Zasada „trening na niespójnym
łańcuchu to zmarnowana noc" zostaje twarda. Wyjątek jest wąski,
dopasowany po nazwie kontroli — gdy nazwa się zmieni, dopasowanie
przestanie działać i noc się zatrzyma. Pomyłka idzie więc w tę stronę,
w którą jest bezpieczniej.

**Gdzie mieszka.** TEST 15f w `diag.py`, obsługa awarii diagu w `noc.sh`.

---

### Pomiar przepustowości mierzył inny kod niż noc

**Objaw.** Noc 06.10, `out/noc_20261006_0004_przepustowosc.log`: „sam
model na karcie" 2593 próbek/s (98,7 ms/krok), „trening" 1275 próbek/s
(200,7 ms/krok). Ten sam trening tej samej nocy szedł 31 ms/krok, czyli
~8258 próbek/s. Narzędzie, które miało rozstrzygnąć, co wyznacza tempo,
pokazywało liczby 3–6 razy za niskie. Werdykt („karta") wyszedł dobry
przypadkiem, a ostrzeżenie „traci się na styku" było fałszywe.

**Przyczyna, podwójna.**
- Pomiar B liczyła własna pętla (`tf.function` + `GradientTape`). Keras 3
  w `model.fit()` kompiluje krok przez XLA (`jit_compile="auto"`), a ta
  pętla nie. Czyli pomiar INNEGO kodu niż ten, który trenuje.
- Pomiar C liczył całe `fit()` od pierwszego kroku, razem z budową grafu
  i kompilacją XLA (kilka sekund rozłożone na 120 kroków).

**Co pomogło.** B i C idą teraz przez to samo `model.fit()` co noc,
z rozgrzewką 10 kroków. Czas mierzy callback od końca ostatniego kroku
rozgrzewki do końca ostatniego kroku — w JEDNYM `fit()`, bo każde nowe
`fit()` od nowa napełnia bufor shuffle (50 tys. próbek). Callback czyta
`loss`, więc czeka na wykonanie kroku, a nie tylko na jego wysłanie.

**Zmierzone 08.10, RTX 3050, partia 256, dwa przebiegi:**

| | było (06.10) | jest | jest, powtórka |
|---|---|---|---|
| A. potok | 363 874/s | 340 717/s | 364 932/s |
| B. model | 2 593/s | **8 954/s** (28,6 ms) | 8 904/s (28,8 ms) |
| C. trening | 1 275/s | **8 741/s** (29,3 ms) | 8 669/s (29,5 ms) |

C zgadza się z nocą (31 ms/krok — reszta to pasek postępu i walidacja).
C ≈ B, a potok jest 40 razy szybszy: **wąskim gardłem jest karta**, teraz
już z pomiaru, nie przypadkiem.

**Ogólniejsza nauka.** Pomiar składnika ma używać tej samej ścieżki
wykonania co całość. „Mniej więcej to samo" (własna pętla zamiast
`fit()`) potrafi różnić się kompilatorem, a wtedy porównanie A–B–C nie
ma sensu.

**Gdzie mieszka.** `tools/przepustowosc.py` (`zmierz_fit`), etap
PRZEPUSTOWOSC w `noc.sh`.

---

### Noc 28.09 liczyła starym kodem i nic tego nie pokazało

**Objaw.** Noc 28.09 (`out/noc_20260928_2349.log`) stanęła na `diag.py`
82/83: `zbiór na raty`, `TypeError: unhashable type: 'list'`. To był
dokładnie błąd z nocy 24.09, poprawiony już wtedy (wpis wyżej: „Noc
przepadła na kontroli drogi, której ta noc nie używała"). Noc wykonała
kod sprzed poprawki, a żaden log nie mówił, jaki kod się wykonał.

**Przyczyna.** Kod żył w dwóch miejscach: powstawał na komputerze
w pracy, a na maszynę treningową jechał pendrakiem przez `na_hdd.bat`.
Na HDD powstał w międzyczasie katalog `.git` (`hdd_repo.sh`), a
`na_hdd.bat` na jego widok przestawał kopiować kod, żeby nie nadpisać
repozytorium. `git pull` tam też nie działał (wpis wyżej o `hdd_repo.sh`).
Kod na HDD zamarzł, a skrypt kopiujący kończył się bez błędu.

**Co pomogło, w dwóch krokach.**
1. 05–08.10, gałąź `przebudowa-wersji`: paszport na starcie nocy.
   Najpierw był to spis odcisków plików (`wersja.sh`, `WERSJA.txt`)
   sprawdzany twardo, bo na HDD nie było działającego gita.
2. 08.10 komputer w pracy odpadł i wszystko przeszło na maszynę
   treningową: jedno repozytorium git tam, gdzie liczy noc. Przyczyna
   zniknęła, bo nie ma już drugiej kopii kodu, która mogłaby się
   rozjechać. Spis odcisków zastąpił commit. `na_hdd.bat`, `z_hdd.bat`,
   `bvn*.bat`, `hdd_repo.sh`, `mount_pendrak.sh`, `porzadki.sh`,
   `wersja.sh` i `git bundle` zostały usunięte.

**Decyzja operatora, 08.10.** Kod niezgodny z commitem NIE zatrzymuje
nocy. Zapomniany commit nie może kosztować nocy. Zamiast tego pełna
różnica idzie do `out/noc_<stempel>_kod.diff`, więc wynik nadal da się
przypisać do dokładnego kodu.

**Ogólniejsza nauka.** Wynik, o którym nie wiadomo, jaki kod go policzył,
nie jest pomiarem. Wersja musi stać przy wyniku, nie tylko w logu: logi
giną, a model i zbiór są używane przez kolejne noce.

**Gdzie mieszka.** Paszport w `noc.sh` (commit, gałąź, `*_kod.diff`).
`train_rtx.py` `kod_wersja()` zapisuje wersję w każdej części zbioru
(pole `kod`) i w `state.json` (lista `kod`: od której epoki jaka wersja).
`noc.bat` robi `git push` z Windows, bo tylko tam są poświadczenia.

---

### Noc brała stary venv, choć nowy był gotowy

**Objaw.** Ta sama noc 28.09: `już aktywne — nie ruszam`, venv spod
`/mnt/c/...` i `import tensorflow: 61 s` przy każdym etapie. Szybki
`~/venv_gpu` (2–3 s, wpis wyżej o drvfs) był już założony.

**Przyczyna, podwójna.**
- `noc.sh` miał skrót: jeśli w powłoce działa już Python z TensorFlow,
  środowiska nie ruszał. W powłoce akurat był aktywny stary venv, więc
  wygrał bez śladu w logu.
- Rada „wpisz `export CW_VENV=...` do `.bashrc`" nie mogła zadziałać
  z `noc.bat`. Ten uruchamia `wsl.exe -- ./noc.sh` bez interaktywnej
  powłoki, a wtedy `~/.bashrc` w ogóle nie jest czytany. Przy ręcznym
  `./noc.sh` działało, więc wyglądało na poprawne.

**Co pomogło.** Środowisko wybiera ZAWSZE `srodowisko/rtx3050_setenv.sh`,
bez skrótu, a ścieżka nie zależy od zmiennej. Kolejność: `$CW_VENV`, potem
`~/venv_gpu`, potem venv projektu. Wygrywa pierwszy KOMPLETNY (ma
`bin/activate` i zainstalowany tensorflow), sprawdzany po katalogu, nie
importem. Poprzedni venv jest czyszczony z `PATH` i `LD_LIBRARY_PATH`.
Paszport wypisuje wszystkich kandydatów i powód wyboru.

**Ogólniejsza nauka.** Skrót „już działa, nie ruszam" zamienia jawny wybór
w przypadkowy stan powłoki. Skrypt bezobsługowy ma za każdym razem ustalać
środowisko tą samą regułą i mówić w logu, co wybrał.

**Gdzie mieszka.** `srodowisko/rtx3050_setenv.sh`, etap 0 i paszport
w `noc.sh`.

---

### Dekoder fcn: powtórzenia znikły u źródła — ale syntetyk i radio się rozjechały

**Objaw, który to otworzyło.** Odczyty typu `CCQQQCCCC` (wpis „Powtórzenia
tego samego znaku") i gubienie krótkich znaków przy dekodowaniu
przesuwanym oknem (wpis „Znaki krótkie są gubione" — OBALONE). Oba
wskazywały na etap dekodowania, nie na sieć.

**Co zrobiono, 09.10.** Architektura `fcn`: sam splot, wyjście = znak na
każdy krok 40 ms (64 kroki na okno), uczona na etykietach na ramkę (`yf`,
z konstrukcji generatora — bez CTC). Odczyt: cisza dzieli odcinki, znak
odcinka to głos całego odcinka (`dekoduj_kroki`), nagranie zszywane ze
środków okien. Miara porównywalna z `dpu`: „środkowy znak".

**Zmierzone:**

| | dpu 10 mln | fcn próba (0,4 mln, 5 ep.) | fcn 10 mln (12 ep.) |
|---|---|---|---|
| środkowy znak (syntetyk) | 99,46% | 98,19% | **99,69%** |
| mic2 + mic3 (9 słów) | 5/9 | **9/9** | 7/9 |
| nagrania z USB IC-7300 (15 słów) | — | 1/15 | 1/15 |
| koperta tonu (≥ 50%) | 650–850 Hz | 650–750 Hz | **750 Hz** |
| krok treningu | 31 ms | 66 ms | 62 ms |

Powtórzenia zniknęły: scalanie 0,35 s nie zmienia już odczytu (różnica
0 słów). Na syntetyku fcn 10 mln ma błąd 0,31% wobec 0,54% dpu.

**Ale.** Dłuższy trening na większym syntetyku POGORSZYŁ odczyt
prawdziwych nagrań z mikrofonu (9/9 → 7/9: zgubione oba SQ2BVN na mic3)
i ZWĘZIŁ kopertę tonu do jednej kolumny 750 Hz. To ten sam wzór co we
wpisie „Więcej danych zwęża kopertę tonu": lepsze dopasowanie do
rozkładu generatora = ostrzejsze milczenie poza nim. 9 słów to mało
(różnica 2 słów), więc to obserwacja, nie prawo — ale kierunek jest
zgodny z wcześniejszym.

**Nagrania z pasma przez USB (IC-7300, 09.10) czytają oba modele fcn
źle**: z 15 słów tylko LA3WAA w całości. Przyczyna zmierzona na
wodospadzie: jasny pas szumu filtra p.cz., AGC w przerwach, trzaski
kluczowania — czego generator nie robi (NOTATKI, 09.10 wieczór).

**Wniosek.** Wąskim gardłem przestał być dekoder, a stał się GENERATOR:
dalsze trenowanie na tym samym syntetyku poprawia syntetyk i nic albo
gorzej na antenie. Następny krok to prawdziwe tło w danych (bank szumu
`probki/szum/`), a test to nagrania z pasma, nie walidacja.

**Gdzie mieszka.** `--arch fcn` w `train_rtx.py`, `dsp/sekwencja.py`,
`tools/nagrania.py` (sam rozpoznaje fcn), `diag.py` TEST 18,
`noc.sh` 10. pozycja = arch.

---

### Prawdziwe tło w generatorze — pierwsza noc (10.10)

**Co zrobiono.** Połowa klipów dostaje tło z `probki/szum` (pusty kanał
IC-7300 przez USB, 80 i 40 m, filtry 200–1250 Hz): pas szumu filtra przy
tonie stacji, ARW zawsze, jasność pasa 0–17 dB, stacja przez SNR −3…20 dB
(TEST 19). Noc: 4 mln próbek w `czesci_rt/`, fcn, 12 epok
(`runs/fcn_4000000_czesci_rt`).

**Zmierzone** — ten sam komplet nagrań dla trzech modeli fcn (w całości /
w kolejności):

| | fcn próba | fcn 10 mln | **fcn 4 mln + prawdziwe tło** |
|---|---|---|---|
| mic2 + mic3 (9 słów) | 9/9 · 9/9 | 7/9 · 7/9 | 7/9 · 8/9 |
| SN0FMT ×3 (12 słów) | 0 · 4 | 0 · 4 | **6 · 8** |
| F5OHS, DP75HSC, LA3WAA, LA4CIR, OK1FPS (5) | 2 · 2 | 2 · 2 | 0 · **4** |
| **radio razem (17 słów)** | 2 · 6 | 2 · 6 | **6 · 12** |

Na odczytach widać różnicę jakościową: znacznie mniej ciągów `0` (pas
szumu przestał być kreską), SN0FMT czytany jako
`…QSTDEEN0FMQ…SN000FMT…TEST`, w LA4CIR pojawiają się kawałki tekstu QSO
(`NAME`, `CALL`).

**Uczciwie o tym, czego to NIE dowodzi.**
- Prawie cały zysk „w całości" jest na SN0FMT — a bank szumu nagrano tego
  samego wieczoru, na tej samej częstotliwości (3565 kHz). To przeciek
  warunków (nie samych próbek) i wynik jest tam optymistyczny.
- Na pozostałych pięciu stacjach „w całości" spadło z 2/5 do 0/5
  (DP75HSC i LA3WAA, czytane wcześniej w całości, teraz tylko we
  fragmentach), a „w kolejności" wzrosło z 2/5 do 4/5. Wynik mieszany.
- Nowy błąd: podwojone litery (`1FFPSS1FFP` przy OK1FPS) — odczyt dzieli
  jeden znak na dwa odcinki. Hipoteza: ARW i pas szumu robią krótkie
  „dziury" w środku elementu, a `dekoduj_kroki` dzieli odcinki ciszą.
- Walidacja na syntetyku (97,77% środkowego znaku) nie jest porównywalna
  z wcześniejszymi — połowa zbioru ma teraz trudne, prawdziwe tło.

**Wniosek.** Kierunek jest dobry (radio „w kolejności" 6 → 12 z 17,
koniec ciągów `0`), ale bank szumu jest za mały i za jednorodny (jeden
wieczór, dwa pasma), a test częściowo z tych samych warunków. Następne:
nagrania szumu i stacji z innych pasm i pór (20 m), i test tylko na
stacjach spoza warunków banku.

**Gdzie mieszka.** `dsp/radio.py` i `train_rtx.py` (`bank_szumu`,
`szum_prawdziwy`, `receive`), stałe `REAL_NOISE_*` w `dsp/config.py`,
TEST 19 w `diag.py`, bank `probki/szum/`.

---

### Czy prosta baza szumów pomaga — doświadczenie kontrolne (11.10)

**Pytanie operatora:** czy prosta baza szumów „lewaruje" dekodowanie —
jeśli słabo, szkoda na nią czasu na tym etapie.

**Układ.** Dwa modele różniące się TYLKO tłem: fcn, 4 mln próbek, 12 epok,
ten sam kod odczytu (ze sklejaniem liter).
- `runs/fcn_4000000_czesci_4m` — stare części, szum syntetyczny
  (`czesci_4m/` = dowiązania twarde do `czesci/..._00-19`)
- `runs/fcn_4000000_czesci_rt` — połowa klipów z bankiem `probki/szum`

**Zmierzone** (w całości / w kolejności):

| grupa nagrań | bez banku | z bankiem |
|---|---|---|
| stacje spoza warunków banku: F5OHS, DP75HSC, LA4CIR, OK1FPS (4 słowa) | 1 / 2 | 0 / 2 |
| warunki jak w banku: SN0FMT ×3 (3565 kHz, ten sam wieczór), LA3WAA (7007, 10 min po szumie 7025) — 13 słów | 1 / 3 | **6 / 10** |
| mikrofon: mic2 + mic3 (9) | 7 / 8 | 7 / 8 |

**Wniosek.** Zysk z banku jest TYLKO tam, gdzie warunki nagrania pokrywają
się z warunkami nagrania szumu (to samo pasmo i wieczór). Na stacjach
niezależnych efektu nie widać — przy 4 słowach nie da się go ani
potwierdzić, ani wykluczyć. Bank z jednego wieczoru uczy sieć TAMTEGO
tła, nie tła w ogóle. Czy bank z wielu pasm i pór przeniósłby się na
inne warunki — hipoteza, niesprawdzona.

**Decyzja (11.10):** priorytet przechodzi na projekt Grawana_od_rana
(zasilanie domu). Dekoder wstrzymany w stanie: generator z bankiem
działa i jest przetestowany (diag 106/106), modele i dane zostają. Wrócić
z niezależnym zestawem testowym (nagrania z 20 m i innych pór).

---

### Poziom zamiast normalizacji, jedna miara S/N (12.10)

**Teza operatora.** Telegrafista nie podkręca szumu do poziomu sygnału.
Za mocno — tłumik; średnio — nic; słabo — filtr. Amplituda i S/N mają
być parametrem treningu, a nie czymś, co „normalizacja wyrówna".

**Pomiar 1 — `auto_gain` przy odczycie** (fcn_4000000_czesci_rt /
fcn_proba, 26 słów, w całości / w kolejności):

| | jak dziś | tylko tłumik | bez korekty |
|---|---|---|---|
| model z bankiem | 13/19 | 13/19 | **3/9** |
| model bez banku | 11/15 | 11/15 | 11/11 |

Na wszystkich 16 nagraniach korekta była ujemna (−2,8…−12,7 dB) —
`auto_gain` działał wyłącznie jako tłumik, „podciągania" przy odczycie
nie ma. Tłumik jest potrzebny; bez niego model z bankiem się sypie.

**Pomiar 2 — X-Ray w jednej skali** (`out/xray_sn.png`). Klipy z bankiem
mają tło jasne, często prawie jak znaki — to generator „podkręca" szum
(jasność pasa 0–17 dB). Nagrania: tło ciemne, znaki jasne. 28% klipów
ze stacją ma szczyt na granicy skali (24 dB), w klipach z bankiem ponad
połowa. S/N syntetyku i banku liczono różnie (0–4 kHz vs filtr p.cz.) —
nieporównywalne.

**Co zrobione.** Jedna miara jak w katalogu odbiornika dla CW: moc nośnej
do mocy szumu w 500 Hz wokół tonu (`C.SN_REF_BW_HZ`, `dsp/sn.py`).
Dwie drogi: ze składowych (generator) i z gotowego audio (nagranie:
ramki znak/przerwa progiem Otsu w wąskim paśmie). Walidacja: od 0 dB
w górę błąd ±0,6 dB (szum biały, różowy, bank; 12–30 WPM); poniżej 0 dB
zawyża do +5 dB. Skoki ARW w banku udawały stację (do +12 dB) — odcina
je warunek koncentracji przyrostu mocy w wąskim paśmie (≥ 0,8).
`tools/sn.py` mierzy nagrania, TEST 20 w `diag.py` pilnuje miary.

**Zmierzone S/N (mediana okien 5 s, 500 Hz):**

| | S/N |
|---|---|
| radio USB: LA4CIR / LA3WAA / DP75HSC / f200 | 4,8 / 7,2 / 10,8 / 10,0 dB |
| radio USB: SN0FMT ×3 / OK1FPS | 14,4–15,5 / 15,6 dB |
| Nagrywanie (7)–(9) (IC-7300, F5OHS…) | 2,8–7,0 dB |
| mikrofon mic1–3 | 26–29 dB |
| trening, syntetyk (po kanale) | mediana 20, 25–75%: 13–26 dB |
| trening, bank (po kanale) | mediana 6,8, 95%: 13,8 dB; 38% klipów < 0 dB lub bez pomiaru |

Klipy z bankiem kończą się na ~14 dB — stacje mocniejsze (SN0FMT,
OK1FPS) przy prawdziwym tle sieć widziała tylko na brzegu. Syntetyk
pokrywa je, ale z innym tłem.

**Przy okazji.** `probki/szum/szum_ic7300_usb_7025kHz_20261009_2135_230Hz_013-047s.wav`
ma ton o S/N 7–12 dB w 7 z 12 okien (koncentracja 0,99). Dekoder czyta
głównie „9"/„0" — nośna albo stacja, nie rozstrzygnięte. Do odsłuchu.

**Generator po zmianie (12.10, ten sam dzień).** `receive()` w
`dsp/radio.py` i `train_rtx.py` (zgodne bit w bit, TEST 12):

1. Stacja powstaje z amplitudą 1; amplitudę dostaje z S/N losowanego
   w jednej mierze, `SN_DB = (-3, 30)`, dla obu rodzajów tła. Usunięte:
   `NOISE_RMS_SPREAD`, `REAL_NOISE_SNR_DB`, `REAL_NOISE_TLO_DB`,
   `REAL_NOISE_RMS_MAX` (poziom szumu wynika teraz z S/N i gałki).
2. Gałka AF osobno: poziom tonu stacji w obrazie `POZIOM_DB = (2, 20)`,
   z `TON_DB_AMP1 = 28,2` (zmierzone). Szeroki szum przy małym S/N —
   ściszenie do `POZIOM_RMS_MAX`.
3. Tłumik jak przy odczycie: szczyt klipu nie wyżej niż `TLUMIK_DB = 17`
   — ta sama stała w `wav2net` i `mic2wav`. Bez niego silna druga stacja
   albo trzask przesterowywały 11,6% klipów.

diag 110/110 (TEST 19 przepisany: S/N z audio = wylosowane, 0% nasycenia,
szczyt ≤ 17 dB). Arkusz `out/xray_sn_po.png`: tło z banku ciemne, S/N
widać jako stopniowanie od „ledwo widać" do „czysto"; klipy z bankiem
przy 10–30 dB wyglądają jak nagrania. Syntetyk przy małym S/N ma szum na
całej szerokości obrazu (odbiornik bez wąskiego filtra) — nagrania tak
nie wyglądają, ale to możliwy przypadek. Poziom stacji w treningu:
mediana ~9 dB, nagrania po tłumiku ~17 dB — pokryte, nie wyśrodkowane.

**Ocena okiem operatora (12.10, wieczór)** — `tools/xray_ocena.py`,
przeglądarka pygame z polem „zły", partia 320 klipów + 32 okna PRAWDZIWYCH
nagrań wymieszane bez oznaczeń (kontrola oka). Wynik `out/ocena/partia1`:

| | złych |
|---|---|
| bank, wąski filtr p.cz. (200–500 Hz) | 84% |
| bank, szeroki filtr (1250 Hz) | 8% |
| syntetyk | 5% |
| **prawdziwe nagrania** | **31/32** |

Kryterium operatora: „czego nie widać na wodospadzie IC-7300". Wodospad
pokazuje widmo PRZED filtrem p.cz., gniazdo USB — ZA filtrem; czarne pola
nad i pod pasem filtra są więc prawdziwe (dowód: `out/ocena/zle_a_prawdziwe.png`).
Ocena trafiła za to w trzy realne błędy, które pod tym kryterium było widać:

1. **Gałka AF zaczepiona o stację** — przy słabej stacji szum szedł „po
   sam dach", jaśniejszy niż jakakolwiek stacja. Prawdziwe radio trzyma
   w ryzach SZUM. Teraz losowany jest poziom szumu w obrazie (`TLO_DB`,
   miara `dsp/sn.tlo_db`: pasma przy tonie, ramki przerw), stacja = szum
   + S/N, potem tłumik 17 dB. Nagrania radiowe tą miarą: 5% −15,0 | 50%
   −5,7 | 95% +3,7 dB; generator (bank, S/N < 16) po zmianie: −14,5 |
   −5,3 | +1,3 dB.
2. **QRM i trzaski dokładane za filtrem p.cz.** — świeciły w czarnym polu
   poza pasem („widma z innego świata"). Teraz przy tle z banku idą przez
   filtr odczytany z widma tego szumu (`radio.filtr_jak_szum`, −38 dB poza
   pasem). QRM ma S/N z `SN_DB` jak nasza stacja (było zawsze 16–33 dB,
   tłumik wtedy chował naszą stację).
3. **Rozbieg ARW** — detektor startował od zera, lewa krawędź świeciła;
   **karta** — szczyt w czasie 4–8× pełnej skali, ogranicznik robił szum
   na całym obrazie. Teraz detektor startuje z poziomu sygnału, a szczyt
   ≤ `KARTA_MAX` (operator ścisza).

Usunięte: `POZIOM_DB`, `TON_DB_AMP1`, `POZIOM_RMS_MAX`, `QRM_AMP`.
diag 113/113. Nowa partia do oceny: `out/ocena/partia2.npz`.

**Partia 2 po poprawkach** (`out/ocena/partia2`): operator przejrzał
22/22 strony — **0 złych na 352** (generator 0/320, nagrania 0/32).

### Znak ucięty krawędzią okna dostaje pełną etykietę (12.10)

**Objaw (operator).** Okno zaczyna się w połowie kreski albo kropki —
„niejednoznaczność i wprowadzanie sieci w błąd".

**Przyczyna.** `frontend.frame_labels` daje każdej widocznej ramce znaku
jego PEŁNĄ klasę, także gdy połowa znaku jest poza oknem. Pół kreski na
lewej krawędzi to równie dobrze końcówka „T", „M", „O" czy „0" — a sieć
fcn uczy się nazywać ją jedną z nich.

**Zmierzone** (`python -m tools.xray_ocena ucinki`): `czesci_rt` część 00,
pierwsze 20 000 okien — znak ucięty lewą krawędzią w 28%, prawą w 29%;
ucięty cel (znak etykiety) 21 razy. Partia 2 (nowy generator): 26% / 27%.

**Stan:** nierozwiązane. Do decyzji: ucięte ramki jako „nie licz"
(maska w funkcji straty) zamiast pełnej klasy. Przy odczycie nagrań
i tak liczą się środki okien (`kroki_nagrania`), ale trening wymusza
zgadywanie na krawędziach. Hipoteza: to szkodzi — niesprawdzona.

**Narzędzie.** `tools/xray_ocena.py przegladaj` otwiera partię, część
zbioru treningowego (pierwsze N bez rozpakowywania całości) i nagranie
WAV tą samą drogą co odczyt (przestrojenie + tłumik), w jednej skali.
Pod obrazkiem pasek etykiety (cel żółty, ucięty czerwony); powiększenie
z kursorem, suwakami, odczytem dB i profilami. Dwie kopie obok siebie:
`--kol 2 --pozycja x,y`. Barwy z `tools/xray_magma.npy` — bez matplotlib.

**Dalej:** decyzja o etykietach uciętych znaków, potem noc porównawcza
na warunkach kontroli z 10.10 (4 mln, 12 epok, fcn), zbiór `czesci_sn`.

### Sklejone znaki: przerwa międzyznakowa za krótka (13.10)

**Objaw (operator, partia 2, #6 „JVD").** J i V sklejone — „przerwa
między znakami była krótka, powinna być jak 3 t".

**Zmierzone.** Z zapisanych granic znaków: J–V 7,8 ramki = **1,8 t**
(13,8 WPM, kropka 4,35 ramki), V–D 2,6 t. Na 4000 klipach losowanych jak
w `receive()`: przerwa międzyznakowa min 1,56 t, 1,7% < 2 t, 10% < 2,4 t.
Rozjazdy się MNOŻĄ: 3 t × (1 − fist 0,25) × (1 − gap_jitter 0,4) ×
(1 − drift) ≈ 1,15 t w najgorszym razie. W czasie przerwa wewnątrz znaku
(max 1,39 t) jeszcze ich nie myliła, ale okno FFT 64 ms rozmywa krawędzie
— przy 14 WPM przerwa w obrazie jest krótsza o ~0,7 t i 1,8 t wygląda
jak 1 t.

**Co pomogło.** `GAP_MIN_T = 2,5` — przerwa międzyznakowa nie krótsza niż
2,5 t w lokalnym tempie operatora (`morse.keying_envelope` i kopia
w `train_rtx.py`); rozrzut w górę zostaje. Po zmianie: min 2,06 t
w jednostkach nominalnych (wolne błądzenie tempa), 0% < 2 t. Nie zmienia
kolejności losowań. TEST 21 w `diag.py`.

**Decyzja operatora (13.10, tego samego dnia): timing KEYERA.** „Na
kluczu sztorcowym mało kto nadaje, większość ma manipulator dwudźwigniowy
i keyer, znaki wychodzą równo; odległości powinny być nawet dłuższe, a jak
ktoś klei, to źle nadaje — QLF." Cel: „jeśli zacznie odbierać prawidłowo
dobrze nadane znaki, to będzie sukces i punkt wyjścia do następnych
kroków". Zmiany: `FIST = FIST_DRIFT = (0, 0)` (klucz ręczny zostaje
w kodzie, wyłączony), przerwa międzyznakowa tylko DŁUŻSZA — 3 t ×
(1 + U(0, 0,5)), `GAP_MIN_T = 3`. Zmierzone na 4000 klipach: przerwa
w znaku 1,00 t; między znakami min 3,00, mediana 3,28 t. Kontrola
widoczności w TEST 9 przy S/N < 3 dB wymaga > 5 dB kontrastu (było 10 —
próg z czasów samych czystych stacji). diag 114/114. Partia do oceny:
`out/ocena/partia4.npz`.

### Noc 14.10 (nowy generator) gorsza — sieć rozpoznaje stację po CHIRPIE

**Wynik nocy** (`runs/fcn_4000000_czesci_sn`, warunki kontroli z 10.10):
walidacja 92,29%, środkowy znak 97,24%, CER 14,08%; nagrania w całości /
w kolejności **6/26 / 14/26** (stary generator z bankiem: 13/26 / 19/26).
mic2 3/3 → 1/3, mic3 6/6 → 2/6. Odczyty pełne błędnych znaków.

**Odrzucone (pomiar, bez treningu):** „znaki w ciszy" — na pustym szumie
nowy model daje znak w 0–0,2% kroków przy tle od −28 do +4 dB, jak stary.
Na klipach z generatora (pełny realizm) nowy model 87,3%, stary 87,2% —
miara poprawna. Przy pełnym realizmie wysokie S/N nie szkodzi (25–30 dB:
92,4%).

**Znalezione (test „po jednym", S/N 15–30 dB, zjawiska wyłączone):**

| | nowy | stary (bank) |
|---|---|---|
| wszystko wyłączone | 41,8% | 59,1% |
| + chirp (0–25 Hz) | **86,4%** | 88,1% |
| + zapadanie amplitudy | 63,3% | 73,0% |
| + zanik / ARW / bank / przerwy / dryf | 41–69% | 59–67% |
| chirp stały 0 / 0,5 / 2 / 5 / 25 Hz | 42 / 50 / 70 / 88 / 90% | 59 / 66 / 81 / 90 / 91% |

Przy chirpie 0 model NIE WIDZI silnej, czystej stacji — odpowiada „cisza"
przez całe okno. **Przyczyna w kodzie:** `add_qrm` (druga stacja, klasa 0)
syntetyzuje BEZ chirpu i zapadania, a z rozrzutem ręki (`fist` 0–0,2);
stacja docelowa ma chirp U(0, 25 Hz) prawie zawsze, od 13.10 do tego timing
keyera. Sieć nauczyła się skrótu: „czysty ton bez chirpu (i z ręcznym
timingiem) = obca stacja = ignoruj". Nagrania z mikrofonu (keyer, bez
chirpu) i dobre nadajniki na paśmie wyglądają jak QRM. Stary generator
miał ten sam skrót słabiej (stacja też miała rozrzut ręki), stąd mniejsza
szkoda.

**Co zrobione (14.10).** `radio.losuj_nadajnik` (i kopia w `train_rtx.py`)
— JEDEN rozkład operatora i nadajnika dla stacji docelowej i QRM; QRM ma
też to samo tempo (`wpm_klipu`) i zanik. Zapadanie zasilacza (chirp + spadek
amplitudy, jedna przyczyna) tylko w `ZAPADANIE_PROB = 0,2` nadań. TEST 22
w `diag.py`; diag 116/116.

**Strażnik w nocy.** `tools/skroty.py` — test „po jednym" po każdym treningu
fcn, etap SKROTY w `noc.sh`, tabela w `RANO.txt`, ostrzeżenie, gdy czysta
stacja wypada o > 10 pkt gorzej niż pełny realizm. Na modelu z 14.10:
czysta 39,1%, zapadanie zawsze 89,7%, dłuższe przerwy 74,0% (drugi, słabszy
skrót: QRM z ręką miał krótsze przerwy), pełny realizm 87,4% — UWAGA.

**Wynik po poprawce — noc 14/15.10** (`runs/fcn_4000000_czesci_nad`,
warunki kontroli z 10.10: 4 mln, 12 epok, fcn; wznowiona po zaniku prądu):
walidacja 91,80%, środkowy znak 96,81%, CER 14,86%. SKRÓTY: czysta stacja
91,3%, pełny realizm 89,7% — skrótu nie widać (dzień wcześniej 39,1 / 87,4).

| nagrania (słowa w całości / w kolejności) | czesci_rt (10.10) | czesci_sn (14.10) | **czesci_nad** |
|---|---|---|---|
| RAZEM (26) | 13 / 19 | 6 / 14 | **16 / 24** |
| stacje spoza warunków banku: F5OHS, DP75HSC, LA4CIR, OK1FPS (4) | 0 / 2 | 0 / 1 | 1 / 3 |
| warunki banku: SN0FMT ×3, LA3WAA (13) | 6 / 9 | 2 / 9 | **8 / 13** |
| mikrofon: mic2, mic3 (9) | 7 / 8 | 3 / 4 | 7 / 8 |

Pierwszy raz OK1FPS w całości i DP75HSC w kolejności; SN0FMT w kolejności
12/12. Mikrofon wrócił do poziomu sprzed zmian. Zastrzeżenie bez zmian:
stacji niezależnych są 4 słowa — to wskazówka, nie dowód.

**Koperta tonu — sprostowanie (15.10).** Uwaga „750–750 Hz (dzień wcześniej
750–850), zawężenie do sprawdzenia" była błędna: siatka koperty ma kroki
co 100 Hz, a trening losuje ton 750 ± 80 Hz, więc 650 i 850 Hz leżą POZA
treningiem w obu nocach. „750–850" brało się z jednej komórki (13 WPM,
850 Hz: 50%) modelu ze skrótem chirpowym, który przy 750 Hz czytał tylko
35–88%. Pomiar co 25 Hz (`tools.koperta`, sygnał idealny, % trafień):

| | 675–825 Hz, 13–27 WPM | 10 WPM | 30 WPM |
|---|---|---|---|
| czesci_nad | **100 w każdej komórce** | 2–10 | 58–80 |
| czesci_rt (10.10) | 12–100, z dziurami | 68–92 | 25–68 |

Ton: nowy model czyta cały zakres treningu, równo. Słaby punkt to TEMPO
poza treningiem (20 ± 7 = 13–27 WPM): stary czytał 10 WPM, bo rozrzut ręki
i błądzenie tempa poszerzały mu zakres; po przejściu na timing keyera
(13.10) tego nie ma.

**Poszerzenie tempa (15.10).** `WPM = 22, WPM_JITTER = 10` → 12–32 WPM.
Dolna granica z okna: „0" z przerwami to 25 jednostek, okno 2,56 s → poniżej
11,7 WPM znak się nie mieści (TEST 9 w `diag.py`). Górna: przy 35 WPM kropka
(34 ms) jest krótsza niż krok wyjścia fcn (40 ms). Wolniejsze stacje — do
rozważenia ściśnięcie obrazu w czasie przy odczycie, bez zmiany treningu.
Koperta ma od 15.10 siatkę tempa 10-35 WPM i tonu gęsto przy granicach
treningu. Punkt odniesienia — model czesci_nad (trenowany 13-27 WPM) na
nowej siatce: 12-28 WPM i 675-825 Hz 95-100%, 32 WPM 0-5%, 10 WPM 0-15%.
diag 116/116 (12 WPM: „0" z przerwami 2,50 s w oknie 2,56 s).

### Noc tempa 12-32 WPM (15/16.10)

`runs/fcn_4000000_czesci_tempo` (warunki jak `czesci_nad`, zmienione tylko
tempo): walidacja 92,56%, środkowy znak 96,61%, CER 13,91%; SKRÓTY OK
(czysta 90,5 / pełny 90,2). Koperta (sygnał idealny): 12-32 WPM przy
675-825 Hz **100% w każdej komórce**, 10 i 35 WPM 32-60% (wcześniej 32 WPM
0-5%, 10 WPM 0-15%).

Nagrania 12/26 w całości, 21/26 w kolejności (czesci_nad: 16 / 24). Różnice
po jednym słowie: LA3WAA, OK1FPS, SN0FMT_b, mic2 (20 WPM — w zakresie obu
treningów). **Na syntetyku w 12-28 WPM oba modele identyczne** (pełny
realizm, 3000 tych samych klipów, trafne kroki):

| WPM | 12-15 | 15-18 | 18-22 | 22-25 | 25-28 | 28-32 | razem |
|---|---|---|---|---|---|---|---|
| czesci_nad | 66,8 | 82,2 | 91,0 | 96,0 | 94,6 | 62,0 | 81,8 |
| czesci_tempo | 67,3 | 81,1 | 90,6 | 95,7 | 94,8 | 95,1 | 86,7 |

Wniosek (hipoteza): 4 słowa różnicy na nagraniach to rozrzut między
treningami, nie skutek tempa — ROZRZUTU MIARY NAGRAŃ NIE ZNAMY; bez niego
każde porównanie nocy o kilka słów jest niepewne. Nowe: wolne tempo przy
pełnym realizmie słabe w obu modelach (12-15 WPM 67%, 15-18 WPM 82%), choć
na sygnale idealnym 100%. mic1 (12 WPM) i mic3 (15 WPM) są w tym zakresie.

### Pomiar tempa z bufora — pod normalizację tempa (16.10)

**Pomysł operatora.** Zamiast łatać skutki stałego okna (maska, podwajanie
okien — „cerowanie"): okno w JEDNOSTKACH, nie w sekundach. „Do bufora
nabierać ileś elementów, z tego szacować tempo i normalizować do tempa
standardowego, tak jak się normalizuje amplitudę." Krok 1 — sam pomiar
(`dsp/tempo.py`), bez zmiany treningu.

**Metoda.** Obwiednia ±50 Hz przy tonie, bloki 5 ms, Otsu → odcinki;
dopasowanie jednostki u i wagi b: znak = k·u + b (k 1,3), przerwa = k·u − b
(k 1,3,7), błąd względny. Pierwszy przebieg przy wygładzeniu 10/20/30/45 ms,
drugi z wygładzeniem u/2, wygrywa najmniejszy błąd poza granicą siatki.

**Walidacja na syntetyku** (bufor 8 s, 10-35 WPM, 6 buforów na komórkę):
w ±15% — czysto i ARW 100%, bank 95-100%, QRM 93-98%, pogrubione znaki
(waga +0,3 u) 100%, przy S/N 5, 10 i 20 dB. Po drodze: jeden przebieg
zawyżał wolne tempo przy S/N 5 dB o 40-60% (szum tnie długie znaki); stała
wygładzenia 10 ms dobijała na nagraniach z radia do granicy siatki (40 WPM).

**Nagrania** (bufor 8 s co 4 s): stabilnie — mic1 14,6, mic2 19,5, mic3
20,0 WPM (nazwa „15wpm" zawyżona; niezależny pomiar z 14.10: jednostka
60 ms = 20 WPM), SN0FMT ×3 15-17, radio1 20,9, f200 18,7. Szeroki rozrzut
(~20-38 WPM) w nagraniach z kilkoma stacjami (DP75HSC, LA3WAA, LA4CIR,
OK1FPS, Nagrywanie 6-9) — mediana ~27 WPM typowa dla zawodów; czy to
prawda nagrania, czy błąd pomiaru — nie rozstrzygnięte (brak znanego
tempa tych stacji). TEST 23 w `diag.py`; diag 117/117.

### Dwuetapowy dekoder: najpierw zbocza, potem znaki — punkt 1 (17.10)

**Pomysł operatora.** Rozdzielić odzyskiwanie zboczy kluczowania od
rozpoznawania znaków, jak przy kolorowaniu czarno-białych zdjęć. Pierwsza
sieć patrzy na widmo i odtwarza, gdzie BYŁO zbocze, zanim tor nadawczy,
filtr p.cz., ARW i szum je przesunęły i rozmyły („kość" na końcach znaków).
Druga sieć albo tabela Morse'a czyta już przebieg wyprostowany, przez
grzebień i normalizację tempa.

**Generator — rytm bliższy nagraniom** (hipoteza z nocy czesci_norm):
- WAGA kluczowania `WAGA_ZAKRES = (-0,1, 0,5)` u: znak dłuższy, przerwa
  po nim krótsza o tyle samo, takt bez zmian (`morse.keying_envelope`,
  losowana w `losuj_nadajnik` — ta sama dla stacji i QRM);
- przerwa między literami 3-10 t (`GAP_JITTER = (0, 2,33)`; było do 4,5 t)
  — nadawanie z klawiatury; klejenia nadal nie ma (QLF);
- zapas błędu tempa `TEMPO_BLAD` 0,15 → 0,25.
diag: kontrola wagi w TEST 21 (kropka 1,30 u, przerwa 0,70 u przy wadze 0,3);
120/120, TEST 12 bit w bit (46 stałych).

**Etykieta zboczy.** `radio.receive(..., z_kluczem=True)` oddaje prawdziwy
przebieg klucza (`synth_cw(return_env=True)`), wyrównany do klipu, oraz wagę
nadajnika. Bez wpływu na losowania, więc kopia w `train_rtx` go nie potrzebuje.
`dsp/zbocza.py`: widmo co 5 ms z oknem 32 ms (N_FFT 256), 512 ramek
(2,56 s), tempo PRAWDZIWE; klucz uśredniony ±2,5 ms wokół środka ramki.
Arkusz kontrolny (8 klipów, klucz na obrazie) — etykiety leżą na znakach.

**Odniesienie — próg klasyczny** (przekrój ±60 Hz przy tonie, Otsu, jak
grzebień; 400 klipów na przedział, tolerancja pół jednostki):

| S/N | błąd mediana / 90% | pominięte | fałszywe | narastające / opadające |
|---|---|---|---|---|
| -3..3 dB | 5,0 / 13,9 ms | 10% | 126% | -2,5 / +1,7 ms |
| 3..10 dB | 4,9 / 9,0 ms | 5% | 19% | -4,7 / +4,4 ms |
| 10..20 dB | 6,7 / 10,1 ms | 3,5% | 4,7% | -7,5 / +6,5 ms |
| 20..31 dB | 9,9 / 12,5 ms | 4,3% | 2,7% | -10,1 / +9,5 ms |

**„Kość" zmierzona.** Próg poszerza znak SYMETRYCZNIE, a o ile — zależy od
poziomu: silniejszy sygnał rozlany przez okno wcześniej przekracza próg.
Przy dobrym S/N błąd jest prawie czysto systematyczny (IQR 2 ms przy
przesunięciu ±10 ms), więc da się go oddać — zgodnie z intuicją operatora
o „wyprzedzaniu". Przy słabym sygnale błąd jest przypadkowy (fałszywe
zbocza z szumu), tam pomóc może tylko wiedza o kształcie zbocza.

**Narzędzie:** `tools/zbocza_trening.py` (generuj / trenuj / ocen).
Sieć DPU-bezpieczna: pasma zwijane pulowaniem (1,2), sploty po czasie
z dylatacją 1-32 (zasięg ±0,65 s), wyjście logit na ramkę.

**Pierwszy trening (17.10, `runs/zbocza_1`):** 200 tys. klipów (2 × 100 tys.,
~2 min generowania na część), 20 epok po ~9 min, val_loss 0,083. Ocena na
150 nowych klipach na przedział, te same klipy dla obu metod:

| S/N | klasyczna: mediana / 90% | **sieć: mediana / 90%** | fałszywe kl. / sieć | przesunięcie sieci |
|---|---|---|---|---|
| -3..3 dB | 4,8 / 13,9 ms | **2,6 / 9,9 ms** | 115% / 22% | -0,3 / +0,2 ms |
| 3..10 dB | 4,8 / 8,9 ms | **0,9 / 3,1 ms** | 21% / 5,0% | -0,1 / +0,1 ms |
| 10..20 dB | 6,9 / 10,1 ms | **0,4 / 1,3 ms** | 2,5% / 1,7% | -0,1 / +0,1 ms |
| 20..31 dB | 9,9 / 12,3 ms | **0,2 / 0,8 ms** | 3,1% / 5,9% | -0,1 / +0,1 ms |

Przesunięcie „kości" znikło całkowicie, błąd przy dobrym S/N spadł ~40×,
fałszywe zbocza przy słabym — 5×. Uwaga: to SYNTETYK z tego samego
generatora, co trening; na nagraniach prawdy nie znamy — sprawdzian to
grzebień na wyjściu sieci (pewność, odczyt tabelą) i ucho operatora.
Przy 20-31 dB fałszywych więcej niż przy 10-20 (5,9 vs 1,7%) — do zbadania
(QRM silniejsza od stacji? przesterowanie?).

**Na nagraniach — `tools/tester_zbocza`** (grzebień na wyjściu sieci kontra
grzebień na progu klasycznym, te same okna 8 s, ta sama bramka, tabela Morse'a
z nut; `out/zbocza/tester_1*.txt`). Sieć zboczy_1 na surowych nagraniach:
słowa 2/26 (klasyczna 13/26) — przegrana. Składanie okien sprawdzone
(pojedyncze okno = przebieg z zakładek, `out/zbocza/nagrania_p.png`).
**Przyczyna: obrazy czystsze niż w treningu.** LA3WAA (szum w tle) — zbocza
prawie idealne; mic1-3 i SN0FMT po wyrównaniu poziomu mają tło czarne
(mikrofon: tło -30...-20 dB, zmierzone 12.10; trening: od -12 dB, S/N do
30 dB). Sprawdzian: szum dosypany do S/N 25 dB → słowa sieć 11/26 (klasyczna
16/26), **mic3 6/6 (klasyczna 1/6), mic2 3/3, mic1 prawie bez błędu**
(„CQCQCQDESQ2BVNPSEK…", klasyczna „…CQOTO?0BVN…").
**SN0FMT nadal 0/4** — w KRÓTKICH przerwach zostaje energia przy tonie,
szersza od tonu (pasma 7-19 przy tonie 12-16), w długiej przerwie tło
czarne (`out/zbocza/sn0fmt_p.png`); sieć daje tam 0,5-0,75. Hipotezy (bez
pomiaru): twarde kluczowanie nadajnika (kliksy — generator miał zawsze
narastanie 5 ms), ARW, pogłos/wielodrogowość. Do oceny przez operatora.
**Poprawka (zboczy_2):** dla sieci zboczy szersze zakresy
(`zbocza.ZAKRESY_ZBOCZY`: S/N do 45 dB, tło od -32 dB, narastanie 2-8 ms —
IC-7300 ma 2/4/6/8 ms), przez `radio.receive(zakresy=...)`; bez nich
losowania jak wcześniej (TEST 12).

**Zboczy_2 (17.10, `runs/zbocza_2`, 200 tys., 20 epok, val_loss 0,079).**
Syntetyk — jak zboczy_1, plus nowy przedział 31-46 dB: sieć 0,1 / 0,5 ms
(klasyczna 10,7 / 13,9 ms, przesunięcie -11 / +10 ms). Nagrania SUROWE, bez
dosypywania szumu (`out/zbocza/tester_2.txt`):

| | próg klasyczny + grzebień | **sieć zboczy + grzebień** |
|---|---|---|
| słowa w całości / w kolejności | 13/26 / 16/26 | **21/26 / 21/26** |
| SN0FMT (trzy nagrania) | 10/12 | **11/12** |
| mic3 / mic2 / DP75HSC | 1/6 / 1/3 / 0/1 | **6/6 / 2/3 / 1/1** |
| pewność grzebienia (mediana) | 0,85-0,98 | 0,92-0,99 |

To SAMA TABELA Morse'a z nut — bez drugiej sieci. Dla porównania najlepszy
odczyt główną siecią: czesci_tempo z normalizacją przy odczycie 19/26 / 22/26.
mic1 (bez pliku .txt): „QCQCQBVNSQ2BVDS?2BVNPSEKCQCQCQDESQ2BVNPSEK…".
**Słabo:** radio1 (dwie stacje) — 0 okien przez bramkę (klasyczna 17,
ale czyta śmieci); OK1FPS 1 okno z 10; Nagrywanie (7), (9) — 0 okien.
Hipoteza SN0FMT (kliksy/narastanie) — po rozszerzeniu zakresów sieć czyta
SN0FMT poprawnie; które z trzech rozszerzeń zadziałało, nie rozdzielone.

**Regenerat „źle się słucha" (operator, 17.10).** `grzebien.regeneruj`
stawiał znaki na siatce, ale początki znaków zostawiał jak w nagraniu
i brał jedno tempo (medianę) na całość: przerwy między literami 4-10 u,
cisza tam, gdzie okno nie przeszło bramki (mic1: pierwsze 11 s). Znaki
same były dobre (zmierzone z obwiedni pliku). Nowy
`grzebien.regeneruj_wzorcowo`: każdy znak w jednostce swojego okna,
między literami 3 u, między słowami 7 u (przerwa >= 5 u), bez luk —
pliki `*_wzorcowo.wav`. mic1 zmierzony z pliku: 1/3/1/3/7 u ±0,1.

**Ton w ciszy po przestrojeniu (operator, 17.10, mic1 26,0-26,3 s).**
W nagraniu w tym miejscu jest tylko wolne falowanie podłogi (kilka Hz,
-39 dB względem znaku); `tune.retune` przesuwał CAŁE widmo o +53 Hz i robił
z niego ton ~53 Hz narastający i opadający. Przy dużym przesunięciu
(ton 400 -> 750 Hz) przydźwięk 50/100 Hz wszedłby w pasmo CW. Poprawka:
sygnał analityczny bez składowych < `RETUNE_FMIN_HZ` = 200 Hz
(`out/zbocza/mic1_26s_zrodlo.png`). Odczyt nagrań bez zmian (21/26
i 13/26 — ton był poza pasmem obrazu 400-1200 Hz), diag 120/120.

**Histereza i odrzucanie krótkich impulsów (pomysły operatora, 17.10).**
Deformacja w regeneracie mic1 ~16 s (nagranie 27-31 s, „BVN SQ" czytane
„BVD S?"): wyjście sieci po kropce N wisiało na 0,5-0,9 (doklejony element),
w kresce Q spadło do 0,35 (kreska rozcięta). `zbocza.odcinki_z_przebiegu`:
histereza (włącz 0,6 / wyłącz 0,3) i opcjonalnie sklejanie odcinków
krótszych niż 0,5 u (u z pomiaru tempa okna). `out/zbocza/tester_4.txt`:

| | jeden próg 0,5 | histereza | histereza + krótkie |
|---|---|---|---|
| słowa (w całości / w kolejności) | 21/26 21/26 | 21/26 21/26 | 21/26 21/26 |
| mic1 ~16 s | „BVDS?2BVN" | **„BVNSQ2BVN"** | „BVTSQ2BVN" + „2BKN" |
| okna SN0FMT (bramka) | 23 / 17 / 28 | 33 / 18 / 30 | 34 / 19 / 30 |
| radio1, Nagrywanie (8): bez stacji do czytania | cisza | prawie cisza | **śmieci** („EMRITITEEM…") |

Histereza: naprawia wskazane miejsce, więcej okien przechodzi bramkę,
pewność grzebienia w górę, regresji w słowach brak. Odrzucanie krótkich
impulsów w tej postaci szkodzi: wygładza szum w „znaki" (więcej okien
przechodzi bramkę tam, gdzie stacji nie ma) i zjada prawdziwe kropki
(N -> T). Słowa ze znanych plików nie rozróżniają wariantów — za mało
słów; różnice widać w tekście.

**Odrzucanie krótkich impulsów, wersja 2** (decyzja operatora: przerobić,
nie usuwać). Jednostka z GRZEBIENIA okna, czyszczenie tylko w oknach po
bramce (bramka decyduje na przebiegu z samą histerezą), potem grzebień
jeszcze raz (`zbocza.sklej_krotkie`, `tester_zbocza.droga(czysc=...)`;
`out/zbocza/tester_5.txt`). Śmieci na nagraniach bez stacji znikły
(Nagrywanie (8): cisza; radio1: 1 okno, jak sama histereza). mic1: „2BKN"
naprawione, ale „BVN SQ" nadal „BVT SQ" — tam, gdzie histereza sama dawała
dobrze. Hipoteza (bez pomiaru): histereza skraca przerwę między kreską
a kropką N poniżej 0,5 u i sklejanie łączy je w jeden element. Słowa 21/26
jak bez czyszczenia; pewność grzebienia o 0,01-0,04 wyżej. Zysku nie ma.

**Hipoteza sprawdzona — i odrzucanie USUNIĘTE** (operator: „zmierz najpierw
tę przerwę w N"). mic1, N od 27,34 s, u grzebienia 79 ms: próg 0,5 — kreska
250 ms, przerwa 25 ms + okruchy; histereza — kreska 285 ms (3,6 u), przerwa
35 ms (0,44 u), kropka 105 ms; sklejanie (< 40 ms) — jeden element 425 ms
= T. Przerwę skraca już SIEĆ: po kresce P schodzi powoli (0,54 0,54 0,31
0,32 0,60), do zera tylko na ~20 ms. Sama histereza czyta dobrze, bo
grzebień dociąga 3,6/0,44/1,3 u do 3/1/1. Sklejanie niszczy to, co grzebień
umie naprawić — usunięte (`zbocza.sklej_krotkie`, droga `siec_hk`).

**Ogon po elemencie — pomiar** (`tools/ogon.py`, `out/zbocza/ogon.txt`,
wykres `ogon.png`). Obwiednia przy tonie po zboczu opadającym, dB względem
znaku, mediana po zboczach; syntetyk tym samym torem schodzi do -79 dB
po 20 ms (dowolne narastanie 2-8 ms):

| ms po zboczu | 10 | 20 | 30 | 40 | 60 | 120 | podłoga |
|---|---|---|---|---|---|---|---|
| mic1 | -26 | -28 | -28 | -31 | -36 | -37 | -37 |
| mic1 PRZED narastającym (lustro) | -35 | -37 | -37 | -38 | -38 | -37 | |
| mic2 | -29 | -29 | -24 | -30 | -31 | -33 | -32 |
| SN0FMT | -24 | -25 | -28 | -28 | -31 | -25 | -20 |

Mikrofon: po zboczu ~8-10 dB nad podłogą przez ~40 ms, z garbem przy
15-30 ms; przed zboczem narastającym tego nie ma — ogon jest niesymetryczny,
czyli pogłos albo wybrzmiewanie (hipoteza: pomieszczenie, ~10 dB na 50 ms,
RT60 ~0,3 s). Radio (SN0FMT): po ~80 ms poziom ROŚNIE do -18 dB przy 200 ms
— hipoteza: ARW odzyskuje wzmocnienie i podciąga szum. Generator sieci zboczy
nie ma pogłosu; ARW ma (`apply_agc`), ale nie wiadomo, czy z takim czasem.

**Zboczy_3 — pogłos w generatorze** (`radio.dodaj_poglos`: sygnał + splot
z szumem gasnącym wykładniczo; `ZAKRESY_ZBOCZY["POGLOS"]`: 40% klipów,
poziom -35...-18 dB, stała 20-80 ms, opóźnienie 0-15 ms — zakres obejmuje
ogon mic1 z obu stron; domyślna droga bit w bit). 200 tys., 20 epok,
val_loss 0,080. Syntetyk jak zboczy_2 (0,1-3,0 ms mediana).
**mic1, N od 27,34 s:** kreska 240 ms (3,03 u), przerwa 75-80 ms (1,0 u),
kropka 80 ms (1,01 u); P sieci czyste 0/1 (zboczy_2: przerwa 25-35 ms,
ogon 0,3-0,6). Przyczyna usunięta u źródła — mic1 czytany dobrze już
z jednym progiem, bez histerezy. Nagrania (`out/zbocza/tester_6.txt`):
słowa 21/26 jak zboczy_2; więcej okien przez bramkę (SN0FMT jeden próg
28 / 18 / 29, było 23 / 17 / 28), pewność mic1 0,99. **Gorzej:** na nagraniach
bez czytelnej stacji histereza przepuszcza śmieci — radio1 3 okna
(„R?GTEEEEEEMNMRKE?", zboczy_2: 1 okno), Nagrywanie (7) 5 okien.

### Znaki spoza alfabetu: „=" i „?" w każdej łączności (18.10)

**Objaw.** Operator odsłuchał regenerat mic1 (zboczy_3) w całości: typowa
pierwsza relacja („SP2MSA DE SQ2BVN = GA DR OM = TKS FER UR CALL = UR RST IS
559 559 … HW? …"). Odczyt tabelą miał w miejscach „=" znak „?" (kod nieznany).
**Przyczyna.** `C.ALPHABET` — i 37 klas głównej sieci — to spacja, A-Z, 0-9.
Żadnych znaków przestankowych ani skrótów proceduralnych. **Zmierzone w nutach
mic1:** 11 x -...- („="), 2 x ..--.. („?" po HW, wg ITU). Odczyt teraz:
„SP2 MSA DE SQ2 BVN = = = GA DR OM = = TKS FER C ALL = … HW ?? SP2MSA DESQ2BVN K".
**Zrobione:** `grzebien.ZNAKI_DODATKOWE` (= ? / , . + KN SK, kody ITU-R
M.1677-1), błąd HH = 7+ kropek („więcej niż 6" — operator), kod nieznany
teraz „*". Opis `probki/mic1_manipulator_12wpm.txt` dostał treść (`nadane:`) —
mianownik słów w testerze rośnie, sumy sprzed 18.10 niewprost porównywalne.
**Otwarte:** główna sieć tych znaków nie zna (decyzja operatora: zmiana klas,
zbiór, trening, model pod KV260). Operator podał „? = --..--, ! = ..--.." —
w ITU --..-- to przecinek, ..--.. to „?", i ..--.. jest w nagraniu po HW.
Operator zapisał „+" po QRN i po QTH; w nagraniu w obu miejscach jest -...-.
**Decyzja operatora (18.10): dodać znaki do głównej sieci, kody wg ITU**
(„mam jakieś nawyki z wojska"). `C.ALPHABET` + `= ? / + ( ~` (KN = „(",
SK = „~"), dopisane na końcu — stare modele czytają się dalej. Pierwsza
wersja bez przecinka i kropki; operator: „mogą się pojawiać w tekstach" —
noc przerwana w generowaniu (przed pierwszą częścią), dodane: 45 klas.
Najdłuższe znaki „0" i „," (19 jedn.), okno bez zmian. Noc `czesci_znaki`.

### Ile znaków poprawnie — nasz dekoder vs klasyczny (19.10)

Pytanie operatora: „w ilu procentach udaje się zdekodować wszystkie znaki" —
miara ZNAKOWA na pełnym tekście (odległość edycyjna), nagrania o treści
znanej co do znaku. Dekoder klasyczny (`tools/klasyczny.py`, jak na
mikrokontrolerze): filtr ±100 Hz, prostowanie, wygładzenie 10 ms, komparator
z progiem adaptacyjnym (szczyt/tło) i histerezą 10%, jednostka adaptacyjna,
kropka < 2 u, przerwy 2 / 5 u, drzewko. `out/klasyczny_1.txt`.
Błędy na 100 nadanych znaków:

| nagranie | S/N | klasyczny | **nasz** |
|---|---|---|---|
| kluczownik IC-7300 (USB) | czyste | 8,0 | **0,4** |
| syntetyk + szum pasma | 20 / 10 / 6 dB | 0 / 0 / 9,7 | **0 / 0 / 0** |
| syntetyk + szum pasma | 3 / 0 dB | 79 / 349 | **0 / 0,7** |
| kluczownik + szum pasma | 6 / 3 dB | 55 / 250 | **0 / 0** |
| kluczownik + szum pasma | 0 / -3 dB | 464 / 685 | **2,1 / 13,4** |

(> 100 = więcej fałszywych znaków niż nadanych.) Nasz: 11/15 czystych nagrań
kluczownika całkiem bez błędu; część z 0,4% to wada nagrania (12:24 —
pierwsza kreska ucięta przed poprawką startu rejestratora: oba czytają „AN").
Klasyczny na prawdziwym podsłuchu IC-7300 gubi się już bez szumu (twarde
zbocza, wahania amplitudy). Ręka: treść nieznana co do znaku — nie liczone.

### Druga stacja obok (QRM) — najsłabszy punkt drogi (19.10)

**Test** (`tools/test_qrm.py`, `probki/qrm`): stacja docelowa z kluczownika
(bez anteny, tekst znany) na środku filtra szumu, druga stacja = inne
nagranie kluczownika (to samo tempo 17 WPM!) 30/60/120/200 Hz obok,
-6/0/+6/+12 dB względem docelowej, S/N docelowej 10 dB. Cała droga
(`tester_viterbi`): **65/257 słów** (tabela 38). Nawet -6 dB 60 Hz obok: 0/27.
**Dwie przyczyny** (`out/viterbi_qrm_1.txt`):
1. pętla tonu na obcej stacji w 9/16 — środek filtra liczony względem
   SZCZYTU 5. percentyla, a stacje nadające bez przerwy go podnosiły: błąd
   środka mediana 27,6 Hz. Odniesienie = 60. percentyl wzdłuż pasma
   („płaskowyż" szumu): 5,6 Hz; test_tonu 90,3 -> **92,2%**, druga stacja
   73 -> 77%; QRM: pętla na celu 9/16. Okno kandydatów 15 dB zamiast 10:
   QRM 12/16, ale test_tonu 80% (słabe przypadkowe blisko środka) — odrzucone.
2. sieć zboczy miesza obie stacje, nawet gdy pętla jest na celu (60 Hz,
   -6 dB, pętla 100%: 0/27). **Wąski filtr przed siecią** (±100 / ±60 Hz,
   `tester_viterbi --filtr`): 0/257 i 6/257 — sieć nie widziała takich
   obrazów; obejście DSP odrzucone. **Do treningu:** `ZAKRESY_ZBOCZY["QRM"]`
   (60% klipów, połowa 30-200 Hz obok, -10...+12 dB względem naszej;
   `radio.receive` + `add_qrm(ton=)`; domyślna droga bit w bit) -> zboczy_4.

**Zboczy_4** (`runs/zbocza_4`, 200 tys., 20 epok, val_loss 0,100 — zboczy_3:
0,080; `out/zbocza/bieg_4.log`, `out/viterbi_3_zb3.txt` vs `out/viterbi_4_zb4.txt`):

| zestaw (Viterbi) | zboczy_3 | zboczy_4 |
|---|---|---|
| QRM | 75/257 | 81/257 |
| pasmo + mikrofon | 56/62 | 54/62 |
| kluczownik | 282/282 | 281/282 |
| zaszumione syntetyczne | 469/470 | 459/470 |
| zaszumione prawdziwe | 425/456 | 414/456 |
| RAZEM | **1321/1541** | 1303/1541 |

Syntetyk: błąd zbocza nieco większy (np. 10-20 dB: 0,5 / 1,9 ms vs 0,4 / 1,5).
**Odrzucone** — domyślna zostaje zboczy_3; zakres QRM w `ZAKRESY_ZBOCZY`
zostawiony do następnych prób. Test QRM ma najtrudniejszy przypadek (obie
stacje: ten sam kluczownik, to samo tempo). Hipoteza (bez pomiaru): samych
danych za mało — potrzebna drobniejsza skala częstotliwości wokół stacji
(pasma mel ~48 Hz przy 750 Hz) albo jawny kanał „to jest nasza stacja".
**Diagnoza (zboczy_3, pętla na celu w 9/16 przypadkach)** — słowa tam, gdzie
pętla jest DOBRZE: 30 Hz +6 dB 6/17; 60 Hz +6 dB 3/17; **60 Hz -6 dB 0/27**;
120 Hz 0 / +6 / -6 dB 7/7, 11/27, 7/7; 200 Hz 0 / +6 / -6 dB 12/17, 7/7,
14/28. Nawet słabsza i odległa obca stacja zabiera słowa — głównym
ograniczeniem jest SIEĆ ZBOCZY (miesza obce elementy w przebieg), nie pętla.
Następny krok (do omówienia z operatorem): porównać P sieci z obcą stacją
i bez niej na tym samym nagraniu (gdzie wchodzą fałszywe elementy), potem
drobna skala częstotliwości wokół stacji / kanał „nasza stacja" w wejściu.

### Przeciwsygnał: obca stacja odtworzona i odjęta — QRM 75 -> 104 (19.10)

**Pomysł operatora** („generowanie przeciwsygnału"), obok maskowania ze
śledzeniem i notcha. Operator: najpierw przeciwsygnał, notch jako porównanie.

**Najpierw sufit.** Zestaw QRM odtworzony tym samym ziarnem ze SKŁADOWYMI
osobno (nasza + szum, obca). Bez obcej stacji: **145/257**, nie 257 —
`test_qrm` ucina naszą stację do 40 s, a „nadane:" ma cały tekst. 75/257
to więc 75 ze 145.

**Druga obserwacja: sieć zboczy myli się od obcej stacji wszędzie w swoim
wejściu**, nie tylko tuż obok: przy 120 i 200 Hz w paśmie ±50 Hz naszej
stacji SIR 15-45 dB, a słowa giną (120 Hz +12 dB: 4/27 przy suficie 11).
Obcą trzeba usunąć z całego pasma, łącznie z wstęgami jej zboczy.

**Porównanie** (częstotliwości z opisu nagrania, cała droga;
`out/viterbi_qrm_przeciwsygnal.txt`, przypadek po przypadku):

| sposób | słowa (sufit 145) |
|---|---|
| nic | 75 |
| notch ±10 / ±20 / ±30 Hz na obcej | 115 / 131 / 138 |
| przeciwsygnał, kształt z maski kluczowania | 111 |
| przeciwsygnał, rzut ±40 Hz na fazę obcej | 138 |
| **przeciwsygnał, rzut ±80 Hz, pasmo naszej wycięte z estymaty** | **142** |
| + wyuczony wzorzec zbocza obcej (kliknięcia) | bez zysku (SIR), odrzucone |

Notch ±30 Hz i przeciwsygnał różnią się tam, gdzie najtrudniej — obca
blisko i mocna: 30 Hz +6 / +12 dB: 10 i 4 vs 12 i 6; 60 Hz +12 dB: 5 vs 6.
Szerszy notch zjada naszą stację, przeciwsygnał odejmuje TYLKO obcą (jej
amplituda i faza uśredniane 0,4 s w chwilach wciśnięcia; nasza, odległa
o >= 25 Hz, wiruje w tym układzie i uśrednia się do zera).

**Wersja automatyczna** (obce z widma kontrastu kluczowania, nasza z pętli
tonu): 104. Straty TYLKO w 7/16 przypadkach, w których pętla wybrała obcą
stację (wtedy kasujemy naszą); w pozostałych 9 — tyle co z opisu.

**Pierwsza regresja** (bez progów): 1336/1541, QRM +29, ale reszta −14
słów — kasowała „stacje", których nie ma: stałe garby banku szumów (521,
607, 689 Hz) i szum pasma 20 m; w nagraniach ręcznych parę 420/1100 Hz
wokół 760 (nasz produkt). Dwa progi (`out/viterbi_kasuj_v1.txt` ->
`out/viterbi_kasuj.txt`):
- **spójność nośnej** (|Σz·e^{-j2πδt}| / Σ|z| w chwilach wciśnięcia,
  najlepsze δ w ±3 Hz) >= 0,5: prawdziwe obce w teście QRM 0,58-0,96,
  fałszywe 0,05-0,37;
- **korelacja obwiedni z naszą** <= 0,5: ręka 420/1100 Hz +0,68,
  prawdziwe obce -0,02...+0,33.

**Wynik: 1350/1541 (było 1321)** — QRM 75 -> 104, a poza QRM ŻADNE
nagranie nie zmieniło liczby słów (porównanie plik po pliku z
`out/viterbi_3_zb3.txt`). `radio1` (dwie stacje 55 Hz, mikrofon,
przesterowane, treść nieznana): stacja 845 Hz po odjęciu 789 Hz daje
pierwsze strzępy treści („UR 5NN", „RZAQSO"), bez — sam szum liter.

**Gdzie mieszka:** `dsp/przeciwsygnal.py` (`kasuj_obce`), wołane
w `tools/odczyt.przygotuj` między pętlą tonu a przestrojeniem;
`tester_viterbi` domyślnie z nim, `--bez-kasowania` — droga sprzed.

**Sprawdzone przy okazji i odrzucone: środek filtra z szerszego pasma**
(150-2500 Hz zamiast 400-1200; w QRM środek 800 zamiast 826-834 przy
filtrze szerszym niż pasmo analizy). QRM: pętla na celu 9 -> 11/16, ale
test_tonu 92,2 -> 91,0% — przy szerokich filtrach środek jest słabo
określony (zbocze szumu przesuwa go o 25-30 Hz; błąd > 20 Hz: 7 -> 33%
przypadków). Pewny środek da radio: ton CW (CW pitch) przez CI-V.

**Co zostaje:** wybór, KTÓRA stacja jest nasza (7/16 przypadków: obca
+12 dB wypycha naszą z okna `KL_REMIS_DB`, albo obie po równo od środka
filtra). Rozdzielanie już działa — to decyzja reguły, nie sygnału.
Obca stacja z dryfem > 3 Hz w nagraniu nie przejdzie progu spójności
(nie kasujemy — kierunek bezpieczny); śledzenie dryfu obcej — do zrobienia,
gdy zobaczymy to na paśmie.

### Grzebień z cepstrum + Viterbi — odczyt bez bramki (19.10) — PRZEŁOM

**Pomysł operatora:** jednostka z cepstrum („fizyka"), grzebień przykładany
do przebiegu, w komórkach prawdopodobieństwa zepsutego elementu, odczyt =
najbardziej prawdopodobny ciąg zgodny z gramatyką Morse'a („najkorzystniejsza
tabela prawdopodobieństw"; jak maska przesuwana po zapisie F2F w kasowniku);
bez bramki. `dsp/viterbi.py`: semi-Markow po odcinkach (kropka 1 u, kreska
3 u, przerwy 1 / ~3,5 / 7 u, cisza >= 10 u), wynik odcinka = 0,3 x suma
log P ramek (filtr dopasowany do długości) + kara log-normalna za długość;
siatka „sprężysta" (każdy odcinek własna długość). `tools/tester_viterbi.py`.
Wejście: P z sieci zboczy_3, pętla tonu = ton kluczowany.
**Pierwsza wersja (u tylko z cepstrum)** `out/viterbi_1.txt`: 777/828
(tabela 685); dwa nagrania przy 20 dB po 0 słów — cepstrum dało 110 ms
zamiast 67 ms. **Jednostka też jest parametrem** (`dekoduj_najlepiej`:
kandydatki z cepstrum i z grzebienia, wygrywa lepszy wynik Viterbiego na
ramkę), `out/viterbi_2.txt`:

| | tabela + bramka | **Viterbi** |
|---|---|---|
| pasmo + mikrofon (probki) | 53/62 | **56/62** |
| kluczownik | 281/282 | **282/282** |
| ręka | 14/14 | 14/14 |
| zaszumione 0-20 dB | 337/470 | **469/470** |
| w tym 0 dB / 3 dB | 0/94 / 55/94 | **93/94 / 94/94** |
| **RAZEM** | 685/828 | **821/828** |

Nowe na paśmie: F5OHS w Nagrywanie (6) (pierwszy raz kiedykolwiek), OK1FPS,
mic2. Spacje między słowami z odcinków PS/CI wychodzą same („CQ CQ CQ DE
SQ2BVN SQ2BVN PSE K"). Do zrobienia: gramatyka liter (tylko istniejące
kody, bez „*"), strojenie wag na osobnym zestawie, odczyt pliku -> tekst
całą nową drogą, praca na żywo.

**Zaszumione PRAWDZIWE nagrania** (`probki/zaszumione_real`: ręka CQ
i SP2KKB, kluczownik qso2 i qso5, szum banku 7025 i 3566 kHz FIL 500 Hz;
`tools/zaszum.py --katalog`, treść z pola „nadane:" opisu; zmierzone S/N
~1 dB niższe od zadanego), `out/viterbi_real_1.txt`:

| S/N | tabela + bramka | **Viterbi** | ręka | kluczownik |
|---|---|---|---|---|
| +6 dB | 114/114 | **114/114** | 28/28 | 86/86 |
| +3 dB | 2/114 | **114/114** | 28/28 | 86/86 |
| 0 dB | 0/114 | **109/114** | 28/28 | 81/86 |
| -3 dB | 0/114 | **88/114** | 28/28 | 60/86 |

**Viterbi po LITERACH** (litera = sztywny wzór 1/3/1 u, skala 0,9-1,1,
tylko istniejące kody) — sprawdzony 19.10 i USUNIĘTY: tekst prawie ten sam,
na kluczowniku gorszy początek („ANQ" zamiast „CQ"), wynik dopasowania na
ramkę gorszy (-0,020 vs -0,007) — sztywny wzór gorzej pasuje do zboczy niż
elastyczne odcinki; wybór po wyniku i tak brałby odcinki. Szybszy (2-3x).

**Strojenie** (`tools/viterbi_strojenie.py`, przebiegi raz -> siatka na
procesorze; `out/viterbi_strojenie.txt`): WAGA_EMISJI 0,15/0,3/0,6 x
KARA_DLUGOSCI 2/4/8 — od 1313 do 1321/1541, płasko; obecne (0,3; 4)
w czołówce. Wagi nie są wąskim gardłem — zostają.

Ręka liczona łagodnie (słowo raz, powtórzeń wiele). `tools/odczyt.py`:
plik -> tekst całą nową drogą (ton kluczowany, sieć zboczy, cepstrum +
grzebień, Viterbi).

### Cepstrum przebiegu kluczowania — drugi, niezależny pomiar jednostki (19.10)

**Pomysł operatora** (prędkościomierz mikrofalowy, 2008): cepstrum (widmo
z logarytmu widma) pokazuje rytm powtarzających się odstępów — „prążek na
jeden i na trzy". Liczone na P(klucz) z sieci zboczy_3, okna 8 s co 4 s
(okno Hanninga, FFT x4, log mocy, odwrotna FFT), wykres
`out/cepstrum_obwiedni.png`.
**Zmierzone:** ostre UJEMNE szpilki dokładnie przy 1 u, 3 u i 7 u (prostokąt
długości T ma zera widma co 1/T -> okresowe dołki w log widma -> ujemny
szczyt cepstrum przy T), słaby dodatni przy 4 u — na kluczowniku, ręce
i mikrofonie (mic1: 80/240 ms). Autokorelacja daje to samo słabiej.
Najgłębszy punkt sam: mediana dobra w 4 z 5 nagrań, okna często 2u/3u.
**Suma cepstrum przy u + 3u + 0,5·7u** (proporcje Morse'a), siatka 1 ms:

| nagranie | grzebień u | cepstrum u | okna zgodne ±5% |
|---|---|---|---|
| kluczownik qso1 / qso2 | 70 / 70 ms | 70 / 70 ms | 4/4, 24/24 |
| ręka SP2KKB / CQ (różne odstępy) | 70 / 70 ms | 70 / 70 ms | 34/34, 27/28 |
| mic1 (mikrofon, pogłos) | 80 ms | 80 ms | 63/63 |

Zastosowania (do zrobienia): fuzja z grzebieniem (zgodne -> pewność,
niezgodne -> okno podejrzane; zamiast części bramki S/N), punkt startowy
grzebienia, cepstrum AUDIO do echa/wielodrogowości (pogłos mic, ogon DX).

### Pętla tonu: „ton kluczowany" — zestaw z prawdą i nowa metoda (18/19.10)

**Zestaw** (`tools/test_tonu.py`, decyzja operatora: zamiast sumatora
rezystorowego — dodawanie nagrań programowo): stacja = nagranie kluczownika
IC-7300MK2 przesunięte na ton w środku filtra (±30 Hz, tak dostraja
operator), szum naturalny (bank + szum własny odbiornika; pas -6 dB zmierzony;
stacja w pasie), 0-5 s szumu przed nadaniem, obok: nic / druga stacja /
stała nośna 60-200 Hz dalej, -10 / 0 / +6 dB. 150 przypadków; plik banku
z wąskim prążkiem (7025 kHz FIL 230 Hz: 18,5 dB przy 770 Hz) wykluczony.
Miara: udział czasu nadania z pętlą w ±10 Hz od tonu stacji.

**Metoda** (`tune.track_tone_kluczowany`): w oknie 6 s dla każdej
częstotliwości kontrast kluczowania (90. - 10. percentyl mocy) — stacja CW
kluczowana, szum skacze, nośna/przydźwięk stałe. Kontrast >= 10 dB tylko
ODSIEWA; spośród kluczowanych (nie słabszych niż 10 dB od najgłośniejszego)
wygrywa bliższy środka filtra; bez przeskoku, gdy poprzedni wciąż kandyduje.
Po drodze (każde zmierzone na zestawie):

| wersja | na tonie | bez obok | stacja obok | nośna obok |
|---|---|---|---|---|
| obecna `track_tone` | 31,7% | 65,7% | 12,6% | 6,6% |
| kontrast wybiera | 48,8% | 59,2% | 42,7% | 41,4% |
| kontrast odsiewa, poziom wybiera | 62,5% | 99,7% | 33,0% | 44,0% |
| odniesienie mocy jedno, nie mediana bloku | 79,5% | 100% | 33,1% | 100% |
| bliższy środka filtra wygrywa | 78,4% | 100% | 31,8% | 97,8% |
| środek filtra z cichego percentyla (20.) | 83,9% | 98,4% | 51,7% | 97,8% |
| **środek z 5. percentyla** | **90,3%** | **98,5%** | **72,5%** | **97,8%** |

Kluczowe błędy po drodze: (1) kliknięcia twardych zboczy podsłuchu są
„kluczowane" na całym paśmie — kontrast nie może wybierać, tylko odsiewać;
(2) mediana pasma w bloku skacze 36 dB przy kluczowaniu — stała nośna
względem niej wyglądała na kluczowaną (kontrast 30 dB); (3) środek filtra
ze średniego widma = najsilniejsza stacja; z 20. percentyla przy szybkim
kluczowaniu nadal obca stacja (blok 200 ms prawie zawsze łapie element).
Docelowo przy pracy z radiem środek = ton podsłuchu (CW pitch) z CI-V.
**Na prawdziwych nagraniach:** kluczownik i ręka — ten sam ton co obecna
(760 Hz), EEEE 19:09: obecna 794 Hz (źle), nowa 760; różne tony na DP75HSC,
LA3WAA, LA4CIR, Nagrywanie (6), (7), radio1 — rozstrzyga odczyt (tester
z obiema pętlami). **Tester na prawdziwych nagraniach:** słowa sieci
zboczy identyczne w KAŻDYM nagraniu (pasmo + mikrofon 53/62, ręka 14/14),
więcej okien przez bramkę (mic1 62 -> 64, mic2 43 -> 45, CQ ręka 24 -> 26,
EEEE 3 -> 11). **Od 19.10 `tune.track_tone` = ton kluczowany**; pętla fazowa
usunięta (opis w nagłówku tune.py i tutaj). diag 120/120. Na nieprzyciętych
oryginałach kluczownika (3-4 s szumu przed nadaniem, 6 m: wcześniej 0 okien)
poza tonem 0-1% czasu -> **obejście `nagraj_kluczownik.przytnij` usunięte**
(nagrania już przycięte zostają, pole „przycięte" w ich opisach).

### Drugi etap — sieć na wyjściu sieci zboczy, pierwszy trening (18/19.10)

`dsp/etap2.py`, `tools/etap2_trening.py`. Wejście: P(klucz) z sieci zboczy_3
przeskalowane do 4 próbek na jednostkę (u z grzebienia; w treningu prawdziwe
u x (1 ± 10%)), okno 64 u; wyjście co 1 u: klasa znaku (45) albo przerwa.
Zbiór: klipy generatora sieci zboczy przepuszczone przez PRAWDZIWĄ sieć zboczy
(2 okna na klip), 400 tys., ~23 min na 200 tys.; 20 epok po ~71 s;
dokładność kroków: walidacja 89,2% (na 20 tys. próbek: 87% — zbiór x20 prawie
nic). Ocena: wszystkie nagrania z treścią (358 słów), te same nuty co tabela:

| odczyt | słowa |
|---|---|
| tabela (grzebień) | 348/358 |
| drugi etap, sklejanie „każda zmiana klasy = znak" | 347/358 |
| **drugi etap, sklejanie odcinkami** | **350/358** |

Sklejanie: odcinek kroków niezerowych = jeden znak (klasa większościowa),
przerwa krótsza niż 2 kroki go nie dzieli. Naprawiło qso5 („579" -> „5709":
9 = ----. dostawało na początku kroki „0"). Lepiej od tabeli: OK1FPS, mic2,
qso4; gorzej: mic1 (31 vs 32: „KK", „GG", „7" na początku — rozcięcia
przerwą >= 2 kroków; pogłos mikrofonu?). Różnica 350 vs 348 mieści się
w rozrzucie — na razie REMIS. Ograniczenie: dokładność kroków 89% to głównie
granice znaków (który krok to jeszcze znak), nie rozpoznanie.

### Drugi etap — główna sieć znaków na regeneracie: gorzej (18.10, wieczór)

15 nagrań kluczownika (20 m, 6 m, bez anteny; 282 słowa), słowa w całości:

| droga | czesci_norm (37 znaków) | czesci_znaki (45 znaków) |
|---|---|---|
| główna sieć na SUROWYM nagraniu (normalizacja tempa przy odczycie) | 238 | **256** |
| główna sieć na regeneracie wzorcowym z nut sieci zboczy (+ szum 25 dB) | 140 | 161 |
| **sieć zboczy + grzebień + tabela** | — | **281** |

Nowa sieć znaków na nagraniach lepsza od czesci_norm (256 vs 238), choć na
syntetyku gorsza (CER 14,2 vs 5,3%). Regenerat jako wejście głównej sieci
— spadek o ~40% (jak 17.10: 4/13): regenerat (czysty sinus, idealne
narastanie, przerwy 3/7 u) jest poza tym, co główna sieć widziała w treningu.
**Wniosek:** drugi etap nie przez regenerat i główną sieć, tylko sieć
uczona na WYJŚCIU sieci zboczy (przebieg P(klucz), tempo znormalizowane),
z błędami takimi, jakie sieć zboczy naprawdę robi. Do tego czasu najlepszy
odczyt: sieć zboczy + grzebień + tabela. Skrypt: scratch `drugi_etap.py`.

### Lepsze próbki testowe: kluczownik IC-7300MK2 + naturalny szum pasma (18.10)

**Dlaczego.** 26 słów i rozrzut ±2 to za mało, żeby rozróżniać warianty
(histereza: różnice widać tylko w tekście). Operator: tekst znany co do znaku,
nadany automatem, potem z ręki z różnymi przerwami, „bez wychodzenia w eter".
**Radio** (źródła w rozmowie 18.10): podsłuch CW na karcie USB — SET >
Connectors > USB AF/IF Output > AF Beep/Speech… Output = ON (RemoteTx,
IC-7300MK2; domyślnie OFF i podsłuchu na USB nie ma — zgodne z obserwacją
operatora). Tekst do kluczownika przez CI-V 0x17 (Hamlib `icom_send_morse`:
do 30 znaków w ramce, FA = bufor pełny, 0x17 0xFF = stop); break-in odczyt
0x16 0x47 (00 wył., 01 pół, 02 pełny). Przy BK-IN OFF kluczownik gra tylko
podsłuch (instrukcja IC-7300).
**Zrobione:** `probki/teksty/qso*.txt` (5 tekstów, wszystkie 44 znaki);
`tools/radio/kluczownik.ps1` (URUCHAMIA OPERATOR; odmawia przy break-in
innym niż 00, trybie innym niż CW, braku odpowiedzi; `-Sucho` = same
bezpieczniki; parsowanie odpowiedzi sprawdzone na podstawionym porcie);
`tools/zaszum.py` (czysty sygnał + szum z `probki/szum` przy zadanym S/N,
ton na środku pasa -6 dB szumu; `--syntetyk WPM` = sygnał kluczownika
z tekstu, dopóki nie ma nagrań). Pierwszy zestaw: 5 tekstów x S/N 20/10/6/3/0
dB, szum 7025 kHz FIL 500 Hz; S/N zmierzone = zadane ±0,5 dB (przy 0 dB
+1,1...+1,4 — miara zawyża przy słabym sygnale).

**Pierwsze nagrania z kluczownika (18.10, 17 WPM, podsłuch USB, 14,040 MHz
CW-R).** Po drodze:
- `mic2wav` ucinał nagranie (27 s -> 10,7 s): pętla kończyła się po czasie
  zegarowym, a bloki czekały w kolejce — przeliczanie co blok nie nadążało
  przy generowaniu zbioru na wszystkich rdzeniach. Teraz kolejka dopisywana
  po zamknięciu strumienia; `latency="high"` (były dziury 1-13 ms w znakach).
- Początek ucięty: karta oddaje dźwięk ~2-3 s po otwarciu strumienia —
  kluczownik rusza 3 s po „NAGRYWANIE TRWA".
- „VVV =" w jednym nagraniu nadawał operator RĘCZNIE — podsłuch klucza
  ręcznego też idzie na USB (nagrania z ręki tą samą drogą). Moja hipoteza
  o stacji na paśmie była błędna.
- Podsłuch IC-7300MK2 nie jest idealny: zbocza ~1 ms (kliknięcia do 3 kHz),
  amplituda siada w elemencie (0,55 -> 0,4) i różni się między elementami
  (hipoteza: ARW), szum pasma 20 m w filtrze CW w przerwach.
**Wynik** (`tester_zbocza` na `probki/kluczownik`, 94 słowa): próg klasyczny
+ grzebień 66/94, **sieć zboczy_3 + grzebień 84/94**; pewność 0,95-1,00;
grzebień: 16,9 WPM przy 17 zadanych.

**Pętla tonu na nagraniu zaczynającym się od szumu — OTWARTE.** Zestaw
zaszumiony (sekunda szumu banku przed nadaniem): `tune.track_tone`
zaczepiała się w pierwszym bloku na szpilce szumu (556 Hz; szpilki w bloku
200 ms wystają 15-17 dB ponad medianę, próg zaczepu 8 dB) i trzymała szum
do końca — stacji 40 dB silniejszej przy 756 Hz nie przejęła (strefa
zaczepu ±12 Hz). Do tego cyfrowa cisza (same zera) dawała zaczep na
dowolnym prążku — **naprawione** (blok zer = brak sygnału; prawdziwe
nagrania: tor tonu bit w bit bez zmian). Sprawdzone i odrzucone:
(1) próg zaczepu 15 dB — szpilki szumu go przekraczają; (2) zaczep po 0,3 s
w tym samym miejscu — mic3: wygrał stały przydźwięk 539 Hz zamiast stacji
750 Hz; (3) pierwszy zaczep z całego nagrania (80. percentyl widma) —
naprawia zestaw, ale zmienia ton na 4 z 16 nagrań z kilkoma sygnałami
(DP75HSC, OK1FPS, Nagrywanie 6 i 9), bez miary, który wybór jest dobry.
Na razie `zaszum` zaczyna nadanie od pierwszej próbki.
Ten sam problem na PRAWDZIWYM nagraniu: kluczownik na 6 m (50,009 MHz, CW,
FIL3) — pusty odbiornik z ARW daje głośny szum (szczyt ~0,25 przy sygnale
0,6, za to równy: szczyty nagrań 0,80-0,86, na 20 m 0,64-0,98); rejestrator
nagrywa ~3-4 s szumu przed kluczownikiem; pętla zaczepiła się na szumie
(672 Hz, potem wędrówka 645-677 Hz) — qso1: 0 okien, qso2: 19 z 26.
Sprawdzone i odrzucone jeszcze dwa: (4) zaczep tylko na prążku wąskim
(ponad sąsiedztwo 40-150 Hz) — bez zmian, bo to nie kliknięcie, a szum;
(5) puszczanie zaczepu, gdy trafienia z ostatniej sekundy rozjeżdżają się
> 5 Hz — naprawia 6 m i syntetyk, ale zmienia ton na 9 prawdziwych
nagraniach stacji (dryf, chirp, zanik też „wędrują"). Do zrobienia porządnie:
zestaw z tonem znanym co do herca (np. nagrania kluczownika ze znanym tonem
podsłuchu + nagrania z pasma z opisem) i dopiero na nim zmiana pętli.
**Obejście w testach:** `nagraj_kluczownik.przytnij` — nagranie zaczyna się
0,1 s przed pierwszym znakiem (0,3 s to wciąż za dużo: blok 200 ms samego
szumu wystarczył do zaczepu); oryginał `surowe/` pełny, przycięcie w opisie.
**Po przycięciu** (`out/zbocza/tester_kluczownik.txt`, 20 m + 6 m, 188 słów):
próg klasyczny + grzebień 149, **sieć zboczy + grzebień 186, z histerezą
187**. Na 20 m skok 84 -> 93/94 tylko przez właściwy zaczep tonu — słabość
pętli kosztowała też na 20 m (tam ~2 s szumu przed kluczownikiem).
**Bez anteny** (6 m, antena odłączona, szum własny odbiornika z ARW; szczyt
0,51 we wszystkich pięciu): sieć zboczy + grzebień **94/94**, próg klasyczny
+ grzebień **2/94** (pewność grzebienia 0,94, a tekst śmieciowy — próg Otsu
na równym szumie odbiornika; nie badane). Razem 282 słowa: sieć 280-281,
klasyczna 151.

**Zestaw zaszumiony — wynik** (`out/zbocza/tester_zaszumione.txt`; sygnał
kluczownika z tekstu 18 WPM + szum banku 7025 kHz FIL 500 Hz; 94 słowa na
poziom; sieć zboczy_3 + grzebień, z histerezą):

| S/N zadane (zmierzone) | +20 | +10 | +6 | +3 (3,3-3,6) | 0 (1,0-1,4) |
|---|---|---|---|---|---|
| słowa | 94/94 | 94/94 | 94/94 | 56/94 | 0/94 |

Przy 3 i 0 dB odcina BRAMKA (okno czytane, gdy S/N >= 3 dB): przy +3 dB
część okien odpada (qso1: 3 z 5), przy 0 dB wszystkie. Odczytu okien, które
przeszły, nie psuje szum — ograniczeniem jest próg bramki, nie sieć.
Próg klasyczny + grzebień: 334/470 (sieć 338/470).

### Grzebień — odzyskanie taktu, nuty, regenerator, tester (17.10)

**Pomysł operatora.** Jak odzyskiwanie zegara: wewnątrz znaku krawędzie na
siatce jednostki, przerwy między znakami sprowadzone do 3 u, „wirtualny
przebieg prostokątny" przyłożony jak grzebień do gwintu. Wynik = „nuty",
z których: regenerator (czysty sygnał, „idealny filtr CW"), tester sieci
(sieć na regeneracie vs tabela Morse'a z tych samych nut), odczyt
klasyczny; docelowo fuzja dróg z wagami jak w metrologii.

**Zrobione.** `dsp/grzebien.py` (dopasowanie u i wagi b do położeń
krawędzi, odporna funkcja błędu, pewność), `tools/tester_grzebien.py`.

**Syntetyk:** tempo mediana < 3% (stałe +1,5-2,5% przy dobrym S/N — do
wyjaśnienia); odczyt klasyczny z nut CER 0% od 10 dB, mediana 0% przy
3-5 dB, 37% przy 0 dB; pewność 0,93-0,97 / ~0,85 przy 0 dB.

**Nagrania** (model czesci_tempo): sieć surowy 19/22, tabela 14/16, sieć
na regeneracie 4/13. Regenerat musi mieć szum jak w treningu (pierwsza
wersja bez szumu — tło ~-40 dB poza rozkładem). W SN0FMT_c tabela i
regenerat dają ciągi „DDDD" — **operator odsłuchał: to naprawdę D (ktoś
stroił nadajnik)**. Grzebień miał rację, sieć na surowym czytała tam
przypadkowe znaki: podział winy działa (tu: SYGNAŁ). Pewność grzebienia
odróżnia nagrania: SN0FMT 0,92-0,98, zawody/mic/F5OHS 0,72-0,79.

### Noc normalizacji tempa (16/17.10) — syntetyk skok, nagrania nie

`runs/fcn_4000000_czesci_norm` (4 mln, 12 epok, fcn, generator z normalizacją):

| | czesci_tempo | **czesci_norm** |
|---|---|---|
| walidacja | 92,6% | **96,9%** |
| CER | 13,9% | **5,3%** |
| okno w całości | 76,6% | **91,2%** |
| SKRÓTY czysta / pełny | 90,5 / 90,2 | **98,7 / 96,0** |
| trafność całe 10-16 / 16-25 / 25-40 WPM | 94,0 / 96,2 / 94,4 (*) | **97,6 / 96,4 / 94,4** |
| nagrania (w całości / w kolejności) | 19 / 22 (*) | 14 / 22 |

(*) czesci_tempo odczytane z normalizacją tempa przy odczycie (bez niej
12 / 21). Koperta czesci_norm: 10-35 WPM.

**Nagrania gorzej w całości** — głównie SN0FMT 3565 (4/4 → 1/4, w kolejności
4/4 oba): „SN0FMT" czytane „SN9FM", „SN8T". **Test czułości na tempo**
(obraz liczony z tempem zmierzonym × 0,75…1,3): nowy model wąsko dostrojony
— przy ×0,75 i ×1,3 milknie (0/0), stary czyta dalej; nowy ma maksimum
często OBOK tempa zmierzonego (SN0FMT ×0,85: 3/4, ×1,0: 1/4).

**Hipoteza (do sprawdzenia):** niedopasowanie RYTMU prawdziwych nagrań do
treningu — (1) błąd pomiaru tempa większy niż zapas TEMPO_BLAD ±15%,
(2) WAGA kluczowania, której generator nie ma: zmierzone w nagraniach
pogrubienie znaków +0,1…+0,5 u (pogłos, ARW, ustawienie keyera). Stary model
widział wiele długości bezwzględnych i był na to przypadkiem odporny.

### Trening w tempie znormalizowanym — przebudowa generatora (16/17.10)

**Decyzja operatora.** „Trenowanie do prędkości znormalizowanej plus minus
margines na błąd normalizacji — bardzo wolne i bardzo szybkie tempo w oknie
albo się nie mieści, albo nabiera za dużo; zamiast uczyć różnych temp, uczyć
małego zakresu, ale różnych wykwintnych zniekształceń."

**Jak.** Stacja (i QRM) nadaje w PRAWDZIWYM tempie z `TEMPO_ZAKRES =
(10, 40)` — zjawiska fizyczne (narastanie klucza, ARW, zanik) mają stały
czas w sekundach. Obraz dostaje tempo „zmierzone" = prawdziwe × (1 ±
`TEMPO_BLAD = 0,15`) i jest liczony w tempie wzorcowym (`C.WPM` = 22):
widmo co 5 ms, ramka = tyle czasu, ile przy 22 WPM trwa 20 ms. Długość
klipu dobrana tak, żeby okno zajmowało ten sam ułamek klipu
(`CLIP_SAMPLES · 22 / tempo`). Jedna droga klip → obraz → etykiety ramek:
`dsp/klip.py` (kopia w `train_rtx.py`, TEST 12 bit w bit: 12 ziaren,
różnica 0, 45 stałych). Na nią przepięte: generator, koperta, skróty,
trafność, X-Ray ocena; odczyt nagrań (`wav2net`) przez `tempo.tor_tempa`.
`TEMPO_NORM = False` = droga sprzed 16.10.

**Po drodze.**
- Plik banku 10 s krótszy niż klip przy 8,5 WPM (~10,4 s) — przedłużanie
  odbiciem w czasie, bez dodatkowych losowań.
- **Generowanie 5,5× szybsze:** 11 procesów odpalało każdy wątki BLAS na
  wszystkie rdzenie. Jeden wątek na proces przy `generate` (przed importem
  numpy): 20 000 próbek 355 s → 65 s; część 200 tys. ~11 min (było ~25 —
  stare noce też to miały).
- Strażnik `_sprawdz_tempo`: trening na zbiorze niepasującym do
  `TEMPO_NORM` (brak/obecność `wpm_norm`) kończy się błędem — inaczej trening
  i odczyt liczyłyby obraz inaczej i nic by tego nie zgłosiło. Najpierw
  pominął drogę „jeden plik" (wracała przed sprawdzeniem) — poprawione;
  potem wywrócił TEST 15 (części testowe bez kolumny) — dostały kolumnę.
- TEST 24: ten sam znak zajmuje tyle samo ramek przy 10, 22 i 40 WPM;
  tempo, błąd i długość klipu zgodne z configiem. diag 119/119.

**Stary model (czesci_tempo, bez normalizacji w treningu) na nowej drodze:**
koperta 10-35 WPM 100% (wcześniej 10 i 35 WPM ~50%); trafność 10-16 WPM:
ucięte znaki 6% kroków (było 42%), całe 94%; nagrania 19/26 / 22/26.
Arkusz obrazów: `out/xray_tempo_norm.png` (10-39 WPM, znaki w oknie jak
~22 WPM). Przedziały tempa w `tools/trafnosc` od teraz 10-16 / 16-25 /
25-40 (prawdziwe tempo) — tabele sprzed 16.10 niewprost porównywalne.

**Noc do uruchomienia:** `noc.bat 200000 12 256 czesci_norm 40 3000 20 0
auto fcn` — porównać z czesci_tempo (12/21, 12/19; z normalizacją przy
odczycie 19/22) i maską.

### Normalizacja tempa przy odczycie — BEZ trenowania (16.10, wieczór)

**Uwaga operatora.** Telegrafista też nie wie, czy pierwszy element to
kropka, czy kreska — rozstrzyga przerwa za nim i kolejne znaki; dekodery
na rynku „najpierw pokazują bzdury, potem przepisują bufor", kursy dają
VVV =, żeby słuchacz złapał tempo. Adaptacja do tempa jest normalna.

**Zrobione.** `dsp/tempo.tor_tempa` (pomiar w buforze 8 s co 2 s, w obie
strony nagrania, mediana z 5) i `obraz_w_tempie_wzorcowym`: widmo co 5 ms,
moc uśredniana w przedziale, który przy 22 WPM trwałby 20 ms — każda stacja
wygląda w oknie jak 22 WPM. Wejście sieci bez zmian (128×32).

**Nagrania, modele bez zmian** (w całości / w kolejności, z 26):

| | (a) obraz jak dotąd | (b) nowa droga, 22 na sztywno | (c) tempo zmierzone |
|---|---|---|---|
| czesci_tempo (bez maski) | 12 / 21 | 12 / 21 | **19 / 22** |
| czesci_tempo_maska | 17 / 21 | 16 / 23 | 14 / **25** |

(b) to kontrola: samo uśrednianie ramek nie zmienia sumy. Zysk (c) bez
maski rozłożony na 6 nagrań (LA3WAA, OK1FPS, SN0FMT ×3, mic2) — więcej niż
rozrzut dwóch identycznych treningów (do 2 słów na nagraniu). Z maską
normalizacja daje rekord w kolejności, ale mniej w całości — dwie poprawki
nie sumują się. Zmierzone tempa: SN0FMT 15-20, mic 19-20, stacje zawodowe
28-37 WPM.

**Dalej:** trening na obrazach znormalizowanych (generator zna prawdziwe
tempo; resztkowy rozrzut ±15-20% na błąd pomiaru), odczyt nagrań przez
`tor_tempa`; potem tryb na żywo z przepisaniem bufora.

### Maska uciętych znaków — pierwszy trening (16.10)

`runs/fcn_4000000_czesci_tempo_maska`: zbiór, ustawienia i ziarno 42 jak
`czesci_tempo`, plus `--maska-uciete`. Środkowy znak 96,69%; CER 20,61%
i SKRÓTY 80% — obie liczby liczą też znaki ucięte, których ten model z
założenia nie uczył się nazywać (SKRÓTY: czysta 79,4 / pełny 80,0 — OK).

**Nagrania: 17/26 w całości (najwięcej w projekcie), 21/26 w kolejności**
(bez maski: 12 i 12 / 21 i 19). Zysk na SN0FMT_c 1→3, SN0FMT_b 2→3, LA3WAA
0→1, mic2 1→2; strata: LA4CIR i DP75HSC w kolejności.

**Syntetyk** (`tools/trafnosc`, 800 klipów na przedział; „środek" =
kroki 24-39, dokładnie to, co bierze `kroki_nagrania`):

| tempo | całe: bez maski | całe: maska | środek: bez maski | środek: maska |
|---|---|---|---|---|
| 12-18 WPM | 94,0 / 94,4 | 93,1 | 95,9 / 96,1 | **94,2** |
| 18-25 WPM | 96,7 / 96,3 | 95,1 | 97,4 / 96,9 | 96,5 |
| 25-32 WPM | 95,4 / 95,2 | 95,3 | 95,8 / 95,4 | 95,6 |

Na uciętych model z maską trafia 2-6% (nie nazywa ich — tak miało być).
**Nierozstrzygnięte:** syntetyk mówi „nieznacznie gorzej" (−1,8 pkt przy
wolnym tempie, więcej niż różnica dwóch treningów bez maski), nagrania
„lepiej" (+5 słów; dwa identyczne treningi bez maski różniły się o 0 w
sumie, o 2 na nagraniu). Jeden trening z maską to za mało. Maska zostaje
opcją, nie domyślną. Rozstrzygnie: druga noc z maską i/lub więcej nagrań.

**Druga para — rozstrzygnięcie (16.10, 20:49).** `czesci_tempo_maska_s7`
(ziarno 7 jak powtórka bez maski): środkowy znak 96,65%, SKRÓTY OK.

| nagrania (w całości / w kolejności) | bez maski | z maską |
|---|---|---|
| ziarno 42 | 12 / 21 | 17 / 21 |
| ziarno 7 | 12 / 19 | 14 / 24 |
| średnio | 12 / 20 | **15,5 / 22,5** |

W OBU parach maska czyta w całości LA3WAA (0→1), SN0FMT_c (1→3), SN0FMT_b
(2→3) — powtarzalnie; bez maski żaden z dwóch treningów. Syntetyk też
powtarzalny: środek okna 12-18 WPM 94,2 i 94,2 z maską wobec 95,9 i 96,1
bez (−1,8 pkt); 18-25 WPM −0,6…−0,9; 25-32 bez różnicy. Oba efekty są
prawdziwe. Hipoteza: przy wolnym tempie długi znak ucięty krawędzią okna
sięga do środka okna, a model z maską go tam nie nazywa — to dotyczy
tylko wolnych stacji (nagrania z zapisaną treścią są głównie 15-25 WPM).

**Poprawka miary.** `tools/trafnosc` liczył „środek" jako kroki 16-47
(połowa okna); odczyt nagrań bierze 24-39. Poprawione — tabele z 16.10
przed 13:50 mają szerszy środek.

### Rozrzut miary nagrań — powtórka treningu (16.10)

`runs/fcn_4000000_czesci_tempo_s7`: TE SAME ustawienia i zbiór co
`czesci_tempo`, inne ziarno podziału (wagi startowe i tak losowe przy
każdym treningu). Środkowy znak 96,72% (96,61), CER 14,09% (13,91), SKRÓTY
OK (czysta 89,9 / pełny 89,5). Nagrania:

| | w całości | w kolejności |
|---|---|---|
| czesci_tempo | 12/26 | 21/26 |
| czesci_tempo_s7 (powtórka) | 12/26 | 19/26 |

Pod sumami słowa się przetasowały: SN0FMT_c w kolejności 3/4 → 1/4,
DP75HSC i OK1FPS w kolejności 1/1 → 0/1, mic2 1/3 → 2/3 w całości i 1/3 →
3/3 w kolejności. **Między dwoma identycznymi treningami pojedyncze
nagranie potrafi przeskoczyć o 2 słowa, a suma o 2.** Z jednej pary nie da
się podać rozrzutu, ale wniosek praktyczny: różnice nocy o 2-3 słowa na
26 to szum. Różnica czesci_nad (16/24) wobec czesci_tempo (12/21, 12/19)
— 4 słowa w całości — jest na granicy; nie rozstrzygnięta. Porównania
modeli opierać na `tools/trafnosc` (syntetyk, stałe klipy), a nagrania
traktować jako sprawdzian kierunku, dopóki nie przybędzie nagrań.

### Wolne tempo „słabe" — to głównie ucięte znaki (16.10)

**Objaw.** Na syntetyku przy pełnym realizmie 12-15 WPM 67%, 15-18 WPM 82%
trafnych kroków, a koperta (środkowy znak, sygnał idealny) — 100%.

**Test „po jednym" w dwóch przedziałach tempa** (czesci_tempo, S/N 15-30):
już CZYSTA stacja 79,4% przy 12-18 WPM wobec 99,7% przy 22-28; żadne
zjawisko kanału nie tłumaczy różnicy (wszystkie wiersze −14…−28 pkt).

**Przyczyna (zmierzone, 400 klipów):**

| | wszystkie kroki | znaki całe | ucięte krawędzią | środek okna | udział uciętych |
|---|---|---|---|---|---|
| czysto 12-18 WPM | 78,0 | **97,1** | 47,8 | 94,8 | **38,7%** |
| czysto 22-28 WPM | 99,8 | 99,9 | 90,6 | 99,9 | 1,2% |
| realizm 12-18 WPM | 72,3 | **94,0** | 45,1 | 91,4 | **44,3%** |
| realizm 22-28 WPM | 96,8 | 96,9 | 81,9 | 96,6 | 0,7% |

Przy wolnym tempie znaki są długie i prawie połowa kroków ze znakiem należy
do znaków uciętych krawędzią okna — na nich model zgaduje (~46%). Na znakach
całych wolne tempo 94-97%. „Słabość" to w dużej części miara licząca ucięte
znaki; prawdziwa różnica w środku okna (z niego składa się odczyt nagrań)
to 91-95 wobec 97-100%. Trening uczy jednak sieć zgadywania na prawie
połowie kroków ze znakiem przy wolnym tempie — to argument za maską.

**Co zrobione.** `train_rtx.py train --maska-uciete` (`_maska_uciete`):
kroki znaku dotykającego krawędzi okna mają w TRENINGU wagę 0; walidacja
i pomiary bez maski (porównywalne z nocami przed). Maska liczona z etykiet
kroków — bez nowego zbioru. Test jednostkowy i trening próbny obiema drogami
(z rotacją i bez) na CPU; diag 116/116.

### Zanik zasilania w trakcie generowania zbioru (14.10)

**Objaw.** Noc `czesci_nad` (start 18:14) przerwana zanikiem zasilania
w trakcie szóstej części (126 500 / 200 000 próbek, po 20:24).

**Stan po awarii (sprawdzone).** `git fsck` bez błędów; paszport nocy bez
różnicy od commita; części 00–04 całe (CRC archiwum OK, po 200 000 próbek,
ziarna 12345…16345); części 05 nie ma wcale — plik powstaje dopiero po
wygenerowaniu całości. Treningu jeszcze nie było.

**Wznowienie było już w kodzie.** `train_rtx.py generate --shards` pomija
części kompletne; `noc.sh` przy 5 z 20 częściach woła generowanie, które
dorabia brakujące 15.

**Co poprawione.** Zapis części szedł prosto pod docelową nazwę — awaria
w samym zapisie zostawiłaby ucięty plik. Teraz `<część>.npz.part`, fsync,
`os.replace` (`.part` nie pasuje do wzorca `*.npz`). Sprawdzone poza
repozytorium: 2 części, potem ucięta trzecia i śmieciowy `.part` —
wznowienie pominęło dwie, ucięta „NIEKOMPLETNA — generuje od nowa",
po zapisie bez resztek. diag 116/116.

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

### Brak notcha — ZAMKNIĘTE 19.10 przeciwsygnałem

Luka była taka: przy dwóch stacjach obok siebie operator sięga po notch,
układ nie miał czym. Zmierzone 19.10 („Przeciwsygnał: obca stacja odtworzona
i odjęta"): notch ±10 / ±20 / ±30 Hz 115 / 131 / 138 ze 145, przeciwsygnał
142; w drodze jest przeciwsygnał (`dsp/przeciwsygnal.py`). Przewidywanie
stąd („notch musi być węższy niż ~30 Hz, żeby nie zjadać wstęg") potwierdzone:
±30 Hz traci właśnie przy obcej 30 Hz obok.

### Model zapamiętuje zbiór

`val_loss` ma minimum w epoce 13 i potem rośnie, przy dokładności
treningowej bliskiej 100%. Pięciokrotne zwiększenie zbioru (200 tys. ->
1 mln) zmniejszyło błąd o połowę, więc dźwignia nie jest wyczerpana.

Przy 4 mln (06.10) to samo: minimum `val_loss` 0,0288 w epoce 15, na
koniec 0,0574. Najlepsza dokładność w epoce 32, ale epoki po 20 dały
niewiele. Wniosek: 20 epok zamiast 40, a zaoszczędzony czas na dane.

Przy 10 mln (08/09.10) zapamiętywanie prawie znikło: minimum `val_loss`
0,0211 w epoce 16, na koniec 0,0231 (+10%, przy 4 mln było +100%).
Najlepsza dokładność w epoce 19 z 20 — model jeszcze nie skończył się
uczyć, więc kilka epok więcej na tym samym zbiorze może coś dać.

### Więcej danych zwęża kopertę tonu

Nieoczekiwane i zmierzone: przy 650 Hz model na 200 tys. trafiał 40–82%,
model na 1 mln trafia 0–20%. Lepsze dopasowanie do rozkładu treningowego
(750 ± 80 Hz) oznacza ostrzejsze milczenie poza nim. To nie jest regres,
tylko wybrany kompromis selektywność–odporność — ale **zależność od
`dsp/tune.py` przez to wzrosła**.

**Przy 10 mln trend się odwrócił — ale tylko na brzegach** (siatka
koperty, `out/koperta_dpu_20261006_0004.txt` kontra `..._20261008_2220.txt`):

| | 4 mln | 10 mln |
|---|---|---|
| 650 Hz przy 25 / 27 / 30 WPM | 2 / 15 / 0% | 50 / 62 / 55% |
| 850 Hz przy 10 / 13 WPM | 18 / 50% | 48 / 70% |
| 650 i 850 Hz przy 15–20 WPM | 0–8% | 0–8% |
| 750 Hz przy 30 WPM | 100% | 90% |

`RANO.txt` pisze „ton 650–850 Hz", bo bierze skrajne komórki ≥ 50%.
To myli: w środku zakresu tempa (15–22 WPM) model nadal czyta wyłącznie
750 Hz. Zależność od `tune.py` się nie zmniejszyła.
