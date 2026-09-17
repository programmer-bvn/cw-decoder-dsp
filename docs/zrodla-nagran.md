# Źródła nagrań — co którym da się zwalidować

Osiem dostępnych źródeł. To nie jest lista „więcej danych" — to macierz
walidacyjna dla modelu kanału, bo **żaden parametr nadajnika ani ARW
w `dsp/config.py` nie jest dziś zmierzony**. Wszystkie pochodzą z wiedzy
operatora o fizyce zjawiska, spisanej we wrześniu 2026, i są rozsądne —
ale rozsądne to nie to samo co zmierzone. `WPM_JITTER` też był rozsądny,
dopóki nie okazał się najdroższą pomyłką tego projektu.

## Dostępne źródła

| # | źródło | tor | co daje |
|---|---|---|---|
| 1 | keyer + manipulator, IC-746 | audio | nadawanie własne, elementy maszynowe |
| 2 | odbiór z pasma, IC-746 | audio | obce stacje, ARW IC-746 |
| 3 | **klucz sztorcowy, TS-520** | audio | nadajnik lampowy: chirp, dryf, sag |
| 4 | odbiór z pasma, TS-520 | audio | obce stacje, ARW lampowy |
| 5 | manipulator dwudźwigniowy, IC-7300MK2 | audio | nadawanie własne |
| 6 | keyer, IC-7300MK2 | audio | timing idealny (odniesienie) |
| 7 | odbiór z pasma, IC-7300MK2 | audio | obce stacje, rozstrojenia |
| 8 | **IC-7300MK2 przez USB** | cyfrowy | to samo bez toru mikrofonowego |

## Trzy rzeczy, które ta lista rozstrzyga

### Tor mikrofonowy kontra cyfrowy (źródło 8)

Wszystkie cztery obecne próbki przeszły przez głośnik, powietrze,
mikrofon i ARW karty dźwiękowej. Skutki są zmierzone i opisane
w `probki/*.txt`: ton rozmyty na 200 Hz, tło podniesione o 15 dB,
przesterowanie o 12 dB wymagające korekty `auto_gain_db()`.

**Źródło 8 usuwa cały ten tor.** Nagranie tej samej treści co `mic3`,
przez USB, daje pierwszą próbkę odniesienia bez znanej wady — i przy
okazji mierzy, ILE ten tor kosztował, bo różnica będzie policzalna.

To jest najtańsza i najpewniej najbardziej opłacalna rzecz z całej listy.

### Wady nadajnika (źródło 3, TS-520)

W `dsp/config.py` stoi:

```python
CHIRP_HZ   = (0.0, 25.0)     # ile herców spada ton w trakcie elementu
SAG_DB     = (0.0, 3.0)      # ile decybeli spada amplituda elementu
SAG_TAU_MS = (20.0, 120.0)   # stała czasowa zapadania i powrotu
DRIFT_HZ   = 8.0             # dryf tonu na długość klipu
```

Mechanizm opisał operator: naduszenie klucza rozładowuje kondensator
w zasilaczu, napięcie siada, a to przestraja generator LC — stąd ton
750 Hz na początku kreski i 730 Hz na jej końcu. Fizyka się zgadza, ale
**liczby nie są znikąd zmierzone**.

TS-520 to konstrukcja lampowa z generatorem VFO i dokładnie takim
zasilaczem. Jest jedynym dostępnym źródłem, na którym te cztery liczby
da się zweryfikować, bo IC-746 i IC-7300 są syntezowane i tego zjawiska
praktycznie nie mają.

Do zmierzenia z nagrania: przebieg częstotliwości chwilowej wewnątrz
pojedynczej kreski (`dsp/tune.py` ma już do tego maszynerię) oraz
obwiednia amplitudy elementu.

### Trzy rodzaje timingu (źródła 3, 5, 6)

Generator ma trzy osobne pokrętła i każde odpowiada innemu sposobowi
nadawania:

| pokrętło | co opisuje | źródło do walidacji |
|---|---|---|
| `FIST` | rozjazd długości elementów | 3 — klucz sztorcowy |
| `GAP_JITTER` | rozjazd samych przerw | 5 — manipulator + keyer |
| oba zerowe | timing maszynowy | 6 — keyer z pamięci |

To jest rzadka sytuacja: parametry, które zwykle trzeba by szacować,
mają tu po jednym źródle, które izoluje dokładnie jeden z nich.
Klucz sztorcowy rozjeżdża elementy i przerwy; manipulator z keyerem ma
elementy maszynowo poprawne, a ludzkie tylko przerwy; pamięć keyera nie
rozjeżdża nic.

## Odbiór z pasma (źródła 2, 4, 7)

Trzy różne odbiorniki to trzy różne ARW i trzy różne kształty filtru.
`AGC_TAU_MS = (50, 500)` i `AGC_DEPTH = (0.2, 0.8)` są dziś zgadnięte.

Nagrania z pasma nie mają prawdy naziemnej, ale w tym projekcie
obowiązuje zasada, że odniesieniem jest operator: **to, co odczytasz ze
słuchu i wpiszesz w pole `nadane:`, JEST etykietą.** Wystarczy fragment,
choćby sam znak wywoławczy — `tools/nagrania.py` wyciąga z tego pola
ciągi pisane wielkimi literami i liczy je tak samo jak pełną treść.

Tu też siedzi materiał, którego dziś brakuje najbardziej: **sygnały
rozstrojone**. Wszystkie cztery obecne próbki mają ton w zakresie
700–839 Hz, a koperta modelu to 670–830 Hz — czyli nigdy nie sprawdziliśmy,
co się dzieje poza nią. Na paśmie rozstrojonych stacji jest pełno.

## Kolejność, gdyby robić po jednym

1. **Źródło 8** — ta sama treść co `mic3`, przez USB. Zastępuje próbkę
   odniesienia wersją bez znanej wady toru i mierzy koszt tego toru.
2. **Źródło 3** — TS-520, klucz sztorcowy, kilkadziesiąt sekund. Jedyna
   droga do zmierzenia chirpu, sagu i dryfu.
3. **Źródło 7 albo 2** — pasmo, ze szczególnym oczekiwaniem na stacje
   rozstrojone. Odczyt ze słuchu w `nadane:`, choćby częściowy.
4. Reszta jako uzupełnienie macierzy.

## Jak dołożyć nagranie

Plik `.wav` do `probki/`, potem:

```bash
python -m tools.probki
```

Narzędzie dopisze część mierzoną (ton, tempo, szerokość, szczyt, korekta
poziomu) i **zostawi ręczną nietkniętą**. W ręcznej wypełnia się to,
czego z pliku nie da się odczytać: `nadane`, `nadajnik`, `odbiornik`,
`klucz`, `pasmo`, `filtr`, `tor`, `uwagi`.

Od następnej nocy nagranie liczy się w mierze `w całości / w kolejności`
razem z pozostałymi — pod warunkiem, że `nadane:` zawiera cokolwiek
pisane wielkimi literami.
