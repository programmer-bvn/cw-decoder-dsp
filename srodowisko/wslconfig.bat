@echo off
rem ===========================================================================
rem  wslconfig.bat  --  limit pamieci i rdzeni dla WSL2
rem
rem  Uzycie (z Windows, na maszynie z karta):
rem      srodowisko\wslconfig.bat            24 GB, rdzenie host-2
rem      srodowisko\wslconfig.bat 28         inny limit pamieci
rem      srodowisko\wslconfig.bat 28 10      i jawna liczba rdzeni
rem
rem  PO CO. WSL2 domyslnie bierze POLOWE pamieci hosta. Maszyna z 32 GB daje
rem  wiec w WSL okolo 15 GB, a straznik pamieci w train_rtx.py liczy z tego,
rem  co widzi WSL -- czyli przepuszcza najwyzej 5 czesci zbioru (1 mln
rem  probek), choc maszyna unioslaby dwa razy tyle.
rem
rem      limit WSL   bezpieczny zbior
rem      15 GB       5 czesci = 1,0 mln
rem      24 GB       8 czesci = 1,6 mln
rem      28 GB      10 czesci = 2,0 mln
rem
rem  NIE NADPISUJE ISTNIEJACEGO PLIKU. .wslconfig dotyczy WSZYSTKICH
rem  dystrybucji na maszynie i moze zawierac ustawienia, o ktorych ten
rem  skrypt nie wie. Gdy plik juz jest, skrypt pokazuje jego tresc i mowi,
rem  co dopisac -- decyzje zostawia czlowiekowi.
rem
rem  'processors' USTAWIA TYLKO NA ZADANIE, drugim argumentem.
rem  Bez niego maszyna wirtualna bierze wszystkie rdzenie i tak ma
rem  zostac: ta maszyna nie robi nic poza mieleniem danych, wiec
rem  kazdy rdzen oddany hostowi to czyste spowolnienie.
rem
rem  KIEDY WARTO SIEGNAC PO TEN ARGUMENT. Przy generowaniu wszystkie
rem  rdzenie ida na 100%% i potrafi sie ROZLACZYC WiFi -- Windows nie
rem  ma na czym obsluzyc stosu sieciowego. Dopoki maszyna tylko
rem  mieli, nic to nie kosztuje: siec jest potrzebna dopiero
rem  w etapie 7 (commit, paczka, push), godziny pozniej, gdy rdzenie
rem  sa juz wolne.
rem
rem  Ale jesli maszyna ma byc w tym czasie do czegokolwiek uzywana
rem  -- pulpit zdalny, przegladanie wynikow, cokolwiek przez siec --
rem  to dwa watki oddane hostowi kosztuja okolo kwadransa na dwie
rem  godziny generowania i zalatwiaja sprawe:
rem      srodowisko\wslconfig.bat 24 10
rem ===========================================================================

setlocal

set GB=%~1
if "%GB%"=="" set GB=24

rem  Liczba rdzeni dla maszyny wirtualnej. PUSTE = nie ustawiamy jej
rem  wcale, czyli WSL bierze wszystkie. %NUMBER_OF_PROCESSORS% to
rem  watki logiczne HOSTA -- podglad, ile ich w ogole jest.
set RDZENIE=%~2

set PLIK=%USERPROFILE%\.wslconfig
if not "%~3"=="" set PLIK=%~3

echo ===========================================================================
echo  LIMIT PAMIECI WSL2
echo ===========================================================================
echo  plik:      %PLIK%
echo  ustawiam:  memory=%GB%GB
if not "%RDZENIE%"=="" echo             processors=%RDZENIE%  (host ma %NUMBER_OF_PROCESSORS% watkow)
if "%RDZENIE%"=="" echo             processors: NIE ustawiam, WSL bierze wszystkie %NUMBER_OF_PROCESSORS%
echo.

if exist "%PLIK%" goto :juz_jest

echo Plik nie istnieje -- tworze.
>  "%PLIK%" echo [wsl2]
>> "%PLIK%" echo memory=%GB%GB
if not "%RDZENIE%"=="" >> "%PLIK%" echo processors=%RDZENIE%
echo.
echo Zapisane:
type "%PLIK%"
goto :po_zapisie

:juz_jest
echo UWAGA: plik JUZ ISTNIEJE i go nie ruszam.
echo.
echo Obecna tresc:
echo ---------------------------------------------------------------------------
type "%PLIK%"
echo ---------------------------------------------------------------------------
echo.
echo Jesli nie ma tam linii 'memory=', dopisz w sekcji [wsl2]:
echo.
echo     memory=%GB%GB
if not "%RDZENIE%"=="" echo     processors=%RDZENIE%
echo.
echo Jesli jest, ale mniejsza -- popraw wartosc.
echo.
echo Otworz go czym chcesz, np.:
echo     notepad "%PLIK%"

:po_zapisie
echo.
echo ===========================================================================
echo  ZMIANA NIE DZIALA, DOPOKI MASZYNA WIRTUALNA NIE WSTANIE OD NOWA:
echo.
echo      wsl --shutdown
echo.
echo  Sprawdzenie po ponownym wejsciu do WSL:
echo      free -h          ma pokazac okolo %GB%Gi
if not "%RDZENIE%"=="" echo      nproc            ma pokazac %RDZENIE%
echo ===========================================================================
echo.
choice /C TN /N /M "Zrobic 'wsl --shutdown' teraz? [T/N] "
if errorlevel 2 goto :koniec
wsl --shutdown
echo.
echo Zrobione. Przy nastepnym wejsciu do WSL limit bedzie juz nowy.

:koniec
echo.
pause
endlocal
