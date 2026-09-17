@echo off
rem ===========================================================================
rem  wslconfig.bat  --  podnosi limit pamieci WSL2
rem
rem  Uzycie (z Windows, na maszynie z karta):
rem      srodowisko\wslconfig.bat            24 GB
rem      srodowisko\wslconfig.bat 28         inna wartosc
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
rem  NIE USTAWIA 'processors'. Domyslnie WSL dostaje wszystkie rdzenie
rem  i tak ma zostac: generowanie zbioru chodzi na 11 procesach i kazdy
rem  zabrany rdzen to dluzsze generowanie.
rem ===========================================================================

setlocal

set GB=%~1
if "%GB%"=="" set GB=24
set PLIK=%USERPROFILE%\.wslconfig
if not "%~2"=="" set PLIK=%~2

echo ===========================================================================
echo  LIMIT PAMIECI WSL2
echo ===========================================================================
echo  plik:      %PLIK%
echo  ustawiam:  memory=%GB%GB
echo.

if exist "%PLIK%" goto :juz_jest

echo Plik nie istnieje -- tworze.
>  "%PLIK%" echo [wsl2]
>> "%PLIK%" echo memory=%GB%GB
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
