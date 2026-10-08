@echo off
rem ===========================================================================
rem  z_hdd.bat  --  WYNIKI z HDD na pendraka, po treningu
rem
rem  Uzycie:
rem      z_hdd.bat                     -> zrodlo domyslne C:\AI_DSP
rem      z_hdd.bat D:\praca\AI_DSP
rem      z_hdd.bat C:\AI_DSP test      -> tylko pokaz, co by zrobil
rem
rem  Kierunek odwrotny do na_hdd.bat. To JEDYNY skrypt, ktory pisze na
rem  pendraka, i pisze malo plikow: modele, logi treningu, wykresy.
rem  exFAT bez wear-levelingu nie lubi tysiacy malych zapisow, wiec zbior
rem  treningowy (*.npz, 800 MB) i katalog venv tu NIE ida -- jedno i drugie
rem  odtwarza sie na miejscu.
rem
rem  KODU Z HDD JUZ NIE WOZIMY (od 08.10). Wczesniej ten skrypt kopiowal
rem  z powrotem "kod zmieniony na HDD" -- i to byla pompa smieci: kazdy
rem  dorazny skrypt napisany na maszynie treningowej przy szukaniu bledu
rem  (GPU_check.sh, libsRTX3050.sh.sh, dwie kopie setenv...) jechal na
rem  pendraka, a stamtad na BD-R. Kod idzie teraz w JEDNA strone:
rem  repozytorium (GitHub albo pendrak) -> HDD. Poprawke robi sie
rem  w repozytorium, a noc.sh sprawdza spis wersji i niezgodny plik
rem  zatrzyma noc.
rem
rem  Rowniez bez /MIR: nic na pendraku nie jest kasowane.
rem ===========================================================================

setlocal
set CEL=%~dp0
set ZRODLO=%~1
if "%ZRODLO%"=="" set ZRODLO=C:\AI_DSP
set TRYB=%~2

set OPCJE=/E /NFL /NDL /NJH /NP /R:2 /W:2

if /I "%TRYB%"=="test" (
    set OPCJE=%OPCJE% /L
    echo *** TRYB TESTOWY -- nic nie zostanie skopiowane ***
    echo.
)

echo ===========================================================================
echo  WYNIKI  ->  PENDRAK
echo ===========================================================================
echo  zrodlo: %ZRODLO%
echo  cel:    %CEL%
echo.
echo  kopiowane:     runs\ (modele, log.csv, state.json), out\ (logi nocne,
echo                 wykresy, paczki .bundle)
echo  NIE kopiowane: kod (idzie tylko w strone HDD), *.npz, .venv
echo.

if not exist "%ZRODLO%\dsp\config.py" (
    echo BLAD: nie widze %ZRODLO%\dsp\config.py
    echo       Podaj katalog projektu na HDD jako pierwszy argument.
    exit /b 1
)

rem  TRYB "auto": bez pytania. Potrzebny dla noc.bat, ktory ma dzialac
rem  bez nadzoru -- wtykasz pendraka, Enter, i idziesz do innej roboty.
rem  Z reki zostaje pytanie, bo to jedyny moment, w ktorym mozna sie
rem  zorientowac, ze cel jest zly, PRZED zapisem.
if /I "%TRYB%"=="auto" echo  *** TRYB AUTO -- bez pytania ***
if /I not "%TRYB%"=="test" if /I not "%TRYB%"=="auto" (
    choice /C TN /N /M "Kopiowac na pendraka? [T/N] "
    if errorlevel 2 (
        echo przerwane
        exit /b 1
    )
)

rem --- MODELE I LOGI: to jedyny nieodtwarzalny wynik treningu.
rem     Kopiujemy tylko rozniace sie, zeby nie przepisywac 12 MB modelu
rem     przy kazdym uruchomieniu.
echo [1/2] runs (modele i logi treningu)...
if exist "%ZRODLO%\runs" (
    robocopy "%ZRODLO%\runs" "%CEL%runs" %OPCJE%
) else (
    echo   brak runs\ -- pomijam
)

rem --- WYKRESY I OBRAZY: krzywe uczenia, X-Ray. Male, warto miec.
echo [2/2] out (logi nocne, wykresy, X-Ray)...
if exist "%ZRODLO%\out" (
    rem  *.log     -- logi nocne. NAJWAZNIEJSZY plik do przeczytania rano.
    rem              Bez tego wracaly tylko wtedy, gdy spakowalo sie caly
    rem              katalog tarem, czyli przypadkiem.
    rem  *.bundle  -- commity z nocy. W WSL nie ma Credential Managera,
    rem              wiec noc.sh nie wypchnie ich sam; paczka wraca tu
    rem              i wypycha ja maszyna, ktora token ma.
    robocopy "%ZRODLO%\out" "%CEL%out" *.png *.txt *.html *.csv *.log *.err *.bundle %OPCJE%
) else (
    echo   brak out\ -- pomijam
)


echo.
echo ===========================================================================
echo  GOTOWE.  Wyniki na pendraku: %CEL%
echo.
echo  Na koniec warto zrobic archiwum jednym plikiem (mniej zapisow do FAT):
echo      tar czf AI_DSP_%%DATE:~0,10%%.tar.gz --exclude=.venv --exclude=__pycache__ ^
echo          --exclude=*.npz --exclude=*.whl -C %ZRODLO%\.. AI_DSP
echo ===========================================================================
exit /b 0
