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
rem  pendraka, i pisze malo plikow: model, logi treningu, wykresy, kod
rem  zmieniony na HDD. exFAT bez wear-levelingu nie lubi tysiacy malych
rem  zapisow, wiec zbior treningowy (*.npz, 800 MB) i katalog venv tu
rem  NIE ida -- jedno i drugie odtwarza sie na miejscu.
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
echo  kopiowane:     runs\ (modele, log.csv, state.json), out\*.png, kod
echo  NIE kopiowane: *.npz (zbior, 800 MB), .venv, __pycache__
echo.

if not exist "%ZRODLO%\dsp\config.py" (
    echo BLAD: nie widze %ZRODLO%\dsp\config.py
    echo       Podaj katalog projektu na HDD jako pierwszy argument.
    exit /b 1
)

if /I not "%TRYB%"=="test" (
    choice /C TN /N /M "Kopiowac na pendraka? [T/N] "
    if errorlevel 2 (
        echo przerwane
        exit /b 1
    )
)

rem --- MODELE I LOGI: to jedyny nieodtwarzalny wynik treningu.
rem     Kopiujemy tylko rozniace sie, zeby nie przepisywac 12 MB modelu
rem     przy kazdym uruchomieniu.
echo [1/3] runs (modele i logi treningu)...
if exist "%ZRODLO%\runs" (
    robocopy "%ZRODLO%\runs" "%CEL%runs" %OPCJE%
) else (
    echo   brak runs\ -- pomijam
)

rem --- WYKRESY I OBRAZY: krzywe uczenia, X-Ray. Male, warto miec.
echo [2/3] out (wykresy, X-Ray)...
if exist "%ZRODLO%\out" (
    robocopy "%ZRODLO%\out" "%CEL%out" *.png *.txt *.html *.csv %OPCJE%
) else (
    echo   brak out\ -- pomijam
)

rem --- KOD: gdyby cos bylo poprawiane na HDD w trakcie treningu.
rem     Tu NIE nadpisujemy zawsze -- na pendraku moze byc nowsza wersja.
rem     Robocopy skopiuje tylko to, co sie rozni.
echo [3/3] kod zmieniony na HDD...
robocopy "%ZRODLO%" "%CEL%." *.py *.md *.bat %OPCJE% /XD .venv venv_gpu __pycache__ out runs .git /XF *.npz *.whl *.pyc

echo.
echo ===========================================================================
echo  GOTOWE.  Wyniki na pendraku: %CEL%
echo.
echo  Na koniec warto zrobic archiwum jednym plikiem (mniej zapisow do FAT):
echo      tar czf AI_DSP_%%DATE:~0,10%%.tar.gz --exclude=.venv --exclude=__pycache__ ^
echo          --exclude=*.npz --exclude=*.whl -C %ZRODLO%\.. AI_DSP
echo ===========================================================================
exit /b 0
