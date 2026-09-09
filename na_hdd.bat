@echo off
rem ===========================================================================
rem  na_hdd.bat  --  KOD z pendraka na HDD, przed treningiem
rem
rem  Uzycie:
rem      na_hdd.bat                     -> cel domyslny C:\AI_DSP
rem      na_hdd.bat D:\praca\AI_DSP     -> wlasny cel
rem      na_hdd.bat C:\AI_DSP test      -> tylko pokaz, co by zrobil
rem
rem  DWIE RZECZY, KTORE TEN SKRYPT ROBI SWIADOMIE
rem
rem  1. Zrodlo bierze z WLASNEGO polozenia (%~dp0), a nie z litery dysku.
rem     Pendrak dostaje rozne litery na roznych maszynach (u nas byl J:,
rem     potem I:) i wpisana litera bylaby pomylka czekajaca na okazje.
rem
rem  2. NIE UZYWA robocopy /MIR ani /PURGE. Mirror usuwa z celu wszystko,
rem     czego nie ma w zrodle -- czyli skasowalby na HDD zbior treningowy
rem     (800 MB), wytrenowane modele w runs\ i wyniki w out\. Kopiujemy
rem     tylko do przodu, nigdy nie kasujemy.
rem
rem  Czego NIE kopiuje i dlaczego:
rem      .venv, venv_gpu   odtwarzalne, tysiace malych plikow
rem      __pycache__       odtwarzalne, i jest z innej wersji Pythona
rem      *.npz             zbior treningowy, generowany na miejscu (2 min)
rem      *.whl             kola pipa, do sciagniecia
rem      out\              wyniki, powstaja na HDD
rem      runs\             WYTRENOWANE MODELE -- na HDD sa nowsze niz na
rem                        pendraku; kopiowanie ich tutaj nadpisalo by
rem                        wynik treningu. Do archiwizacji sluzy z_hdd.bat
rem ===========================================================================

setlocal
set ZRODLO=%~dp0
set CEL=%~1
if "%CEL%"=="" set CEL=C:\AI_DSP
set TRYB=%~2

set OPCJE=/E /NFL /NDL /NJH /NP /R:2 /W:2
set WYKLUCZ_KAT=/XD .venv venv_gpu __pycache__ out runs .git
set WYKLUCZ_PLIK=/XF *.npz *.whl *.pyc *.keras *.h5

if /I "%TRYB%"=="test" (
    set OPCJE=%OPCJE% /L
    echo *** TRYB TESTOWY -- nic nie zostanie skopiowane ***
    echo.
)

echo ===========================================================================
echo  KOD  ->  HDD
echo ===========================================================================
echo  zrodlo: %ZRODLO%
echo  cel:    %CEL%
echo.
echo  kopiowane:    *.py *.md *.txt *.sh requirements.txt
echo  NIE kopiowane: .venv, __pycache__, out\, runs\, *.npz, *.whl
echo.

if not exist "%ZRODLO%dsp\config.py" (
    echo BLAD: nie widze %ZRODLO%dsp\config.py
    echo       Uruchom ten skrypt z katalogu projektu na pendraku.
    exit /b 1
)

rem  TRYB "auto": bez pytania. Potrzebny dla noc.bat, ktory ma dzialac
rem  bez nadzoru -- wtykasz pendraka, Enter, i idziesz do innej roboty.
rem  Z reki zostaje pytanie, bo to jedyny moment, w ktorym mozna sie
rem  zorientowac, ze cel jest zly, PRZED zapisem.
if /I "%TRYB%"=="auto" echo  *** TRYB AUTO -- bez pytania ***
if /I not "%TRYB%"=="test" if /I not "%TRYB%"=="auto" (
    choice /C TN /N /M "Kopiowac? [T/N] "
    if errorlevel 2 (
        echo przerwane
        exit /b 1
    )
)

if not exist "%CEL%" mkdir "%CEL%"

rem --- KOD: nadpisujemy ZAWSZE (/IS /IT), nie po znacznikach czasu.
rem     exFAT ma rozdzielczosc 2 s i nie trzyma strefy czasowej, wiec
rem     porownanie czasow bywa zawodne -- wlasnie na tym stracilismy
rem     pliki przy przenoszeniu. Kod ma kilkaset kB, kopiowanie za kazdym
rem     razem nic nie kosztuje, a daje pewnosc.
echo [1/2] kod...
robocopy "%ZRODLO%." "%CEL%" *.py *.md *.txt *.sh *.cfg *.bat %OPCJE% /IS /IT %WYKLUCZ_KAT% %WYKLUCZ_PLIK%
set RC1=%ERRORLEVEL%

rem --- PROBKI: nagrania, tylko rozniace sie. To 110 MB i nie zmieniaja
rem     sie czesto; kopiowanie za kazdym razem zajechaloby pendraka
rem     odczytami bez potrzeby.
echo [2/2] probki (nagrania)...
if exist "%ZRODLO%probki" (
    robocopy "%ZRODLO%probki" "%CEL%\probki" %OPCJE%
    set RC2=%ERRORLEVEL%
) else (
    echo   brak katalogu probki -- pomijam
    set RC2=0
)

echo.
echo ===========================================================================
rem robocopy: 0 = nic do zrobienia, 1 = skopiowano, >=8 = blad
if %RC1% GEQ 8 goto :blad
if %RC2% GEQ 8 goto :blad
echo  GOTOWE.  Kod w %CEL%
echo.
echo  Dalej, na HDD:
echo      cd /d %CEL%
echo      python diag.py
echo      python train_rtx.py generate --n 200000
echo      python train_rtx.py train --epochs 100 --batch 256 --mixed --require-gpu
echo ===========================================================================
exit /b 0

:blad
echo  BLAD kopiowania (robocopy %RC1% / %RC2%)
echo  Kody ^>= 8 to bledy. Sprawdz, czy cel nie jest zajety.
echo ===========================================================================
exit /b 1
