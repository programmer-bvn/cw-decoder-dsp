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
echo  kopiowane:    *.py *.md *.txt *.sh *.cfg *.bat LICENSE .gitignore .gitattributes
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
rem --- JESLI CEL JEST REPOZYTORIUM, NIE NADPISUJEMY KODU.
rem     robocopy z /IS /IT nadpisuje pliki BEZWARUNKOWO, wiec w drzewie
rem     roboczym gita zrobilby z kazdego pliku zmiane -- nawet gdy tresc
rem     jest ta sama, bo zmienia znaczniki czasu, a przy roznicy choc
rem     jednego bajtu skasowalby poprawke zrobiona po tamtej stronie.
rem     Tam kod odswieza sie przez 'git pull --ff-only'.
rem     Ustawienie repozytorium: ./srodowisko/hdd_repo.sh
rem UWAGA NA %ERRORLEVEL% W NAWIASACH. cmd.exe rozwija zmienne przy
rem PARSOWANIU calego bloku, a nie przy wykonaniu, wiec
rem     if ... ( robocopy ... & set RC=%ERRORLEVEL% )
rem zapisuje kod SPRZED bloku. Dlatego ponizej sa skoki, a nie nawiasy.
rem Ten sam blad siedzial tu wczesniej przy RC2 i powodowal, ze awaria
rem kopiowania nagran nigdy nie byla wykrywana.
if exist "%CEL%\.git" goto :bez_kodu
echo [1/2] kod...
rem  LICENSE, .gitignore i .gitattributes sa WYMIENIONE Z NAZWY, bo nie
rem  maja rozszerzenia i zaden wzorzec *.cos ich nie lapie. Bez nich
rem  kopia na HDD rozni sie od repozytorium w sposob, ktory szkodzi:
rem    .gitignore   -- bez niego "git status" na HDD pokazuje tysiace
rem                    nieśledzonych plikow (czesci/*.npz, venv_gpu),
rem                    wiec kontrola roznic i porzadki.sh przestaja
rem                    cokolwiek znaczyc
rem    .gitattributes -- bez niego zakonczenia linii normalizuja sie
rem                    inaczej po obu stronach i git pokazuje roznice
rem                    w plikach, ktorych nikt nie ruszal
rem    LICENSE      -- repozytorium jest publiczne na GPL-3.0
robocopy "%ZRODLO%." "%CEL%" *.py *.md *.txt *.sh *.cfg *.bat LICENSE .gitignore .gitattributes %OPCJE% /IS /IT %WYKLUCZ_KAT% %WYKLUCZ_PLIK%
set RC1=%ERRORLEVEL%
goto :po_kodzie

:bez_kodu
echo [1/2] kod... POMIJAM -- cel jest repozytorium git
echo        odswiezenie kodu tam:  git pull --ff-only
set RC1=0

:po_kodzie

rem --- PROBKI: nagrania, tylko rozniace sie. To 110 MB i nie zmieniaja
rem     sie czesto; kopiowanie za kazdym razem zajechaloby pendraka
rem     odczytami bez potrzeby. Nagran NIE MA w repozytorium, wiec ta
rem     czesc dziala tak samo, gdy cel jest repozytorium.
echo [2/2] probki (nagrania)...
if not exist "%ZRODLO%probki" goto :bez_probek
robocopy "%ZRODLO%probki" "%CEL%\probki" %OPCJE%
set RC2=%ERRORLEVEL%
goto :po_probkach

:bez_probek
echo   brak katalogu probki -- pomijam
set RC2=0

:po_probkach

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
