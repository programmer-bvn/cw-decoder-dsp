@echo off
rem ===========================================================================
rem  noc.bat  --  jeden Enter z Windows: kod na HDD, noc w WSL, wyniki wracaja
rem
rem  Uzycie:
rem      noc.bat                             cel domyslny, ustawienia domyslne
rem      noc.bat C:\AI_DSP                   wlasny katalog roboczy na HDD
rem      noc.bat C:\AI_DSP 200000 40         + argumenty dla noc.sh
rem
rem  PO CO TO ISTNIEJE
rem  Praca idzie w przerwach miedzy inna robota: wtykasz pendraka, klepiesz
rem  Enter, idziesz do zadan fizycznych, a wyniki ogladasz jak noc zapadnie.
rem  Bez tego pliku trzeba: uruchomic na_hdd, wejsc do WSL, przejsc do
rem  katalogu, odpalic noc.sh, a rano jeszcze zgrac wyniki. Piec operacji
rem  zamiast jednej, kazda z wlasna okazja do pomylki.
rem
rem  CO ROBI, PO KOLEI
rem      1. kopiuje kod z pendraka na HDD           (na_hdd.bat, tryb auto)
rem      2. uruchamia noc.sh w WSL na kopii z HDD
rem      3. zgrywa wyniki z HDD na pendraka         (z_hdd.bat, tryb auto)
rem      4. wypisuje out\RANO.txt na ekran
rem
rem  DLACZEGO PRACA IDZIE NA HDD, A NIE NA PENDRAKU
rem  exFAT bez wear-levelingu ma rzedu 50 tys. cykli zapisu. Trening pisze
rem  tysiace malych plikow, wiec pendrak by tego nie przezyl. Na pendraka
rem  wraca tylko wynik: modele, logi, pomiary, paczka git.
rem
rem  TRYB AUTO w obu skryptach kopiujacych wylacza pytanie "Kopiowac? [T/N]".
rem  Z reki pytanie zostaje -- to jedyny moment, w ktorym mozna zauwazyc zly
rem  cel PRZED zapisem. Tutaj nie ma komu odpowiedziec.
rem
rem  LOGI. Faza kopiowania idzie do pliku na PENDRAKU, zeby zostala nawet
rem  wtedy, gdy HDD padnie. Faza WSL loguje sie sama -- noc.sh trzyma
rem  wszystko w out\noc_*.log przez tee, a przy awarii dosypuje czarna
rem  skrzynke ze stanem maszyny.
rem ===========================================================================

setlocal

set ZRODLO=%~dp0
set CEL=%~1
if "%CEL%"=="" set CEL=c:\Users\ADMIN\PyCharmMiscProject\FT1-TF2-convert\CLAU

rem Uzytkownik WSL. Konfiguracja karty na tej maszynie powstala jako root
rem (symlinki do /usr/lib/wsl/lib), wiec taki jest domyslny.
if "%WSLUSER%"=="" set WSLUSER=root

rem Argumenty dla noc.sh: wszystko po pierwszym.
set ARGS=
if not "%~2"=="" set ARGS=%~2 %~3 %~4 %~5 %~6 %~7

rem  Znacznik czasu przez PowerShell, nie przez %DATE%.
rem  ZMIERZONE: na tej maszynie %DATE% daje "09.09.2026", wiec podzial
rem  na tokeny dawal "202609" -- bez dnia i w zlej kolejnosci. Format
rem  %DATE% zalezy od ustawien regionalnych, a nazwa pliku z logiem nie
rem  moze od nich zalezec. Get-Date -Format jest jednoznaczny.
set STEMPEL=
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmm"') do set STEMPEL=%%i
if "%STEMPEL%"=="" set STEMPEL=bez_daty
if not exist "%ZRODLO%out" mkdir "%ZRODLO%out"
set LOGB=%ZRODLO%out\noc_bat_%STEMPEL%.log

echo ===========================================================================
echo  NOC TRENINGOWA  --  jeden Enter
echo ===========================================================================
echo  pendrak: %ZRODLO%
echo  HDD:     %CEL%
echo  WSL:     uzytkownik %WSLUSER%
if not "%ARGS%"=="" echo  noc.sh:  %ARGS%
echo  log:     %LOGB%
echo.

rem --- czy jestesmy tam, gdzie myslimy ------------------------------------
if not exist "%ZRODLO%dsp\config.py" (
    echo BLAD: nie widze %ZRODLO%dsp\config.py
    echo       Ten plik ma lezec w katalogu projektu na pendraku.
    goto :koniec_blad
)
if not exist "%ZRODLO%noc.sh" (
    echo BLAD: nie widze %ZRODLO%noc.sh
    goto :koniec_blad
)

rem --- czy WSL w ogole jest -----------------------------------------------
where wsl.exe >nul 2>&1
if errorlevel 1 (
    echo BLAD: nie znalazlem wsl.exe w PATH.
    echo       Bez WSL nie ma na czym trenowac -- karta jest widziana
    echo       tylko z Linuksa.
    goto :koniec_blad
)

rem --- czy WSL ma zainstalowana dystrybucje -------------------------------
rem  wsl.exe istnieje w System32 nawet wtedy, gdy zadna dystrybucja nie jest
rem  zainstalowana -- sam plik nic nie dowodzi. Bez tego sprawdzenia proba
rem  uruchomienia noc.sh wypluwa cala pomoc wsl w UTF-16, dwa razy, i nie
rem  da sie tego przeczytac. Sonda: uruchom "true" i patrz na kod wyjscia.
rem  Pomoc leci na STDOUT, nie na stderr, wiec trzeba wygluszyc oba.
wsl.exe -e true >nul 2>&1
if errorlevel 1 (
    echo BLAD: WSL nie ma zainstalowanej dystrybucji albo nie odpowiada.
    echo.
    echo       Sprawdzenie:   wsl -l -v
    echo       Instalacja:    wsl --install -d Ubuntu
    echo.
    echo       Trening idzie tylko z Linuksa -- karta nie jest widziana
    echo       z samego Windows.
    goto :koniec_blad
)

rem === 1. KOD NA HDD =====================================================
echo [1/4] kod na HDD...
call "%ZRODLO%na_hdd.bat" "%CEL%" auto >> "%LOGB%" 2>&1
if errorlevel 1 (
    echo       NIE UDALO SIE. Szczegoly w %LOGB%
    goto :koniec_blad
)
echo       gotowe
echo.

rem === 2. NOC W WSL ======================================================
rem  --cd przyjmuje sciezke Windows i sam ja tlumaczy na /mnt/c/...
rem  Gdyby ta wersja WSL go nie znala, ponizej jest wariant z wslpath.
echo [2/4] noc.sh w WSL -- to potrwa; wszystko idzie do out\noc_*.log
echo.
wsl.exe --cd "%CEL%" -u %WSLUSER% -- ./noc.sh %ARGS%
set RC=%ERRORLEVEL%

if %RC% EQU 0 goto :po_wsl
if %RC% NEQ 1 goto :po_wsl
rem Kod 1 moze znaczyc "noc.sh przerwal etapem 1-3" ALBO "wsl nie zna --cd".
rem Rozroznienie: czy powstal jakikolwiek log nocy na HDD.
if exist "%CEL%\out\noc_*.log" goto :po_wsl
echo.
echo  --cd nie zadzialal -- probuje wariantem z wslpath
wsl.exe -u %WSLUSER% -- bash -lc "cd \"$(wslpath -a '%CEL%')\" && ./noc.sh %ARGS%"
set RC=%ERRORLEVEL%

:po_wsl
echo.
if %RC% EQU 0 (
    echo       noc.sh zakonczyl sie bez bledu
) else (
    echo       noc.sh zakonczyl sie kodem %RC%
    echo       Wyniki i tak zgrywam -- logi powiedza, co poszlo nie tak.
)
echo.

rem === 3. WYNIKI NA PENDRAKA =============================================
rem  Bezwarunkowo, takze po bledzie: logi awarii i czarna skrzynka sa
rem  wtedy najcenniejsza rzecza na tym dysku.
echo [3/4] wyniki na pendraka...
call "%ZRODLO%z_hdd.bat" "%CEL%" auto >> "%LOGB%" 2>&1
if errorlevel 1 (
    echo       kopiowanie zglosilo blad -- patrz %LOGB%
) else (
    echo       gotowe
)
echo.

rem === 4. CO WYSZLO ======================================================
echo [4/4] podsumowanie
echo.
if exist "%ZRODLO%out\RANO.txt" (
    type "%ZRODLO%out\RANO.txt"
) else (
    echo  Nie ma out\RANO.txt -- noc.sh nie doszedl do podsumowania.
    echo  Zajrzyj do najnowszego:
    dir /b /o-d "%ZRODLO%out\noc_*.log" 2>nul
)
echo.
echo ===========================================================================
echo  Pendrak mozna wyjac. Log tego skryptu: %LOGB%
echo ===========================================================================
pause
endlocal
exit /b 0

:koniec_blad
echo.
echo ===========================================================================
echo  PRZERWANE -- nic nie zostalo policzone.
echo ===========================================================================
pause
endlocal
exit /b 1
