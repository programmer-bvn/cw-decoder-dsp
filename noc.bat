@echo off
rem ===========================================================================
rem  noc.bat  --  jeden Enter z Windows: noc w WSL, wyniki na GitHub
rem
rem  Uzycie (z katalogu projektu albo podwojnym kliknieciem):
rem      noc.bat                                  ustawienia domyslne noc.sh
rem      noc.bat 200000 20 256 - 40 3000 50       argumenty dla noc.sh
rem
rem  PUSTY ARGUMENT = KRESKA "-", nie "". Pusty argument znika po drodze
rem  przez cmd i wsl.exe i przesuwa wszystkie nastepne o jedno miejsce.
rem
rem  CO ROBI, PO KOLEI
rem      1. uruchamia noc.sh w WSL w TYM katalogu (tu lezy repozytorium)
rem      2. git push -- noc.sh zrobil commit wynikow, wypycha go Windows
rem      3. wypisuje out\RANO.txt na ekran
rem
rem  DO 08.10 bylo tu jeszcze pobieranie kodu z GitHuba albo z pendraka,
rem  kopiowanie na HDD (na_hdd.bat) i zgrywanie wynikow z powrotem
rem  (z_hdd.bat). Kod i trening byly na dwoch maszynach. Teraz jest jedna:
rem  kod, git i wyniki leza w tym samym katalogu na SSD W:, wiec nie ma
rem  czego wozic. Jaki kod sie wykonal, mowi paszport w noc.sh (commit
rem  i out\noc_*_kod.diff).
rem
rem  DLACZEGO PUSH STAD, A NIE Z noc.sh. Poswiadczenia GitHuba ustawia
rem  operator po stronie Windows (Credential Manager). W WSL ich nie ma --
rem  tam lezalyby czystym tekstem. GCM_INTERACTIVE=never: bez poswiadczen
rem  push ma sie wywrocic i powiedziec, a nie czekac do rana na okno
rem  logowania.
rem ===========================================================================

setlocal

rem  "%~dp0." zamiast "%~dp0": sciezka konczy sie ukosnikiem, a ukosnik
rem  przed cudzyslowem cmd i wsl.exe biora za znak ucieczki.
set KAT=%~dp0.
cd /d "%KAT%"

rem Uzytkownik WSL. Konfiguracja karty na tej maszynie powstala jako root
rem (symlinki do /usr/lib/wsl/lib), a ~/venv_gpu jest roota. Domyslny
rem uzytkownik WSL to bvn -- stad jawne -u.
if "%WSLUSER%"=="" set WSLUSER=root

rem Argumenty dla noc.sh: WSZYSTKIE. Petla z shift zbiera dowolnie wiele
rem (do 08.10 bylo %~2..%~7, czyli najwyzej szesc, a noc.sh ma dziewiec).
rem Konczy sie na pierwszym pustym -- stad kreska "-".
set ARGS=
:zbierz_arg
if "%~1"=="" goto :zebrane_arg
set ARGS=%ARGS% %~1
shift
goto :zbierz_arg
:zebrane_arg
if defined ARGS set ARGS=%ARGS:~1%

rem  Znacznik czasu przez PowerShell, nie przez %DATE%.
rem  ZMIERZONE: %DATE% daje tu "09.09.2026", a podzial na tokeny dawal
rem  "202609" -- bez dnia i w zlej kolejnosci. Format %DATE% zalezy od
rem  ustawien regionalnych, a nazwa pliku z logiem nie moze od nich zalezec.
set STEMPEL=
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmm"') do set STEMPEL=%%i
if "%STEMPEL%"=="" set STEMPEL=bez_daty
if not exist "out" mkdir "out"
set LOGB=%KAT%\out\noc_bat_%STEMPEL%.log

echo ===========================================================================
echo  NOC TRENINGOWA  --  jeden Enter
echo ===========================================================================
echo  katalog: %KAT%
echo  WSL:     uzytkownik %WSLUSER%
if not "%ARGS%"=="" echo  noc.sh:  %ARGS%
echo  log:     %LOGB%
echo.

rem --- czy jestesmy tam, gdzie myslimy ------------------------------------
if not exist "noc.sh" (
    echo BLAD: nie widze noc.sh w %KAT%
    echo       noc.bat ma lezec w katalogu projektu, obok noc.sh.
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
rem  da sie tego przeczytac. Pomoc leci na STDOUT, wiec wygluszone oba.
wsl.exe -e true >nul 2>&1
if errorlevel 1 (
    echo BLAD: WSL nie ma zainstalowanej dystrybucji albo nie odpowiada.
    echo.
    echo       Sprawdzenie:   wsl -l -v
    echo       Instalacja:    wsl --install -d Ubuntu
    goto :koniec_blad
)

rem === 1. NOC W WSL ======================================================
rem  --cd przyjmuje sciezke Windows i sam ja tlumaczy na /mnt/...
echo [1/3] noc.sh w WSL -- to potrwa; wszystko idzie do out\noc_*.log
echo.
wsl.exe --cd "%KAT%" -u %WSLUSER% -- ./noc.sh %ARGS%
set RC=%ERRORLEVEL%
echo.
if %RC% EQU 0 (
    echo       noc.sh zakonczyl sie bez bledu
) else (
    echo       noc.sh zakonczyl sie kodem %RC% -- logi w out\ powiedza dlaczego
)
echo.

rem === 2. NA GITHUB ======================================================
rem  Bezwarunkowo, takze po bledzie: noc.sh commituje logi awarii tak samo
rem  jak wyniki, a te sa wtedy najcenniejsze.
echo [2/3] git push...
where git.exe >nul 2>&1
if errorlevel 1 (
    echo       brak git.exe w PATH Windows -- pomijam; wyniki sa w commicie
    goto :rano
)
set GCM_INTERACTIVE=never
set GIT_TERMINAL_PROMPT=0
git push origin HEAD >> "%LOGB%" 2>&1
if errorlevel 1 (
    echo       NIE przeszedl -- nic nie zginelo, commit jest lokalnie.
    echo       Powod w %LOGB%. Recznie:  git push origin HEAD
) else (
    echo       wypchniete
)
echo.

:rano
rem === 3. CO WYSZLO ======================================================
echo [3/3] podsumowanie
echo.
if exist "out\RANO.txt" (
    type "out\RANO.txt"
) else (
    echo  Nie ma out\RANO.txt -- noc.sh nie doszedl do podsumowania.
    echo  Zajrzyj do najnowszego:
    dir /b /o-d "out\noc_*.log" 2>nul
)
echo.
echo ===========================================================================
pause
rem  W jednej linii: po samym endlocal zmiennej RC juz nie ma.
endlocal & exit /b %RC%

:koniec_blad
echo.
echo ===========================================================================
echo  PRZERWANE -- nic nie zostalo policzone.
echo ===========================================================================
pause
endlocal
exit /b 1
