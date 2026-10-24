@echo off
rem Budowa Dekodera CW dla innych operatorow (np. SP2MSA): exe + instalator.
rem 1) wagi sieci w numpy z aktualnego modelu (sprawdza zgodnosc z TensorFlow),
rem 2) PyInstaller -> dist\DekoderCW (bez TensorFlow),
rem 3) Inno Setup 7 -> dist\DekoderCW_setup_<wersja>.exe (to sie wysyla ludziom),
rem    oraz dist\DekoderCW.zip (wersja bez instalacji).
rem Wersja programu: WERSJA_APLIKACJI w tools\aplikacja.py.
cd /d "%~dp0"
set TF_CPP_MIN_LOG_LEVEL=2
set ISCC=C:\Program Files\Inno Setup 7\ISCC.exe
python -m dsp.siec_numpy runs\zbocza_5\best.keras runs\zbocza_5\siec.npz || goto blad
for /f "delims=" %%v in ('python -c "from tools.aplikacja import WERSJA_APLIKACJI as w; print(w)"') do set WERSJA=%%v
for /f "delims=" %%v in ('git describe --always --tags --dirty') do set GIT=%%v
>wersja.txt echo %GIT%, zbudowano %DATE%
echo Wersja %WERSJA% (%GIT%)
python -m PyInstaller --noconfirm --clean --name DekoderCW --windowed --onedir --paths . ^
  --icon tools\instalator\dekoder_cw.ico ^
  --add-data "runs\zbocza_5\siec.npz;." --add-data "tools\radio\civ_odczyt.ps1;tools\radio" ^
  --add-data "wersja.txt;." --add-data "tools\instalator\dekoder_cw.ico;." ^
  --exclude-module tensorflow --exclude-module keras --exclude-module tf_keras --exclude-module matplotlib ^
  --exclude-module torch --exclude-module IPython --exclude-module pandas --exclude-module h5py ^
  tools\aplikacja.py || goto blad
copy /y tools\CZYTAJ_DekoderCW.txt dist\DekoderCW\CZYTAJ.txt >nul || goto blad
copy /y tools\README_EN_DekoderCW.txt dist\DekoderCW\README_EN.txt >nul || goto blad
powershell -NoProfile -Command "Compress-Archive -Force -Path dist\DekoderCW -DestinationPath dist\DekoderCW.zip" || goto blad
if not exist "%ISCC%" (
    echo Brak Inno Setup 7 ^(%ISCC%^) - instalator pominiety, jest tylko dist\DekoderCW.zip
    goto koniec
)
"%ISCC%" /Q /DWersja=%WERSJA% tools\instalator\DekoderCW.iss || goto blad
echo.
echo GOTOWE: dist\DekoderCW_setup_%WERSJA%.exe  (oraz dist\DekoderCW.zip)
:koniec
pause
exit /b 0
:blad
echo.
echo BUDOWA NIE POWIODLA SIE - komunikat wyzej
pause
exit /b 1
