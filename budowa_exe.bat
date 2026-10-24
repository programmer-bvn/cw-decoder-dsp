@echo off
rem Budowa aplikacji Dekoder CW (dla innych operatorow, np. SP2MSA) - tools\aplikacja.py.
rem Bez TensorFlow: siec zboczy w numpy (dsp\siec_numpy.py), wagi runs\zbocza_5\siec.npz.
rem Wynik: dist\DekoderCW\DekoderCW.exe (caly katalog do skopiowania) i dist\DekoderCW.zip.
cd /d "%~dp0"
set TF_CPP_MIN_LOG_LEVEL=2
rem wagi z aktualnego modelu (sprawdza zgodnosc z TensorFlow)
python -m dsp.siec_numpy runs\zbocza_5\best.keras runs\zbocza_5\siec.npz || goto blad
for /f "delims=" %%v in ('git describe --always --tags --dirty') do set WERSJA=%%v
>wersja.txt echo %WERSJA% (zbudowano %DATE%)
python -m PyInstaller --noconfirm --clean --name DekoderCW --windowed --onedir --paths . ^
  --add-data "runs\zbocza_5\siec.npz;." --add-data "tools\radio\civ_odczyt.ps1;tools\radio" --add-data "wersja.txt;." ^
  --exclude-module tensorflow --exclude-module keras --exclude-module tf_keras --exclude-module matplotlib ^
  --exclude-module torch --exclude-module IPython --exclude-module pandas --exclude-module h5py ^
  tools\aplikacja.py || goto blad
copy /y tools\CZYTAJ_DekoderCW.txt dist\DekoderCW\CZYTAJ.txt >nul || goto blad
powershell -NoProfile -Command "Compress-Archive -Force -Path dist\DekoderCW -DestinationPath dist\DekoderCW.zip" || goto blad
echo.
echo GOTOWE: dist\DekoderCW.zip
pause
exit /b 0
:blad
echo.
echo BUDOWA NIE POWIODLA SIE - komunikat wyzej
pause
exit /b 1
