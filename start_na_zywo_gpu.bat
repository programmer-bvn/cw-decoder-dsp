@echo off
rem Odczyt CW na zywo z karta graficzna: dekoder w WSL (GPU), karta dzwiekowa
rem IC-7300 i CI-V w Windows (WSL2 ich nie widzi). Most przez localhost:7355.
rem Sesja (dzwiek, opis, dziennik) zapisywana przez dekoder do probki\sesje.
rem Tekst pojawia sie w oknie "Dekoder CW (WSL)". Tylko odbior.
rem Karta: sama szuka "USB Audio" (WASAPI); inna: start_na_zywo_gpu.bat --device NR
rem Lista kart:          python -m tools.mic2wav --list
rem Proba bez radia:     start_na_zywo_gpu.bat --wav probki\kluczownik\NAGRANIE.wav
cd /d "%~dp0"
start "Dekoder CW (WSL)" wsl -u root bash -c "cd /mnt/w/PyCharmMiscProject/cw-decoder-dsp && set +u && source srodowisko/rtx3050_setenv.sh >/dev/null 2>&1 && TF_CPP_MIN_LOG_LEVEL=2 python -u -m tools.na_zywo serwer; echo; read -p 'Enter zamyka okno '"
set TF_CPP_MIN_LOG_LEVEL=2
set TF_ENABLE_ONEDNN_OPTS=0
rem most czeka do 2 min, az dekoder w WSL wstanie (TensorFlow laduje sie kilka s)
if "%~1"=="" (
    python -m tools.na_zywo most --civ
) else (
    python -m tools.na_zywo most --civ %*
)
pause
