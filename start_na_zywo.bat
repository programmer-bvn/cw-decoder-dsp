@echo off
rem Odczyt CW na zywo z IC-7300 (USB audio + CI-V, tylko odbior) - tools\na_zywo.py.
rem Windows, nie WSL: WSL2 nie widzi kart dzwiekowych ani portow COM Windows.
rem Liczy na samym procesorze; 21.10 operator: nadazalo (HISTORIA "Odczyt na zywo").
rem Inna karta niz 15:   start_na_zywo.bat --device NR
rem Lista kart:          python -m tools.mic2wav --list
rem Ctrl+C konczy. Sesja (dzwiek, opis, dziennik) idzie do probki\sesje;
rem bez zapisu:        start_na_zywo.bat --device 15 --bez-zapisu
cd /d "%~dp0"
rem bez komunikatow TensorFlow (I0000..., oneDNN) - zaslanialy tekst
set TF_CPP_MIN_LOG_LEVEL=2
set TF_ENABLE_ONEDNN_OPTS=0
if "%~1"=="" (
    python -m tools.na_zywo karta --device 15 --civ
) else (
    python -m tools.na_zywo karta --civ %*
)
pause
