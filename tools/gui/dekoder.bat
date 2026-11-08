@echo off
rem Dekoder CW w oknie Qt: Otworz plik... -> odczyt WAV, widok sygnalu z kursorami.
cd /d "%~dp0..\.."
python -m tools.gui.dekoder %*
if errorlevel 1 pause
