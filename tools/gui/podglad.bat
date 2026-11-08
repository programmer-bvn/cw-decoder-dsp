@echo off
rem Podglad okna Qt z ruchomymi atrapami (po zapisaniu w Designerze).
cd /d "%~dp0..\.."
python -m tools.gui.podglad
if errorlevel 1 pause
