@echo off
rem Dekoder CW w oknie Qt: Sluchaj (na zywo, zrodlo z zakladki Zrodlo) albo Otworz plik...
cd /d "%~dp0..\.."
python -m tools.gui.dekoder %*
if errorlevel 1 pause
