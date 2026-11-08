@echo off
rem Qt Designer z okno.ui Dekodera CW. Nazw elementow (objectName) nie zmieniac.
cd /d "%~dp0"
for /f "delims=" %%d in ('python -c "import PySide6,os;print(os.path.dirname(PySide6.__file__))"') do set PS6=%%d
start "" "%PS6%\designer.exe" "%~dp0okno.ui"
