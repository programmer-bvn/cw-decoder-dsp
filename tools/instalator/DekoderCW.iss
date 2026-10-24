; Instalator Dekodera CW (24.10) — Inno Setup 7. Buduje budowa_exe.bat:
;   ISCC /DWersja=0.9.0 tools\instalator\DekoderCW.iss
; Wejście: dist\DekoderCW (PyInstaller). Wyjście: dist\DekoderCW_setup_<wersja>.exe.
; Instalacja bez uprawnień administratora (domyślnie dla bieżącego użytkownika):
; operator 24.10 — ma się instalować u ludzi nie-informatyków bez pytań o zgodę.
; Nowa wersja instaluje się na starą (ten sam AppId); ustawienia (rejestr HKCU)
; i dane (Dokumenty\DekoderCW: sesje, raporty, słownik osobisty) zostają.

#ifndef Wersja
  #define Wersja "0.0.0"
#endif
#define Nazwa "Dekoder CW SQ2BVN"
#define Exe "DekoderCW.exe"

[Setup]
AppId={{DA52DC2A-77EF-4A5B-A173-91ECCD3A787A}
AppName={#Nazwa}
AppVersion={#Wersja}
AppVerName={#Nazwa} {#Wersja}
AppPublisher=SQ2BVN
AppPublisherURL=https://github.com/programmer-bvn/cw-decoder-dsp
AppSupportURL=https://github.com/programmer-bvn/cw-decoder-dsp/issues
AppUpdatesURL=https://github.com/programmer-bvn/cw-decoder-dsp/releases
DefaultDirName={autopf}\DekoderCW
DefaultGroupName={#Nazwa}
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
LicenseFile=..\..\LICENSE
OutputDir=..\..\dist
OutputBaseFilename=DekoderCW_setup_{#Wersja}
SetupIconFile=dekoder_cw.ico
UninstallDisplayIcon={app}\{#Exe}
WizardStyle=modern
WizardImageFile=kreator.bmp
WizardSmallImageFile=kreator_maly.bmp
Compression=lzma2/ultra64
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=yes
DisableProgramGroupPage=yes

[Languages]
Name: "polski"; MessagesFile: "compiler:Languages\Polish.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "pulpit"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\..\dist\DekoderCW\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\CZYTAJ_DekoderCW.txt"; DestDir: "{app}"; DestName: "CZYTAJ.txt"; Flags: ignoreversion
Source: "..\README_EN_DekoderCW.txt"; DestDir: "{app}"; DestName: "README_EN.txt"; Flags: ignoreversion

[Icons]
Name: "{group}\{#Nazwa}"; Filename: "{app}\{#Exe}"
Name: "{group}\{cm:ProgramOnTheWeb,{#Nazwa}}"; Filename: "https://github.com/programmer-bvn/cw-decoder-dsp"
Name: "{group}\{cm:UninstallProgram,{#Nazwa}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#Nazwa}"; Filename: "{app}\{#Exe}"; Tasks: pulpit

[Run]
Filename: "{app}\{#Exe}"; Description: "{cm:LaunchProgram,{#Nazwa}}"; Flags: nowait postinstall skipifsilent
