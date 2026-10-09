# Nagranie pustego pasma: CI-V (tylko odczyt) przed i po, potem mic2wav.
# Nazwa pliku i opis biora filtr z CI-V, nie z tego, co ktos napisal.
param([int]$Sekundy = 150)
$civ = Join-Path $PSScriptRoot "civ_odczyt.ps1"
$kat = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

function Stan {
    $o = & $civ -Porty COM5 -Bauds 115200
    $f = ($o | Select-String 'adres').ToString() -replace '.*03 ((?:[0-9A-F]{2} ){5})FD.*', '$1'
    $b = $f.Trim().Split(' ') | ForEach-Object { [Convert]::ToInt32($_, 16) }
    $hz = 0; $mn = 1
    foreach ($x in $b) { $hz += ($x -band 0x0F) * $mn; $mn *= 10; $hz += (($x -shr 4) -band 0x0F) * $mn; $mn *= 10 }
    $t = ($o | Select-String 'tryb').ToString() -replace '.*04 ([0-9A-F]{2}) ([0-9A-F]{2}) FD.*', '$1 $2'
    $tryb, $fil = $t.Split(' ')
    [pscustomobject]@{ Hz = $hz; Tryb = $tryb; Fil = [int]$fil }
}

$przed = Stan
$stempel = Get-Date -Format "yyyyMMdd_HHmm"
$nazwa = "szum_ic7300_usb_{0}kHz_FIL{1}_{2}" -f [int]($przed.Hz / 1000), $przed.Fil, $stempel
"przed: {0} Hz, tryb {1}, FIL{2}  ->  {3}" -f $przed.Hz, $przed.Tryb, $przed.Fil, $nazwa
$env:PYTHONIOENCODING = "utf-8"
Push-Location $kat
python -m tools.mic2wav --device 15 --seconds $Sekundy --surowe --out "probki/szum/$nazwa.wav" 2>&1 |
    ForEach-Object { $_ -split "`r" } | Select-String 'nagrane|zapisano|orygina|UWAGA'
Pop-Location
$po = Stan
"po:    {0} Hz, tryb {1}, FIL{2}" -f $po.Hz, $po.Tryb, $po.Fil
if ($po.Hz -ne $przed.Hz -or $po.Fil -ne $przed.Fil) { "UWAGA: stan radia zmienil sie w trakcie nagrania" }
@"
# SZUM PASMA - bez sygnalu; do banku tla generatora.
czestotliwosc: $($przed.Hz) Hz (CI-V przed), $($po.Hz) Hz (po)
tryb/filtr:    $($przed.Tryb) / FIL$($przed.Fil) (CI-V przed), FIL$($po.Fil) (po)
tor:           USB audio z IC-7300MK2, AF Output Level 50%, wejscie Windows 50%
czas:          $stempel, $Sekundy s
"@ | Set-Content -Encoding utf8 "$kat\probki\szum\$nazwa.txt"
