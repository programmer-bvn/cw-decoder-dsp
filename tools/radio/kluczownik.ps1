# Tekst z pliku -> kluczownik IC-7300MK2 przez CI-V (polecenie 0x17), 18.10.
#
#   powershell -File tools\radio\kluczownik.ps1 probki\teksty\qso2_relacja.txt
#   powershell -File tools\radio\kluczownik.ps1 plik.txt -Sucho     (tylko bezpieczniki)
#   powershell -File tools\radio\kluczownik.ps1 plik.txt -Porty COM5 -Bauds 115200
#
# URUCHAMIA OPERATOR. Claude ten skrypt napisał, ale radia nie kluczuje.
#
# Bezpieczniki — każdy przerywa PRZED pierwszym znakiem:
#   - break-in (0x16 0x47) musi być WYŁĄCZONY (00). Wtedy kluczownik gra
#     tylko podsłuch, bez nadawania (instrukcja IC-7300: przy BK-IN OFF
#     w gotowości słychać podsłuch CW bez nadawania). 01 = pół-, 02 = pełny
#     break-in — wtedy radio by NADAWAŁO; skrypt odmawia.
#   - tryb (0x04) musi być CW albo CW-R (03 / 07);
#   - brak odpowiedzi radia = odmowa.
# DTR i RTS wyłączone PRZED otwarciem portu (w Icomach bywają PTT/kluczem).
# Ctrl+C albo błąd: w bloku finally idzie 0x17 0xFF (stop kluczownika).
#
# Format wg Hamlib (rigs/icom/icom.c, icom_send_morse): 0x17 + tekst ASCII,
# najwyżej 30 znaków w ramce; radio odrzuca ramkę (FA), gdy bufor pełny —
# wtedy ponawiamy. Pisownia projektu: "~" (SK) -> "^SK" (zapis Icom: znaki
# po "^" bez przerwy międzyznakowej); "(" = KN, "+" = AR idą bez zmian.
param(
    [Parameter(Mandatory = $true)][string]$Plik,
    [string[]]$Porty = @("COM5", "COM4"),
    [int[]]$Bauds = @(115200, 19200),
    [switch]$Sucho
)

$ErrorActionPreference = "Stop"

function Ramka($sp, [byte]$adr, [byte[]]$dane) {
    $sp.DiscardInBuffer()
    [byte[]]$r = @(0xFE, 0xFE, $adr, 0xE0) + $dane + @(0xFD)
    $sp.Write($r, 0, $r.Length)
    $koniec = (Get-Date).AddMilliseconds(600)
    $bufor = New-Object System.Collections.Generic.List[byte]
    while ((Get-Date) -lt $koniec) {
        Start-Sleep -Milliseconds 20
        $n = $sp.BytesToRead
        if ($n -gt 0) {
            $b = New-Object byte[] $n
            [void]$sp.Read($b, 0, $n)
            $bufor.AddRange($b)
            # odpowiedź do nas (E0) od radia, zakończona FD; echo naszej ramki pomijamy
            $s = ($bufor | ForEach-Object { '{0:X2}' -f $_ }) -join ' '
            if ($s -match "FE FE E0 ([0-9A-F]{2}) (.*?) ?FD") {
                $x = $matches[2].Trim()
                if ($x -eq "") { return ,([byte[]]@()) }
                $o = [byte[]]($x.Split(' ') | ForEach-Object { [Convert]::ToByte($_, 16) })
                # FB = OK, FA = odmowa; inaczej pierwszy bajt to echo polecenia
                # (04 03 01 -> tryb 03; 16 47 00 -> 47 00) - zdejmujemy go.
                if ($o[0] -eq 0xFB -or $o[0] -eq 0xFA) { return ,$o }
                if ($o.Count -eq 1) { return ,([byte[]]@()) }
                return ,([byte[]]$o[1..($o.Count - 1)])
            }
        }
    }
    return $null
}

$tekst = (Get-Content -Raw -Encoding UTF8 $Plik).Trim() -replace "\s+", " "
$tekst = $tekst.ToUpper().Replace("~", "^SK")
$dozwolone = '^[A-Z0-9 /?.,:''()=+"@^-]*$'
if ($tekst -notmatch $dozwolone) { throw "Znaki spoza kluczownika w: $tekst" }

# --- znajdź radio (jak civ_odczyt.ps1) ---
$sp = $null; $adr = 0; $sukces = $false
foreach ($p in $Porty) {
    foreach ($b in $Bauds) {
        $s = New-Object System.IO.Ports.SerialPort $p, $b, ([System.IO.Ports.Parity]::None), 8, ([System.IO.Ports.StopBits]::One)
        $s.DtrEnable = $false; $s.RtsEnable = $false
        $s.Handshake = [System.IO.Ports.Handshake]::None
        try { $s.Open() } catch { continue }
        foreach ($a in 0xB6, 0x94) {
            if ($null -ne (Ramka $s ([byte]$a) @([byte]0x03))) { $sp = $s; $adr = $a; break }
        }
        if ($sp) { break }
        $s.Close()
    }
    if ($sp) { break }
}
if (-not $sp) { throw "Radio nie odpowiada na CI-V (porty: $Porty) — nic nie wysłano." }
"radio: {0} {1} bod, adres 0x{2:X2}" -f $sp.PortName, $sp.BaudRate, $adr

try {
    # --- bezpieczniki ---
    $tryb = Ramka $sp ([byte]$adr) @([byte]0x04)
    if ($null -eq $tryb -or $tryb.Count -lt 1 -or ($tryb[0] -ne 0x03 -and $tryb[0] -ne 0x07)) {
        throw ("Radio nie jest w CW (tryb: {0}) — nic nie wysłano." -f (($tryb | ForEach-Object { '{0:X2}' -f $_ }) -join ' '))
    }
    $bk = Ramka $sp ([byte]$adr) @([byte]0x16, [byte]0x47)
    if ($null -eq $bk -or $bk.Count -lt 2 -or $bk[0] -ne 0x47) { throw "Nie udało się odczytać break-in — nic nie wysłano." }
    if ($bk[1] -ne 0x00) {
        throw ("Break-in WŁĄCZONY (0x{0:X2}: 01 pół, 02 pełny) — radio by NADAWAŁO. Wyłącz BK-IN i uruchom ponownie." -f $bk[1])
    }
    "tryb CW, break-in wyłączony — kluczownik zagra tylko podsłuch"
    "tekst ($($tekst.Length) znaków): $tekst"
    if ($Sucho) { "-Sucho: bezpieczniki przeszły, nic nie wysłano"; $sukces = $true; return }

    # --- wysyłanie w ramkach po 30 znaków ---
    for ($i = 0; $i -lt $tekst.Length; $i += 30) {
        $kawal = $tekst.Substring($i, [Math]::Min(30, $tekst.Length - $i))
        $dane = @([byte]0x17) + [System.Text.Encoding]::ASCII.GetBytes($kawal)
        $proby = 0
        while ($true) {
            $o = Ramka $sp ([byte]$adr) $dane
            if ($null -ne $o -and $o.Count -ge 1 -and $o[0] -eq 0xFB) { break }   # OK
            if (++$proby -gt 600) { throw "Kluczownik nie przyjmuje tekstu (ostatnia odpowiedź: $o)." }
            Start-Sleep -Milliseconds 200                                           # FA = bufor pełny
        }
        "  wysłane: $kawal"
    }
    "gotowe - kluczownik dogrywa bufor"
    $sukces = $true
}
finally {
    if ($sp.IsOpen) {
        # Przerwane (Ctrl+C, błąd) w trakcie wysyłania: stop kluczownika.
        if (-not $sukces) { [void](Ramka $sp ([byte]$adr) @([byte]0x17, [byte]0xFF)) }
        $sp.Close()
    }
}
