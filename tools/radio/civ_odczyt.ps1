# Odczyt stanu IC-7300MK2 przez CI-V. TYLKO polecenia odczytu: 03 (częstotliwość)
# i 04 (tryb + filtr). DTR i RTS jawnie wyłączone PRZED otwarciem portu —
# w Icomach mogą być przypisane do PTT (USB SEND) albo kluczowania CW.
param([string[]]$Porty = @("COM5", "COM4"), [int[]]$Bauds = @(115200, 19200))

function Zapytaj($sp, [byte]$adr, [byte]$cmd) {
    $sp.DiscardInBuffer()
    [byte[]]$ramka = 0xFE, 0xFE, $adr, 0xE0, $cmd, 0xFD
    $sp.Write($ramka, 0, $ramka.Length)
    Start-Sleep -Milliseconds 250
    $n = $sp.BytesToRead
    if ($n -le 0) { return $null }
    $buf = New-Object byte[] $n
    [void]$sp.Read($buf, 0, $n)
    return $buf
}

foreach ($port in $Porty) {
    foreach ($baud in $Bauds) {
        $sp = New-Object System.IO.Ports.SerialPort $port, $baud, ([System.IO.Ports.Parity]::None), 8, ([System.IO.Ports.StopBits]::One)
        $sp.DtrEnable = $false
        $sp.RtsEnable = $false
        $sp.Handshake = [System.IO.Ports.Handshake]::None
        $sp.ReadTimeout = 500
        try { $sp.Open() } catch { "$port ${baud}: nie da sie otworzyc ($($_.Exception.Message))"; continue }
        try {
            foreach ($adr in 0xB6, 0x94, 0x00) {
                $o = Zapytaj $sp ([byte]$adr) 0x03
                if ($o) {
                    "$port $baud adres 0x{0:X2}: {1}" -f $adr, (($o | ForEach-Object { '{0:X2}' -f $_ }) -join ' ')
                    $m = Zapytaj $sp ([byte]$adr) 0x04
                    if ($m) { "  tryb: " + (($m | ForEach-Object { '{0:X2}' -f $_ }) -join ' ') }
                    return
                }
            }
            "$port ${baud}: brak odpowiedzi"
        } finally { $sp.Close() }
    }
}
