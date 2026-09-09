#!/bin/bash
# =============================================================================
#  mount_pendrak.sh  --  ręczne montowanie nośnika w WSL2
#
#      sudo ./srodowisko/mount_pendrak.sh d          # D: -> /mnt/d
#      sudo ./srodowisko/mount_pendrak.sh i /mnt/d888
#
#  PO CO TO ISTNIEJE
#  WSL montuje dyski obecne w chwili startu. Nośnik wetknięty PÓŹNIEJ nie
#  pojawia się w /mnt/ i wszystko, co się do niego odwołuje, kończy się:
#
#      wsl: Failed to translate 'd:\AI_DSP'
#
#  Ustawienie [automount] enabled=true w /etc/wsl.conf tego NIE załatwia —
#  dotyczy startu, nie podłączenia w trakcie pracy.
#
#  LITERA DYSKU NIE JEST STAŁA. Ten sam pendrak bywa D: na jednej maszynie
#  i I: na innej, zależnie od tego, co już jest zajęte. Dlatego litera jest
#  argumentem, a skrypt nie zgaduje.
#
#  Alternatywa bez skryptu, z Windows: `wsl --mount` z parametrem --bare dla
#  dysków fizycznych. Dla nośników z systemem plików, który Windows już
#  obsługuje (exFAT, NTFS), drvfs jak niżej jest prostsze.
# =============================================================================

set -u

LITERA="${1:-}"
PUNKT="${2:-}"

if [ -z "$LITERA" ]; then
    echo "użycie: sudo $0 <litera> [punkt montowania]"
    echo "przykład: sudo $0 d"
    echo
    echo "Litery widoczne z Windows sprawdzisz przez:"
    echo "    powershell.exe -c 'Get-Volume | Format-Table DriveLetter,FileSystemLabel,Size'"
    exit 1
fi

# Jedna litera, bez dwukropka, małą literą.
LITERA="$(echo "$LITERA" | tr 'A-Z' 'a-z' | tr -d ':')"
case "$LITERA" in
    [a-z]) : ;;
    *) echo "BŁĄD: '$LITERA' nie jest literą dysku."; exit 1 ;;
esac

[ -n "$PUNKT" ] || PUNKT="/mnt/$LITERA"

if [ "$(id -u)" != "0" ]; then
    echo "BŁĄD: montowanie wymaga roota."
    echo "    sudo $0 $LITERA ${2:-}"
    exit 1
fi

# Już zamontowane? Nie dubluj — drugi mount na tym samym punkcie przesłania
# pierwszy i potem nie wiadomo, co się właściwie widzi.
if mountpoint -q "$PUNKT" 2>/dev/null; then
    echo "$PUNKT jest już zamontowany:"
    findmnt -no SOURCE,FSTYPE,OPTIONS "$PUNKT" | sed 's/^/    /'
    echo
    echo "Zawartość:"
    ls "$PUNKT" 2>/dev/null | head -8 | sed 's/^/    /'
    exit 0
fi

mkdir -p "$PUNKT"

# metadata: pozwala zapisywać prawa dostępu w systemie plików Windows.
# Bez tego wszystko jest 0777 i chmod +x na skryptach nie działa — a to
# potrafi zmarnować pół godziny na "Permission denied" bez powodu.
if mount -t drvfs "${LITERA}:" "$PUNKT" -o metadata,uid=1000,gid=1000; then
    echo "zamontowane: ${LITERA^^}: -> $PUNKT"
    findmnt -no SOURCE,FSTYPE,OPTIONS "$PUNKT" | sed 's/^/    /'
    echo
    ls "$PUNKT" 2>/dev/null | head -8 | sed 's/^/    /'
else
    RC=$?
    echo
    echo "NIE ZAMONTOWANO (kod $RC). Sprawdź po kolei:"
    echo "  1. czy Windows widzi ten dysk pod tą literą:"
    echo "     powershell.exe -c 'Get-Volume | Format-Table DriveLetter,FileSystemLabel'"
    echo "  2. czy nośnik nie został wysunięty"
    echo "  3. czy litera jest właściwa — ten sam pendrak bywa pod inną"
    exit $RC
fi
