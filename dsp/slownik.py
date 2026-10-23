"""SŁOWNIK — spacje rozstrzygane słownikiem krótkofalarskim (24.10).

Ręka (HISTORIA „Przerwy dopasowane do nadawcy"): przerwa międzyliterowa
operatora mediana 3,4 u (P10-P90 2,6-4,3), międzywyrazowa 6,9 u (5,4-8,8) —
rozkłady zachodzą na siebie, przerwy ~5 u z samego czasu nie rozstrzygnie
się. Odczyt: „CQCQCQDESQ2BVN", „MYNAME", „TKSF ER", „SP2 MSA".

Decyzja operatora (24.10): słowa znalezione w słowniku (albo pasujące do
wzoru znaku wywoławczego, raportu RST) ZIELONE, spacje wokół nich poprawione;
LITER NIE POPRAWIAMY — inaczej nie będzie widać, czy dekoder pracuje.

Spację wolno przestawić tylko tam, gdzie przerwa była niejednoznaczna:
rozdzielić — przerwa >= ROZDZIEL_U, skleić — przerwa <= SKLEJ_U.
Podział: programowanie dynamiczne po literach; słowo znane zyskuje tyle,
ile ma liter (>= 2), każda przestawiona spacja kosztuje KARA_ZMIANY.
"""

from __future__ import annotations

import re
from pathlib import Path

# Zmierzone 24.10 (114 nagrań p_zb5, całe słowa): rozdziel >= 3,0 / sklej <= 6,0 —
# ręka 25 -> 38/45, pasmo + mikrofon 45 -> 51/62, kluczownik 362 -> 363/369,
# zaszumione 470 + 434 i QRM 141 bez zmian; sklej <= 7: kluczownik 359 (-4).
ROZDZIEL_U = 3.0
SKLEJ_U = 6.0
KARA_ZMIANY = 0.3
NAJDLUZSZE = 14
DLUGIE = 4                  # znane słowo „długie”: może samo uzasadnić rozdzielenie
# K, R — same w sobie słowem (koniec nadania, potwierdzenie): „PSEK” -> „PSE K”
POJEDYNCZE = {"K", "R"}
ZYSK_POJEDYNCZEJ = 0.5

# Q-kody, skróty i słowa łączności CW — z praktyki krótkofalarskiej, NIE
# z tekstów nagrań testowych (inaczej miara całych słów byłaby oszukana).
# EL, OB, OT, GG, AA, AB wróciły ze SLANG (decyzja operatora) — bywają zielone
# w śmieciach przed pierwszą stacją („EL”, 24.10); w pomiarze bez zmian.
OGOLNY = """
QRA QRB QRG QRH QRI QRK QRL QRM QRN QRO QRP QRQ QRS QRT QRU QRV QRX QRZ
QSA QSB QSD QSK QSL QSM QSO QSP QST QSV QSX QSY QTC QTH QTR QRR
CQ DE PSE KN SK BK AR AS BT TU TNX TKS THX TX RX TRX RIG ANT PWR WX
OM YL XYL OP NAME QTH RST UR MY HR HW HPE CPY CPI AGN RPT SRI NIL DR
GM GA GE GN GD GB GL GUD FB UFB VY NW SO ES ND FER FOR THE AND IS AT TO
IN ON OF ALL ABT BCNU CUL CUAGN CU 73 88 55 DX DXCC IOTA SOTA POTA
WKD WKG WRK BURO VIA DIRECT LOTW EQSL MGR QSLL QSLS
NR AGE YRS ELE DIPOLE VERTICAL YAGI BEAM LW GP LOOP WIRE
WATTS WATT KW HI HIHI OK SURE TEST CONTEST CW SSB FT8 RTTY
SUNNY CLOUDY RAIN RAINY SNOW WINDY FOG FOGGY COLD WARM HOT TEMP
NICE GOOD FINE BAD VERY MANY THANKS CALL REPORT SIGS SIG STRONG WEAK
BAND BANDS CONDX CONDITIONS LOCAL SKED MIN HOURS NOW HERE THERE
MSG
"""
# Slang krótkofalarski — tabela z Wikipedii („Slang krótkofalarski”), wkleił
# operator 24.10; same skróty (pierwsza kolumna, warianty rozdzielone).
SLANG = """
AA AB ABT AC ADR AER AF AGN AHD ALNG AM AMP ANI ANS ANT AS AT AUD AUSSIE AVC
AWH AWS BA BCI BCL BCUS BECUS BD BEAM BEST BF BFRE B4 BFO BK BLV BOX BT BTR
BTU BTWN BU BUG BUK BURO BUREAU BY CALL CB CC CFM CL CLD CLG CMG CN CNT CO
CONGRATS CP CPY CPI CQ CRD CTS CUAGN CUA CUD CUDNT CUL CUM CW DWN DBL DC DCN
DE DNT DR DUNNO DX EL ELS EM ENAF EU EX FAN FB FD FR FOR FEW FM FONE FRD FREQ
FQ FRM FROM FWD FYI GA GANG GB GD GE GEN GESS GG GNG GL GLD GM GMT GN GND GUD
GV GVG HAM HF HLO HP HPI HQ HR HRD HT HUM HV HVE HVI HVNT HW HWS HWSAT IF IM
IN INFO INPT IRPT IS KC KHZ KN KNW KW KY LAT LDSKPR LS LF LFT LIC LIS LKG LOG
LONG LOTSA LP LR LSN LSTN LT LTR LW MA MAG MC MF MFD MHZ MIKE MILL MILS MIN
MISD MNI MNL MO MOD MSG MST MTR MY MI NG NIL NITE NM NN NR NVR NW OB OC ODX
OFT OK OM ONLI OP OPS OSC OT OUTPUT PA PART PDC PER PIRATE PM POB PR PSE PSED
PT PWR RCD RCTF RX RCVR RDI RIG RITE RMKS ROK RST RPRT RPT SA SEC SET SIG SIGS
SIGN SK SN SOS SPK SRI SSB STDI STN SUM SURE SVC SW SWL SWR TB TEST TFC THRU
TILL TIL TIME TK TKS TNX THX TKU TMW TMR TONE TONITE 2NITE TOO TR TRUB TRX TU
TVI TX TXT UR UHF UL UNLIS UNSTDI UP UT VALVE VFO VIA VY WAT WATSA WEAK WEN WK
WRK WKD WKG WT WW WX XCUSE XCUS XTAL XTL XYL YDAY YET YL YR YSTN 55 73 88 99
"""
# Kody Q (lista z opisami, wkleił operator 24.10) — same kody.
QKODY = """
QAZ QHL QHM QLH QLM QMH QML QQQ QRA QRAR QRB QRD QRE QRF QRG QRH QRI QRJ QRK
QRL QRM QRN QRO QRP QRQ QRR QRRR QRS QRSS QRT QRU QRV QRW QRX QRY QRZ QSA QSB
QSD QSI QSK QSL QSLL QSLN QSM QSN QSO QSP QSQ QSR QSS QST QSU QSUF QSV QSW QSX
QSY QSZ QTA QTB QTC QTG QTH QTN QTR QTS QTT QTU QTV QTX
"""
OSOBISTY = Path(__file__).with_name("slownik_osobisty.txt")

ZNAK = re.compile(r"^(?:[A-Z]{1,2}|[0-9][A-Z]|[A-Z][0-9])[0-9]{1,2}[A-Z]{1,4}"
                  r"(?:/(?:P|M|MM|AM|QRP|[0-9]))?$")
RST = re.compile(r"^(?:[1-5][1-9][1-9N]|5NN)$")
LITERA = re.compile(r"[A-Z0-9/]")


def _slowa() -> set[str]:
    s = set(OGOLNY.split()) | set(SLANG.split()) | set(QKODY.split())
    if OSOBISTY.exists():
        for w in OSOBISTY.read_text(encoding="utf-8").splitlines():
            w = w.split("#", 1)[0].strip().upper()
            s.update(w.split())
    return s


SLOWA = _slowa()


# Dwuliterowe tylko najczęstsze: ze slangiem świeciły kawałki złych słów —
# „T MA YNAMIS”, „MO O M” (okno na radiu 24.10). Rzadkie dwuliterowe
# (MA, MO, EL, OB, ...) słownik pomija: ani zielone, ani nie przestawiają spacji.
DWULITEROWE = set("""
CQ DE UR OM GM GA GE GN ES HR FB TU GL VY WX TX RX OK OP IS MY NR HW DX YL
SK KN BK CU DR GD 73 88 55 99 IN AT TO SO NW AB AS AR
""".split())
QSET = set(QKODY.split())
# Kod Q wolno skleić przy dłuższej przerwie: „QS B”, „QR Z”, „QS L” (okno 24.10)
SKLEJ_Q_U = 9.0


def znane(w: str) -> bool:
    if len(w) == 1:
        return w in POJEDYNCZE
    if len(w) == 2:
        return w in DWULITEROWE
    return w in SLOWA or bool(ZNAK.match(w)) or bool(RST.match(w))


def podziel(litery: list[str], spacja: list[bool], przerwa_u: list[float],
            rozdziel_u: float = ROZDZIEL_U, sklej_u: float = SKLEJ_U,
            sklej_q_u: float | None = None) -> list[tuple[str, bool]]:
    """litery[i]; spacja[k], przerwa_u[k] — granica PRZED literą k (k >= 1;
    [0] nieużywane), przerwa w jednostkach (nan = nieznana: nie ruszamy).
    -> [(słowo, czy znane)] po kolei; słowa nieznane to ciągi liter między
    spacjami tak jak w odczycie (poza granicami wymuszonymi przez znane)."""
    n = len(litery)
    if not n:
        return []
    if sklej_q_u is None:
        sklej_q_u = SKLEJ_Q_U

    def granica(k: int, chce_spacji: bool) -> float | None:
        """Koszt ustawienia granicy k; None = niedozwolone."""
        if chce_spacji == spacja[k]:
            return 0.0
        g = przerwa_u[k]
        if g != g:                                   # nan
            return None
        if chce_spacji and g >= rozdziel_u:
            return KARA_ZMIANY
        if not chce_spacji and g <= sklej_u:
            return KARA_ZMIANY
        return None

    NEG = float("-inf")
    # f[i][z]: najlepszy wynik dla liter [0, i); z — ostatni kawałek: 0 nieznany,
    # 1 znane słowo krótkie (< DLUGIE liter), 2 znane długie.
    f = [[NEG] * 3 for _ in range(n + 1)]
    skad: list[list[tuple | None]] = [[None] * 3 for _ in range(n + 1)]
    f[0][0] = 0.0

    def rozdzielenie_wolno(zp: int, zc: int) -> bool:
        # „MAR IUS” -> „M AR IUS” (AR w słowniku, 24.10): rozdzielić ciąg z odczytu
        # wolno tylko między dwoma znanymi albo obok długiego znanego (znak wywoławczy)
        return (zp > 0 and zc > 0) or zp == 2 or zc == 2

    def przed(i: int, zp: int, zc: int) -> float | None:
        """Koszt granicy przed kawałkiem zaczynającym się w i (poprzedni zp, ten zc)."""
        if i == 0:
            return 0.0
        if zp == 0 and zc == 0:
            return 0.0                                # jak w odczycie
        k = granica(i, True)
        # obok „=”, „?”, „+” granica naturalna („OM=” -> „OM =”, okno 24.10)
        interp = not LITERA.match(litery[i]) or not LITERA.match(litery[i - 1])
        if k is None or (k > 0 and not interp and not rozdzielenie_wolno(zp, zc)):
            return None
        return k

    for i in range(n):
        for zp in (0, 1, 2):
            if f[i][zp] == NEG:
                continue
            # 1) pojedyncza litera, nieznana
            k = przed(i, zp, 0)
            if k is not None and f[i][zp] - k > f[i + 1][0]:
                f[i + 1][0] = f[i][zp] - k
                skad[i + 1][0] = (i, zp, False)
            # 2) słowo znane litery[i:j]: w środku bez spacji
            if not LITERA.match(litery[i]):
                continue
            wew = 0.0
            tylko_q = False                   # w środku przerwa w (sklej_u, sklej_q_u]
            for j in range(i + 1, min(n, i + NAJDLUZSZE) + 1):
                if not LITERA.match(litery[j - 1]):
                    break
                if j - 1 > i:
                    k = granica(j - 1, False)
                    if k is None:
                        g = przerwa_u[j - 1]
                        if not (spacja[j - 1] and g == g and g <= sklej_q_u and litery[i] == "Q"):
                            break
                        k, tylko_q = KARA_ZMIANY, True
                    wew += k
                w = "".join(litery[i:j])
                if not znane(w) or (tylko_q and w not in QSET):
                    continue
                zc = 2 if len(w) >= DLUGIE else 1
                k = przed(i, zp, zc)
                if k is None:
                    continue
                wyn = f[i][zp] + (len(w) if len(w) >= 2 else ZYSK_POJEDYNCZEJ) - wew - k
                if wyn > f[j][zc]:
                    f[j][zc] = wyn
                    skad[j][zc] = (i, zp, True)
    # odtworzenie
    z = max(range(3), key=lambda q: f[n][q])
    j, kaw = n, []
    while j > 0:
        i, zp, zn = skad[j][z]
        kaw.append((i, j, zn))
        j, z = i, zp
    kaw.reverse()
    # kawałki -> słowa: znane osobno; nieznane litery sklejane wg spacji z odczytu
    wynik: list[tuple[str, bool]] = []
    for i, j, zn in kaw:
        w = "".join(litery[i:j])
        if zn:
            wynik.append((w, True))
        elif wynik and not wynik[-1][1] and not spacja[i]:
            wynik[-1] = (wynik[-1][0] + w, False)
        else:
            wynik.append((w, False))
    return wynik


def z_znakow(znaki: list[tuple[str, float, float]], u: float) -> tuple[list[str], list[bool], list[float]]:
    """[(znak, od, do)] z odczytu (spacje jako osobne wpisy) -> wejście podziel()."""
    litery, spacja, przerwa = [], [], []
    byla_spacja, ost_do = False, None
    for z, a0, b0 in znaki:
        if z == " ":
            byla_spacja = True
            continue
        litery.append(z)
        spacja.append(byla_spacja)
        przerwa.append((a0 - ost_do) / u if ost_do is not None and u > 0 else float("nan"))
        byla_spacja, ost_do = False, b0
    return litery, spacja, przerwa


def koloruj(znaki: list[tuple[str, float, float]], u: float) -> list[tuple[str, bool]]:
    """Odczyt [(znak, od, do)] -> [(słowo, czy znane)] ze spacjami ze słownika."""
    lit, spa, prz = z_znakow(znaki, u)
    return podziel(lit, spa, prz)
