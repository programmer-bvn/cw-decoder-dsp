TEKSTY TESTOWE DO NADANIA KLUCZOWNIKIEM IC-7300MK2 (18.10)

Pliki qso*.txt: jedna relacja na plik, pisownia projektu (C.ALPHABET):
  =  ?  /  ,  .  +  (AR)   (  (KN)   ~  (SK)
tools/radio/kluczownik.ps1 zamienia "~" na "^SK" (zapis kluczownika Icom:
znaki po "^" bez przerwy międzyznakowej); reszta idzie do radia bez zmian.

Nagranie z podsłuchu przez USB: SET > Connectors > USB AF/IF Output >
AF Beep/Speech... Output = ON (RemoteTx, IC-7300MK2). Break-in WYŁĄCZONY —
kluczownik gra wtedy tylko podsłuch, bez nadawania (instrukcja IC-7300).
Opis nagrania: tools/probki.py; pole "nadane:" = treść pliku qso*.txt.
