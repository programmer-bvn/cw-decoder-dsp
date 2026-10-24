"""Teksty okienka i aplikacji: polski / angielski (24.10, operator: „co najmniej
wybór między polskim a angielskim”). Klucze po polsku jak reszta kodu;
T("klucz", **pola). Brak tłumaczenia — tekst polski (nie wywraca programu)."""

from __future__ import annotations

JEZYK = "pl"

TEKSTY: dict[str, dict[str, str]] = {
    # okienko na żywo
    "start": {"pl": "start...", "en": "starting..."},
    "laduje": {"pl": "ton CW: {ton} Hz; ładuję sieć...", "en": "CW pitch: {ton} Hz; loading network..."},
    "nasluch": {"pl": "ton CW: {ton} Hz; nasłuch", "en": "CW pitch: {ton} Hz; listening"},
    "koniec_pracy": {"pl": "dekoder zakończył pracę", "en": "decoder stopped"},
    "blad_okno": {"pl": "BŁĄD — szczegóły w oknie", "en": "ERROR — details in the window"},
    "zamykam": {"pl": "zamykam sesję...", "en": "closing session..."},
    "ton_nad_szumem": {"pl": "ton nad szumem", "en": "tone above noise"},
    "blokada": {"pl": "blokada szumów [dB nad szumem]; 0 = wyłączona",
                "en": "noise squelch [dB above noise]; 0 = off"},
    "wod_wej": {"pl": "wejście z radia {a}-{b} Hz", "en": "radio input {a}-{b} Hz"},
    "wod_syg": {"pl": "do sieci (obce skasowane, nasza na {t} Hz) {a}-{b} Hz",
                "en": "to the network (other stations removed, ours at {t} Hz) {a}-{b} Hz"},
    "zglos": {"pl": "Zgłoś problem z dekodowaniem…", "en": "Report a decoding problem…"},
    "otworz": {"pl": "Otwórz nagranie…", "en": "Open recording…"},
    "wroc": {"pl": "Wróć do radia", "en": "Back to radio"},
    "tempo": {"pl": "tempo", "en": "speed"},
    "nagrania": {"pl": "Nagrania", "en": "Recordings"},
    "odtwarzam": {"pl": "odtwarzam nagranie: {p}", "en": "playing recording: {p}"},
    "nowa_wersja": {"pl": "Dostępna nowa wersja {w} — kliknij, aby pobrać",
                    "en": "New version {w} available — click to download"},
    # okno startowe
    "ust_tytul": {"pl": "Dekoder CW — ustawienia", "en": "CW Decoder — settings"},
    "ust_jezyk": {"pl": "Język / Language:", "en": "Language / Język:"},
    "ust_karta": {"pl": "Karta dźwiękowa (wejście z radia):", "en": "Sound card (input from the radio):"},
    "ust_civ": {"pl": "Ton CW z radia przez CI-V (Icom; tylko odczyt)",
                "en": "CW pitch from the radio via CI-V (Icom; read only)"},
    "ust_ton": {"pl": "albo ton CW wpisany ręcznie [Hz] (puste = sam szuka):",
                "en": "or CW pitch entered manually [Hz] (empty = search):"},
    "ust_zapis": {"pl": "Zapisuj sesje (dźwięk + tekst) do {k}", "en": "Record sessions (audio + text) to {k}"},
    "ust_slownik": {"pl": "Słownik osobisty: {p}", "en": "Personal dictionary: {p}"},
    "ust_aktualizacje": {"pl": "Sprawdzaj nowe wersje przy starcie (GitHub)",
                         "en": "Check for new versions at start (GitHub)"},
    "ust_start": {"pl": "Start", "en": "Start"},
    # raport
    "rap_tytul": {"pl": "Zgłoś problem z dekodowaniem", "en": "Report a decoding problem"},
    "rap_wstep": {"pl": "Wystarczy kliknąć przycisk na dole — nagranie (ostatnie {m} min), odczyt, wersję programu,\n"
                        "ustawienia i stan radia program dopisze sam. Pola poniżej są NIEOBOWIĄZKOWE.",
                  "en": "Just click the button below — the recording (last {m} min), decoded text, program version,\n"
                        "settings and radio state are added automatically. The fields below are OPTIONAL."},
    "rap_znak": {"pl": "Twój znak (opcjonalnie):", "en": "Your callsign (optional):"},
    "rap_nadane": {"pl": "Co było nadane (jeśli wiesz):", "en": "What was sent (if known):"},
    "rap_opis": {"pl": "Uwagi (np. co było źle):", "en": "Remarks (e.g. what went wrong):"},
    "rap_utworz": {"pl": "Utwórz raport i wyślij", "en": "Create report and send"},
    "rap_brak_sesji": {"pl": "Zapis sesji jest wyłączony — włącz go w ustawieniach i nagraj problem jeszcze raz.",
                       "en": "Session recording is off — enable it in the settings and record the problem again."},
    "wys_wstep": {"pl": "Raport gotowy. Utwórz nową wiadomość w swojej poczcie (także w przeglądarce: Gmail,\n"
                        "Onet, WP...), wklej adres i temat, a plik raportu przeciągnij do wiadomości.\n"
                        "Masz program pocztowy (Outlook, Thunderbird)? Kliknij „Otwórz program pocztowy”.",
                  "en": "The report is ready. Create a new message in your e-mail (also web mail: Gmail...),\n"
                        "paste the address and subject and drag the report file into the message.\n"
                        "Using a mail program (Outlook, Thunderbird)? Click \"Open mail program\"."},
    "wys_brak_adresu": {"pl": "Raport zapisany. Adres do wysyłania raportów nie jest jeszcze ustawiony w tej\n"
                              "wersji programu — zachowaj plik, adres podamy w następnej wersji.",
                        "en": "Report saved. The address for reports is not set in this version yet —\n"
                              "keep the file, the address will come with the next version."},
    "wys_adres": {"pl": "Adres:", "en": "Address:"},
    "wys_temat": {"pl": "Temat:", "en": "Subject:"},
    "wys_plik": {"pl": "Plik:", "en": "File:"},
    "wys_kopiuj": {"pl": "Kopiuj", "en": "Copy"},
    "wys_pokaz": {"pl": "Pokaż plik raportu", "en": "Show report file"},
    "wys_program": {"pl": "Otwórz program pocztowy", "en": "Open mail program"},
    "wys_zamknij": {"pl": "Zamknij", "en": "Close"},
    "rap_temat": {"pl": "[CW-RAPORT] {znak} {wersja}", "en": "[CW-REPORT] {znak} {wersja}"},
    "rap_tresc": {"pl": "Raport z Dekodera CW w załączniku ({plik}).\n\n{opis}\n",
                  "en": "CW Decoder report attached ({plik}).\n\n{opis}\n"},
}


def T(klucz: str, **pola) -> str:
    t = TEKSTY.get(klucz, {})
    s = t.get(JEZYK) or t.get("pl") or klucz
    return s.format(**pola) if pola else s
