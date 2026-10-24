CW DECODER SQ2BVN — live Morse code reading (neural network + Viterbi)
Trial version. Recordings the program could not read help a lot!

INSTALLATION
Run DekoderCW_setup_<version>.exe. Windows may say "Windows protected your
PC" (the program is not signed) — click "More info", then "Run anyway".
No administrator rights needed. Each version installs NEXT TO the previous
ones (separate shortcut with the version number) so you can compare how the
old and the new one read the same signal. Settings and recordings are shared.
Remove old versions in "Add or remove programs". A plain microphone next to
the radio speaker works too.

FIRST START
In the settings window choose the language and the sound card that carries
the audio from the radio (usually "USB Audio … WASAPI" for a radio with USB,
or the line input of the computer). CW pitch:
- Icom radio with a CI-V/USB cable: tick "CW pitch from the radio via CI-V"
  and choose the COM port (the program ONLY reads frequency, mode and CW
  pitch — it never changes settings and never keys the radio),
- other radios: enter the CW pitch in Hz, e.g. 600,
- empty field: the program looks for the strongest keyed station.
Settings, window position and squelch level are remembered.
The program needs about 20 s to load.

WINDOW
- Bright text: confirmed reading. Grey: preliminary (may still change).
  Green: a word from the ham radio dictionary, a callsign or an RST report —
  spaces around them are corrected. Letters are NEVER corrected: what you
  see is what the decoder decoded.
- Waterfalls: top — audio from the radio; bottom — cleaned signal (other
  stations removed, received station at 750 Hz), as the network sees it.
- "Tone above noise" and the noise squelch slider: between transmissions
  noise may produce single E/I/T. The slider blocks reading below the level;
  0 = off. Strong station: about 15-20 dB; weak station: keep it low.

OPEN RECORDING
"Open recording…" decodes a saved session (sesje\*.wav) instead of the radio —
e.g. the same recording in a new version, to check whether your report helped.
Speed 1× like live, 2×/4×/max faster. "Back to radio" switches back.

REPORT A DECODING PROBLEM
When the program misreads a station that sounds fine, click "Report a
decoding problem…", enter what was sent (if known) and a short description.
The program prepares a report (last 15 min of audio + decoded text) and
opens an e-mail. If the file is not attached automatically, drag it into
the message. Session recording must be on (it is by default).

FILES (Documents\DekoderCW)
- sesje — audio and text of each session,
- raporty — prepared reports,
- slownik_osobisty.txt — your own words (name, QTH, friends) shown in green,
- dziennik.txt — program log (useful when something goes wrong).

The program only RECEIVES. Closing the window ends and saves the session.
Source code (GPL-3.0): https://github.com/programmer-bvn/cw-decoder-dsp
