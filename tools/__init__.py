"""Narzędzia uruchamiane jako moduły: python -m tools.<nazwa>

    generator   generator treningowy    audio -> npz
    xray        przeglądarka X-Ray      npz / wav -> png
    wav2net     konwerter wav -> sieć   wav -> wejście modelu
    mic2wav     konwerter mic -> wav    mikrofon -> wav
    trainer     trening modelu          npz -> .keras

Żadne z nich nie ma własnych stałych — wszystkie czytają dsp/config.py.
"""
