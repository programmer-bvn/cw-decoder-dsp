"""Wspólna warstwa projektu: config + front-end + synteza CW + kanał + model.

Narzędzia w tools/ nie mają własnych stałych ani własnego przetwarzania.
Wszystko idzie stąd, żeby generator, podgląd i dekoder widziały ten sam obraz.

IMPORTY SĄ LENIWE
-----------------
Ten plik świadomie NIE importuje podmodułów na starcie. Wcześniej robił to
zachłannie, więc samo `from dsp import config` ciągnęło frontend, a razem
z nim librosę (a ta numbę i scipy) — kilkaset megabajtów na odczytanie
jednej stałej.

Zobaczyliśmy to na tools/profil.py: proces nadrzędny, który potrzebował
tylko config, trzymał librosę i przez to proces potomny z TensorFlow nie
miał już miejsca i kończył się "OpenBLAS: Memory allocation failed".

Dostęp `dsp.frontend` nadal działa — moduł doładuje się przy pierwszym
użyciu (PEP 562).
"""

import importlib

_LAZY = ("config", "melref", "frontend", "morse", "radio", "model")

__all__ = list(_LAZY)


def __getattr__(name):
    """Doładowanie podmodułu przy pierwszym odwołaniu."""
    if name in _LAZY:
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"moduł {__name__!r} nie ma atrybutu {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
