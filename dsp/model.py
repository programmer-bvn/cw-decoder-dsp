"""Architektura sieci. Kształt wejścia bierze się z config.INPUT_SHAPE.

DOCELOWY SPRZĘT: AMD Kria KV260 (Zynq UltraScale+, DPUCZDX8G).
To akcelerator SPLOTOWY. Nie obsługuje warstw rekurencyjnych — RNN w Vitis
AI ma osobne nakładki (DPURADR16L na Alveo U25, DPURAHR16L na U50LV),
których na Zynq nie ma. Model z GRU/LSTM albo się nie skompiluje, albo
kompilator rozetnie graf i część policzy ARM, co przy strumieniu audio
zabija przepustowość.

Stąd dwie architektury:

    arch="dpu"   sam splot — TA idzie na KV260 (domyślna)
    arch="gru"   z dwukierunkowym GRU — punkt odniesienia, ile dokładności
                 kosztuje rezygnacja z rekurencji; NIE do wdrożenia

DLACZEGO NIE GlobalAveragePooling PO OSI CZASU (w obu wersjach)
--------------------------------------------------------------
W klasyfikacji obrazów uśrednienie po całej mapie cech jest standardem.
Tutaj byłoby błędem: w Morse'ie informacja SIEDZI W KOLEJNOŚCI. "..-" i
"-.." mają identyczny zestaw elementów i identyczną średnią energię —
uśrednienie po czasie zrównuje U z D.

DLACZEGO NIE Flatten + Dense(512) NA PEŁNEJ ROZDZIELCZOŚCI
----------------------------------------------------------
Tak było w Trainer_v5_6_ULTRA.py: po trzech blokach zostaje 16x4x128 = 8192
cechy, a Dense(512) to 4,2 mln wag — stąd plik modelu 51 MB. Te wagi uczą
się osobno dla każdej pozycji w czasie, więc znak przesunięty o dwie ramki
trafia w inny zestaw wag.
"""

from __future__ import annotations

import keras
from keras import layers

from . import config as C

# Typy warstw, które przechodzą przez kompilator Vitis AI na DPUCZDX8G.
# Lista celowo wąska. Nie zastępuje kompilatora — on sprawdzi jeszcze
# rozmiary jąder, liczbę kanałów i kolejność warstw.
DPU_SAFE_LAYERS = {
    "InputLayer", "Conv2D", "DepthwiseConv2D", "SeparableConv2D",
    "Conv2DTranspose", "BatchNormalization", "Activation", "ReLU",
    "MaxPooling2D", "AveragePooling2D", "GlobalAveragePooling2D",
    "Add", "Concatenate", "Flatten", "Reshape", "Dense", "Dropout",
}


def build_model(n_classes: int = C.N_CLASSES,
                input_shape: tuple = C.INPUT_SHAPE,
                dropout: float = 0.3,
                gru_units: int = 96,
                learning_rate: float = 1e-3,
                arch: str = "dpu") -> keras.Model:
    inputs = keras.Input(shape=input_shape, name="obraz")
    x = inputs

    if arch == "gru":
        # Splot redukuje głównie CZĘSTOTLIWOŚĆ (ton jest wąskopasmowy),
        # oś czasu zostaje i przechodzi do warstwy rekurencyjnej.
        for filters, pool in ((32, (2, 2)), (64, (2, 2)), (128, (1, 2))):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            x = layers.MaxPooling2D(pool)(x)
        t, f, ch = x.shape[1], x.shape[2], x.shape[3]
        x = layers.Reshape((t, f * ch), name="czas_cechy")(x)
        x = layers.Bidirectional(layers.GRU(gru_units), name="gru")(x)
        name = "morse_crnn"

    elif arch == "dpu":
        # Zasięg widzenia po osi czasu. Liczony narastająco, z wkładem
        # KAŻDEJ warstwy — także redukujących, o czym łatwo zapomnieć:
        #   rf = 1, skok = 1
        #   conv3   rf =   3    pool2  rf =   4, skok 2
        #   conv3   rf =   8    pool2  rf =  10, skok 4
        #   conv3   rf =  18    pool2  rf =  22, skok 8
        #   conv3   rf =  38    pool2  rf =  46, skok 16
        #   conv3   rf =  78
        #   conv3   rf = 110 ramek = 2,20 s
        # Najdłuższy znak ('0') zajmuje 1,17 s, a z przerwami międzyznakowymi
        # 1,50 s. Bez pokrycia obu granic naraz neuron nie ma z czego
        # odczytać długości znaku. Ostatni splot jest właśnie po to: bez
        # niego zasięg to 78 ramek = 1,56 s, czyli 0,06 s zapasu — za mało,
        # żeby cokolwiek na tym oprzeć. Pilnuje tego TEST 13 w diag.py.
        #
        # Sploty rozszerzone (dilation) dałyby ten zasięg taniej, ale na
        # DPUCZDX8G mają ograniczenia zależne od konfiguracji rdzenia,
        # więc zasięg budujemy zwykłym stosem.
        for filters, pool in ((32, (2, 2)), (48, (2, 2)), (64, (2, 2)),
                              (96, (2, 2)), (128, None), (128, None)):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            if pool is not None:
                x = layers.MaxPooling2D(pool)(x)

        # Głowa MUSI być zależna od pozycji: etykietą jest ŚRODKOWY z trzech
        # nadanych znaków, więc "znajdź jakikolwiek znak" to zła odpowiedź.
        # Maksimum po czasie zwracałoby najmocniejszy znak w oknie, czyli
        # często sąsiada. Flatten po 8 zgrubnych krokach czasu jest zależny
        # od pozycji, ale zgrubnie — a środek znaku z etykiety błądzi
        # +/-24 ramki, czyli +/-1,5 kroku na tym poziomie.
        x = layers.Flatten()(x)
        x = layers.Dense(128, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        name = "morse_cnn_dpu"

    else:
        raise ValueError(f"nieznana architektura: {arch!r} "
                         f"(dostępne: 'dpu', 'gru')")

    x = layers.Dropout(dropout)(x)
    # dtype="float32": przy mixed_float16 softmax musi liczyć się w pełnej
    # precyzji. Na DPUCZDX8G softmax ma własny blok sprzętowy.
    outputs = layers.Dense(n_classes, activation="softmax", name="znak",
                           dtype="float32")(x)

    model = keras.Model(inputs, outputs, name=name)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def check_dpu_compatible(model, verbose: bool = True) -> list[str]:
    """Lista warstw, których DPUCZDX8G nie obsługuje. Pusta = w porządku."""
    bad = [f"{l.name} ({type(l).__name__})" for l in model.layers
           if type(l).__name__ not in DPU_SAFE_LAYERS]
    if verbose:
        if bad:
            print("UWAGA: warstwy nieobsługiwane przez DPUCZDX8G (KV260):")
            for b in bad:
                print(f"    {b}")
        else:
            print("kontrola DPUCZDX8G: wszystkie typy warstw obsługiwane")
    return bad


def load_model(path=C.MODEL_PATH) -> keras.Model:
    """Wczytuje zapisany model. Obsługuje .keras i stare .h5."""
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"brak modelu: {p}\nNajpierw: python -m tools.trainer"
        )
    return keras.models.load_model(str(p))
