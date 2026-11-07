# Male sieci testowe: ktora operacja liczy sie na DPU inaczej niz vai_q.
import numpy as np, tensorflow as tf
from tensorflow_model_optimization.quantization.keras import vitis_quantize
L = tf.keras.layers
rng = np.random.default_rng(1)

def m_splot():
    x = i = L.Input((512, 32, 1)); x = L.ReLU()(L.Conv2D(16, (5, 3), padding="same")(x))
    return tf.keras.Model(i, L.Conv2D(1, 1)(x)), 1
def m_pool():
    x = i = L.Input((512, 32, 1)); x = L.ReLU()(L.Conv2D(16, (5, 3), padding="same")(x))
    x = L.MaxPooling2D((1, 2))(x); return tf.keras.Model(i, L.Conv2D(1, 1)(x)), 1
def m_dil():
    x = i = L.Input((512, 4, 16)); x = L.ReLU()(L.Conv2D(16, (5, 1), padding="same", dilation_rate=(8, 1))(x))
    return tf.keras.Model(i, L.Conv2D(1, 1)(x)), 1
def m_dil1():
    x = i = L.Input((512, 1, 16)); x = L.ReLU()(L.Conv2D(16, (5, 1), padding="same", dilation_rate=(4, 1))(x))
    return tf.keras.Model(i, L.Conv2D(1, 1)(x)), 1
def m_zlacz():
    a = L.Input((512, 32, 1)); b = L.Input((512, 32, 1))
    x = L.Concatenate()([L.ReLU()(L.Conv2D(8, 3, padding="same")(a)), L.ReLU()(L.Conv2D(8, 3, padding="same")(b))])
    return tf.keras.Model([a, b], L.Conv2D(1, 1)(x)), 2
def m_suma():
    x = i = L.Input((512, 1, 16)); r = L.ReLU()(L.Conv2D(16, (5, 1), padding="same")(x))
    x = L.Add()([x, r]); return tf.keras.Model(i, L.Conv2D(1, 1)(x)), 1

for nazwa, f in [("splot", m_splot), ("pool", m_pool), ("dil", m_dil), ("dil1", m_dil1), ("zlacz", m_zlacz), ("suma", m_suma)]:
    m, n = f()
    sh = [tuple(t.shape[1:]) for t in m.inputs]
    kal = [rng.random((64,) + s, np.float32) for s in sh]
    tst = [rng.random((4,) + s, np.float32) for s in sh]
    q = vitis_quantize.VitisQuantizer(m).quantize_model(calib_dataset=kal if n > 1 else kal[0], calib_batch_size=16)
    q.save(f"/t/{nazwa}.h5")
    y = q.predict(tst if n > 1 else tst[0], verbose=0)
    np.savez(f"/t/{nazwa}.npz", y=y, **{f"x{k}": t for k, t in enumerate(tst)})
    print(nazwa, "ok", y.shape)
