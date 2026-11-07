"""SIEĆ ZBOCZY NA DPU KV260 — okna testowe -> logity, porównanie z vai_q i float (07.11).

NA PŁYTCE (VART z srodowisko/kv260/buduj_vart.sh, xmutil loadapp kv260-benchmark-b4096):
    sudo ~/venv/bin/python dpu_proba.py zbocza_5.xmodel tst_dpu.npz
tst_dpu.npz: a, b (okna widmo/drobny z out/dpu/okna.npz, tst_*), fl (logity numpy),
vq (out/dpu/tst_logit_vaiq.npy). DPU ma się zgadzać z vai_q prawie co do bitu.
"""
import sys, time
import numpy as np
import vart, xir

g = xir.Graph.deserialize(sys.argv[1])
sg = [s for s in g.get_root_subgraph().toposort_child_subgraph()
      if s.has_attr("device") and s.get_attr("device") == "DPU"]
print("podgrafy DPU:", len(sg))
r = vart.Runner.create_runner(sg[0], "run")
wej, wyj = r.get_input_tensors(), r.get_output_tensors()
for t in wej + wyj:
    print(" ", t.name, tuple(t.dims), t.dtype, "fix_point", t.get_attr("fix_point"))
d = np.load(sys.argv[2])
zrodla = {"widmo": d["a"], "drobny": d["b"]}
out = []
t0 = time.time()
for i in range(d["a"].shape[0]):
    bufy = []
    for t in wej:
        x = next(v for k, v in zrodla.items() if k in t.name)[i:i + 1]
        s = 2.0 ** t.get_attr("fix_point")
        bufy.append(np.clip(np.round(x * s), -128, 127).astype(np.int8).reshape(tuple(t.dims)))
    y = np.empty(tuple(wyj[0].dims), np.int8)
    jid = r.execute_async(bufy, [y]); r.wait(jid)
    out.append(y.astype(np.float32).reshape(-1) * 2.0 ** -wyj[0].get_attr("fix_point"))
dt = time.time() - t0
p = np.stack(out)
sig = lambda z: 1 / (1 + np.exp(-z))
n = p.shape[0]
print(f"{n} okien w {dt:.3f} s ({1000 * dt / n:.2f} ms/okno; okno = 512 ramek = 2,56 s)")
for nazwa, ref in (("vai_q (PC)", d["vq"]), ("float (numpy)", d["fl"])):
    dp = np.abs(sig(p) - sig(ref))
    print(f"DPU vs {nazwa}: |dP| średnio {dp.mean():.4f}, maks {dp.max():.3f}, ramki zgodne {np.mean((sig(p) > .5) == (sig(ref) > .5)):.4%},"
          f" logity identyczne {np.mean(np.isclose(p, ref)):.2%}")
print("DPU logity: min/max/średnia", p.min(), p.max(), p.mean(), " unikalnych", np.unique(p).size)
print("vai_q logity:", d["vq"].min(), d["vq"].max(), d["vq"].mean())
print("korelacja DPU~vai_q", np.corrcoef(p.ravel(), d["vq"].ravel())[0, 1])
print("okno 3, ramki 200..215 DPU :", np.round(p[3, 200:216], 1))
print("okno 3, ramki 200..215 vaiq:", np.round(d["vq"][3, 200:216], 1))
