import sys, pathlib
import numpy as np
sys.path.insert(0, r"D:\temp\opencode\layerdump")
from moe_stage1_router import dequant, MODEL, RUN, LOSSLESS
from freetoken.models.gguf.reader import iter_gguf_tensors

W = None
for t in iter_gguf_tensors(MODEL):
    if t.name == "blk.0.ffn_gate_inp.weight":
        W = dequant(t)
        break

mi_f = np.load(RUN / "free.npz")["mlp_in_0"].astype(LOSSLESS)
mi_l = np.fromfile(RUN / "llama" / "mlp_in_000.f32", dtype=np.float32).reshape(18, -1).astype(LOSSLESS)

for name, rl in [("free", mi_f @ W.T), ("llama", mi_l @ W.T)]:
    p = np.exp(rl - rl.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    top10 = np.sort(p, axis=1)[:, -10:]
    s = top10.sum(1)
    print(name, "top10 sum:", np.round(s, 4).tolist(), " mean", round(float(s.mean()), 4))
    print(name, "renorm factor (1/s) mean", round(float((1 / s).mean()), 4),
          "range", round(float((1 / s).min()), 4), "-", round(float((1 / s).max()), 4))
