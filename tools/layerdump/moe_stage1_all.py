# -*- coding: utf-8 -*-
"""MoE stage-1 (all layers): router-logits + top-k agreement between FreeToken and llama.

Same router weights from the GGUF, mlp_in from the per-layer captures on both sides.
If top-k agrees everywhere, routing is exonerated and any mlp_out diff is expert-GEMM numerics.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, r"E:\_AI\FreeToken\python")

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
MODEL = r"C:\Users\TM\.lmstudio\models\Qwen3.8-Flash-Next-UD-Q3_K_XL-merged.gguf"
RUN = HERE / "ft_div"
TOK = 18


def dequant(t) -> np.ndarray:
    if t.ggml_type == 0:
        return t.packed().numpy().view("<f4").reshape(t.shape).astype(np.float64)
    if t.ggml_type == 8:
        R, C = t.shape
        blob = t.packed().numpy().view(np.uint8).reshape(R, (C // 32) * 34)
        rows = np.empty((R, C), np.float64)
        for r in range(R):
            for cb in range(C // 32):
                blk = blob[r, cb * 34 : cb * 34 + 34]
                sc = np.frombuffer(blk[:2].tobytes(), "<f2")[0].astype(np.float64)
                q = np.frombuffer(blk[2:34].tobytes(), "i1").astype(np.float64)
                rows[r, cb * 32 : cb * 32 + 32] = sc * q
        return rows
    raise ValueError(f"unsupported ggml type {t.ggml_type}")


def main() -> int:
    from freetoken.models.gguf.reader import iter_gguf_tensors

    routers = {}
    for t in iter_gguf_tensors(MODEL):
        nm = t.name
        if nm.startswith("blk.") and nm.endswith(".ffn_gate_inp.weight"):
            layer = int(nm.split(".")[1])
            routers[layer] = dequant(t)

    free = np.load(RUN / "free.npz")
    n_layer = 48
    bad = 0
    for li in range(n_layer):
        mi_f = free[f"mlp_in_{li}".format(li)].astype(np.float64)
        mi_l = np.fromfile(RUN / "llama" / f"mlp_in_{li:03d}.f32", dtype=np.float32).reshape(TOK, -1).astype(np.float64)
        W = routers[li]  # [n_expert, hidden]
        rl_f = mi_f @ W.T
        rl_l = mi_l @ W.T
        sig = np.sqrt((rl_l ** 2).mean())
        rel = float(np.sqrt(((rl_f - rl_l) ** 2).mean()) / sig)
        tf = np.argsort(-rl_f, axis=1)[:, :8]
        tl = np.argsort(-rl_l, axis=1)[:, :8]
        ov1 = np.mean([len(set(a) & set(b)) for a, b in zip(tf[:, :1], tl[:, :1])])
        ov4 = np.mean([len(set(a[:4]) & set(b[:4])) for a, b in zip(tf, tl)])
        flag = "" if (ov1 == 1.0 and ov4 == 1.0) else "  <-- ROUTING DIVERGES"
        if flag:
            bad += 1
        print(f"layer {li:2d}: mlp_in rel {rel:7.4f}  top1_ov {ov1:5.3f} top4_ov {ov4:5.3f}{flag}")
    print(f"\nlayers with routing divergence: {bad}/{n_layer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
