# -*- coding: utf-8 -*-
"""MoE decomposition (review item 2): split mlp_out into shared*sig(gate) and implied routed,
on both FreeToken and llama captures, to test the renormalize (norm_topk_prob) hypothesis.

shared_raw = down( silu(gate@x) * (up@x) )            # SwiGLU shared expert
shared      = shared_raw * sigmoid(x @ shared_gate_w)  # per-token scalar gate
routed      = mlp_out - shared
If llama renormalizes top-10 weights (norm_w=true, factor approx 1/sum(top10) ~3.7) and
FreeToken does NOT (norm_topk_prob=False), then routed_llama/routed_free ~= 1/sum(top10) per token.
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
DIM = 2560
INT = 640


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


def silu(x):
    return x / (1 + np.exp(-x))


def main() -> int:
    from freetoken.models.gguf.reader import iter_gguf_tensors

    want = {}
    for t in iter_gguf_tensors(MODEL):
        if t.name in (
            "blk.0.ffn_gate_shexp.weight",
            "blk.0.ffn_up_shexp.weight",
            "blk.0.ffn_down_shexp.weight",
            "blk.0.ffn_gate_inp_shexp.weight",
        ):
            want[t.name] = dequant(t)

    G, U, D = want["blk.0.ffn_gate_shexp.weight"], want["blk.0.ffn_up_shexp.weight"], want["blk.0.ffn_down_shexp.weight"]
    g_w = want["blk.0.ffn_gate_inp_shexp.weight"].ravel()  # [DIM]

    free = np.load(RUN / "free.npz")
    mlp_out_free = free["mlp_out_0"].astype(np.float64)
    mlp_in_free = free["mlp_in_0"].astype(np.float64)
    mlp_out_ll = np.fromfile(RUN / "llama" / "mlp_out_000.f32", dtype=np.float32).reshape(TOK, -1).astype(np.float64)
    mlp_in_ll = np.fromfile(RUN / "llama" / "mlp_in_000.f32", dtype=np.float32).reshape(TOK, -1).astype(np.float64)

    def decompose(x):
        gate = silu(x @ G.T)          # [T, INT]
        up = x @ U.T                  # [T, INT]
        shared_raw = (gate * up) @ D.T
        gate_sig = 1 / (1 + np.exp(-(x @ g_w)))   # [T]
        shared = shared_raw * gate_sig[:, None]
        return shared, shared_raw

    sh_free, _ = decompose(mlp_in_free)
    sh_ll, _ = decompose(mlp_in_ll)
    rt_free = mlp_out_free - sh_free
    rt_ll = mlp_out_ll - sh_ll

    def rel(a, b):
        return float(np.sqrt(((a - b) ** 2).mean()) / np.sqrt((b ** 2).mean()))

    print(f"shared: free-vs-llama rel = {rel(sh_free, sh_ll):.4f} (input only differs 2.5%)")
    print(f"mlp_out: free-vs-llama rel = {rel(mlp_out_free, mlp_out_ll):.4f}")
    print(f"routed: free-vs-llama rel  = {rel(rt_free, rt_ll):.4f}")
    print(f"routed signal rms: free {np.sqrt((rt_free**2).mean()):.4f}  llama {np.sqrt((rt_ll**2).mean()):.4f}")
    # per-token ratio routed_llama/routed_free (mean over dims) vs renorm factor 1/sum(top10)
    rl = np.sqrt((rt_ll ** 2).mean(1)) / np.sqrt((rt_free ** 2).mean(1))
    print("routed ll/free ratio per token:", np.round(rl, 2).tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
