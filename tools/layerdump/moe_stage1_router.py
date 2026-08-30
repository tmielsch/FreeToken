# -*- coding: utf-8 -*-
"""MoE stage-1 A/B using already-captured data: compare router logits + top-k selection
between FreeToken and llama.cpp at layer 0 WITHOUT new captures.

router_logits = mlp_in @ ffn_gate_inp.weight^T   (same GGUF Q8_0 tensor, dequant -> f64)
  - FreeToken router input: free.npz['mlp_in_0']            (captured)
  - llama     router input: ft_div/llama/mlp_in_000.f32     (captured)
Both share the same real router weight from the GGUF.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, r"E:\_AI\FreeToken\python")

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
MODEL = r"C:\Users\TM\.lmstudio\models\Qwen3.8-Flash-Next-UD-Q3_K_XL-merged.gguf"
RUN = HERE / "ft_div"
LOSSLESS = np.float64


def dequant(t) -> np.ndarray:
    """Dequant GGUF tensor to f64 row-major (handles F32=0 and Q8_0=8)."""
    if t.ggml_type == 0:  # F32
        return t.packed().numpy().view("<f4").reshape(t.shape).astype(np.float64)
    if t.ggml_type == 8:  # Q8_0 (2-byte fp16 scale, 34-byte blocks)
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

    W = None
    for t in iter_gguf_tensors(MODEL):
        if t.name == "blk.0.ffn_gate_inp.weight":
            W = dequant(t)  # [n_expert, hidden]
            break
    if W is None:
        raise SystemExit("ffn_gate_inp.weight not found")

    n_tok = 18
    free = np.load(RUN / "free.npz")
    mi_f = free["mlp_in_0"].astype(LOSSLESS)          # [T, 2560]
    mi_l = np.fromfile(RUN / "llama" / "mlp_in_000.f32", dtype=np.float32).reshape(n_tok, -1).astype(LOSSLESS)

    print("mlp_in rel(free, llama) =", float(
        np.sqrt(((mi_f - mi_l) ** 2).mean()) / np.sqrt((mi_l ** 2).mean())))

    rl_f = mi_f @ W.T     # [T, n_expert] router logits (fp64 on the SAME weights)
    rl_l = mi_l @ W.T

    sig = np.sqrt((rl_l ** 2).mean())
    rel = float(np.sqrt(((rl_f - rl_l) ** 2).mean()) / sig)
    print(f"router_logits rel(free, llama) = {rel:.5f}   (signal rms {sig:.3f})")
    # top-k agreement (FreeToken uses n_expert_used; try k in {1,4,8})
    for k in (1, 4, 8):
        tf = np.argsort(-rl_f, axis=1)[:, :k]
        tl = np.argsort(-rl_l, axis=1)[:, :k]
        agree = np.mean([len(set(a) & set(b)) for a, b in zip(tf, tl)]) / k
        print(f"top-{k} mean overlap = {agree:.3f}")
        if k == 4:
            for i in range(n_tok):
                if len(set(tf[i]) & set(tl[i])) < 4:
                    print(f"  tok {i}: free={tf[i].tolist()} llama={tl[i].tolist()}")
    # top-k weights agreement
    sw = np.exp(rl_f - rl_f.max(1, keepdims=True)); sw /= sw.sum(1, keepdims=True)
    sw_l = np.exp(rl_l - rl_l.max(1, keepdims=True)); sw_l /= sw_l.sum(1, keepdims=True)
    print("softmax router prob rel =", float(np.sqrt(((sw - sw_l) ** 2).mean())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
