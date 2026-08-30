# -*- coding: utf-8 -*-
"""Offline fp32 replay of layer-0's hyper-connection ATTENTION mix, comparing against the
captured llama.cpp (f32 reference) and FreeToken (bf16/quantized GEMM path) states.

Goal: decide whether the layer-0 attn_in divergence (~1%) is the *compute precision* of the
quantized GEMM path (FreeToken) vs f32 (llama), by replaying Q8_0 weights in pure fp32 with
real bytes and checking whether fp32-replay matches llama (vs FreeToken).

Latency: < 30 s CPU.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
MODEL = r"C:\Users\TM\.lmstudio\models\Qwen3.8-Flash-Next-UD-Q3_K_XL-merged.gguf"
RUN = HERE / "ft_div"
N_HC, DIM = 4, 2560


def dequant_q8_0(t, dtype=np.float64) -> np.ndarray:
    """Dequant a GGUF Q8_0 tensor [R, C] to row-major [R, C].

    Q8_0 block = { fp16 scale (2 bytes), int8 qs[32] } = 34 bytes per 32 cols, rows padded
    to 32-col blocks (C is a multiple of 32 here).
    """
    import numpy as np

    R, C = t.shape
    packed = t.packed().numpy().view(np.uint8)
    assert C % 32 == 0
    blocks_per_row = C // 32
    blob = packed.reshape(R, blocks_per_row * 34)
    rows = []
    for r in range(R):
        row = np.zeros(C, dtype=np.float64)
        for cb in range(blocks_per_row):
            blk = blob[r, cb * 34 : cb * 34 + 34]
            scale = np.frombuffer(blk[:2].tobytes(), dtype="<f2")[0].astype(np.float64)
            q = np.frombuffer(blk[2:34].tobytes(), dtype="i1").astype(np.float64)
            row[cb * 32 : cb * 32 + 32] = scale * q
        rows.append(row)
    return np.stack(rows)


def main() -> int:
    import sys

    sys.path.insert(0, r"E:\_AI\FreeToken\python")
    from freetoken.models.gguf.reader import iter_gguf_tensors

    want = {
        "blk.0.hc_attn_norm.weight": None,
        "blk.0.hc_attn_down.weight": None,
        "blk.0.hc_attn_inject.weight": None,
        "blk.0.hc_attn_up.weight": None,
    }
    qtypes = {}
    for t in iter_gguf_tensors(MODEL):
        if t.name in want:
            want[t.name] = t
            qtypes[t.name] = t.ggml_type
        if "hc_attn_down" in t.name or "hc_attn_inject" in t.name:
            pass
    for k, v in want.items():
        assert v is not None, f"missing {k}"
    print("qtypes:", qtypes)

    down = dequant_q8_0(want["blk.0.hc_attn_down.weight"])        # [320, 10240]
    up = dequant_q8_0(want["blk.0.hc_attn_up.weight"])             # [10240, 320]
    inject = want["blk.0.hc_attn_inject.weight"].packed().numpy().view("<f4").reshape(4, 10240).astype(np.float64)
    w_norm = want["blk.0.hc_attn_norm.weight"].packed().numpy().view("<f4").reshape(10240).astype(np.float64)
    eps = 1e-6

    meta = json.loads((RUN / "llama" / "meta.json").read_text(encoding="utf-8"))
    n = meta["n_tokens"]
    R = np.fromfile(RUN / "llama" / "layer_000.f32", dtype=np.float32).reshape(n, 4 * DIM).astype(np.float64)
    llama_attn_in = np.fromfile(RUN / "llama" / "attn_in_000.f32", dtype=np.float32).reshape(n, DIM).astype(np.float64)
    free = np.load(RUN / "free.npz")
    free_attn_in = free["attn_in_0"].astype(np.float64)

    # fp32 replay of the hc_attn mix (mirrors llama build_hc_mix AND FreeToken _mix_torch)
    Rn = R.reshape(n, N_HC, DIM)
    rms = np.sqrt((Rn * Rn).mean(axis=2, keepdims=True) + eps)
    xn = (Rn / rms) * w_norm.reshape(1, N_HC, DIM)        # [T,4,2560], folded (1+w)
    xflat = xn.reshape(n, N_HC * DIM)
    lora = xflat @ down.T                                 # [T,320]
    s = xflat @ inject.T                                  # [T,4]
    silu = (lora / N_HC) / (1 + np.exp(-(lora / N_HC)))   # silu(x/hc)
    gate = silu @ up.T                                    # [T,10240] (per-stream gate)
    gated = (1 / (1 + np.exp(-gate))) * xflat             # sigmoid(gate) * xn
    x = gated.reshape(n, N_HC, DIM).mean(axis=1)          # [T,2560] mean over streams

    def rel(a, b):
        d = a - b
        return float(np.sqrt((d * d).mean()) / max(np.sqrt((b * b).mean()), 1e-9))

    print(f"fp32-replay  vs llama attn_in rel = {rel(x, llama_attn_in):.6f}")
    print(f"FreeToken    vs llama attn_in rel = {rel(free_attn_in, llama_attn_in):.6f}")
    print(f"fp32-replay  vs FreeToken      rel = {rel(x, free_attn_in):.6f}")
    # also compare the inject logits? we don't have llama's inject captured; skip
    print(f"llama attn_in rms={np.sqrt((llama_attn_in**2).mean()):.5f} free rms={np.sqrt((free_attn_in**2).mean()):.5f} replay rms={np.sqrt((x**2).mean()):.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
