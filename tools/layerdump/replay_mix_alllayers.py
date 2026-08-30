# -*- coding: utf-8 -*-
"""Per-layer HC-mix replay vs llama's captured attn_in, on llama's OWN residual inputs.

For every layer k, replay attn_in_k = hc_attn_mix(residual_before_layer_k) in fp64 using the
real GGUF Q8_0 weights and llama's captured residual (layer_00k), then compare against llama's
captured attn_in_00k. If llama's mix formula were canonical, the replay should match ~1e-4;
a constant ~1% per-layer gap means llama.cpp's HC-mix compute deviates from canonical.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, r"E:\_AI\FreeToken\python")

import numpy as np
from freetoken.models.gguf.reader import iter_gguf_tensors

HERE = pathlib.Path(__file__).resolve().parent
MODEL = r"C:\Users\TM\.lmstudio\models\Qwen3.8-Flash-Next-UD-Q3_K_XL-merged.gguf"
RUN = HERE / "ft_div"
SYSTEM = 4
EPS = 1e-6


def dequant_q8_0(t) -> np.ndarray:
    R, C = t.shape
    packed = t.packed().numpy().view(np.uint8)
    blocks = C // 32
    blob = packed.reshape(R, blocks * 34)
    rows = []
    for r in range(R):
        row = np.zeros(C, dtype=np.float64)
        for cb in range(blocks):
            blk = blob[r, cb * 34 : cb * 34 + 34]
            sc = np.frombuffer(blk[:2].tobytes(), dtype="<f2")[0].astype(np.float64)
            q = np.frombuffer(blk[2:34].tobytes(), dtype="i1").astype(np.float64)
            row[cb * 32 : cb * 32 + 32] = sc * q
        rows.append(row)
    return np.stack(rows)


def main() -> int:
    meta = json.loads((RUN / "llama" / "meta.json").read_text(encoding="utf-8"))
    n, W = meta["n_tokens"], meta["n_embd_out"]
    DIM = W // SYSTEM

    attn_in = {}
    residual = {}
    for li in range(48):
        attn_in[li] = np.fromfile(RUN / "llama" / f"attn_in_{li:03d}.f32", dtype=np.float32).reshape(n, DIM).astype(np.float64)
        residual[li] = np.fromfile(RUN / "llama" / f"layer_{li:03d}.f32", dtype=np.float32).reshape(n, W).astype(np.float64)

    want = {}
    for t in iter_gguf_tensors(MODEL):
        nm = t.name
        if nm.startswith("blk.") and "hc_attn_" in nm and nm.endswith(".weight"):
            want[nm] = t

    rels = []
    for li in range(48):
        base = f"blk.{li}.hc_attn_"
        down = dequant_q8_0(want[base + "down.weight"])     # [320, W]
        up = dequant_q8_0(want[base + "up.weight"])         # [W, 320]
        inject = want[base + "inject.weight"].packed().numpy().view("<f4").reshape(SYSTEM, W).astype(np.float64)
        w_norm = want[base + "norm.weight"].packed().numpy().view("<f4").reshape(W).astype(np.float64)

        R = residual[li]
        Rn = R.reshape(n, SYSTEM, DIM)
        rms = np.sqrt((Rn * Rn).mean(2, keepdims=True) + EPS)
        xn = (Rn / rms) * w_norm.reshape(1, SYSTEM, DIM)
        xflat = xn.reshape(n, W)
        lora = xflat @ down.T
        silu = (lora / SYSTEM) / (1 + np.exp(-(lora / SYSTEM)))
        gate = silu @ up.T
        gated = (1 / (1 + np.exp(-gate))) * xflat
        x = gated.reshape(n, SYSTEM, DIM).mean(1)

        d = x - attn_in[li]
        rel = float(np.sqrt((d * d).mean()) / max(np.sqrt((attn_in[li] ** 2).mean()), 1e-12))
        rels.append(rel)

    print("per-layer replay(ep64) vs llama attn_in rel:")
    for li, r in enumerate(rels):
        print(f"{li:2d} {r:.4f}")
    print(f"min={min(rels):.4f} max={max(rels):.4f} mean={sum(rels)/len(rels):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
