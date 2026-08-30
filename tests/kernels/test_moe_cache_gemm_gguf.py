# -*- coding: utf-8 -*-
"""EXPERTS REAL-GGUF integration test: production routed-prefill MoE path
vs direct-kernel reference, both on REAL bytes read from the checkpoint.

  cache-fed:  layer._prefill_routed (materialize_layer(ids=) + copy_missing ->
              _copy_routed_prefill_ids dedupe/num_indices -> slot cache -> gemm)
  reference:  ggml_moe_a8_vec directly over the source bytes (no copy)

Real expert rows are read from the GGUF (layer 0: gate/up IQ3_XXS, down IQ4_NL)
with the exact engine row geometry. Usage: .venv python ft_cache_gemm_integration.py
"""
import sys

import torch

sys.path.insert(0, r"E:\_AI\FreeToken\python")
from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.moe.offload_cache import OffloadMoeCache
from freetoken.layers.activation import silu_and_mul
from freetoken.models.gguf.dequant import row_bytes
from freetoken.kernel.gguf import ggml_moe_a8_vec

torch.manual_seed(0)
dev = torch.device("cuda")
if try_get_tp_info() is None:
    set_tp_info(rank=0, size=1)

MODEL = r"C:\Users\TM\.lmstudio\models\Qwen3.8-Flash-Next-UD-Q3_K_XL-merged.gguf"
H, I, GU_ROWS, DN_ROWS = 2560, 640, 1280, 2560
E, TOPK, T = 512, 10, 4


def align64(n):
    return (n + 63) // 64 * 64


def make_real_banks():
    import gguf

    reader = gguf.GGUFReader(MODEL)
    gu_meta = dn_meta = None
    for t in reader.tensors:
        if t.name == "blk.0.ffn_gate_exps.weight":
            gu_meta = t
        elif t.name == "blk.0.ffn_up_exps.weight":
            up_meta = t
        elif t.name == "blk.0.ffn_down_exps.weight":
            dn_meta = t
    assert gu_meta is not None and up_meta is not None and dn_meta is not None
    gu_type = int(gu_meta.tensor_type)
    dn_type = int(dn_meta.tensor_type)
    gu_rb = row_bytes(H, gu_type)
    dn_rb = row_bytes(I, dn_type)
    gu_stride = align64(GU_ROWS * gu_rb)
    dn_stride = align64(DN_ROWS * dn_rb)
    half = I * gu_rb  # gate / up each occupy I rows
    with open(MODEL, "rb") as f:
        src_gu = torch.zeros(E, gu_stride, dtype=torch.uint8)
        for e in range(E):
            f.seek(int(gu_meta.data_offset) + e * half)
            f.readinto(memoryview(src_gu[e, :half].numpy()))
            f.seek(int(up_meta.data_offset) + e * half)
            f.readinto(memoryview(src_gu[e, half:2 * half].numpy()))
        src_dn = torch.zeros(E, dn_stride, dtype=torch.uint8)
        for e in range(E):
            f.seek(int(dn_meta.data_offset) + e * DN_ROWS * dn_rb)
            f.readinto(memoryview(src_dn[e, : DN_ROWS * dn_rb].numpy()))
    src_gu = src_gu.pin_memory()
    src_dn = src_dn.pin_memory()
    return src_gu, src_dn, gu_type, dn_type, gu_stride, dn_stride


def direct_kernel_ref(x, topk_ids, src_gu, src_dn, gu_type, dn_type, gu_stride, dn_stride):
    """Reference = ggml_moe_a8_vec directly over GPU copies of the source bytes."""
    guw = src_gu.to(dev)
    dnw = src_dn.to(dev)
    gu = ggml_moe_a8_vec(x, guw, topk_ids, TOPK, gu_type, GU_ROWS, T, gu_stride).contiguous()
    flat = topk_ids.reshape(-1, 1)
    inter = silu_and_mul(gu)
    dn = ggml_moe_a8_vec(inter, dnw, flat, 1, dn_type, DN_ROWS, T * TOPK, dn_stride)
    return (dn.reshape(T, TOPK, DN_ROWS) * topk_weights.reshape(T, TOPK, 1).to(dn.dtype)).sum(1)


from freetoken.layers.moe import OffloadMoELayer  # noqa: E402

src_gu, src_dn, gu_type, dn_type, gu_stride, dn_stride = make_real_banks()
gu_rb = row_bytes(H, gu_type)
dn_rb = row_bytes(I, dn_type)
print(f"real types: gate_up={gu_type} down={dn_type} (gu_stride={gu_stride})")

x = torch.randn(T, H, dtype=torch.bfloat16, device=dev)
topk_ids = torch.randint(0, E, (T, TOPK), dtype=torch.int32, device=dev)
topk_ids[0, 1] = topk_ids[0, 0]  # duplicate
topk_weights = torch.full((T, TOPK), 0.1, dtype=torch.float32, device=dev)
topk_weights[:, 0] = 0.5

cache = OffloadMoeCache(num_layers=1, num_experts=E, cache_size=E, device=dev, quant_format="gguf")
cache.bank_sources["gate_up"] = [src_gu]
cache.bank_sources["down"] = [src_dn]
cache.bank_caches["gate_up"] = torch.zeros(E, gu_stride, dtype=torch.uint8, device=dev)
cache.bank_caches["down"] = torch.zeros(E, dn_stride, dtype=torch.uint8, device=dev)
cache.banks.append(([src_gu], cache.bank_caches["gate_up"]))
cache.banks.append(([src_dn], cache.bank_caches["down"]))
cache._build_copy_plan()
print("fused copy ok:", bool(cache._copy_fused_ok))

layer = OffloadMoELayer(layer_id=0, num_experts=E, top_k=TOPK, hidden_size=H, intermediate_size=I)
layer.offload_cache = cache
layer.gguf_gate_up_type = gu_type
layer.gguf_down_type = dn_type
layer.gguf_gate_up_rows = GU_ROWS
layer.gguf_down_rows = DN_ROWS

out = layer._prefill_routed(x, topk_weights, topk_ids)
ref = direct_kernel_ref(x, topk_ids, src_gu, src_dn, gu_type, dn_type, gu_stride, dn_stride)

print("out finite:", torch.isfinite(out).all().item(), "ref finite:", torch.isfinite(ref).all().item())
d = (out.float() - ref.float()).abs()
rel = d / ref.float().abs().clamp_min(1e-3)
print(f"cache-fed vs direct-kernel: max_abs={d.max().item():.5f} max_rel={rel.max().item():.5f}")
ok = d.max().item() < 0.5 and torch.isfinite(out).all()
print("RESULT:", "PASS" if ok else "FAIL")
assert ok
