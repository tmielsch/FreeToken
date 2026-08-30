"""Multi-token indexing test for ``ggml_moe_a8_vec`` (routed-expert GGUF GEMV).

Multi-token calls must equal the concatenation of per-token calls bit-for-bit
(chunk invariance) for every routed-expert quant-type pair the Qwen4Exp
UD checkpoint uses (gate/up: IQ3_XXS/IQ4_XS; down: IQ4_NL/Q8_0). This guards
against regressions in the token/row indexing (topk_ids offsets, expert-stride
addressing, y-row strides) that only manifest with >1 token.
"""
from __future__ import annotations

import pytest
import torch

from freetoken.kernel.gguf import ggml_moe_a8_vec
from freetoken.models.gguf.dequant import BLOCK_SHAPE, row_bytes

H = 2560  # hidden
I = 640  # expert intermediate
GU_ROWS = 2 * I  # gate+up output rows
DN_ROWS = H  # down output rows
TOKENS = 5
TOPK = 10
E = 64

# (gate/up type, down type) pairs actually present in the merged UD-Q3_K_XL model
TYPE_PAIRS = {
    "IQ3_XXS/IQ4_NL": (18, 20),
    "IQ3_XXS/Q8_0": (18, 8),
    "IQ4_XS/IQ4_NL": (23, 20),
    "IQ4_XS/Q8_0": (23, 8),
}


def _sanitize(w: torch.Tensor, qtype: int) -> torch.Tensor:
    """Random uint8 gives NaN/inf fp16 in the per-block ``d`` scales; overwrite the
    leading 2 bytes of every block with fp16(0.001)."""
    _, type_size = BLOCK_SHAPE[qtype]
    w = w.clone()
    v = w.view(-1, type_size)
    v[:, 0] = 0x60
    v[:, 1] = 0x22
    return w


def _align64(n: int) -> int:
    return (n + 63) // 64 * 64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("name,types", sorted(TYPE_PAIRS.items()))
def test_moe_vec_multi_token_chunk_invariance(name: str, types) -> None:
    gu_qtype, dn_qtype = types
    gu_rb = row_bytes(H, gu_qtype)
    dn_rb = row_bytes(I, dn_qtype)
    gu_stride = _align64(GU_ROWS * gu_rb)
    dn_stride = _align64(DN_ROWS * dn_rb)
    dev = torch.device("cuda")

    x = torch.randn(TOKENS, H, dtype=torch.bfloat16, device=dev)
    topk_ids = torch.randint(0, E, (TOKENS, TOPK), dtype=torch.int32, device=dev)
    w_gu = _sanitize(torch.randint(0, 256, (E, gu_stride), dtype=torch.uint8, device=dev), gu_qtype)
    w_dn = _sanitize(torch.randint(0, 256, (E, dn_stride), dtype=torch.uint8, device=dev), dn_qtype)

    kgu = ggml_moe_a8_vec(x, w_gu, topk_ids, TOPK, gu_qtype, GU_ROWS, TOKENS, gu_stride)
    chunks = [
        ggml_moe_a8_vec(
            x[t : t + 1], w_gu, topk_ids[t : t + 1], TOPK, gu_qtype, GU_ROWS, 1, gu_stride
        )
        for t in range(TOKENS)
    ]
    assert (kgu.float() - torch.cat(chunks, dim=0).float()).abs().max().item() < 1e-3

    inter = torch.randn(TOKENS * TOPK, I, dtype=torch.bfloat16, device=dev)
    flat_ids = topk_ids.reshape(-1, 1)
    kdn = ggml_moe_a8_vec(inter, w_dn, flat_ids, 1, dn_qtype, DN_ROWS, TOKENS * TOPK, dn_stride)
    dchunks = [
        ggml_moe_a8_vec(
            inter[r : r + 1], w_dn, flat_ids[r : r + 1], 1, dn_qtype, DN_ROWS, 1, dn_stride
        )
        for r in range(TOKENS * TOPK)
    ]
    assert (kdn.float() - torch.cat(dchunks, dim=0).float()).abs().max().item() < 1e-3
