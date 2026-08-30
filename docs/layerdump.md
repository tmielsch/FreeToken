# Layer-Dump Harness — Loop-Divergence Localization (Qwen3.8-Flash-Next GGUF)

Status: 2026-08-30, branch `qwen38-flash-next-gguf`.
Full investigation + captures live in `D:\temp\opencode\layerdump\` (RESULTS.md, harness,
replay scripts, per-layer captures `ft_div` / `ft_len3` / `ft_len4`).

## Why this exists

FreeToken's engine reproduces the user's "loop" (repeating output) with default sampling
while llama.cpp matches it only poorly: on 15+ diverse prompt tokens the **first-token logit
distributions differ by ~9–14 logprob units** from the llama.cpp CPU reference, while at 1–4
tokens they match to ~1–2 (≈ noise). Component-level tests (GDN / QSA / HC / PLE / MoE) are all
fp32-clean, so the open question was: which decoder layer first diverges with REAL GGUF weights?

## What was built

- **Part A (FreeToken)** — flag-gated per-layer dump in `python/freetoken/models/qwen4_exp/model.py`
  (commit `447dc0f`): with flag `D:\temp\opencode\ft_hidden_dump.flag` it captures the fp32
  hyper-connection residual `[T, hc*4*2560]` after every layer plus mid-block states
  (`attn_in` / `attn_out` / `mlp_in` / `mlp_out`) into a keyed NPZ (prefill only, warmup-safe).
- **Part B (llama.cpp)** — custom C++ tool `llama-layerdump` on the ggml staging API
  (`llama_get_embeddings_layer_inp`) with a small fork patch for wide hyper-connection residuals;
  CPU reference (`-ngl 0`). Build: `C:\Users\TM\.unsloth\llama.cpp\build-layerdump`.
- **Harness/analysis** — `layerdump_harness.py` (free/llama/diff), offline fp64 replays on real
  GGUF Q8_0 bytes (`replay_hc_mix.py`, `replay_mix_alllayers.py`).

## Finding (mechanically proven)

**The divergence enters at layer 0 and is length-independent; it has two sources.**

1. **Secondary — llama.cpp quantizes f32 activations to Q8_0 in the dense GGUF GEMMs (HC/GDN).**
   A fp64 replay of the layer-0 HC mix on llama's own weights/input matches llama to **rel 1e-4
   only when the f32 activation is Q8_0-quantized per 32-block before each GEMM**; plain fp32 is
   1.1% off. Source: `ggml/src/ggml-cpu/ggml-cpu.c` `ggml_compute_forward_mul_mat`
   (`vec_dot_type`=Q8_0 ⇒ `quantize_row_q8_0` applied to the f32 activation). `GGML_PREC_F32` is
   a no-op on CPU. FreeToken computes these in fp32 (matches the canonical fp64 replay to 0.26%).
   A new `f32_dense_mm` cparam (`build_lora_mm` casts Q8_0→F32 via `ggml_cast`) closes this:
   layer-0 `attn_in` drops 1.1% → 0.3%.
2. **DOMINANT — the MoE block output (`mlp_out`).** Even after the dense-path fix, `mlp_out`
   stays ~30% relative (abs ~0.013 ≈ 4× the input error) at layer 0, length-independent (3/4/18
   tokens), and it propagates: residual rel grows 0.25 → 0.75 over layers 0..47 (~100× depth
   amplification), which is what produces the 9–14 logprob delta. The MoE path does not use
   `build_lora_mm`: FreeToken's prefill dequantizes the I-quant (IQ4) experts to **BF16** +
   torch GEMM, while llama uses quantized expert kernels with quantized activations.

Neither is a FreeToken bug — both are llama.cpp/ggml numerical-path behaviors that a fp32-clean
FreeToken cannot reproduce exactly.

## Reproduce / extend

```
# FreeToken side (engine on 8890): capture per-layer dumps for a prompt
python D:\temp\opencode\layerdump\layerdump_harness.py free --prompt-name div15 --outdir ft_div
# llama reference (default control):
... layerdump_harness.py llama  --outdir ft_div
# llama reference with fp32 dense GEMMs (A/B):
... layerdump_harness.py llama  --outdir ft_div --f32
... layerdump_harness.py diff   --outdir ft_div [--f32]
```

## Open questions / next steps (good for review input)

1. **MoE A/B with equalized input** — feed the same `mlp_in` through both MoE implementations and
   compare `mlp_out` to separate *routing sensitivity* (top-k flips on tiny input diffs, corr
   ≈0.96 suggests partial not total) from *expert numerics* (BF16 I-quant dequant vs quantized
   kernels; is FP32 expert dequant the fix?). Highest-value next experiment.
2. **Reference choice** — the user loop may compare against a **GPU** llama (f16 activations
   instead of Q8_0/I-quant activation paths); that may shrink the whole divergence.
3. Do we want FreeToken to *match* llama CPU bit-numerics (deliberately quantize activations), or
   is llama/ggml supposed to match the canonical fp32 path instead?

## Artifacts

- Repo: instrumentation commit `447dc0f` (`perf(qwen38-gguf): flag-gated per-layer hidden/mid-state
  dump for layer harness`).
- `D:\temp\opencode\layerdump\`: `RESULTS.md` (full), harness/replay scripts, captures `ft_div`
  (`llama/` + `llama_f32/`), llama.cpp fork patches + `build-layerdump`.
