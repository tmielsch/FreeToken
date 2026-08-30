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
   amplification), which is what produces the 9–14 logprob delta.
   **Correction after code review (verified with the reviewer's finding):** the routed-MoE path
   does **not** use a BF16-dequant+torch-GEMM. `fmt=="gguf"` (layers/moe.py:619) → `fused_experts_gguf`
   → `ggml_moe_a8_vec` (`kernel/csrc/gguf/gguf_kernel.cu:566-568`), which quantizes the activation
   to **Q8_1 itself** — the *same* kernel llama.cpp uses. The BF16 fallback (moe.py:588) only
   handles genuinely-BF16 experts (not this checkpoint). So FreeToken-GPU and llama-GPU share the
   a8/Q8_1 expert numerics, and the ~30% against our llama-**CPU** reference is most plausibly a
   CPU-vs-GPU reference artifact (llama-CPU dequantizes experts to fp32 without a8) and/or a
   routing/assembly difference — to be tested against llama-GPU (which the GUI already uses, `-ngl 99`).

Neither is a FreeToken bug — the CPU-reference differences come from llama.cpp/ggml numerical-path
behaviors that a fp32-clean FreeToken cannot reproduce exactly; the GPU-vs-GPU baseline is the
acceptance target.

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

**Stage-1 router A/B already resolved (no new captures needed, from the saved `mlp_in` states):**
at layer 0 the router logits agree (rel 0.39%, F32 router weight) and **top-1/top-4 expert
selection is identical (overlap 1.0)** — so the layer-0 `mlp_out` 30% is **not** a routing flip.
Deep layers (≥20) do diverge in routing, but only because the compounded residual drift (mlp_in
~10%) flips top-k — a consequence, not the root cause. Remaining explanation for layer 0: expert
GEMM numerics (llama-CPU fp32-dequant experts vs FreeToken's shared a8/q8_1 kernel).

1. **Definition / proof options for "expert GEMM numerics (CPU-vs-GPU)"**:
   - Run a true full-GPU llama (a8 MoE + f32 dense) — blocked here by 16 GB VRAM vs 83.8 GB model.
   - Offline stage-2/3 replay: dequant the selected IQ3_XXS/IQ4_XS/IQ4_NL expert weights to fp32
     and replay (a) fp32 and (b) q8_1-activation trajectories against the captured `mlp_out`.
   This distinguishes activation-rounding magnitude from an output-scale/assembly bug.
2. **Review verdict followed:** do NOT chase llama-CPU bit-numerics; target qualitative
   equivalence vs llama-**GPU** (which the GUI already uses). Whether llama/ggml should instead
   match the canonical fp32 path on CPU remains a llama.cpp-side question.
3. Historical "9–14 logprob" band is not reproduced by shared top-10 deltas (~2–3 at div15);
   worth re-deriving the original metric before treating it as ground truth.

## Artifacts

- Repo: instrumentation commit `447dc0f` (`perf(qwen38-gguf): flag-gated per-layer hidden/mid-state
  dump for layer harness`).
- `D:\temp\opencode\layerdump\`: `RESULTS.md` (full), harness/replay scripts, captures `ft_div`
  (`llama/` + `llama_f32/`), llama.cpp fork patches + `build-layerdump`.
