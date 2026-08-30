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

## Finding (mechanically proven, updated 2026-08-30)

**The loop-divergence root cause is a REAL FreeToken bug: the routed-MoE top-10 router weights
are not renormalized on the GGUF path.** Qwen MoE (`norm_topk_prob=True`, llama.cpp qwen4exp
hardcodes `norm_w=true`) rescales the selected top-k softmax weights to sum to 1; FreeToken's
GGUF parse set `norm_topk_prob=False`, so its routed contribution was systematically ~2–6× too
small (softmax top-10 sums to ~0.29 on average ⇒ factor ~3.7). This skewed `mlp_out` at every
layer (and, through the shared-expert add, the whole decoder), producing the observed first-token
divergence and the generation loop.

Evidence (layer 0, 18-token div15, real GGUF bytes; scripts in `tools/layerdump/`):
- **Decomposition**: `shared` expert output free-vs-llama rel = 0.007 (matches; only the 2.5%
  `mlp_in` diff propagates); `routed` rel = 0.716, with **routed signal rms 0.0056 (free) vs
  0.0186 (llama)** — llama's routed part is ~3.3× larger.
- **Per-token routed ratio `routed_llama/routed_free`** `[5.78, 3.36, 2.28, ...]` matches the
  per-token renormalization factor `1/sum(top10 softmax)` `[5.76, 3.34, 2.26, ...]` almost exactly.
- **Fix reconstruction**: `shared + routed × (1/sum(top10))` reproduces llama's `mlp_out` to
  **rel 0.015** (vs 0.302 raw). => the entire 30% `mlp_out` gap is the missing renormalization.
- **Acceptance after fix** (`gguf.py` → `norm_topk_prob=True`, + regression test): div15 first-token
  top-1 id 271 lnprobs −0.085 vs llama-CPU −0.185 (±0.10, was ±1.39); 9/10 shared top-10 ids; max
  shared-id delta 1.48 (was 2.3). Greedy generation no longer repeats (dup_frac 0, proper countdown);
  default-sampling runs produce sane text with no degenerate loop.

Secondary/reference-path observations (kept for context):
- llama.cpp **CPU** Q8_0 dense GEMMs quantize the f32 activation to Q8_0 (ggml `mul_mat`, ~1% per
  dense GEMM; `GGML_PREC_F32` is a CPU no-op). FreeToken is fp32-clean there; a `f32_dense_mm`
  A/B lever (`build_lora_mm` casts Q8_0→F32) closes layer-0 `attn_in` 1.1% → 0.3%. This is a
  llama-CPU-only artifact and NOT the loop driver.
- Routed-MoE expert compute uses the **shared `ggml_moe_a8_vec` kernel** (Q8_1 activation) on
  FreeToken-GPU and llama-GPU alike — its numerics match by construction (bit-identical test
  `dd7aaf2`). Remaining small residual after the fix is expert-numerics/CPU-vs-GPU noise.

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

The review's finding on top-k was correct and is now the committed fix:
- **Stage-1 (top-10, exact `fused_topk` semantics):** top-10 IDs match; the missing
  **renormalization** of the top-10 weights is the whole `mlp_out` gap (see Finding).
- **`mlp_out` ≠ routed-MoE:** confirmed — it is `routed + shared·sigmoid(gate)`. Decomposition
  shows `shared` matches (rel 0.007) and the entire gap sits in `routed` (renorm factor match
  1/sum(top10), reconstruction rel 0.015).
- **Prefill-vs-decode invariance test (review priority 1):** designed (`invariance_test.py`).
  Blocked on the HTTP path: FreeToken OpenAI API rejects token-id prompts ("pass text prompt
  strings instead") and text-join of a greedy continuation does not re-tokenize identically.
  Next step: drive the model in-process (exact ids) with a prefill-vs-incremental-decode harness
  to confirm the decode-state rollforward (GDN/PLE/KV) reproduces the full-prefill states —
  now that the routing-weight scale is fixed, this is the remaining generation-side check.
- **After invariance PASS:** stage-2/3 selected-expert replay (Layer 0, 1 token, 10 experts)
  exclusively to quantify the a8/q8_1-vs-fp32 rounding magnitude (CPU-vs-GPU reference noise).

## Artifacts

- Repo: instrumentation commit `447dc0f` (`perf(qwen38-gguf): flag-gated per-layer hidden/mid-state
  dump for layer harness`).
- `D:\temp\opencode\layerdump\`: `RESULTS.md` (full), harness/replay scripts, captures `ft_div`
  (`llama/` + `llama_f32/`), llama.cpp fork patches + `build-layerdump`.
