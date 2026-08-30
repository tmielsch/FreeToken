# Layer-Dump Harness (Loop-Divergence localization)

Self-contained scripts behind `docs/layerdump.md`. They reproduce the FreeToken-vs-llama.cpp
per-layer comparison that localized the Qwen3.8-Flash-Next loop divergence (see the doc for the
findings).

## Contents

| file | purpose |
|---|---|
| `layerdump_harness.py` | orchestrate the two captures + diff (`free`/`llama`/`llama --f32`/`diff`) |
| `replay_hc_mix.py` | fp64 replay of the layer-0 hyper-connection mix on real GGUF bytes (proves the Q8_0-activation effect) |
| `replay_mix_alllayers.py` | same replay for every layer (constant ~0.5-1% llama-vs-canonical gap) |
| `moe_stage1_router.py` | router-logits + top-k A/B at layer 0 using only captured `mlp_in` states |
| `moe_stage1_all.py` | same top-k agreement check for all 48 layers |
| `llama_layerdump.cpp` | llama.cpp per-layer dump tool source (built inside the llama.cpp fork tree; see below) |
| `golden_layer0.json` | tiny golden vector: the key measured numbers at layer 0 (no big NPZ needed) |

## What this measures

- FreeToken side: flag-gated per-layer dump in `python/freetoken/models/qwen4_exp/model.py`
  (flag `D:\temp\opencode\ft_hidden_dump.flag` → keyed NPZ with `layer_<i>`, `attn_in/out_<i>`,
  `mlp_in/out_<i>`; prefill only).
- llama.cpp side: `llama_layerdump` (built from `examples/layerdump/layerdump.cpp` in the fork
  `C:\Users\TM\.unsloth\llama.cpp\`, CPU-only `build-layerdump`, plus the fork patches listed in
  `docs/layerdump.md`) dumps the same per-layer states + `-f32` variant (fp32 dense GEMMs).

## Reproduce (Windows, repo venv)

```
# 1. FreeToken engine running on 8890 (GPU), then:
python tools/layerdump/layerdump_harness.py free --prompt-name div15 --outdir D:\temp\opencode\layerdump\ft_div
python tools/layerdump/layerdump_harness.py llama --outdir D:\temp\opencode\layerdump\ft_div
python tools/layerdump/layerdump_harness.py llama --outdir D:\temp\opencode\layerdump\ft_div --f32
python tools/layerdump/layerdump_harness.py diff  --outdir D:\temp\opencode\layerdump\ft_div --f32
```

Paths at the top of each script (`MODEL`, `LLAMA_EXE`, capture dirs) default to this machine but
are plain constants/args — change them for your environment. The full per-layer captures live in
`D:\temp\opencode\layerdump\{ft_div,ft_len3,ft_len4}` (too large to commit).

## Golden vector

`golden_layer0.json` holds the decisive layer-0 numbers (FreeToken vs llama-CPU, 18-token div15)
so external reviewers can sanity-check the doc without the multi-MB dumps.
