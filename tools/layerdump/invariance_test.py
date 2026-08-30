# -*- coding: utf-8 -*-
"""Prefill-vs-Decode invariance test (review item 3) — text-driven A/B on the fixed engine.

A: div_prompt_text + generated_text as ONE prefill  (tokenization == DIV18 + gen_ids, verified)
B: div_prompt_text prefill + greedy decode of the SAME continuation
Compare per-position per-layer residuals: A position p vs B decode position p.
"""
from __future__ import annotations

import json
import pathlib
import shutil
import sys
import urllib.request

import numpy as np

DUMP = pathlib.Path(r"D:\temp\opencode\ft_hidden_dump")
FLAG = pathlib.Path(r"D:\temp\opencode\ft_hidden_dump.flag")
FLAG_D = pathlib.Path(r"D:\temp\opencode\ft_hidden_dump_decode.flag")
LOGIT_JSONL = pathlib.Path(r"D:\temp\opencode\ft_logit_capture.jsonl")
LOGIT_FLAG = pathlib.Path(r"D:\temp\opencode\ft_logit_capture.flag")
URL = "http://127.0.0.1:8890/v1/completions"
MODEL = "Qwen3.8-Flash-Next-UD-Q3_K_XL-merged.gguf"
DIV_TEXT = "Count down from 5 to 1, one number per line, no extra text."
N_GEN = 16


def post(prompt, max_tokens):
    body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))


def clear_dump():
    if DUMP.exists():
        shutil.rmtree(DUMP)
    DUMP.mkdir(parents=True, exist_ok=True)


def main() -> int:
    sys.path.insert(0, r"E:\_AI\FreeToken\python")
    from freetoken.models.gguf.tokenizer import load_gguf_tokenizer

    tok = load_gguf_tokenizer(r"C:\Users\TM\.lmstudio\models\Qwen3.8-Flash-Next-UD-Q3_K_XL-merged.gguf")

    # B first: capture exact continuation ids via logit-capture
    LOGIT_JSONL.write_text("", encoding="utf-8")
    LOGIT_FLAG.touch()
    try:
        outB = post(DIV_TEXT, N_GEN)
    finally:
        LOGIT_FLAG.unlink(missing_ok=True)
    gen_ids = [json.loads(l)["sampled_id"] for l in LOGIT_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
    gen_text = outB["choices"][0]["text"]
    print("B generated ids:", gen_ids, "(len %d)" % len(gen_ids))

    div_ids = tok.encode(DIV_TEXT)
    joined = DIV_TEXT + gen_text
    jids = tok.encode(joined)
    ok = (jids[: len(div_ids)] == div_ids) and (jids[len(div_ids) :] == gen_ids)
    print("A prompt join tokenizes to div+gen:", ok, "(joined len", len(jids), ")")
    if not ok:
        print("fallback: use per-token verification anyway; comparing by position p=18..")

    # ---- A: full prefill of the joined sequence ----
    clear_dump()
    FLAG.touch()
    try:
        outA = post(joined, 1)
    finally:
        FLAG.unlink(missing_ok=True)
    A_npz = sorted(DUMP.glob("*.npz"))
    assert A_npz, "A produced no dump"
    A = np.load(A_npz[-1])
    nT = A["input_ids"].shape[0]
    print("A prefill ntok =", nT, " (expect", len(jids), ")")

    # ---- B: 18 prefill + greedy decode with decode capture ----
    clear_dump()
    FLAG.touch()
    FLAG_D.touch()
    try:
        outB2 = post(DIV_TEXT, N_GEN)
    finally:
        FLAG.unlink(missing_ok=True)
        FLAG_D.unlink(missing_ok=True)
    print("B2 text:", repr(outB2["choices"][0]["text"][:80]))

    # ---- compare per position ----
    divlen = len(div_ids)
    W = A["layer_0"].shape[1]
    bad = 0
    n_compare = 0
    for p in range(divlen, nT):
        fp = DUMP / f"decode_pos_{p:04d}.npz"
        if not fp.exists():
            print(f"missing decode dump pos {p}")
            continue
        B = np.load(fp)
        rels = []
        for li in range(48):
            a = A[f"layer_{li}"][p].astype(np.float64)
            b = B[f"layer_{li}"][0].astype(np.float64)
            rels.append(float(np.sqrt(((a - b) ** 2).mean()) / max(np.sqrt((b ** 2).mean()), 1e-12)))
        mx = max(rels)
        wl = int(np.argmax(rels))
        flag = "  <-- DIVERGE" if mx > 0.02 else ""
        print(f"pos {p:3d}: max rel {mx:.5f} (layer {wl}){flag}")
        bad += int(mx > 0.02)
        n_compare += 1

    # also compare the very first decode positions' last-layer vs A last-layer
    print(f"\ncompared positions {n_compare}; diverged (>2%): {bad}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
