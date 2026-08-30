# -*- coding: utf-8 -*-
"""Layer-Dump-Harness: FreeToken vs llama.cpp per-layer hyper-connection residual capture.

Modes (run on the same outdir):
  free   - enable the FreeToken hidden-dump flag, truncate D:\\temp\\opencode\\ft_hidden_dump,
           drive ONE /v1/completions request on the repo engine (default 8890), then copy the
           produced per-layer NPZ (keyed by the engine's REAL input ids) into outdir/free.npz
           and write outdir/tokens.txt (the exact token ids the engine used).
  llama  - write the token ids to llama-layerdump with the CPU reference GGUF and copy the
           per-layer .f32 files into outdir/llama/
  diff   - compare free.npz vs llama/layer_*.f32 per layer boundary, print a table and write
           outdir/diff_report.json

Examples:
  python layerdump_harness.py free  --prompt-name div15 --outdir ft_run
  python layerdump_harness.py llama --outdir ft_run
  python layerdump_harness.py diff  --outdir ft_run
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
DUMP_DIR = pathlib.Path(r"D:\temp\opencode\ft_hidden_dump")
DUMP_FLAG = pathlib.Path(r"D:\temp\opencode\ft_hidden_dump.flag")
PROMPTS = pathlib.Path(r"D:\temp\opencode\ft_ab_prompts_len2.json")
MODEL_NAME = "Qwen3.8-Flash-Next-UD-Q3_K_XL-merged.gguf"
MODEL_PATH = r"C:\Users\TM\.lmstudio\models\Qwen3.8-Flash-Next-UD-Q3_K_XL-merged.gguf"
LLAMA_EXE = pathlib.Path(
    r"C:\Users\TM\.unsloth\llama.cpp\build-layerdump\bin\Release\llama-layerdump.exe"
)


def load_prompts(path: pathlib.Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def http_post_json(url: str, body: dict, timeout: float = 900.0) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


# ---------------------------------------------------------------------------
# free
# ---------------------------------------------------------------------------


def cmd_free(args) -> int:
    prompts = load_prompts(args.prompts)
    match = [p for p in prompts if p["name"] == args.prompt_name]
    if not match:
        sys.exit(f"no prompt named {args.prompt_name} in {args.prompts}")
    text = match[0]["text"]

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if DUMP_DIR.exists():
        shutil.rmtree(DUMP_DIR)
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    DUMP_FLAG.touch()
    try:
        body = {
            "model": MODEL_NAME,
            "prompt": text,
            "max_tokens": args.max_tokens,
            "temperature": 0,
        }
        t0 = time.time()
        out = http_post_json(f"http://127.0.0.1:{args.port}/v1/completions", body)
        print(
            f"[free] request OK in {time.time() - t0:.1f}s, choice text={out['choices'][0]['text'][:40]!r}"
        )
    finally:
        DUMP_FLAG.unlink(missing_ok=True)

    npzs = sorted(DUMP_DIR.glob("*.npz"))
    if not npzs:
        sys.exit(f"[free] no npz produced under {DUMP_DIR}")
    # newest first: our request is the last prefill written
    npz = npzs[-1]
    with open(npz, "rb") as f:
        lb = f.read()
    (outdir / "free.npz").write_bytes(lb)
    print(f"[free] captured {npz.name} -> outdir/free.npz")

    import numpy as np

    ids = np.load(npz)["input_ids"].tolist()
    (outdir / "tokens.txt").write_text(" ".join(str(i) for i in ids), encoding="utf-8")
    print(f"[free] input ids (ntok={len(ids)}): {ids[:24]}{'...' if len(ids) > 24 else ''}")
    return 0


# ---------------------------------------------------------------------------
# llama
# ---------------------------------------------------------------------------


def cmd_llama(args) -> int:
    outdir = pathlib.Path(args.outdir)
    tokens_txt = pathlib.Path(args.tokens) if args.tokens else outdir / "tokens.txt"
    if not tokens_txt.exists():
        sys.exit(f"--tokens {tokens_txt} not found (run the free step first)")
    llama_dir = outdir / ("llama_f32" if args.f32 else "llama")
    if llama_dir.exists():
        shutil.rmtree(llama_dir)
    llama_dir.mkdir(parents=True, exist_ok=True)

    exe = pathlib.Path(args.exe) if args.exe else LLAMA_EXE
    cmd = [
        str(exe), "-m", args.model, "-i", str(tokens_txt), "-o", str(llama_dir),
        "-t", str(args.threads),
    ]
    if args.f32:
        cmd.append("-f32")
    print("[llama] " + " ".join(cmd))
    t0 = time.time()
    import subprocess

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        sys.exit(f"[llama] layerdump failed with rc={proc.returncode}")
    print(f"[llama] done in {time.time() - t0:.1f}s -> {llama_dir}")
    return 0


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------


def cmd_diff(args) -> int:
    import numpy as np

    outdir = pathlib.Path(args.outdir)
    free = np.load(outdir / "free.npz")
    llama_dir = outdir / ("llama_f32" if args.f32 else "llama")

    meta = json.loads((llama_dir / "meta.json").read_text(encoding="utf-8"))
    n_layer, n_tokens, w = meta["n_layer"], meta["n_tokens"], meta["n_embd_out"]
    llama_states = {}
    for il in range(1, n_layer + 1):  # layer_001..layer_048 = after layers 0..47
        raw = np.fromfile(llama_dir / f"layer_{il:03d}.f32", dtype=np.float32)
        llama_states[il - 1] = raw.reshape(n_tokens, w)

    rows = []
    for li in range(n_layer):
        b = llama_states[li].astype(np.float64)
        a = free[f"layer_{li}"].astype(np.float64)
        diff = a - b
        rms_b = float(np.sqrt(np.mean(b * b)))
        rms_d = float(np.sqrt(np.mean(diff * diff)))
        max_abs = float(np.max(np.abs(diff)))
        per_tok = np.max(np.abs(diff), axis=1)
        n_big = int(np.sum(per_tok > args.big_thresh))
        rows.append(
            {
                "layer": li,
                "rms_llama": rms_b,
                "rms_diff": rms_d,
                "rel": rms_d / max(rms_b, 1e-12),
                "max_abs": max_abs,
                "last_token_max": float(per_tok[-1]),
                "worst_token": int(np.argmax(per_tok)),
                "n_big": n_big,
            }
        )

    print("layer rms_llama  rms_diff   rel     max_abs  last_tok worst_tok n_big")
    for r in rows:
        print(
            f"{r['layer']:>5} {r['rms_llama']:10.4f} {r['rms_diff']:9.6f} {r['rel']:8.5f} "
            f"{r['max_abs']:9.4f} {r['last_token_max']:9.4f} {r['worst_token']:>9} {r['n_big']:>5}"
        )

    (outdir / "diff_report.json").write_text(
        json.dumps(rows, indent=1), encoding="utf-8"
    )
    print(f"-> {outdir / 'diff_report.json'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("free")
    f.add_argument("--prompt-name", default="div15")
    f.add_argument("--prompts", type=pathlib.Path, default=PROMPTS)
    f.add_argument("--port", type=int, default=8890)
    f.add_argument("--max-tokens", type=int, default=4)
    f.add_argument("--outdir", required=True)
    f.set_defaults(func=cmd_free)

    l = sub.add_parser("llama")
    l.add_argument("--outdir", required=True)
    l.add_argument("--tokens", type=pathlib.Path, default=None)
    l.add_argument("--exe", type=pathlib.Path, default=None)
    l.add_argument("--model", default=MODEL_PATH)
    l.add_argument("--threads", type=int, default=30)
    l.add_argument("--timeout", type=int, default=3600)
    l.add_argument("--f32", action="store_true", help="A/B: fp32 dense weights (no Q8_0 activation quantization)")
    l.set_defaults(func=cmd_llama)

    d = sub.add_parser("diff")
    d.add_argument("--outdir", required=True)
    d.add_argument("--big-thresh", type=float, default=0.5)
    d.add_argument("--f32", action="store_true", help="diff against llama_f32 (fp32-dense A/B) instead of llama")
    d.set_defaults(func=cmd_diff)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
