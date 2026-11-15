"""Clean latency measurement: japanese-jev bench --config ... --work-dir runs/<dir>

The p50 that evaluate records in passing is unreliable (it mixes in tokenization, warm-up, an unmerged LoRA and
whatever else the machine was doing). This module measures latency properly:
  - the same items (first 200 of test_jcommonsenseqa by default), several settings run in turn inside one process,
    ROUNDS rounds each, p50 / p90 over all samples
  - WARMUP items first; tokenization and the forward pass are timed separately; torch.cuda.synchronize() after
    every forward pass on GPU
  - settings: LoRA unmerged / LoRA merged into the weights (use this for deployment) / merged with 4 in-context
    examples (same prompt length as the untrained 4-shot baseline)
Two more comparisons (both with the last model above, no examples):
  - batching: 1 / 8 / 16 / 32 items per forward pass (left-padded, reading the last position of each item);
    reports the average time per item and checks that the answers match the single-item answers
  - generation: the same model generates a JSON-length answer token by token (the token count of
    `{"answer": "B", "confidence": 0.85}`). Only time is measured, not accuracy: it answers "how much longer does
    it take to generate a structured answer instead of reading logits"
Results go to work_dir/bench.json. Close the browser and other GPU users first, and plug in the laptop.
Torch backend only.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import readout
from .evaluate import fewshot_prefixes

WARMUP, ROUNDS, N_ITEMS, SHOTS = 20, 3, 200, 4


def _forward_times(model, tok, prompts: list[str]):
    import torch

    tok_ms, fwd_ms, n_tok = [], [], []
    cuda = torch.cuda.is_available()
    with torch.no_grad():
        for p in prompts:
            t0 = time.perf_counter()
            enc = tok(p, return_tensors="pt").to(model.device)
            t1 = time.perf_counter()
            readout.last_token_logits(model, enc)[0].float().cpu()
            if cuda:
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            tok_ms.append((t1 - t0) * 1000); fwd_ms.append((t2 - t1) * 1000); n_tok.append(int(enc["input_ids"].shape[1]))
    return tok_ms, fwd_ms, n_tok


JSON_ANSWER = '{"answer": "B", "confidence": 0.85}'
BATCH_SIZES = (1, 8, 16, 32)


def _batched(model, tok, prompts: list[str], n_opts: list[int], bs: int):
    """Left-pad and run one batch. Returns (average ms per item, argmax letter index of each item)."""
    import torch

    ids = readout.letter_ids(tok)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    old_side, tok.padding_side = tok.padding_side, "left"
    picks, total = [], 0.0
    cuda = torch.cuda.is_available()
    try:
        with torch.no_grad():
            for i in range(0, len(prompts), bs):
                chunk = prompts[i:i + bs]
                t0 = time.perf_counter()
                enc = tok(chunk, return_tensors="pt", padding=True).to(model.device)
                last = readout.last_token_logits(model, enc).float().cpu()
                if cuda:
                    torch.cuda.synchronize()
                total += time.perf_counter() - t0
                for row, n in zip(last, n_opts[i:i + bs]):
                    picks.append(int(row[ids[:n]].argmax()))
    finally:
        tok.padding_side = old_side
    return total * 1000 / len(prompts), picks


def _generate_times(model, tok, prompts: list[str], new_tokens: int):
    import torch

    out = []
    cuda = torch.cuda.is_available()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    with torch.no_grad():
        for p in prompts:
            t0 = time.perf_counter()
            enc = tok(p, return_tensors="pt").to(model.device)
            model.generate(**enc, max_new_tokens=new_tokens, min_new_tokens=new_tokens, do_sample=False, pad_token_id=pad)
            if cuda:
                torch.cuda.synchronize()
            out.append((time.perf_counter() - t0) * 1000)
    return out


def main(cfg: dict) -> None:
    import torch

    work, data_dir = Path(cfg["work_dir"]), Path(cfg["data_dir"])
    source = cfg.get("bench_source", "jcommonsenseqa")
    rows = readout.read_jsonl(data_dir / f"test_{source}.jsonl")[:N_ITEMS]
    prefix = fewshot_prefixes(readout.read_jsonl(data_dir / "calib.jsonl"), SHOTS, int(cfg.get("seed", 0)))
    plain = [r["prompt"] for r in rows]
    shots = [prefix.get(r["kind"], "") + r["prompt"] for r in rows]

    model, tok = readout.load_model(cfg, backend="torch")
    settings = []
    has_adapter = hasattr(model, "merge_and_unload")
    if has_adapter:
        settings.append(("LoRA unmerged", model, plain))
    acc = {name: {"tok": [], "fwd": [], "n": []} for name in ("LoRA unmerged", "LoRA merged", f"merged + {SHOTS} shots", "base", f"base + {SHOTS} shots")}

    def run(name, m, prompts):
        _forward_times(m, tok, prompts[:WARMUP])
        for _ in range(ROUNDS):
            a, b, c = _forward_times(m, tok, prompts)
            acc[name]["tok"] += a; acc[name]["fwd"] += b; acc[name]["n"] += c

    if has_adapter:
        run("LoRA unmerged", model, plain)
        model = model.merge_and_unload()
        model.eval()
        for _ in range(1):          # alternate the two prompts to avoid order effects
            run("LoRA merged", model, plain)
            run(f"merged + {SHOTS} shots", model, shots)
    else:
        run("base", model, plain)
        run(f"base + {SHOTS} shots", model, shots)

    # ---- batching: average time per item, and check the answers match the single-item answers ----
    n_opts = [int(r["n"]) for r in rows]
    _batched(model, tok, plain[:32], n_opts[:32], 8)                      # warm-up
    batch, ref = {}, None
    for bs in BATCH_SIZES:
        best, picks = min((_batched(model, tok, plain, n_opts, bs) for _ in range(ROUNDS)), key=lambda x: x[0])
        ref = ref or picks
        batch[str(bs)] = {"per_item_ms": round(best, 2), "same_answer_as_single": round(sum(a == b for a, b in zip(ref, picks)) / len(ref), 4)}

    # ---- generation: how long it takes to generate an output as long as a JSON answer ----
    k = len(tok.encode(JSON_ANSWER, add_special_tokens=False))
    sub = plain[:100]
    _generate_times(model, tok, sub[:5], k)
    gen = {"json_example": JSON_ANSWER, "new_tokens": k, "items": len(sub)}
    for name, n_new in (("1 token (letter only)", 1), (f"{k} tokens (JSON answer)", k)):
        ts = _generate_times(model, tok, sub, n_new)
        gen[name] = {"p50_ms": round(float(np.median(ts)), 2), "p90_ms": round(float(np.percentile(ts, 90)), 2)}

    out = {"source": source, "items": len(rows), "rounds": ROUNDS, "batch": batch, "generate": gen,
           "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "settings": {}}
    print(f"\n{'setting':<16}{'prompt tokens':>10}{'tokenize p50':>10}{'forward p50':>10}{'forward p90':>10}{'total p50':>10}  (ms)")
    for name, d in acc.items():
        if not d["fwd"]:
            continue
        tot = np.array(d["tok"]) + np.array(d["fwd"])
        s = {"prompt_tokens_mean": round(float(np.mean(d["n"])), 1), "tokenize_p50_ms": round(float(np.median(d["tok"])), 2),
             "forward_p50_ms": round(float(np.median(d["fwd"])), 2), "forward_p90_ms": round(float(np.percentile(d["fwd"], 90)), 2),
             "total_p50_ms": round(float(np.median(tot)), 2), "samples": len(d["fwd"])}
        out["settings"][name] = s
        print(f"{name:<16}{s['prompt_tokens_mean']:>10}{s['tokenize_p50_ms']:>10}{s['forward_p50_ms']:>10}{s['forward_p90_ms']:>10}{s['total_p50_ms']:>10}")
    print("\nbatching (no examples)   avg ms per item   same answers as single")
    for bs, d in batch.items():
        print(f"  batch of {bs:>2}      {d['per_item_ms']:>10}   {d['same_answer_as_single']:>10.1%}")
    print(f"\ngeneration (no examples, same model)   p50 ms   p90 ms")
    for name, d in gen.items():
        if isinstance(d, dict):
            print(f"  {name:<24}{d['p50_ms']:>8}{d['p90_ms']:>9}")
    work.mkdir(parents=True, exist_ok=True)
    with open(work / "bench.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[bench] wrote {work / 'bench.json'}")
