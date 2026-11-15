"""Distillation step 1: have the teacher (the Jev API) give a probability for every option of the training items,
and build a second dataset data_distill/ from them.

    export OPENROUTER_API_KEY=sk-or-...
    python scripts/distill_label.py --max 20          # try 20 items first
    python scripts/distill_label.py                   # everything; Ctrl+C at any time and rerun the same command to continue
    python scripts/distill_label.py --build-only      # no requests; rewrite train.jsonl from the answers already collected

Item sources (--inputs, three by default):
  data_ft/train.jsonl              the continued-training set (synthetic data v1 is here; items duplicated below are asked once)
  data/train.jsonl                 the training set written by build
  datasets/distill/*.jsonl         new items with a question but no answer (optional; label may be missing)
valid / calib / test_* / final_* are never sent: they take no part in training.

Requests can fail midway (revoked key, credits used up), so:
  - every answer is written to data_distill/raw.jsonl as soon as it arrives, so nothing is lost; on a rerun,
    items already asked are neither asked nor charged again
  - items are sent round-robin across sources (synthetic data first), so wherever it stops, the collected part
    is balanced across sources
  - it stops by itself after 8 consecutive failures; --budget is the most to spend in this run, in USD (default 3)
  - every run ends by rewriting data_distill/train.jsonl from the answers collected so far

In data_distill/:
  raw.jsonl     raw teacher responses (key = hash of the item content)
  train.jsonl   the original rows + "soft" (the teacher's probabilities, in the same order as options); items without soft are left out
  other files   copied from data/ as they are (valid, calib, test_*, final_*), so evaluation numbers are directly comparable with a model trained without distillation
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_jev_api import LETTERS, VIA, call, probs_from, question_for  # noqa: E402


def key_of(r: dict) -> str:
    s = json.dumps([r["kind"], r["state"], r["question"], r["options"]], ensure_ascii=False)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:20]


def load_inputs(patterns: list[str]) -> list[dict]:
    rows, seen = [], set()
    for pat in patterns:
        for path in sorted(glob.glob(str(ROOT / pat))):
            for line in open(path, encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                if "state" not in r or "options" not in r:
                    continue
                r.setdefault("n", len(r["options"]))
                r["_key"] = key_of(r)
                if r["_key"] not in seen:            # identical items are asked only once
                    seen.add(r["_key"])
                    rows.append(r)
    return rows


def interleave(rows: list[dict]) -> list[dict]:
    """Take items from the sources in turn, synthetic data first: wherever it stops, every source is covered."""
    by = defaultdict(list)
    for r in rows:
        by[r.get("source", "?")].append(r)
    srcs = sorted(by, key=lambda s: (not s.startswith("synthetic"), s))
    out, i = [], 0
    while any(by.values()):
        for s in srcs:
            if i < len(by[s]):
                out.append(by[s][i])
        i += 1
        if i > max(len(v) for v in by.values()):
            break
    return out


def build(rows: list[dict], done: dict, out: Path) -> None:
    n = agree = has_gold = 0
    per, per_gold, per_agree = Counter(), Counter(), Counter()
    with open(out / "train.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            resp = done.get(r["_key"])
            if not resp:
                continue
            _, keys = question_for(r)
            p = probs_from(r, (resp.get("answers") or {}).get("q") or {}, keys)
            if p is None or sum(p) <= 0:
                continue
            s = sum(p)
            row = {k: v for k, v in r.items() if not k.startswith("_")}
            row["soft"] = [round(x / s, 4) for x in p]
            top = LETTERS[max(range(len(p)), key=p.__getitem__)]
            if row.get("label"):
                has_gold += 1
                agree += top == row["label"]
                base = row.get("source", "?").split("+")[0]
                per_gold[base] += 1
                per_agree[base] += top == row["label"]
            else:                                    # new items without a label: the option with the highest teacher probability becomes the answer
                row["label"], row["completion"] = top, " " + top
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            per[row.get("source", "?")] += 1
    for p in (ROOT / "data").glob("*.jsonl"):
        if p.name != "train.jsonl":
            shutil.copy(p, out / p.name)
    print(f"data_distill/train.jsonl: {n} rows ({len(rows)} items in total, {n} with a teacher answer)")
    print("  per source: " + "  ".join(f"{k} {v}" for k, v in sorted(per.items())))
    if has_gold:
        print(f"  teacher agreement with the original labels: {agree / has_gold:.3f} ({has_gold} labeled items)")
        print("  by source: " + "  ".join(f"{k} {per_agree[k] / v:.2f}" for k, v in sorted(per_gold.items())))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--via", choices=list(VIA), default="openrouter")
    ap.add_argument("--model", default=None)
    ap.add_argument("--inputs", nargs="*", default=["data_ft/train.jsonl", "data/train.jsonl", "datasets/distill/*.jsonl"])
    ap.add_argument("--out", default="data_distill")
    ap.add_argument("--max", type=int, default=0, help="maximum number of items to send in this run (for a trial run)")
    ap.add_argument("--budget", type=float, default=3.0, help="maximum to spend in this run, in USD")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--build-only", action="store_true")
    a = ap.parse_args()

    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "raw.jsonl"
    done = {}
    if raw_path.exists():
        for line in open(raw_path, encoding="utf-8"):
            r = json.loads(line)
            done[r["key"]] = r["response"]

    rows = interleave(load_inputs(a.inputs))
    todo = [r for r in rows if r["_key"] not in done]
    if a.max:
        todo = todo[:a.max]
    print(f"items {len(rows)}  already answered {len(rows) - len([r for r in rows if r['_key'] not in done])}  to send now {len(todo)}")

    if todo and not a.build_only:
        url, default_model, env = VIA[a.via]
        model = a.model or default_model
        key = os.environ.get(env)
        if not key:
            sys.exit(f"run export {env}=<your key> first")
        stop = threading.Event()

        def work(r):
            return {"_error": "stopped"} if stop.is_set() else call(url, key, model, r)

        cost, fails, sent = 0.0, 0, 0
        try:
            with open(raw_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(a.workers) as ex:
                for r, resp in zip(todo, ex.map(work, todo)):
                    if resp.get("_error") == "stopped":
                        continue
                    sent += 1
                    if "_error" in resp:
                        fails += 1
                        if fails <= 3 or fails % 50 == 0:
                            print(f"  failed: {resp['_error'][:300]}")
                        if fails >= 8 and not stop.is_set():
                            print("8 consecutive failures (revoked key or no credits?); stopped sending. Everything collected so far is saved")
                            stop.set()
                        continue
                    fails = 0
                    done[r["_key"]] = resp
                    f.write(json.dumps({"key": r["_key"], "response": resp}, ensure_ascii=False) + "\n")
                    f.flush()
                    gw = ((resp.get("provider_metadata") or {}).get("gateway") or {}).get("cost")
                    cost += float(gw or (resp.get("usage") or {}).get("cost") or 0)
                    if cost >= a.budget and not stop.is_set():
                        print(f"reached the budget for this run, ${a.budget}; stopped sending")
                        stop.set()
                    if sent % 500 == 0:
                        print(f"  {sent}/{len(todo)}  total ${cost:.3f}", flush=True)
        except KeyboardInterrupt:
            print("interrupted; everything collected so far is saved")
        print(f"sent {sent} items in this run, cost ${cost:.3f}")

    build(rows, done, out)


if __name__ == "__main__":
    main()
