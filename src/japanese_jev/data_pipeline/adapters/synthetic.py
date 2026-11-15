"""Synthetic data: Japanese decision items of kinds that are rare in public datasets (negation, indirect
expressions, honorific relationships, short business texts, ...). Used for training.

The sentences and labels of v1 (about 1,100 items, train.jsonl) and v2 (about 2,000 items, train_v2.jsonl) were
written by Claude (Anthropic's LLM); the generators are scripts/make_synthetic_v1.py / make_synthetic_v2.py.
To leave this data out, remove "synthetic" from train_sources.
No sentence is shared with evalsets/ (the generators deduplicate against them).

This adapter does not generate data; it only reads the generated files datasets/synthetic/<split>*.jsonl
(both train.jsonl and train_v2.jsonl are read), one item per line:
    {"id": "t001", "kind": "choice", "state": "...", "question": "...",
     "options": [["返金", "返金を求めている"], ["再予約", "..."]], "answer": 0}
    noul options are always [["yes",""],["no",""]]; score options are ordered from the lowest level to the highest.
When no file exists it yields 0 rows and prints a hint, so build still runs without synthetic data.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from ..common import make_row

NAME = "synthetic"
HF_ID = None  # own data, placed in datasets/synthetic/ after generation
KINDS = ("choice", "noul", "score")
LICENSE = "CC BY 4.0 (sentences and labels written by Claude)"
EVAL_ONLY = False

DATA_DIR = Path(__file__).resolve().parents[4] / "datasets" / "synthetic"   # under the project root, independent of the current directory
SPLITS = {"train": "train", "valid": "valid", "test": "test"}


def rows(split: str, seed: int = 0, limit: int | None = None):
    paths = sorted(DATA_DIR.glob(f"{SPLITS[split]}*.jsonl"))      # train.jsonl (v1), train_v2.jsonl (v2), ...: every matching file is read
    if not paths:
        print(f"[synthetic] {DATA_DIR}/{SPLITS[split]}*.jsonl does not exist, skipping (run scripts/make_synthetic_v1.py first)")
        return
    rng = random.Random(f"{NAME}-{split}-{seed}")
    lines = [line for p in paths for line in open(p, encoding="utf-8") if line.strip()]
    if True:
        for i, line in enumerate(lines):
            if limit is not None and i >= limit:
                break
            d = json.loads(line)
            options = [(str(n), str(desc)) for n, desc in d["options"]]
            yield make_row(d["kind"], d["state"].strip(), d["question"].strip(), options,   # whitespace is not collapsed, to match how evalsets are written
                           answer_idx=int(d["answer"]), source=NAME, rid=d.get("id", i),
                           shuffle=(split == "train" and d["kind"] == "choice"), rng=rng)
