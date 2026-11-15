"""PAWS-X (ja): do two sentences mean the same thing. Training set.

HF fields (google-research-datasets/paws-x, config "ja"): id, sentence1, sentence2, label (1 = paraphrase).
The 49401 train rows are machine-translated; validation / test (2000 each) are human-translated. Only train is
used for training; the valid split can serve as in-distribution evaluation.
"""

from __future__ import annotations

import random

from ..common import clean_text, load_split, make_row

NAME = "pawsx_ja"
HF_ID = "google-research-datasets/paws-x"
HF_CONFIG = "ja"
KINDS = ("noul",)
LICENSE = "Open (Google asks that PAWS-X be credited as the source)"
EVAL_ONLY = False

SPLITS = {"train": "train", "valid": "validation", "test": "test"}
QUESTION = "2つの文は同じ意味ですか。"


def rows(split: str, seed: int = 0, limit: int | None = None):
    ds = load_split(HF_ID, HF_CONFIG, SPLITS[split])
    rng = random.Random(f"{NAME}-{split}-{seed}")
    for i, ex in enumerate(ds):
        if limit is not None and i >= limit:
            break
        state = f"文1: {clean_text(ex['sentence1'])}　文2: {clean_text(ex['sentence2'])}"
        yield make_row("noul", state, QUESTION, [("yes", ""), ("no", "")],
                       answer_idx=0 if int(ex["label"]) == 1 else 1,
                       source=NAME, rid=ex.get("id", i), shuffle=False, rng=rng)
