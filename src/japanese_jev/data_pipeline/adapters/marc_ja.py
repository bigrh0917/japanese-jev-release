"""MARC-ja (JGLUE): positive / negative product reviews. Evaluation only: the Amazon review corpus has its own terms of use, so it is not trained on.

HF fields (shunk031/JGLUE, config "MARC-ja"): sentence, label (ClassLabel positive / negative / neutral; only the first two are kept by default), review_id.
Splits: train / validation. This dataset downloads the raw corpus from Amazon S3, which often fails; when it does,
the set is skipped so that it does not block the whole build.
"""

from __future__ import annotations

import random

from ..common import clean_text, label_names, load_split, make_row

NAME = "marc_ja"
HF_ID = "shunk031/JGLUE"
HF_CONFIG = "MARC-ja"
KINDS = ("noul",)
LICENSE = "Amazon Customer Reviews terms (evaluation only, or not at all)"
EVAL_ONLY = True

SPLITS = {"train": "train", "valid": "validation", "test": "validation"}
QUESTION = "このレビューは商品に対して肯定的ですか。"
LABELS = ["positive", "negative", "neutral"]


def rows(split: str, seed: int = 0, limit: int | None = None):
    try:
        ds = load_split(HF_ID, HF_CONFIG, SPLITS[split])
    except RuntimeError as e:
        print(f"[marc_ja] failed to load, skipping this evaluation set: {str(e).splitlines()[0]}")
        return
    names = label_names(ds, "label") or LABELS
    rng = random.Random(f"{NAME}-{split}-{seed}")
    for i, ex in enumerate(ds):
        if limit is not None and i >= limit:
            break
        lab = ex["label"]
        lab = names[lab] if isinstance(lab, int) else str(lab)
        if lab not in ("positive", "negative"):
            continue
        yield make_row("noul", clean_text(ex["sentence"])[:400], QUESTION, [("yes", ""), ("no", "")],
                       answer_idx=0 if lab == "positive" else 1,
                       source=NAME, rid=ex.get("review_id", i), shuffle=False, rng=rng)
