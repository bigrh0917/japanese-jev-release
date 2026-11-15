"""JNLI (JGLUE): premise + hypothesis -> entailment / contradiction / neutral. Training set.

HF fields: sentence_pair_id, yjcaptions_id, sentence1, sentence2, label (ClassLabel: entailment / contradiction / neutral).
Splits: train / validation (JGLUE does not publish test).
Turned into a 3-way choice with shuffled options; "neutral" gets a soft label of 0.5.
"""

from __future__ import annotations

import random

from ..common import clean_text, label_names, load_split, make_row
from . import _jglue

NAME = "jnli"
HF_ID = "shunk031/JGLUE"
HF_CONFIG = "JNLI"
KINDS = ("choice",)
LICENSE = "CC BY-SA 4.0"
EVAL_ONLY = False

LABELS = ["entailment", "contradiction", "neutral"]          # default order of the HF ClassLabel
OPTIONS = {
    "entailment": ("含意", "仮説は前提から論理的に導ける"),
    "contradiction": ("矛盾", "仮説は前提と両立しない"),
    "neutral": ("中立", "前提だけでは仮説が正しいとも間違いとも言えない"),
}
QUESTION = "前提に対して、仮説はどの関係にありますか。"
SPLITS = {"train": "train", "valid": "validation", "test": "validation"}


def rows(split: str, seed: int = 0, limit: int | None = None):
    ds = load_split(HF_ID, HF_CONFIG, SPLITS[split], fallback=lambda: _jglue.github_rows("jnli", split))
    names = label_names(ds, "label") or LABELS
    rng = random.Random(f"{NAME}-{split}-{seed}")
    keys = list(OPTIONS)
    for i, ex in enumerate(ds):
        if limit is not None and i >= limit:
            break
        lab = ex["label"]
        lab = names[lab] if isinstance(lab, int) else str(lab)
        if lab not in OPTIONS:
            continue
        # neutral means "cannot tell", so it gets a soft label: half on neutral, the rest split between the other two (sums to 1); the other two classes get hard labels
        soft = [0.5 if k == "neutral" else 0.25 for k in keys] if lab == "neutral" else None
        yield make_row(
            "choice",
            state=f"前提: {clean_text(ex['sentence1'])}　仮説: {clean_text(ex['sentence2'])}",
            question=QUESTION,
            options=[OPTIONS[k] for k in keys],
            answer_idx=keys.index(lab),
            source=NAME, rid=ex.get("sentence_pair_id", i),
            shuffle=(split == "train"), rng=rng, soft=soft,
        )
