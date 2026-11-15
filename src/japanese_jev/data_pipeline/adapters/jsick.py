"""JSICK: sentence-pair entailment + relatedness 1-5. Training set.

HF fields (hpprc/jsick, config "base"; original / stress / stress-original also exist, with different column names):
id, premise, hypothesis, label (0 entailment / 1 neutral / 2 contradiction), score (relatedness 1.0-5.0).
Splits: train 4500 / test 4927, no validation.
Each sample yields two rows:
  noul   "前提から仮説が導けますか" -> yes for entailment
  score  relatedness rounded to 1-5, mapped to 5 ordered levels (not shuffled; the order is the scale)
"""

from __future__ import annotations

import random

from ..common import clean_text, label_names, load_split, make_row

NAME = "jsick"
HF_ID = "hpprc/jsick"
HF_CONFIG = "base"
KINDS = ("noul", "score")
LICENSE = "CC BY-SA 4.0"
EVAL_ONLY = False

LABELS = ["entailment", "neutral", "contradiction"]
SPLITS = {"train": "train", "valid": "test", "test": "test"}
NOUL_Q = "前提が正しいとき、仮説も必ず正しいと言えますか。"
SCORE_Q = "2つの文の意味はどのくらい関連していますか。"
SCORE_LEVELS = [
    ("無関係", "2つの文は全く別の内容"),
    ("ほぼ無関係", "共通点はわずか"),
    ("やや関連", "話題は近いが意味は違う"),
    ("かなり関連", "意味の大部分が重なる"),
    ("ほぼ同じ", "ほぼ同じ意味"),
]


def rows(split: str, seed: int = 0, limit: int | None = None):
    ds = load_split(HF_ID, HF_CONFIG, SPLITS[split])
    names = label_names(ds, "label") or LABELS
    rng = random.Random(f"{NAME}-{split}-{seed}")
    for i, ex in enumerate(ds):
        if limit is not None and i >= limit:
            break
        state = f"前提: {clean_text(ex['premise'])}　仮説: {clean_text(ex['hypothesis'])}"
        lab = ex["label"]
        lab = names[lab] if isinstance(lab, int) else str(lab)
        yield make_row("noul", state, NOUL_Q, [("yes", ""), ("no", "")],
                       answer_idx=0 if lab == "entailment" else 1,
                       source=NAME, rid=f"{ex['id']}-noul", shuffle=False, rng=rng)
        score = ex.get("score")
        if score is not None:
            level = min(4, max(0, int(round(float(score))) - 1))     # 1–5 → 0–4
            yield make_row("score", state, SCORE_Q, SCORE_LEVELS, answer_idx=level,
                           source=NAME, rid=f"{ex['id']}-score", shuffle=False, rng=rng)
