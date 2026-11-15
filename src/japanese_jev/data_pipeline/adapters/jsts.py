"""JSTS (JGLUE): semantic similarity of two sentences, 0-5. Training set, the second source of score rows.

HF fields (shunk031/JGLUE, config "JSTS"): sentence_pair_id, yjcaptions_id, sentence1, sentence2, label (float 0.0-5.0).
Splits: train 12451 / validation 1457 (JGLUE does not publish test). License CC BY-SA 4.0, same repository as JNLI.
Why it is included: with JSICK as the only source, score does not transfer to other scales. JSTS more than doubles
the score data, and it has more 0-1 samples (completely different) than JSICK, so the model learns that the
lowest level is also a common answer.
Similarity is rounded to 6 ordered levels.
"""

from __future__ import annotations

import random

from ..common import clean_text, load_split, make_row
from . import _jglue

NAME = "jsts"
HF_ID = "shunk031/JGLUE"
HF_CONFIG = "JSTS"
KINDS = ("score",)
LICENSE = "CC BY-SA 4.0"
EVAL_ONLY = False

SPLITS = {"train": "train", "valid": "validation", "test": "validation"}
QUESTION = "2つの文の意味はどのくらい似ていますか。"
LEVELS = [
    ("全く違う", "意味に共通点がない"),
    ("ほとんど違う", "話題がわずかに重なる程度"),
    ("あまり似ていない", "一部の内容だけ共通"),
    ("やや似ている", "大筋は同じだが重要な違いがある"),
    ("かなり似ている", "細部が少し違うだけ"),
    ("同じ意味", "言い方が違うだけで意味は等しい"),
]


def rows(split: str, seed: int = 0, limit: int | None = None):
    ds = load_split(HF_ID, HF_CONFIG, SPLITS[split], fallback=lambda: _jglue.github_rows("jsts", split))
    rng = random.Random(f"{NAME}-{split}-{seed}")
    for i, ex in enumerate(ds):
        if limit is not None and i >= limit:
            break
        level = min(5, max(0, int(round(float(ex["label"])))))
        state = f"文1: {clean_text(ex['sentence1'])}　文2: {clean_text(ex['sentence2'])}"
        yield make_row("score", state, QUESTION, LEVELS, answer_idx=level,
                       source=NAME, rid=ex.get("sentence_pair_id", i), shuffle=False, rng=rng)
