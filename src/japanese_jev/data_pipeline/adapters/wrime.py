"""WRIME ver2: intensity (0-3) of 8 emotions in social media posts. Evaluation only: the license is
research-only with no redistribution, so it is not trained on.

HF fields (shunk031/wrime, config "ver2"): sentence, user_id, datetime,
writer / reader1 / reader2 / reader3 / avg_readers (each a dict of the 8 emotions + sentiment, int8).
Splits: train / validation / test.
The avg_readers intensity (average over readers, the objective annotation) becomes a score item: one emotion is
picked at random per post, with 4 ordered levels 0-3.
"""

from __future__ import annotations

import random

from ..common import clean_text, load_split, make_row

NAME = "wrime"
HF_ID = "shunk031/wrime"
HF_CONFIG = "ver2"
KINDS = ("score",)
LICENSE = "Research use only, no redistribution (evaluation only)"
EVAL_ONLY = True

SPLITS = {"train": "train", "valid": "validation", "test": "test"}
EMOTIONS = {
    "joy": "喜び", "sadness": "悲しみ", "anticipation": "期待", "surprise": "驚き",
    "anger": "怒り", "fear": "恐れ", "disgust": "嫌悪", "trust": "信頼",
}
LEVELS = [("なし", "その感情は読み取れない"), ("弱い", "わずかに感じられる"),
          ("中程度", "はっきり感じられる"), ("強い", "非常に強く表れている")]
ANNOTATOR = "avg_readers"


def rows(split: str, seed: int = 0, limit: int | None = None):
    ds = load_split(HF_ID, HF_CONFIG, SPLITS[split])
    rng = random.Random(f"{NAME}-{split}-{seed}")
    emos = list(EMOTIONS)
    for i, ex in enumerate(ds):
        if limit is not None and i >= limit:
            break
        ann = ex.get(ANNOTATOR) or {}
        emo = rng.choice(emos)                       # one item per post, to avoid an 8x blow-up
        if emo not in ann:
            continue
        level = min(3, max(0, int(round(float(ann[emo])))))
        question = f"この投稿から読み取れる「{EMOTIONS[emo]}」の強さはどの程度ですか。"
        yield make_row("score", clean_text(ex["sentence"]), question, LEVELS, answer_idx=level,
                       source=NAME, rid=f"{i}-{emo}", shuffle=False, rng=rng)
