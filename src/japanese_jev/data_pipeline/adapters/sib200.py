"""SIB-200 (Japanese, jpn_Jpan): 7-class topic classification of Flores-200 sentences. **Used only to fit
temperatures**; neither trained on nor evaluated.

HF fields (Davlan/sib200, config "jpn_Jpan"): index_id, category (string), text.
Splits train 701 / validation 99 / test 204, 1,004 items in total; this project merges all three into one
calibration set (split="all"). License CC BY-SA 4.0.
7 classes: science/technology, travel, politics, sports, health, entertainment, geography.

Why a third-party calibration set: when the temperature is fitted on calib cut from the training sources, the
model is over-confident there (it trained on the same distribution), so T comes out too large; on unseen data
this over-corrects, and ECE got worse after calibration on four out-of-distribution sets.
A T fitted on a dataset that is neither trained on nor evaluated transfers better to other unseen data.
"""

from __future__ import annotations

import random

from ..common import clean_text, label_names, load_split, make_row

NAME = "sib200"
HF_ID = "Davlan/sib200"
HF_CONFIG = "jpn_Jpan"
KINDS = ("choice",)
LICENSE = "CC BY-SA 4.0 (used only to fit temperatures in this project)"
EVAL_ONLY = False

SPLITS = {"train": ["train"], "valid": ["validation"], "test": ["test"], "all": ["train", "validation", "test"]}
QUESTION = "この文の主題はどの分野ですか。"
CATEGORIES = [
    ("science/technology", "科学・技術"), ("travel", "旅行・観光"), ("politics", "政治"),
    ("sports", "スポーツ"), ("health", "健康・医療"), ("entertainment", "エンターテインメント"), ("geography", "地理"),
]
NAMES = [n for n, _ in CATEGORIES]


def rows(split: str, seed: int = 0, limit: int | None = None):
    rng = random.Random(f"{NAME}-{split}-{seed}")
    count = 0
    for hf_split in SPLITS[split]:
        ds = load_split(HF_ID, HF_CONFIG, hf_split)
        names = label_names(ds, "category")
        for i, ex in enumerate(ds):
            if limit is not None and count >= limit:
                return
            cat = ex["category"]
            cat = names[cat] if isinstance(cat, int) and names else str(cat).strip()
            if cat not in NAMES:
                continue
            count += 1
            yield make_row("choice", clean_text(ex["text"]), QUESTION, CATEGORIES, answer_idx=NAMES.index(cat),
                           source=NAME, rid=f"{hf_split}-{ex.get('index_id', i)}", shuffle=False, rng=rng)
