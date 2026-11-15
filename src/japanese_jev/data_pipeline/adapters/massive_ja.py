"""MASSIVE (ja-JP): 60-class intent classification. Evaluation only (out-of-distribution).

HF fields (AmazonScience/massive, config "ja-JP"): id, locale, partition, scenario, intent (ClassLabel, 60 classes),
utt, annot_utt, worker_id, ... Splits: train 11514 / validation 2033 / test 2974. License CC BY 4.0.

60 intents exceed the 16 option letters, so each item is negatively sampled down to K candidates (8 by default:
the correct intent + 7 random wrong ones, fixed seed).
The accuracy here is therefore not comparable to published 60-way numbers.
Intent names are English snake_case (e.g. alarm_set) and are used as option names as they are; descriptions are empty.
"""

from __future__ import annotations

import random

from ..common import clean_text, label_names, load_split, make_row, sample_negatives

NAME = "massive_ja"
HF_ID = "AmazonScience/massive"
HF_CONFIG = "ja-JP"
KINDS = ("choice",)
LICENSE = "CC BY 4.0 (evaluation only in this project)"
EVAL_ONLY = True

SPLITS = {"train": "train", "valid": "validation", "test": "test"}
QUESTION = "この発話の意図（インテント）はどれですか。"
K = 8


def rows(split: str, seed: int = 0, limit: int | None = None):
    ds = load_split(HF_ID, HF_CONFIG, SPLITS[split])
    names = label_names(ds, "intent")
    if names is None:
        raise RuntimeError("massive: the intent column is not a ClassLabel, so intent names are unavailable; check the dataset version")
    rng = random.Random(f"{NAME}-{split}-{seed}")
    for i, ex in enumerate(ds):
        if limit is not None and i >= limit:
            break
        gold = names[int(ex["intent"])]
        cands = [gold] + sample_negatives(names, gold, K, rng)
        rng.shuffle(cands)                          # candidates are shuffled with a fixed seed so the correct one is not always A
        yield make_row("choice", state=clean_text(ex["utt"]), question=QUESTION,
                       options=[(c, "") for c in cands], answer_idx=cands.index(gold),
                       source=NAME, rid=ex.get("id", i), shuffle=False, rng=rng)
