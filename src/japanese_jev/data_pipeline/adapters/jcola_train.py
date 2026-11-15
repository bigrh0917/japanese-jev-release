"""Training split of JCoLA: is a Japanese sentence natural. Used for training, so the yes / no question type
sees more varied content.

It is a different file of the same dataset as the evaluation adapter `jcola`, with no overlapping sentences:
  here    data/jcola-v1.0/in_domain_train-v1.0.json (6,919 sentences)
  jcola   in_domain_valid + out_of_domain_valid (the evaluation set is sampled from these two files)
To be safe, training sentences identical to a sentence in either valid file are dropped.
Once this is added, the evaluation set jcola is no longer "an unseen task" but "same kind of task, unseen
sentences"; say so when reporting numbers.

83% of the raw data is natural, which would teach the model to prefer yes. This adapter balances it: all
unnatural sentences (about 1,150) plus the same number of sampled natural ones.
License CC BY-SA 4.0 (training allowed; share-alike, attribution required). Source https://github.com/osekilab/JCoLA
"""

from __future__ import annotations

import random

from ..common import clean_text, make_row
from . import jcola

NAME = "jcola_train"
HF_ID = None
KINDS = ("noul",)
LICENSE = "CC BY-SA 4.0"
EVAL_ONLY = False
PRESAMPLED = True

TRAIN_FILE = "in_domain_train-v1.0.json"


def rows(split: str, seed: int = 0, limit: int | None = None):
    if split != "train":
        return
    held = {clean_text(ex["sentence"]) for ex in jcola._load()}          # sentences in the two evaluation files
    rng = random.Random(f"{NAME}-{seed}")
    seen, good, bad = set(), [], []
    for ex in jcola._load((TRAIN_FILE,)):
        s = clean_text(ex["sentence"])
        if not s or s in seen or s in held:
            continue
        seen.add(s)
        (good if int(ex["label"]) == 1 else bad).append((ex.get("uid"), s))
    rng.shuffle(good)
    rng.shuffle(bad)
    n = min(len(good), len(bad))
    pairs = [x for g, b in zip(good[:n], bad[:n]) for x in ((g, 0), (b, 1))]
    for i, ((uid, s), ans) in enumerate(pairs):
        if limit is not None and i >= limit:
            break
        yield make_row("noul", s, jcola.QUESTION, [("yes", ""), ("no", "")], answer_idx=ans,
                       source=NAME, rid=f"tr{uid if uid is not None else i}", shuffle=False, rng=rng)
