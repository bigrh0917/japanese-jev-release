"""JCommonsenseQA (JGLUE): 5-way commonsense questions. Evaluation only (out-of-distribution: the model has not seen this task).

HF fields: q_id, question, choice0..choice4, label (ClassLabel choice0..choice4, or an integer in the raw GitHub files).
JGLUE does not publish test, so validation is used. The CC BY-SA 4.0 license allows training; it is kept out of
training here so that it stays an out-of-distribution evaluation set.
"""

from __future__ import annotations

import random

from ..common import clean_text, load_split, make_row
from . import _jglue

NAME = "jcommonsenseqa"
HF_ID = "shunk031/JGLUE"
HF_CONFIG = "JCommonsenseQA"
KINDS = ("choice",)
LICENSE = "CC BY-SA 4.0 (evaluation only in this project)"
EVAL_ONLY = True

SPLITS = {"train": "train", "valid": "validation", "test": "validation"}
QUESTION = "次の質問に対する答えとして最も適切なものはどれですか。"
CHOICE_KEYS = ["choice0", "choice1", "choice2", "choice3", "choice4"]


def rows(split: str, seed: int = 0, limit: int | None = None):
    ds = load_split(HF_ID, HF_CONFIG, SPLITS[split], fallback=lambda: _jglue.github_rows("jcommonsenseqa", split))
    rng = random.Random(f"{NAME}-{split}-{seed}")
    for i, ex in enumerate(ds):
        if limit is not None and i >= limit:
            break
        lab = ex["label"]
        if isinstance(lab, str):                   # "choice3" or "3"
            lab = int(lab.replace("choice", ""))
        options = [(clean_text(ex[k]), "") for k in CHOICE_KEYS]
        yield make_row("choice", state=clean_text(ex["question"]), question=QUESTION, options=options,
                       answer_idx=int(lab), source=NAME, rid=ex.get("q_id", i),
                       shuffle=False, rng=rng)
