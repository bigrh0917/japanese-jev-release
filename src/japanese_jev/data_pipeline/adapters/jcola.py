"""JCoLA: is a Japanese sentence natural (acceptability judgments by linguists). Evaluation only, as the
out-of-distribution set for the yes / no question type.

Does not go through HuggingFace; reads the raw files of the JCoLA repository:
  https://github.com/osekilab/JCoLA  data/jcola-v1.0/{in_domain_valid, out_of_domain_valid}-v1.0.json
Fields: uid, source, label (1 = natural, 0 = unnatural), sentence, ...
License CC BY-SA 4.0. Its training split (a different file, no overlapping sentences) can be added to training
through the `jcola_train` adapter; once it is, this set measures "same kind of task, unseen sentences" rather
than "an unseen task".

In the raw data 73-84% of the sentences are natural, so answering yes every time already scores about 80% and
accuracy says little. This adapter therefore balances the set: all 322 unnatural sentences in the two valid files
are used, the same number of natural sentences is sampled, and yes / no alternate (PRESAMPLED).
The numbers in this project are therefore not comparable to the accuracy in the JCoLA paper (which uses the
original ratio and also reports MCC).
"""

from __future__ import annotations

import json
import random
import urllib.request

from ..common import clean_text, make_row

NAME = "jcola"
HF_ID = None
KINDS = ("noul",)
LICENSE = "CC BY-SA 4.0 (evaluation only in this project)"
EVAL_ONLY = True
PRESAMPLED = True

RAW = "https://raw.githubusercontent.com/osekilab/JCoLA/main/data/jcola-v1.0"
FILES = ("in_domain_valid-v1.0.json", "out_of_domain_valid-v1.0.json")
QUESTION = "この文は日本語として自然ですか。"


def _load(files: tuple[str, ...] = FILES) -> list[dict]:
    out = []
    for f in files:
        with urllib.request.urlopen(f"{RAW}/{f}", timeout=60) as r:
            text = r.read().decode("utf-8").strip()
        out += json.loads(text) if text.startswith("[") else [json.loads(l) for l in text.splitlines() if l.strip()]
    return out


def rows(split: str, seed: int = 0, limit: int | None = None):
    if split == "train":
        return
    rng = random.Random(f"{NAME}-{seed}")
    seen, good, bad = set(), [], []
    for ex in _load():
        s = clean_text(ex["sentence"])
        if not s or s in seen:
            continue
        seen.add(s)
        (good if int(ex["label"]) == 1 else bad).append((ex.get("uid"), s))
    rng.shuffle(good)
    rng.shuffle(bad)
    n = min(len(good), len(bad))
    pairs = [x for g, b in zip(good[:n], bad[:n]) for x in ((g, 0), (b, 1))]      # yes / no alternate, so the first N rows are also half and half
    for i, ((uid, s), ans) in enumerate(pairs):
        if limit is not None and i >= limit:
            break
        yield make_row("noul", s, QUESTION, [("yes", ""), ("no", "")], answer_idx=ans,
                       source=NAME, rid=uid if uid is not None else i, shuffle=False, rng=rng)
