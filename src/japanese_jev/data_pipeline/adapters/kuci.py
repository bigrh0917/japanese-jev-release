"""KUCI (Kyoto University commonsense inference): given a context, pick the following event out of four. Training set.

The HF version JINIAC/kuci only contains items with agreement=4 (train 24682 / validation 2968 / test 3047).
Fields: id, context, choice_a..choice_d, label ('a'-'d'), agreement, core_event_pair.
The text is morpheme-segmented (spaces between words); the spaces are removed.

Option expansion (train only): the other training sources have at most 5 options, so the model never sees letters
after F, which makes 9-way Livedoor an unseen format (ECE 0.19 before calibration). For EXPAND_RATIO of the items,
EXPAND_K options are borrowed from other items as distractors, giving 8-12 options.
The borrowed options are following events of other contexts, so they are very unlikely to be valid by accident;
the correct option is unchanged.
"""

from __future__ import annotations

import random

from ..common import load_split, make_row

NAME = "kuci"
HF_ID = "JINIAC/kuci"
HF_CONFIG = None
KINDS = ("choice",)
LICENSE = "CC BY-SA 4.0"
EVAL_ONLY = False

SPLITS = {"train": "train", "valid": "validation", "test": "test"}
QUESTION = "この文脈の後に続く出来事として最も自然なものはどれですか。"
CHOICE_KEYS = ["choice_a", "choice_b", "choice_c", "choice_d"]
EXPAND_RATIO = 0.5          # share of training items whose options are expanded
EXPAND_K = (4, 8)           # distractors borrowed per item (inclusive range), 4 + 4..8 = 8..12 options


def _unseg(s: str) -> str:
    return "".join(str(s).split())


def expand_options(options: list[tuple[str, str]], pool: list[str], k: int, rng: random.Random) -> list[tuple[str, str]]:
    """Append k options from pool that are not already in options; the index of the correct option is unchanged (it stays in front)."""
    have = {name for name, _ in options}
    extra: list[tuple[str, str]] = []
    tries = 0
    while len(extra) < k and tries < k * 20:
        tries += 1
        cand = rng.choice(pool)
        if cand not in have:
            have.add(cand)
            extra.append((cand, ""))
    return list(options) + extra


def rows(split: str, seed: int = 0, limit: int | None = None):
    ds = load_split(HF_ID, HF_CONFIG, SPLITS[split])
    rng = random.Random(f"{NAME}-{split}-{seed}")
    expand = split == "train"
    pool = [_unseg(ex[k]) for ex in ds for k in CHOICE_KEYS] if expand else []
    for i, ex in enumerate(ds):
        if limit is not None and i >= limit:
            break
        lab = str(ex["label"]).strip().lower()
        if lab not in "abcd" or len(lab) != 1:
            continue
        options = [(_unseg(ex[k]), "") for k in CHOICE_KEYS]
        rid = ex.get("id", i)
        if expand and rng.random() < EXPAND_RATIO:
            options = expand_options(options, pool, rng.randint(*EXPAND_K), rng)
            rid = f"{rid}x{len(options)}"
        yield make_row("choice", state=_unseg(ex["context"]), question=QUESTION, options=options,
                       answer_idx="abcd".index(lab), source=NAME, rid=rid,
                       shuffle=(split == "train"), rng=rng)
