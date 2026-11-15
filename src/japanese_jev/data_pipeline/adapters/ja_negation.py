"""Self-built evaluation set: 100 negation minimal pairs (evalsets/ja_negation/). Evaluation only.

25 groups x 4 types (肯定 / 単純否定 / 二重否定 / 否定の結論), half yes and half no, to tell "really understands double
negation" apart from "answers no whenever a negation word appears".
"""

from __future__ import annotations

from ._evalset import load_items as _load_items, rows_from_items

NAME = "ja_negation"
HF_ID = None
HF_CONFIG = None
KINDS = ("noul",)
LICENSE = "Self-built evaluation set (released under CC BY 4.0; evaluation only in this project, never used to train its models)"
EVAL_ONLY = True


def load_items():
    return _load_items(NAME)


def rows(split: str, seed: int = 0, limit: int | None = None):
    return rows_from_items(NAME, seed, limit)
