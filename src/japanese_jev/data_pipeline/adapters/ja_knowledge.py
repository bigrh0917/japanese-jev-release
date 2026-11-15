"""Self-built evaluation set: 200 items on knowledge of Japan and the Japanese language (evalsets/ja_knowledge/items.jsonl). Evaluation only.

Where ja_ambiguity tests pragmatics, this set tests whether the model knows things. 12 categories: 慣用句・ことわざ,
四字熟語, 敬語の正誤, 漢字の読み, 和製英語, 助数詞, 生活・社会制度（税率、緊急番号、祝日 …）, ビジネスマナー,
地理（県庁所在地 …）, 歴史・年号, 文化・年中行事, 擬音語・擬態語.
170 choice (4-way) + 30 noul (correct / incorrect 敬語 and マナー). All answers are stable facts; nothing time-sensitive
(such as the current prime minister).
"""

from __future__ import annotations

from ._evalset import load_items as _load_items, rows_from_items

NAME = "ja_knowledge"
HF_ID = None
HF_CONFIG = None
KINDS = ("choice", "noul")
LICENSE = "Self-built evaluation set (released under CC BY 4.0; evaluation only in this project, never used to train its models)"
EVAL_ONLY = True


def load_items():
    return _load_items(NAME)


def rows(split: str, seed: int = 0, limit: int | None = None):
    return rows_from_items(NAME, seed, limit)
