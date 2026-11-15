"""Self-built evaluation set: 300 items on ambiguous Japanese expressions (evalsets/ja_ambiguity/items.jsonl). Evaluation only.

It covers the part of Japanese classification that is easiest to get wrong: expressions whose literal and actual
meanings differ. 17 phenomena: 婉曲な断り, the polysemy of 「大丈夫 / 結構 / すみません」, 主語省略, 二重否定, ヘッジ,
暗黙の依頼（空気）, 皮肉・慇懃無礼, 否定疑問への返答, 程度表現, 若者言葉・全ひらがな, 方言, 敬語から読む関係性,
約束の実現度, 婉曲な不満, 緊急度. Scenes: customer support, 社内チャット, メール, 飲食店, 家庭, 友人 LINE, SNS, レビュー.
Items and labels were written by Claude (Anthropic's LLM). `scripts/breakdown.py` breaks accuracy down by
phenomenon / scene / primitive.
"""

from __future__ import annotations

from ._evalset import load_items as _load_items, rows_from_items

NAME = "ja_ambiguity"
HF_ID = None
HF_CONFIG = None
KINDS = ("choice", "noul", "score")
LICENSE = "Self-built evaluation set (released under CC BY 4.0; evaluation only in this project, never used to train its models)"
EVAL_ONLY = True


def load_items():
    return _load_items(NAME)


def rows(split: str, seed: int = 0, limit: int | None = None):
    """split is ignored: this set has a single file."""
    return rows_from_items(NAME, seed, limit)
