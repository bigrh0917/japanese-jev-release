"""MewsC-16 (Japanese part): first sentence of a Wikinews article -> 12 topic categories. Training set, the source of
real many-option choice items.

Source: downstreams/text-clustering/data/mewsc16/ja_*.txt in the studio-ousia/ease repository (EASE, NAACL 2022).
Two text files aligned by line: ja_sentences.txt has one sentence per line, ja_categories.txt one category per line.
1,983 items, 12 classes, very unbalanced (スポーツ 498 ... 環境 12). JMTEB uses it as a clustering benchmark; it
does not overlap with this project's evaluation sets.
License: Wikinews text CC BY 2.5, EASE repository MIT, JMTEB lists it as CC BY-SA 4.0. All allow training and
releasing weights, with attribution.

Why it is included: the other training sources have at most 5 options (the expanded KUCI options are borrowed);
this is a real 12-way task.
Each item yields two rows: all 12 classes (shuffled) + a 6-class subset (containing the correct one, shuffled),
so that items with 6-12 options are all represented.
"""

from __future__ import annotations

import random
import urllib.request
from pathlib import Path

from ..common import clean_text, make_row

NAME = "mewsc16"
HF_ID = None   # not on HF; fetch the raw files from GitHub
HF_CONFIG = None
KINDS = ("choice",)
LICENSE = "Wikinews CC BY 2.5 / EASE repository MIT (JMTEB lists CC BY-SA 4.0)"
EVAL_ONLY = False

RAW = "https://raw.githubusercontent.com/studio-ousia/ease/main/downstreams/text-clustering/data/mewsc16"
FILES = ("ja_sentences.txt", "ja_categories.txt")
CACHE = Path.home() / ".cache" / "japanese_jev" / "mewsc16"
QUESTION = "このニュース文の主題はどの分野ですか。"
CATEGORIES = [   # 12 classes; the names are the strings in the file, the one-line descriptions were written for this project
    ("政治", "選挙・政府・外交"), ("経済", "企業・市場・景気"), ("事件", "犯罪・捜査・裁判"),
    ("事故", "交通事故・災害事故"), ("スポーツ", "試合・選手・大会"), ("文化", "芸術・文学・芸能"),
    ("科学技術", "研究・発明・IT"), ("気象", "天気・台風・地震の観測"), ("健康", "医療・病気・衛生"),
    ("教育", "学校・入試・学術制度"), ("訃報", "著名人の死去"), ("環境", "環境問題・自然保護"),
]
SUBSET_K = 6


def _fetch(name: str) -> list[str]:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / name
    if not p.exists():
        with urllib.request.urlopen(f"{RAW}/{name}", timeout=60) as r:
            p.write_bytes(r.read())
    return p.read_text(encoding="utf-8").splitlines()


def load() -> list[tuple[str, str]]:
    sents, cats = (_fetch(f) for f in FILES)
    if len(sents) != len(cats):
        raise RuntimeError(f"mewsc16: the two files have different line counts: {len(sents)} vs {len(cats)}")
    return [(s, c) for s, c in zip(sents, cats) if s.strip() and c.strip()]


def rows(split: str, seed: int = 0, limit: int | None = None):
    data = load()
    names = [n for n, _ in CATEGORIES]
    rng = random.Random(f"{NAME}-{split}-{seed}")
    train = split == "train"
    for i, (sent, cat) in enumerate(data):
        if limit is not None and i >= limit:
            break
        cat = cat.strip()
        if cat not in names:
            continue
        state = clean_text(sent)
        yield make_row("choice", state, QUESTION, CATEGORIES, answer_idx=names.index(cat),
                       source=NAME, rid=i, shuffle=train, rng=rng)
        if train:   # one more row with a 6-class subset containing the correct class
            others = rng.sample([n for n in names if n != cat], SUBSET_K - 1)
            sub = [c for c in CATEGORIES if c[0] in others or c[0] == cat]
            yield make_row("choice", state, QUESTION, sub, answer_idx=[n for n, _ in sub].index(cat),
                           source=NAME, rid=f"{i}s", shuffle=True, rng=rng)
