"""Livedoor news corpus: 9 news categories. Evaluation only: the license is CC BY-ND 2.1 JP, and ND forbids derivatives, so it cannot be trained on.

HF fields (shunk031/livedoor-news-corpus): url, date, title, content, category (ClassLabel, 9 classes).
Loading needs split ratios: train_ratio / val_ratio / test_ratio, with a fixed random_state.
state is the title + the first 300 characters of the body, to stay under 512 tokens.

Two things to watch:
  1. Script-based datasets go through the parquet branch in datasets 4.x, where the test split is the unshuffled
     last 10%: all 500 items are sports-watch, and a model that always answers sports-watch scores 95%. So no
     ready-made split is trusted here: all three splits are read, then sampled round-robin per category with a fixed seed.
  2. Titles and bodies often contain the site name itself (【Sports Watch】, Peachy, 独女通信 ...), and the option
     names are the site names. These strings are masked with ■ before evaluation.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict

from ..common import clean_text, label_names, load_split, make_row

NAME = "livedoor"
HF_ID = "shunk031/livedoor-news-corpus"
HF_CONFIG = None
KINDS = ("choice",)
LICENSE = "CC BY-ND 2.1 JP (no derivatives; evaluation only)"
EVAL_ONLY = True
PRESAMPLED = True   # rows() already yields a stratified order, so build just takes the first N

SPLITS = {"train": "train", "valid": "validation", "test": "test"}
QUESTION = "この記事はどの媒体（カテゴリ）のものですか。"
DESC = {   # one-line description of each of the 9 source sites; without class names, fall back to empty descriptions in ClassLabel order
    "dokujo-tsushin": "独身女性向けコラム",
    "it-life-hack": "ITライフハック",
    "kaden-channel": "家電チャンネル",
    "livedoor-homme": "男性向けライフスタイル",
    "movie-enter": "映画・エンタメ",
    "peachy": "女性向けライフスタイル",
    "smax": "スマートフォン・ガジェット",
    "sports-watch": "スポーツ",
    "topic-news": "一般ニュース・話題",
}
MAX_CHARS = 300


SITE_PAT = re.compile(
    r"独女通信|独女|ITライフハック|IT\s*life\s*hack|家電チャンネル|livedoor\s*HOMME|ライブドア・?オム|MOVIE\s*ENTER|ムービーエンター|"
    r"Peachy|ピーチィ|S-MAX|エスマックス|smaxjp|Sports\s*Watch|スポーツウォッチ|トピックニュース|"
    r"dokujo-tsushin|kaden-channel|topic-news|sports-watch|movie-enter|it-life-hack|livedoor-homme", re.I)


def mask_site(text: str) -> str:
    """Mask site names with ■ (including 【】 brackets that only wrap a site name)."""
    return SITE_PAT.sub("■", text).replace("【■】", "")


def _load_all():
    """Read all three splits; returns (list of records, list of class names)."""
    records, names = [], None
    for hf_split in ("train", "validation", "test"):
        try:
            ds = load_split(HF_ID, HF_CONFIG, hf_split)
        except RuntimeError:
            continue
        names = names or label_names(ds, "category")
        records.extend(dict(ex) for ex in ds)
    if not records:
        raise RuntimeError("livedoor: no split could be loaded")
    if names is None:
        raise RuntimeError("livedoor: the category column is not a ClassLabel, so class names are unavailable; check the dataset version")
    return records, names


def stratified(records: list[dict], key, rng: random.Random) -> list[dict]:
    """Group by category, shuffle within each group, then take one from each in turn: class counts in the first N differ by at most 1."""
    groups = defaultdict(list)
    for r in records:
        groups[key(r)].append(r)
    for g in groups.values():
        rng.shuffle(g)
    out, keys = [], sorted(groups)
    while any(groups[k] for k in keys):
        for k in keys:
            if groups[k]:
                out.append(groups[k].pop())
    return out


def rows(split: str, seed: int = 0, limit: int | None = None):
    """split is ignored: evaluation only, always a stratified sample of the full data."""
    records, names = _load_all()
    options = [(n, DESC.get(n, "")) for n in names]
    rng = random.Random(f"{NAME}-{seed}")
    for i, ex in enumerate(stratified(records, lambda r: int(r["category"]), rng)):
        if limit is not None and i >= limit:
            break
        body = mask_site(clean_text(ex.get("content", "")))[:MAX_CHARS]
        state = f"タイトル: {mask_site(clean_text(ex.get('title', '')))}　本文: {body}"
        yield make_row("choice", state, QUESTION, options, answer_idx=int(ex["category"]),
                       source=NAME, rid=ex.get("url", i), shuffle=False, rng=rng)
