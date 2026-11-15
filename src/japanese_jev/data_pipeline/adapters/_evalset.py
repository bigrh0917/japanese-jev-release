"""Shared loader for the self-built evaluation sets: evalsets/<name>/items.jsonl -> rows.

Each item: {"id", "kind", "scene", "phenomenon", "state", "question", "options": [[name, desc], ...], "answer": idx}
choice options are shuffled with a fixed seed (in the source file the correct option is mostly first, which would
favor a model that always answers A); noul / score options have a meaningful order and are left as they are.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from ..common import make_row

ROOT = Path(__file__).resolve().parents[4]


def load_items(name: str) -> list[dict]:
    path = ROOT / "evalsets" / name / "items.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"cannot find {path}; evalsets/ must sit in the project next to the code")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def rows_from_items(name: str, seed: int = 0, limit: int | None = None):
    rng = random.Random(f"{name}-{seed}")
    for i, it in enumerate(load_items(name)):
        if limit is not None and i >= limit:
            break
        yield make_row(it["kind"], it["state"], it["question"], [tuple(o) for o in it["options"]],
                       answer_idx=int(it["answer"]), source=name, rid=it["id"],
                       shuffle=(it["kind"] == "choice"), rng=rng)
