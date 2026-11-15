"""Prepare a small dataset for continuing training from an existing LoRA: python scripts/make_continue_set.py [replay rows=5000] [synthetic weight=5] [output dir=data_ft] [extra training sources, comma-separated]

Example: python scripts/make_continue_set.py 6000 3 data_ft_jcola jcola_train     # also add the JCoLA training split (once)

No rebuild (a rebuild would re-cut valid / calib, items from the old training set could land in valid, and the
in-distribution numbers would no longer be clean).
It reads the existing data/ and writes to data_ft/:
  train.jsonl   = synthetic data (datasets/synthetic/train.jsonl, x weight; copies of choice rows get a new option order)
                + replay rows sampled from data/train.jsonl (so the old tasks are not forgotten while learning the new items)
  other files   = copied as they are (valid, calib, test_*, final_*), so evaluation numbers are directly comparable with those before the continued training
"""

from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from japanese_jev.data_pipeline import augment  # noqa: E402
from japanese_jev.data_pipeline.adapters import synthetic  # noqa: E402


def main(replay: int = 5000, weight: int = 5, dst_name: str = "data_ft", extra: str = "", seed: int = 42) -> None:
    src, dst = ROOT / "data", ROOT / dst_name
    dst.mkdir(exist_ok=True)
    rng = random.Random(seed)
    old = [json.loads(l) for l in open(src / "train.jsonl", encoding="utf-8")]
    old = [r for r in old if not r["source"].startswith("synthetic")]
    held = {json.loads(l).get("state") for f in ("valid.jsonl", "calib.jsonl") for l in open(src / f, encoding="utf-8")}
    syn = [r for r in synthetic.rows("train", seed=seed) if r["state"] not in held]
    rows = list(syn)
    for k in range(1, weight):
        for r in syn:
            c = augment.permute_row(r, rng) if r["kind"] == "choice" and r["n"] >= 3 else None
            c = c or dict(r)
            c["source"], c["id"] = "synthetic", f"{r['id']}#{k}"
            rows.append(c)
    from japanese_jev.data_pipeline import adapters
    for name in [x for x in extra.split(",") if x]:
        a = adapters.get(name)
        if a.EVAL_ONLY:
            raise ValueError(f"{name} is an evaluation-only dataset and cannot be added to training")
        got = [r for r in a.rows("train", seed=seed) if r["state"] not in held]
        for r in got:
            r.setdefault("source", name)
        print(f"  extra training source {name}: {len(got)} rows (license: {a.LICENSE})")
        rows += got
    rows += rng.sample(old, min(replay, len(old)))
    rng.shuffle(rows)
    with open(dst / "train.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for p in src.glob("*.jsonl"):
        if p.name != "train.jsonl":
            shutil.copy(p, dst / p.name)
    print(f"wrote {dst}/: train {len(rows)} rows (synthetic {len(syn)} × {weight} + replay {min(replay, len(old))}); other files copied from data/ as they are")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:3]), *sys.argv[3:5])
