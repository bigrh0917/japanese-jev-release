"""Break the accuracy on a self-built evaluation set (ja_ambiguity / ja_knowledge) down by phenomenon / scene / primitive.

    python scripts/breakdown.py runs/<dir>/predictions.jsonl                          # ja_ambiguity by default
    python scripts/breakdown.py runs/<dir>/predictions.jsonl evalsets/ja_knowledge/items.jsonl
The dataset name is the parent directory of items.jsonl; the test file defaults to data/test_<dataset>.jsonl.

predictions.jsonl is written by evaluate (dataset / id / label / pred / p_max per item);
only rows with the matching dataset are used, joined back to items.jsonl by id for phenomenon and scene;
for wrong answers, the predicted option is recovered from test_<dataset>.jsonl written by build (options already shuffled).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


LETTERS = "ABCDEFGHIJKLMNOP"


def main(pred_path: str, items_path: str = "evalsets/ja_ambiguity/items.jsonl",
         test_path: str | None = None) -> None:
    dataset = Path(items_path).parent.name
    test_path = test_path or f"data/test_{dataset}.jsonl"
    items = {}
    with open(items_path, encoding="utf-8") as f:
        for line in f:
            it = json.loads(line)
            items[it["id"]] = it
    shuffled = {}
    if Path(test_path).exists():
        with open(test_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                shuffled[r["id"]] = [o[0] for o in r["options"]]
    preds = [json.loads(l) for l in open(pred_path, encoding="utf-8") if l.strip()]
    preds = [p for p in preds if p.get("dataset") == dataset]
    if not preds:
        sys.exit(f"predictions.jsonl has no rows for {dataset}; add {dataset} to eval_sources, build, then evaluate")

    groups = {"phenomenon": defaultdict(list), "scene": defaultdict(list), "primitive": defaultdict(list)}
    wrong = []
    for p in preds:
        it = items.get(p["id"])
        if it is None:
            continue
        hit = p["pred"] == p["label"]
        groups["phenomenon"][it["phenomenon"]].append((hit, p["p_max"]))
        groups["scene"][it["scene"]].append((hit, p["p_max"]))
        groups["primitive"][it["kind"]].append((hit, p["p_max"]))
        if not hit:
            wrong.append((it, p))

    total = sum(h for g in groups["primitive"].values() for h, _ in g)
    print(f"{dataset}  {len(preds)} items  correct {total}  Acc {total / len(preds):.3f}\n")
    for name, g in groups.items():
        print(f"| {name} | n | Acc | mean confidence |\n| --- | --- | --- | --- |")
        for k, lst in sorted(g.items(), key=lambda kv: sum(h for h, _ in kv[1]) / len(kv[1])):
            if name == "scene" and len(lst) < 5:
                continue                      # scenes are too fragmented; those with 1-4 items are skipped
            acc = sum(h for h, _ in lst) / len(lst)
            conf = sum(c for _, c in lst) / len(lst)
            print(f"| {k} | {len(lst)} | {acc:.3f} | {conf:.2f} |")
        print()

    print(f"wrong answers ({len(wrong)}):")
    for it, p in wrong:
        opts = shuffled.get(it["id"])
        pred_name = opts[LETTERS.index(p["pred"])] if opts else p["pred"]
        gold_name = it["options"][it["answer"]][0]
        print(f"- [{it['id']}] {it['state']}  / {it['question']}  gold={gold_name}  pred={pred_name} (confidence {p['p_max']:.2f})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(*sys.argv[1:4])
