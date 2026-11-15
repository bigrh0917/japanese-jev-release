"""How many API calls are saved, and what the combined accuracy is, when the local model answers first and only the
unsure items go to Jev. No requests are sent; it is computed from two existing prediction files.

    python scripts/cascade.py runs/<local model dir>/predictions.jsonl [runs/jev-api/predictions.jsonl]

Method: for each item take the local model's confidence (p_max - 1/n) / (1 - 1/n); at or above the threshold use
the local answer, otherwise the Jev answer.
Note: the thresholds are compared after the fact on the same test items; a real deployment would set the
threshold on separate items. This only shows the order of magnitude.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
THRESHOLDS = (0.5, 0.7, 0.8, 0.9)


def main(mine: str, jev: str = "runs/jev-api/predictions.jsonl") -> None:
    J = {(r["dataset"], r["id"]): r for r in map(json.loads, open(ROOT / jev if not Path(jev).is_absolute() else jev, encoding="utf-8"))}
    F = [r for r in map(json.loads, open(mine, encoding="utf-8")) if (r["dataset"], r["id"]) in J]
    sets = list(dict.fromkeys(r["dataset"] for r in F))
    print("| set | n | local only | Jev only | " + " | ".join(f"threshold {t}: share answered locally / combined accuracy" for t in THRESHOLDS) + " |")
    print("| --- | --- | --- | --- |" + " --- |" * len(THRESHOLDS))
    for ds in ["all"] + sets:
        rows = [r for r in F if ds == "all" or r["dataset"] == ds]
        n = len(rows)
        ok_j = [J[(r["dataset"], r["id"])]["pred"] == r["label"] for r in rows]
        ok_m = [r["pred"] == r["label"] for r in rows]
        conf = [(r["p_max"] - 1 / r["n"]) / (1 - 1 / r["n"]) for r in rows]
        cells = []
        for t in THRESHOLDS:
            local = [c >= t for c in conf]
            acc = sum(m if k else j for m, j, k in zip(ok_m, ok_j, local)) / n
            cells.append(f"{sum(local) / n:.0%} / {acc * 100:.1f}")
        print(f"| {ds} | {n} | {sum(ok_m) / n * 100:.1f} | {sum(ok_j) / n * 100:.1f} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(*sys.argv[1:3])
