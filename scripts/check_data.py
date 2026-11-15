"""Check every data file after build: label distribution, option-count distribution, and texts shared with other files.

    python scripts/check_data.py [data]

Warnings are printed when:
  - one answer (by option name) makes up 85% or more of a source (this happens when a split was not shuffled)
  - one letter makes up 60% or more of a choice source (options not shuffled)
  - a text from test_* / final_* / valid / calib appears in train, or test and final overlap
"""

from __future__ import annotations

import collections
import glob
import json
import sys
from pathlib import Path

LETTERS = "ABCDEFGHIJKLMNOP"


def main(data_dir: str = "data") -> None:
    files = {Path(f).stem: [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
             for f in sorted(glob.glob(f"{data_dir}/*.jsonl"))}
    warnings = []
    for name, rows in files.items():
        print(f"\n{name}  {len(rows)} rows")
        by = collections.defaultdict(list)
        for r in rows:
            by[(r["source"].split("+")[0], r["kind"])].append(r)
        for (src, kind), rs in sorted(by.items()):
            letters = collections.Counter(r["label"] for r in rs)
            names = collections.Counter(r["options"][LETTERS.index(r["label"])][0] for r in rs if "options" in r)
            ns = collections.Counter(r["n"] for r in rs)
            top_name, top_cnt = names.most_common(1)[0] if names else ("-", 0)
            print(f"  {src:<16}{kind:<7}{len(rs):>6}  option counts {dict(sorted(ns.items()))}  letters {dict(sorted(letters.items()))}")
            print(f"  {'':<29}most common answer '{top_name[:20]}' {top_cnt / len(rs):.0%}, distinct answers: {len(names)}")
            if len(rs) >= 50 and top_cnt / len(rs) >= 0.85:
                warnings.append(f"{name} / {src} / {kind}: answer '{top_name}' makes up {top_cnt / len(rs):.0%}")
            if kind == "choice" and len(rs) >= 50 and max(letters.values()) / len(rs) >= 0.6:
                warnings.append(f"{name} / {src}: letter {letters.most_common(1)[0][0]} makes up {max(letters.values()) / len(rs):.0%} (options not shuffled?)")

    key = lambda r: (r["source"].split("+")[0], r.get("state", r["prompt"]))
    # also: texts of the self-built evaluation sets must not appear in any training source
    train_states = {r.get("state") for r in files.get("train", [])}
    for name, rows in files.items():
        if name.startswith("test_ja_"):
            dup = sum(1 for r in rows if r.get("state") in train_states)
            if dup:
                warnings.append(f"{name}: {dup} texts appear in train (evaluation items leaked into training)")
    train = {key(r) for r in files.get("train", [])}
    for name, rows in files.items():
        if name == "train":
            continue
        dup = sum(1 for r in rows if key(r) in train)
        if dup:
            warnings.append(f"{name}: {dup} rows have a text that is also in train")
    for name in [n for n in files if n.startswith("test_")]:
        fin = files.get("final_" + name[5:])
        if fin:
            dup = len({key(r) for r in files[name]} & {key(r) for r in fin})
            if dup:
                warnings.append(f"{name} and final_{name[5:]} share {dup} texts")

    print("\n" + ("no warnings" if not warnings else "warnings:\n" + "\n".join("  - " + w for w in warnings)))


if __name__ == "__main__":
    main(*sys.argv[1:2])
