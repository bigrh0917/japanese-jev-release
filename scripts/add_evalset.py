"""Add one evaluation set without rebuilding: python scripts/add_evalset.py jcola [more names...]

A rebuild would re-cut valid / calib, and existing in-distribution numbers would no longer be comparable. This
script only does step 6 of build: call the adapter for its test split -> write the first 500 items to
data/test_<name>.jsonl (and final_<name>.jsonl if at least 200 more remain), and copy them to data_ft/ and
data_distill/ (whose evaluation files are copies of data/ anyway).
Then add the name to eval_sources in the config and rerun evaluate.
"""

from __future__ import annotations

import random
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from japanese_jev.data_pipeline import adapters  # noqa: E402
from japanese_jev.data_pipeline.build import MIN_FINAL, _rows, write_jsonl  # noqa: E402


def main(names: list[str], seed: int = 42, n_eval: int = 500) -> None:
    data = ROOT / "data"
    for name in names:
        a = adapters.get(name)
        got = _rows(a, "test", seed)
        if not getattr(a, "PRESAMPLED", False):
            random.Random(f"{name}-{seed}").shuffle(got)
        test, rest = got[:n_eval], got[n_eval:2 * n_eval]
        files = [(f"test_{name}.jsonl", test)] + ([(f"final_{name}.jsonl", rest)] if len(rest) >= MIN_FINAL else [])
        for fname, rows in files:
            write_jsonl(data / fname, rows)
            for d in ("data_ft", "data_distill"):
                if (ROOT / d).is_dir():
                    shutil.copy(data / fname, ROOT / d / fname)
            print(f"{fname}: {len(rows)} rows  answer distribution {dict(sorted(Counter(r['label'] for r in rows).items()))}  license {a.LICENSE}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
