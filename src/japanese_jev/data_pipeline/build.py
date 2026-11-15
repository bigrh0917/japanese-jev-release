"""build stage: call the adapters and write the jsonl files under data/.

Outputs (all in cfg["data_dir"]):
    train.jsonl  valid.jsonl  calib.jsonl   <- from cfg["train_sources"] only
    test_<source>.jsonl                     <- one per cfg["eval_sources"], eval_items_per_source items each
    final_<source>.jsonl                    <- the next batch, not overlapping test (only if at least 200 items remain).
                                               Sealed: tune and inspect errors on test only, and run evaluate --final
                                               once right before reporting numbers. It stops being sealed once you
                                               change the model after looking at it.
    calib_<source>.jsonl                    <- one per cfg["calib_sources"] (optional), the whole set; a third-party
                                               calibration set, neither trained on nor evaluated

Each row: {"kind", "n", "label", "prompt", "completion", "source", "id", "state", "question", "options"[, "soft"]}
  Trainers read only prompt / completion; kind / n / label are for evaluation and calibration; source / id record
  provenance (so problems can be traced back); state / question / options are the raw fields, needed when
  augmentation or the permutation-consistency metric has to render the prompt again.

Steps (main runs them in this order):
  1. Check that train_sources contains no EVAL_ONLY adapter; fail if it does
  2. Call adapter.rows("train") for every training source; cfg["source_limits"] can cap a source (KUCI has 80K+ rows, 10K are taken)
  3. Deduplicate by prompt (a prompt with conflicting answers is dropped entirely)
  4. With cfg["seed"] fixed, keep at most cfg["max_per_kind"] rows per primitive, then cut valid / calib by original sample (source + state); the rest is train
  5. Call augment.augment_rows() on the training set with cfg["augment_ratio"] (training set only)
  6. Call adapter.rows("test") for every evaluation source, sample eval_items_per_source items with a fixed seed, write test_<source>.jsonl
  7. Print the row count and primitive mix of every file; afterwards, look at the format with head -c 600 data/train.jsonl

Optional config keys (no limit if absent): source_limits {"kuci": 10000}, max_per_kind 12000.
"""

from __future__ import annotations

import inspect
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from ..template import KINDS
from . import adapters, augment


MIN_FINAL = 200   # no sealed set when fewer than this many items remain


def check_sources(cfg: dict) -> None:
    """Rule 1: a training source cannot be an evaluation-only dataset; every source must be registered."""
    for name in cfg["train_sources"]:
        a = adapters.get(name)
        if a.EVAL_ONLY:
            raise ValueError(f"{name} is an evaluation-only dataset (license: {a.LICENSE}) and cannot be in train_sources")
    for name in cfg["eval_sources"]:
        adapters.get(name)
    for name in cfg.get("calib_sources") or []:
        adapters.get(name)
        if name in cfg["train_sources"] or name in cfg["eval_sources"]:
            raise ValueError(f"{name} is a calibration set and cannot also appear in train_sources or eval_sources")


def _rows(adapter, split: str, seed: int, limit: int | None = None) -> list[dict]:
    """Call adapter.rows; the registry only requires rows(split), and seed / limit are passed only if the adapter accepts them."""
    params = inspect.signature(adapter.rows).parameters
    kw = {}
    if "seed" in params:
        kw["seed"] = seed
    if "limit" in params and limit is not None:
        kw["limit"] = limit
    out = []
    for r in adapter.rows(split, **kw):
        r.setdefault("source", adapter.NAME)
        out.append(r)
        if limit is not None and "limit" not in params and len(out) >= limit:
            break
    return out


def dedupe(rows: list[dict]) -> list[dict]:
    """Keep one row per prompt; when the same prompt has different answers, drop the whole group (conflicting labels would only mislead the model)."""
    by_prompt: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_prompt[r["prompt"]].append(r)
    kept, conflict_groups, conflict_rows = [], 0, 0
    for group in by_prompt.values():
        if len({r["label"] for r in group}) > 1:
            conflict_groups += 1
            conflict_rows += len(group)
            continue
        kept.append(group[0])
    print(f"  dedup: {len(rows)} → {len(kept)} rows (duplicates {len(rows) - len(kept) - conflict_rows}, "
          f"conflicting answers dropped: {conflict_groups} groups, {conflict_rows} rows)")
    return kept


def cap_per_kind(rows: list[dict], max_per_kind: int | None, rng: random.Random) -> list[dict]:
    """Keep at most max_per_kind rows per primitive, taking from the sources in turn so that every source is represented."""
    if not max_per_kind:
        return rows
    out = []
    for kind in KINDS:
        pool = [r for r in rows if r["kind"] == kind]
        if len(pool) <= max_per_kind:
            out.extend(pool)
            continue
        by_src: dict[str, list[dict]] = defaultdict(list)
        for r in pool:
            by_src[r["source"]].append(r)
        for lst in by_src.values():
            rng.shuffle(lst)
        picked: list[dict] = []
        while len(picked) < max_per_kind:
            for lst in by_src.values():
                if lst and len(picked) < max_per_kind:
                    picked.append(lst.pop())
        out.extend(picked)
    return out


def split_by_group(pool: list[dict], n_valid: int, n_calib: int, rng: random.Random):
    """Cut valid / calib / train after grouping by original sample: rows with the same state (noul + score from JSICK,
    12-class + 6-class from MewsC-16, the same context in KUCI) always land in the same file. Otherwise the model
    would have seen the valid sentences during training and the in-distribution numbers would be inflated."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in pool:
        groups[(r["source"], r.get("state", r["prompt"]))].append(r)
    keys = list(groups)
    rng.shuffle(keys)
    valid, calib, train = [], [], []
    for k in keys:
        if len(valid) < n_valid:
            valid.extend(groups[k])
        elif len(calib) < n_calib:
            calib.extend(groups[k])
        else:
            train.extend(groups[k])
    rng.shuffle(train)
    return valid, calib, train


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def report(name: str, rows: list[dict]) -> None:
    kinds = Counter(r["kind"] for r in rows)
    srcs = Counter(r["source"] for r in rows)
    n = max(len(rows), 1)
    ratio = "  ".join(f"{k} {kinds.get(k, 0)} ({kinds.get(k, 0) / n:.0%})" for k in KINDS)
    print(f"{name:<24}{len(rows):>7} rows   {ratio}")
    print("    sources: " + ", ".join(f"{s} {c}" for s, c in sorted(srcs.items())))


def main(cfg: dict) -> None:
    check_sources(cfg)
    seed = int(cfg.get("seed", 0))
    rng = random.Random(seed)
    data_dir = Path(cfg["data_dir"])
    limits = cfg.get("source_limits") or {}

    # 2. training sources
    pool: list[dict] = []
    for name in cfg["train_sources"]:
        a = adapters.get(name)
        got = _rows(a, "train", seed, limits.get(name))
        print(f"[{name}] train split {len(got)} rows (license: {a.LICENSE})")
        pool.extend(got)
    if not pool:
        raise RuntimeError("no rows from any training source; see the loading errors above")

    # 3. dedup  4. per-primitive cap -> shuffle -> cut valid / calib
    pool = dedupe(pool)
    pool = cap_per_kind(pool, cfg.get("max_per_kind"), rng)
    n_valid = int(cfg["split_sizes"]["valid"])
    n_calib = int(cfg["split_sizes"]["calib"])
    if n_valid + n_calib >= len(pool):
        raise RuntimeError(f"only {len(pool)} rows, not enough to cut valid {n_valid} + calib {n_calib}")
    valid, calib, train = split_by_group(pool, n_valid, n_calib, rng)

    # 4b. Weighting small sources: source_weights {"synthetic": 5} = every row of that source appears 5 times in the training set (copies of choice rows get a new option order).
    #     1K synthetic items are only 2% of 50K rows and are barely learned without a weight. Only train is touched; valid / calib get no copies.
    for src, w in (cfg.get("source_weights") or {}).items():
        base = [r for r in train if r["source"] == src]
        extra = []
        for k in range(1, int(w)):
            for r in base:
                c = augment.permute_row(r, rng) if r["kind"] == "choice" and r["n"] >= 3 else None
                c = c or dict(r)
                c["source"], c["id"] = src, f"{r['id']}#{k}"
                extra.append(c)
        train += extra
        print(f"  weight: {src} × {w}, training set {len(base)} → {len(base) + len(extra)} rows")
    rng.shuffle(train)

    # 5. augment the training set only
    ratio = float(cfg.get("augment_ratio", 0.0))
    if ratio > 0:
        extra = augment.augment_rows(train, ratio, seed)
        print(f"  augment: +{len(extra)} rows ({ratio:.0%}), " + ", ".join(f"{s} {c}" for s, c in sorted(Counter(r['source'] for r in extra).items())))
        train = train + extra
        rng.shuffle(train)

    write_jsonl(data_dir / "train.jsonl", train)
    write_jsonl(data_dir / "valid.jsonl", valid)
    write_jsonl(data_dir / "calib.jsonl", calib)

    # 6. evaluation sources: one test_<source>.jsonl each, sampled with a fixed seed
    n_eval = int(cfg.get("eval_items_per_source", 500))
    test_files, final_files = {}, {}
    for name in cfg["eval_sources"]:
        a = adapters.get(name)
        try:
            got = _rows(a, "test", seed)
        except Exception as e:  # noqa: BLE001  one unavailable evaluation set should not fail the whole build
            print(f"[{name}] failed to load, skipping: {str(e).splitlines()[0]}")
            continue
        if not got:
            print(f"[{name}] 0 rows, skipping")
            continue
        if not getattr(a, "PRESAMPLED", False):     # adapters that already stratify (livedoor) are not shuffled again
            random.Random(f"{name}-{seed}").shuffle(got)
        rest = got[n_eval:2 * n_eval]
        got = got[:n_eval]
        write_jsonl(data_dir / f"test_{name}.jsonl", got)
        test_files[name] = got
        if len(rest) >= MIN_FINAL:           # sealed set: the next batch, not overlapping test; not evaluated normally, only once with evaluate --final before reporting numbers
            write_jsonl(data_dir / f"final_{name}.jsonl", rest)
            final_files[name] = rest

    # 6b. third-party calibration sets: the whole set goes to calib_<source>.jsonl
    calib_files = {}
    for name in cfg.get("calib_sources") or []:
        a = adapters.get(name)
        try:
            got = _rows(a, "all" if "all" in getattr(a, "SPLITS", {}) else "test", seed)
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] failed to load, skipping: {str(e).splitlines()[0]}")
            continue
        if got:
            write_jsonl(data_dir / f"calib_{name}.jsonl", got)
            calib_files[name] = got

    # 7. report
    print(f"\nwrote to {data_dir}/")
    report("train.jsonl", train)
    report("valid.jsonl", valid)
    report("calib.jsonl", calib)
    for name, got in test_files.items():
        report(f"test_{name}.jsonl", got)
    for name, got in calib_files.items():
        report(f"calib_{name}.jsonl", got)
    for name, got in final_files.items():
        report(f"final_{name}.jsonl (sealed)", got)
    print("\nNext: check the format with head -c 600 data/train.jsonl, then run bash scripts/train.sh --config <config> --smoke")
