"""Small helpers shared by the adapters: loading HuggingFace datasets and turning a sample into the unified row format.

Why load_split() exists:
  datasets 4.x no longer supports datasets with loading scripts, and the shunk031 repositories for JGLUE / WRIME /
  Livedoor are all script-based. TRL 1.x requires datasets >= 4.7, so downgrading is not an option. The answer is
  a three-step fallback:
    1. plain load_dataset(hf_id, config, split)
    2. the refs/convert/parquet branch of the same repository (parquet files converted by the HuggingFace bot),
       reading only the directory of the wanted config
    3. a fallback function supplied by the caller (JGLUE fetches the raw JSON from GitHub)
  If all fail, raise an error that states the reasons instead of leaving the user with a datasets traceback.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable

from ..template import LETTERS, render

PARQUET_REV = "refs/convert/parquet"


def pick_parquet_files(files: list[str], config: str | None, split: str) -> list[str]:
    """Pick <config>/<split>/*.parquet out of the file list of the parquet branch.

    The branch layout is <config>/<split>/0000.parquet (large datasets may use partial-<split>).
    When config is None, "default" is used; without "default", a single config is used as is, and several configs
    raise an error asking the adapter to set HF_CONFIG. A plain load_dataset(revision=...) would read the files of
    all configs together and fail with CastError when the columns differ (as with JSICK).
    """
    parquet = [f for f in files if f.endswith(".parquet") and f.count("/") >= 2]
    configs = sorted({f.split("/", 1)[0] for f in parquet})
    if not configs:
        raise RuntimeError("no parquet files in the parquet branch")
    if config is None:
        if "default" in configs:
            cfg = "default"
        elif len(configs) == 1:
            cfg = configs[0]
        else:
            raise RuntimeError(f"the parquet branch has several configs {configs}; set HF_CONFIG in the adapter")
    elif config in configs:
        cfg = config
    else:
        raise RuntimeError(f"the parquet branch has no config {config!r}; available: {configs}")
    for sp in (split, f"partial-{split}"):
        wanted = sorted(f for f in parquet if f.startswith(f"{cfg}/{sp}/"))
        if wanted:
            return wanted
    splits = sorted({f.split("/")[1] for f in parquet if f.startswith(cfg + "/")})
    raise RuntimeError(f"config {cfg!r} has no split {split!r}; available: {splits}")


def _load_parquet_branch(hf_id: str, config: str | None, split: str):
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files(hf_id, repo_type="dataset", revision=PARQUET_REV)
    wanted = pick_parquet_files(files, config, split)
    local = [hf_hub_download(hf_id, f, repo_type="dataset", revision=PARQUET_REV) for f in wanted]
    return load_dataset("parquet", data_files=local, split="train")   # the files carry their features metadata, so ClassLabel is preserved


def load_split(hf_id: str, config: str | None, split: str, fallback: Callable[[], Iterable[dict]] | None = None):
    """Return an iterable of records (dicts). Tries datasets first, then falls back in the order above."""
    from datasets import load_dataset  # lazy import: the tests do not need datasets installed

    errors = []
    try:
        return load_dataset(hf_id, config, split=split) if config else load_dataset(hf_id, split=split)
    except Exception as e:  # noqa: BLE001  datasets raises too many exception types; catch them all
        errors.append(f"default: {type(e).__name__}: {str(e)[:200]}")
    try:
        return _load_parquet_branch(hf_id, config, split)
    except Exception as e:  # noqa: BLE001
        errors.append(f"parquet branch: {type(e).__name__}: {str(e)[:200]}")
    if fallback is not None:
        try:
            return list(fallback())
        except Exception as e:  # noqa: BLE001
            errors.append(f"fallback: {type(e).__name__}: {str(e)[:200]}")
    raise RuntimeError(
        f"loading {hf_id} (config={config}, split={split}) failed; tried:\n  - " + "\n  - ".join(errors)
        + "\nHint: datasets 4.x does not support script-based datasets; check whether the dataset page has a parquet branch, or add a fallback in the adapter."
    )


def label_names(ds, column: str) -> list[str] | None:
    """Names of a ClassLabel column; None if it is not a ClassLabel (or ds is a plain list)."""
    feats = getattr(ds, "features", None)
    if not feats or column not in feats:
        return None
    return list(getattr(feats[column], "names", None) or []) or None


def make_row(kind: str, state: str, question: str, options: list[tuple[str, str]], answer_idx: int,
             source: str, rid, *, shuffle: bool, rng: random.Random, soft: list[float] | None = None) -> dict:
    """Turn one item into a row.

    options  list of (name, description); answer_idx is the index of the correct option in options
    shuffle  choice rows for training are shuffled; noul / score options have a meaningful order and are not; evaluation sets are not shuffled
    soft     optional soft label (probabilities, same length as options) for KL / label smoothing; most rows have none
    """
    opts = list(options)
    order = list(range(len(opts)))
    if shuffle:
        rng.shuffle(order)
        opts = [opts[i] for i in order]
        answer_idx = order.index(answer_idx)
        if soft is not None:
            soft = [soft[i] for i in order]
    n = len(opts)
    row = {
        "kind": kind,
        "n": n,
        "label": LETTERS[answer_idx],
        "prompt": render(kind, state, question, opts),
        "completion": " " + LETTERS[answer_idx],
        "source": source,
        "id": str(rid),
        # the next three are the raw fields: augment.py re-renders after rewriting state, and evaluate.py permutes options to measure consistency
        "state": state,
        "question": question,
        "options": [[name, desc] for name, desc in opts],
    }
    if soft is not None:
        row["soft"] = [round(float(x), 4) for x in soft]
    return row


def sample_negatives(all_names: list[str], gold: str, k: int, rng: random.Random) -> list[str]:
    """Negative sampling: draw k-1 wrong classes from all classes and combine them with the correct one into k candidates."""
    pool = [x for x in all_names if x != gold]
    return rng.sample(pool, min(k - 1, len(pool)))


def clean_text(s: str) -> str:
    """Strip surrounding whitespace and inner newlines; the template is parsed line by line, so state cannot contain bare newlines."""
    return " ".join(str(s).split())
