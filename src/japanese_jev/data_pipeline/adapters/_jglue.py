"""Shared by the three JGLUE tasks (JNLI / JCommonsenseQA / MARC-ja): when datasets fails to load them, fetch the raw JSON from the yahoojapan/JGLUE GitHub repository.

JGLUE only publishes train / valid (test is not public), so "test" here means valid.
The raw files are JSON Lines; a JSON array is also handled in case the repository changes format.
"""

from __future__ import annotations

import json
import urllib.request

RAW = "https://raw.githubusercontent.com/yahoojapan/JGLUE/main/datasets"
VERSIONS = ("1.1", "1.2", "1.3")   # version numbers in the file names, tried in order
SPLIT_FILE = {"train": "train", "valid": "valid", "validation": "valid", "test": "valid"}


def github_rows(task: str, split: str):
    """task is 'jnli' / 'jcommonsenseqa'; yields the raw dicts (same fields as the HF version)."""
    fname = SPLIT_FILE[split]
    last = None
    for ver in VERSIONS:
        url = f"{RAW}/{task}-v{ver}/{fname}-v{ver}.json"
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                text = r.read().decode("utf-8")
        except Exception as e:  # noqa: BLE001
            last = e
            continue
        text = text.strip()
        if text.startswith("["):
            yield from json.loads(text)
        else:
            for line in text.splitlines():
                if line.strip():
                    yield json.loads(line)
        return
    raise RuntimeError(f"no JGLUE/{task} {fname} file found on GitHub (tried versions {VERSIONS}): {last}")
