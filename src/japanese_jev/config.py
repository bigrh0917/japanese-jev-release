"""Read configs/*.json and resolve the relative paths in it against the project root.

Project root = the first directory, walking up from the config file, that contains pyproject.toml or .git.
So "runs/xxx" in configs/xxx.json means <project root>/runs/xxx, not configs/runs/xxx.

Config keys (full example in configs/template.json):
  run_name       run name; scripts/train.sh uses it to name the directory under runs/
  base_model     base model directory: hf download Qwen/Qwen3-0.6B-Base --local-dir models/Qwen3-0.6B-Base
  data_dir       where build writes, and the other stages read, the jsonl files
  work_dir       outputs of this run: adapters/  temperature.json  evaluation.json  train.log
  train_sources  datasets used for training and calibration (NAME in data_pipeline/adapters); EVAL_ONLY=False only
  eval_sources   evaluation-only datasets, each written to data/test_<name>.jsonl and never trained on
  split_sizes    number of valid / calib rows cut from the training data
  eval_items_per_source  items taken from each evaluation set (fixed seed)
  lora / train   training hyperparameters
  calibration    temperature-scaling settings
  serve          address the serve stage listens on
"""

from __future__ import annotations

import json
from pathlib import Path

# the values of these keys are paths; relative paths are resolved against the project root
PATH_KEYS = ("base_model", "data_dir", "work_dir")


def project_root(start: Path) -> Path:
    """Walk up from start to the first directory containing pyproject.toml or .git; fall back to start itself."""
    start = start.resolve()
    for d in (start, *start.parents):
        if (d / "pyproject.toml").exists() or (d / ".git").exists():
            return d
    return start


def resolve(root: Path, p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else root / p


def load(path: str | Path, overrides: dict | None = None) -> dict:
    """Read the JSON config, apply command-line overrides (such as --work-dir), then resolve PATH_KEYS to absolute paths.

    The returned dict has two extra keys: _root (project root) and _config_path (absolute path of the config file).
    """
    path = Path(path).resolve()
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    for k, v in (overrides or {}).items():
        if v is not None:
            cfg[k] = v

    root = project_root(path.parent)
    for k in PATH_KEYS:
        if k in cfg:
            cfg[k] = resolve(root, cfg[k])
    cfg["_root"] = root
    cfg["_config_path"] = path
    return cfg
