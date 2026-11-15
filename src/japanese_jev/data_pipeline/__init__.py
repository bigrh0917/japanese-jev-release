"""Data pipeline: public datasets -> three-primitive jsonl.

    adapters/   one file per dataset; rewrites its records into the unified row format
    build.py    calls the adapters named in cfg train_sources / eval_sources, deduplicates, splits, writes data/*.jsonl
    augment.py  Japanese augmentation: all-hiragana, full-width / half-width swaps, unrelated-sentence insertion

Three data rules are enforced here:
  1. Training and evaluation sets are different datasets; an adapter with EVAL_ONLY=True cannot appear in train_sources
  2. Evaluation sets never appear at any stage of training (including calib, seeds for synthetic data, sources for augmentation)
  3. Training only uses data whose license allows training and releasing the weights; restricted data is evaluation-only
"""
