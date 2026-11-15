"""Calibration: fit a temperature T on data/calib.jsonl that minimizes NLL.

T does not change the argmax, only how confident the probabilities are. The result is written to
cfg["work_dir"]/temperature.json:
    {"choice": 1.19, "noul": 1.23, "score": 1.18,                 <- one per primitive (fallback for serve / evaluate)
     "by_n": {"choice": {"3-5": 1.21, "6-16": 1.02}, ...}}        <- also bucketed by option count; a bucket with fewer than MIN_PER_BUCKET rows is omitted
Why bucket by option count: on 9-way Livedoor the model is under-confident (training has at most 5 options),
while on 3-5 options it is slightly over-confident, and a single T cannot fix both. evaluate.py and serve.py
look values up with temperature_for(temps, kind, n).
calib is used only to fit the temperature and is never mixed with valid / test. Expect T between 1 and 3.

The calibration set can be swapped: cfg["calibration"]["split"] defaults to "calib" (cut from the training
sources, in-distribution) and can also be a third-party set such as "calib_sib200" (written by build from
calib_sources). When a third-party set lacks a primitive (SIB-200 only has choice), that primitive is taken
from cfg["calibration"]["fallback_split"] (default "calib").

The logits on calib are also saved to work_dir/calib_logits.npz, so trying another calibration method
(Platt, vector scaling) does not need another forward pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from . import metrics, readout
from .template import KINDS

T_BOUNDS = (0.3, 6.0)
MIN_PER_KIND = 20
MIN_PER_BUCKET = 50
N_BUCKETS = (("2", 2, 2), ("3-5", 3, 5), ("6-16", 6, 16))


def n_bucket(n: int) -> str:
    for name, lo, hi in N_BUCKETS:
        if lo <= n <= hi:
            return name
    return N_BUCKETS[-1][0]


def temperature_for(temps: dict, kind: str, n: int) -> float:
    """Look up T for (primitive, option-count bucket), fall back to the primitive's T, then to 1.0."""
    by_n = temps.get("by_n") or {}
    t = (by_n.get(kind) or {}).get(n_bucket(n))
    if t is None:
        t = temps.get(kind, 1.0)
    return float(t)


def fit_temperature(L: np.ndarray, y: np.ndarray) -> float:
    res = minimize_scalar(lambda t: metrics.nll(t, L, y), bounds=T_BOUNDS, method="bounded")
    return float(res.x)


def _report(label: str, L, y, T, bins):
    before = metrics.summary(L, y, 1.0, bins)
    after = metrics.summary(L, y, T, bins)
    print(f"  {label:<16} n={len(y):<5} T={T:.3f}   acc {after['accuracy']:.3f}   "
          f"ECE {before['ece']:.3f} → {after['ece']:.3f}   NLL {before['nll']:.3f} → {after['nll']:.3f}")


def main(cfg: dict) -> None:
    work = Path(cfg["work_dir"])
    work.mkdir(parents=True, exist_ok=True)
    cal = cfg.get("calibration", {})
    split = cal.get("split", "calib")
    bins = int(cal.get("ece_bins", 15))
    rows = readout.read_jsonl(Path(cfg["data_dir"]) / f"{split}.jsonl")
    print(f"[calibrate] {split}.jsonl {len(rows)} rows")
    fallback = cal.get("fallback_split", "calib")
    fb_path = Path(cfg["data_dir"]) / f"{fallback}.jsonl"
    if fallback != split and fb_path.exists():
        have = {r["kind"] for r in rows}
        extra = [r for r in readout.read_jsonl(fb_path) if r["kind"] not in have]
        if extra:
            print(f"[calibrate] {split} has no {sorted({r['kind'] for r in extra})}; from {fallback}.jsonl, added {len(extra)} rows")
            rows += extra

    model, tok = readout.load_model(cfg)
    L, y, _ = readout.collect_logits(model, tok, rows, desc="calib")
    kinds = np.array([r["kind"] for r in rows])
    ns = np.array([int(r["n"]) for r in rows])
    np.savez(work / "calib_logits.npz", L=L, y=y, kinds=kinds, ns=ns)

    temps: dict = {}
    per_kind = bool(cal.get("per_kind", True))
    if not per_kind:
        T = fit_temperature(L, y)
        temps = {k: T for k in KINDS}
        _report("all", L, y, T, bins)
    else:
        for kind in KINDS:
            m = kinds == kind
            if m.sum() < MIN_PER_KIND:
                print(f"  {kind}: only {int(m.sum())} rows in calib, too few; T set to 1.0")
                temps[kind] = 1.0
                continue
            temps[kind] = fit_temperature(L[m], y[m])
            _report(kind, L[m], y[m], temps[kind], bins)

    if bool(cal.get("per_n", True)):
        by_n: dict[str, dict[str, float]] = {}
        for kind in KINDS:
            for name, lo, hi in N_BUCKETS:
                m = (kinds == kind) & (ns >= lo) & (ns <= hi)
                if m.sum() < MIN_PER_BUCKET:
                    continue
                T = fit_temperature(L[m], y[m])
                by_n.setdefault(kind, {})[name] = round(T, 4)
                _report(f"{kind} n={name}", L[m], y[m], T, bins)
        temps["by_n"] = by_n

    out = work / "temperature.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({k: (round(v, 4) if isinstance(v, float) else v) for k, v in temps.items()},
                  f, indent=2, ensure_ascii=False)
    print(f"[calibrate] wrote {out}")
