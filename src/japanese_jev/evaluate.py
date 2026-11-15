"""Evaluation: the same metrics on valid.jsonl (in-distribution) and on every test_<source>.jsonl (out-of-distribution).

Reported for every dataset: accuracy, within-1-level accuracy (for score), ECE (15 bins), Brier, NLL, automation
rate (2% / 5% error budget), option-permutation consistency (choice only) and p50 latency; once before calibration
(T=1) and once after (work_dir/temperature.json, T looked up by primitive and option count).
Results go to cfg["work_dir"]/evaluation.json in two sections, in-distribution and out-of-distribution, so every
row states whether the model has seen that kind of data.
Per-item predictions are also written to work_dir/predictions.jsonl (dataset, id, kind, n, label, pred, p_max)
for error analysis.

Baselines:
  untrained base   point --work-dir at a directory without adapters/ (e.g. runs/baseline), run calibrate, then evaluate
  the Jev API      not handled here; see scripts/eval_jev_api.py (it reads the same test_<source>.jsonl files)
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from . import metrics, readout
from .calibrate import temperature_for
from .template import KINDS, LETTERS, render

PERM_ITEMS = 100       # number of choice items sampled for permutation consistency
PERM_TIMES = 4         # number of shuffles per item


def load_temperature(work: Path) -> dict:
    p = work / "temperature.json"
    if not p.exists():
        print(f"[evaluate] no {p}; the calibrated column uses T=1. Run japanese-jev calibrate first")
        return {k: 1.0 for k in KINDS}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def fewshot_prefixes(calib_rows: list[dict], k: int, seed: int) -> dict[str, str]:
    """Build a prefix of k answered examples from calib for each primitive. For the untrained base model: it has never
    seen this template, so without examples it has to guess even that the answer is a single letter, which makes
    the comparison with the trained model unfair. The examples come from calib and never touch an evaluation set."""
    rng = random.Random(f"shots-{seed}")
    out = {}
    for kind in KINDS:
        pool = [r for r in calib_rows if r["kind"] == kind and len(r["prompt"]) < 400]
        rng.shuffle(pool)
        out[kind] = "".join(r["prompt"] + r["completion"] + "\n\n" for r in pool[:k])
    return out


def permutation_consistency(model, tok, rows: list[dict], seed: int, prefix: str = "") -> float | None:
    """Shuffle the options of the same choice item PERM_TIMES times and report how often the argmax points at the same option. noul / score options have a meaningful order and are not tested."""
    cands = [r for r in rows if r["kind"] == "choice" and "options" in r and r["n"] > 2]
    if not cands:
        return None
    rng = random.Random(f"perm-{seed}")
    rng.shuffle(cands)
    agree = total = 0
    for r in cands[:PERM_ITEMS]:
        opts = [tuple(o) for o in r["options"]]
        base = int(np.argmax(readout.option_logits(model, tok, r["prompt"], r["n"])))
        for _ in range(PERM_TIMES):
            order = list(range(len(opts)))
            rng.shuffle(order)
            prompt = r.get("_prefix", "") + render(r["kind"], r["state"], r["question"], [opts[i] for i in order])
            pick = int(np.argmax(readout.option_logits(model, tok, prompt, r["n"])))
            agree += int(order[pick] == base)
            total += 1
    return round(agree / total, 4) if total else None


def evaluate_rows(model, tok, rows: list[dict], temps: dict, bins: int, seed: int, desc: str,
                  pred_out=None) -> dict:
    L, y, times = readout.collect_logits(model, tok, rows, desc=desc)
    kinds = np.array([r["kind"] for r in rows])
    T_row = np.array([temperature_for(temps, r["kind"], int(r["n"])) for r in rows])
    L_cal = L / T_row[:, None]          # each row takes T by its own primitive + option count; the metrics are then computed with T=1
    P_cal = metrics.softmax(L_cal)
    if pred_out is not None:
        for r, p in zip(rows, P_cal):
            pred_out.write(json.dumps({"dataset": desc, "id": r["id"], "source": r["source"], "kind": r["kind"],
                                       "n": r["n"], "label": r["label"], "pred": LETTERS[int(p.argmax())],
                                       "p_max": round(float(p.max()), 4)}, ensure_ascii=False) + "\n")

    out = {
        "n": int(len(rows)),
        "kinds": dict(Counter(kinds.tolist())),
        "sources": dict(Counter(r["source"] for r in rows)),
        "uncalibrated": metrics.summary(L, y, 1.0, bins),
        "calibrated": metrics.summary(L_cal, y, 1.0, bins),
        "p50_latency_ms": round(float(np.median(times)) * 1000, 2),
        "permutation_consistency": permutation_consistency(model, tok, rows, seed),
        "reliability_calibrated": metrics.reliability_table(P_cal, y, bins),
        "per_kind": {},
    }
    for kind in KINDS:
        m = kinds == kind
        if m.any():
            out["per_kind"][kind] = {
                "T": round(float(temps.get(kind, 1.0)), 4),
                "uncalibrated": metrics.summary(L[m], y[m], 1.0, bins),
                "calibrated": metrics.summary(L_cal[m], y[m], 1.0, bins),
            }
    return out


def _line(name: str, r: dict) -> str:
    u, c = r["uncalibrated"], r["calibrated"]
    pc = r["permutation_consistency"]
    return (f"{name:<22}{r['n']:>6}  acc {c['accuracy']:.3f}  ECE {u['ece']:.3f}→{c['ece']:.3f}  "
            f"Brier {c['brier']:.3f}  auto@2% {c['automation_2pct']:.2f}  auto@5% {c['automation_5pct']:.2f}  "
            f"perm {pc if pc is not None else '-'}  p50 {r['p50_latency_ms']:.0f}ms")


def main(cfg: dict) -> None:
    work = Path(cfg["work_dir"])
    work.mkdir(parents=True, exist_ok=True)
    data_dir = Path(cfg["data_dir"])
    bins = int(cfg.get("calibration", {}).get("ece_bins", 15))
    seed = int(cfg.get("seed", 0))
    temps = load_temperature(work)

    model, tok = readout.load_model(cfg)
    adir = readout.resolve_adapter(cfg, None)
    result = {
        "base_model": str(cfg["base_model"]),
        "adapter": str(adir) if adir else None,
        "temperature": temps,
        "in_distribution": {},
        "out_of_distribution": {},
    }

    final = bool(cfg.get("final"))
    shots = int(cfg.get("shots") or 0)
    prefixes = fewshot_prefixes(readout.read_jsonl(data_dir / "calib.jsonl"), shots, seed) if shots else {}
    blank = bool(cfg.get("blank_state"))
    perturb = cfg.get("perturb")
    perturb_fn = None
    if perturb:
        import random as _random

        from .data_pipeline import augment
        fns = augment.transforms(_random.Random(f"perturb-{seed}"))
        if perturb not in fns:
            raise RuntimeError(f"--perturb {perturb} needs a package that is not installed (hiragana needs pykakasi; zenkaku / hankaku need jaconv)")
        perturb_fn = fns[perturb]
        print(f"[evaluate] robustness test: applying the {perturb} transform to [state] of every item before evaluating (question and options unchanged)")
    suffix = (("_final" if final else "") + (f"_shots{shots}" if shots else "") + ("_blank" if blank else "")
              + (f"_{perturb}" if perturb else ""))
    result["perturb"] = perturb
    result["final"], result["shots"] = final, shots
    if final:
        print("[evaluate] sealed sets (final_*.jsonl): run only right before reporting numbers; once you change the model after seeing the results, they are no longer sealed")
    pred_out = open(work / f"predictions{suffix}.jsonl", "w", encoding="utf-8")
    def load(path):
        rows = readout.read_jsonl(path)
        if blank:
            for r in rows:
                r["state"] = "（なし）"
                r["prompt"] = render(r["kind"], r["state"], r["question"], [tuple(o) for o in r["options"]])
        if perturb_fn:
            for r in rows:
                r["state"] = perturb_fn(r["state"])
                r["prompt"] = render(r["kind"], r["state"], r["question"], [tuple(o) for o in r["options"]])
        if prefixes:
            for r in rows:
                r["_prefix"] = prefixes.get(r["kind"], "")
                r["prompt"] = r["_prefix"] + r["prompt"]
        return rows

    valid = data_dir / "valid.jsonl"
    if valid.exists() and not final:
        rows = load(valid)
        result["in_distribution"]["valid"] = evaluate_rows(model, tok, rows, temps, bins, seed, "valid", pred_out)
    for name in cfg["eval_sources"]:
        if perturb and name == "jcola":      # this set asks "is this sentence natural"; rewriting the sentence changes the question itself, so it is skipped in robustness tests
            continue
        p = data_dir / f"{'final' if final else 'test'}_{name}.jsonl"
        if not p.exists():
            print(f"[evaluate] no {p.name}, skipping")
            continue
        rows = load(p)
        result["out_of_distribution"][name] = evaluate_rows(model, tok, rows, temps, bins, seed, name, pred_out)
    pred_out.close()

    out = work / f"evaluation{suffix}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\nIn-distribution (valid cut from the training sources)")
    for name, r in result["in_distribution"].items():
        print("  " + _line(name, r))
    print("Out-of-distribution (datasets not used in training)")
    for name, r in result["out_of_distribution"].items():
        print("  " + _line(name, r))
    print(f"\n[evaluate] wrote {out} (alphabet of up to {len(LETTERS)} options)")
