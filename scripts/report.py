"""Print evaluation.json as a Markdown table.

    python scripts/report.py runs/<dir>/evaluation.json [more files...]

One table per file: in-distribution (valid, broken down by primitive) + out-of-distribution (each test_<source>).
When predictions.jsonl is in the same directory, the two-class average is also reported for score sets dominated
by the lowest level (WRIME).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def row(name: str, r: dict) -> str:
    u, c = r["uncalibrated"], r["calibrated"]
    pc = r.get("permutation_consistency")
    acc = f"{c['accuracy']:.3f}"
    if set(r.get("kinds", {})) == {"score"} and "within1" in c:
        acc += f" (within 1 level {c['within1']:.3f})"
    return (f"| {name} | {r['n']} | {acc} | {u['ece']:.3f} → {c['ece']:.3f} | {c['brier']:.3f} "
            f"| {c['automation_2pct']:.2f} / {c['automation_5pct']:.2f} | {pc if pc is not None else '–'} "
            f"| {r.get('p50_latency_ms', 0):.0f} |")


def kind_row(name: str, r: dict) -> str:
    u, c = r["uncalibrated"], r["calibrated"]
    acc = f"{c['accuracy']:.3f}"
    if "score" in name and "within1" in c:
        acc += f" (within 1 level {c['within1']:.3f})"
    return (f"| {name} | {c['n']} | {acc} | {u['ece']:.3f} → {c['ece']:.3f} | {c['brier']:.3f} "
            f"| {c['automation_2pct']:.2f} / {c['automation_5pct']:.2f} | – | – |")


def report(path: Path) -> str:
    e = json.loads(path.read_text(encoding="utf-8"))
    T = e.get("temperature", {})
    out = [f"### {path.parent.name}" + path.stem.replace("evaluation", "").replace("_", " · "), "",
           f"base `{Path(e['base_model']).name}`, adapter `{e.get('adapter') or 'none (untrained)'}`, "
           f"T = choice {T.get('choice', 1):.2f} / noul {T.get('noul', 1):.2f} / score {T.get('score', 1):.2f}", "",
           "| dataset | n | Acc | ECE before → after | Brier | automation @2% / @5% | permutation consistency | p50 ms |",
           "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in e.get("in_distribution", {}).items():
        out.append(row(f"{name} (in-distribution)", r))
        for kind, kr in r.get("per_kind", {}).items():
            out.append(kind_row(f"└ {kind}", kr))
    for name, r in e.get("out_of_distribution", {}).items():
        out.append(row(f"{name} (out-of-distribution)", r))
    out += two_class_lines(path.with_name(path.name.replace("evaluation", "predictions").replace(".json", ".jsonl")))
    return "\n".join(out)


def two_class_lines(pred_path: Path) -> list[str]:
    """For score sets where the lowest level (A) makes up 60% or more (such as WRIME), accuracy says little: always
    answering the lowest level already scores high. The two-class average is also reported = (share answered
    lowest among items labeled lowest + share not answered lowest among items labeled higher) / 2; guessing at
    random or always answering one side gives 50%."""
    if not pred_path.exists():
        return []
    by: dict[str, list[dict]] = {}
    for line in open(pred_path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("kind") == "score":
            by.setdefault(r["dataset"], []).append(r)
    lines = []
    for name, preds in by.items():
        low = [r for r in preds if r["label"] == "A"]
        high = [r for r in preds if r["label"] != "A"]
        if not low or not high or len(low) / len(preds) < 0.6:
            continue
        v = (sum(r["pred"] == "A" for r in low) / len(low) + sum(r["pred"] != "A" for r in high) / len(high)) / 2
        lines.append(f"{name} two-class average: {v:.3f} (lowest level: {len(low) / len(preds):.0%})")
    return [""] + lines if lines else []


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    print("\n\n".join(report(Path(p)) for p in sys.argv[1:]))
