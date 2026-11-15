"""Serving: FastAPI exposes POST /v1/systemone with the same shape as TypeSafe's.

Request: {"state": "...", "questions": {"<name>": {"type": "noul|choice|score", "instructions": "...", "criteria": ...}}}
  noul    no criteria needed
  choice  criteria is {"option name": "description"} (the description may be an empty string)
  score   criteria is ["description of level 0", "description of level 1", ...], lowest first
For each question: template.render() builds the text, readout.option_logits() reads the logits, and decide() is
called with the T from temperature.json.
Returns the probabilities and confidence of each question.

    japanese-jev serve --config configs/qwen3-0.6b-ja.json
    curl -s -X POST http://127.0.0.1:8000/v1/systemone -H 'content-type: application/json' -d '{...}'
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from . import readout
from .calibrate import temperature_for
from .template import KINDS, LETTERS, render


def build_options(kind: str, q: dict) -> list[tuple[str, str]]:
    if kind == "noul":
        return [("yes", ""), ("no", "")]
    crit = q.get("criteria")
    if kind == "score":
        if not isinstance(crit, list) or len(crit) < 2:
            raise ValueError("criteria for score must be a list of at least 2 levels (lowest first)")
        return [(str(i), str(c)) for i, c in enumerate(crit)]
    if isinstance(crit, dict):
        opts = [(str(k), str(v or "")) for k, v in crit.items()]
    elif isinstance(crit, list):
        opts = [(str(c), "") for c in crit]
    else:
        raise ValueError("criteria for choice must be {option: description} or [option]")
    if not 2 <= len(opts) <= len(LETTERS):
        raise ValueError(f"the number of options must be between 2 and {len(LETTERS)}, got {len(opts)}")
    return opts


def answer_one(model, tok, temps: dict[str, float], state: str, q: dict) -> dict:
    kind = q.get("type")
    if kind not in KINDS:
        raise ValueError(f"type must be one of {KINDS}, got {kind!r}")
    opts = build_options(kind, q)
    prompt = render(kind, state, q.get("instructions", ""), opts)
    p = readout.decide(readout.option_logits(model, tok, prompt, len(opts)), temperature_for(temps, kind, len(opts)))
    conf = readout.confidence(p)
    if kind == "noul":
        return {"type": kind, "probability": p[0], "confidence": conf}
    if kind == "score":
        return {"type": kind, "score": sum(i * pi for i, pi in enumerate(p)), "probabilities": p, "confidence": conf}
    best = max(range(len(p)), key=p.__getitem__)
    return {"type": kind, "answer": opts[best][0],
            "probabilities": {name: pi for (name, _), pi in zip(opts, p)}, "confidence": conf}


def create_app(cfg: dict):
    from fastapi import FastAPI, HTTPException

    work = Path(cfg["work_dir"])
    tpath = work / "temperature.json"
    if tpath.exists():
        with open(tpath, encoding="utf-8") as f:
            temps = json.load(f)
    else:
        print(f"[serve] no {tpath}, using T=1 (run japanese-jev calibrate first for better probabilities)")
        temps = {k: 1.0 for k in KINDS}
    model, tok = readout.load_model(cfg)
    if hasattr(model, "merge_and_unload"):       # merge the LoRA into the base model: same results, about 44 ms -> 25 ms per item (see bench)
        model = model.merge_and_unload()
        model.eval()
    model_name = f"japanese-jev/{cfg.get('run_name', 'dev')}"

    app = FastAPI(title="japanese-jev", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_name, "temperature": temps}

    @app.post("/v1/systemone")
    def systemone(req: dict):
        state = req.get("state")
        questions = req.get("questions")
        if not isinstance(state, str) or not isinstance(questions, dict) or not questions:
            raise HTTPException(400, "state (string) and questions (non-empty object) are required")
        t0 = time.perf_counter()
        answers = {}
        for name, q in questions.items():
            try:
                answers[name] = answer_one(model, tok, temps, state, q)
            except ValueError as e:
                raise HTTPException(400, f"{name}: {e}") from None
        return {"model": model_name, "answers": answers,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}

    return app


def main(cfg: dict) -> None:
    import uvicorn

    s = cfg.get("serve", {})
    app = create_app(cfg)
    uvicorn.run(app, host=s.get("host", "127.0.0.1"), port=int(s.get("port", 8000)))
