"""Send the test items to the Jev API (through Vercel AI Gateway or OpenRouter), save the answers as a
predictions.jsonl in the same format as the local model's, and compare with audit.py.

    export AI_GATEWAY_API_KEY=vck_...            # Vercel; keep the key in the environment only, never in a file or in git
    python scripts/eval_jev_api.py --limit 2      # try 2 items per set first and check the response format
    python scripts/eval_jev_api.py                # everything (items already asked are not asked again)

    export OPENROUTER_API_KEY=sk-or-...          # go through OpenRouter instead
    python scripts/eval_jev_api.py --via openrouter --limit 2

Both providers use the same request and response shape (TypeSafe's systemone):
  vercel       POST https://ai-gateway.vercel.sh/typesafe/v1/systemone   model typesafe-ai/jev
  openrouter   POST https://openrouter.ai/api/v1/systemone              model typesafe/jev-1.14
By default only jcommonsenseqa / massive_ja and the three self-built sets are sent; WRIME and Livedoor have
redistribution restrictions and are not sent. Only test_* is sent, never the sealed final_*.
Outputs in runs/jev-api/:
  raw.jsonl           raw response per item (for resuming and debugging)
  predictions.jsonl   dataset / id / kind / n / label / pred / p_max, same format as evaluate writes
Latency includes the network round trip and is not comparable with local milliseconds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LETTERS = "ABCDEFGHIJKLMNOP"
VIA = {   # name -> (endpoint, default model, environment variable holding the key)
    "vercel": ("https://ai-gateway.vercel.sh/typesafe/v1/systemone", "typesafe-ai/jev", "AI_GATEWAY_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1/systemone", "typesafe/jev-1.14", "OPENROUTER_API_KEY"),
}
DEFAULT_SETS = ["jcommonsenseqa", "massive_ja", "jcola", "ja_ambiguity", "ja_knowledge", "ja_negation"]
YES = {"yes", "はい", "true", "該当する"}


def question_for(row: dict) -> tuple[dict, list[str]]:
    """One item -> the question object for the API, plus keys in the same order as options (to map returned probabilities back to letters)."""
    opts = row["options"]
    names = [o[0] for o in opts]
    if row["kind"] == "noul":
        return {"type": "noul", "instructions": row["question"]}, names
    if row["kind"] == "score":
        crit = [f"{n}：{d}" if d else n for n, d in opts]
        return {"type": "score", "instructions": row["question"], "criteria": crit}, crit
    keys = names if len(set(names)) == len(names) else [f"{LETTERS[i]}. {n}" for i, n in enumerate(names)]
    return {"type": "choice", "instructions": row["question"],
            "criteria": {k: (o[1] or o[0]) for k, o in zip(keys, opts)}}, keys


def probs_from(row: dict, ans: dict, keys: list[str]) -> list[float] | None:
    """Turn the returned answer into probabilities in the same order as options; None if the format is not recognized (the raw response stays in raw.jsonl)."""
    n = row["n"]
    if ans.get("type") == "noul":
        p = float(ans["noul"])
        yes = next((i for i, k in enumerate(keys) if k.lower() in YES), 0)
        out = [1 - p] * n
        out[yes] = p
        return out
    pr = ans.get("probabilities")
    if isinstance(pr, list) and len(pr) == n:
        return [float(x) for x in pr]
    if isinstance(pr, dict):
        if all(k in pr for k in keys):
            return [float(pr[k]) for k in keys]
        if all(str(i) in pr for i in range(n)):                 # score: keys are 0..n-1
            return [float(pr[str(i)]) for i in range(n)]
        if len(pr) == n:                                        # keys do not match but the count does: use the returned order
            return [float(v) for v in pr.values()]
    return None


def call(url: str, key: str, model: str, row: dict) -> dict:
    q, _ = question_for(row)
    body = json.dumps({"model": model, "state": row["state"], "questions": {"q": q}}, ensure_ascii=False).encode()
    for attempt in range(6):
        req = urllib.request.Request(url, data=body, method="POST", headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=60) as r:
                out = json.loads(r.read())
            out["_ms"] = round((time.perf_counter() - t0) * 1000)
            return out
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")[:500]
            if e.code in (429, 500, 502, 503, 529) and attempt < 5:
                time.sleep(2 ** attempt)
                continue
            return {"_error": f"HTTP {e.code}: {msg}"}
        except Exception as e:                                   # network errors and the like
            if attempt < 5:
                time.sleep(2 ** attempt)
                continue
            return {"_error": repr(e)}
    return {"_error": "out of retries"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--via", choices=list(VIA), default="vercel")
    ap.add_argument("--model", default=None, help="defaults to the provider's default model")
    ap.add_argument("--sets", nargs="*", default=DEFAULT_SETS)
    ap.add_argument("--limit", type=int, default=0, help="send only the first N items of each set (for a trial run)")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="runs/jev-api")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    url, default_model, env = VIA[a.via]
    a.model = a.model or default_model
    key = os.environ.get(env)
    if not key:
        sys.exit(f"run export {env}=<your key> first")
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "raw.jsonl"
    done = {}
    if raw_path.exists():
        for line in open(raw_path, encoding="utf-8"):
            r = json.loads(line)
            if "_error" not in r["response"]:
                done[(r["dataset"], r["id"])] = r["response"]

    todo, rows_by_set = [], {}
    for name in a.sets:
        p = ROOT / a.data_dir / f"test_{name}.jsonl"
        if not p.exists():
            print(f"no {p}, skipping")
            continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        rows = rows[:a.limit] if a.limit else rows
        rows_by_set[name] = rows
        todo += [(name, r) for r in rows if (name, r["id"]) not in done]
    print(f"via {a.via}  model {a.model}  to send {len(todo)} items ({len(done)} already answered)")

    cost = 0.0
    with open(raw_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(a.workers) as ex:
        for i, ((name, r), resp) in enumerate(zip(todo, ex.map(lambda t: call(url, key, a.model, t[1]), todo)), 1):
            f.write(json.dumps({"dataset": name, "id": r["id"], "response": resp}, ensure_ascii=False) + "\n")
            f.flush()
            if "_error" in resp:
                print(f"  [{name} {r['id']}] failed: {resp['_error']}")
                if i <= 3:
                    sys.exit("failed within the first few items; see the error above. Not sending more")
            else:
                done[(name, r["id"])] = resp
                gw = ((resp.get("provider_metadata") or {}).get("gateway") or {}).get("cost")      # Vercel reports the cost here
                cost += float(gw or (resp.get("usage") or {}).get("cost") or 0)
            if i % 100 == 0:
                print(f"  {i}/{len(todo)}  cost so far in this run {cost:.4f}", flush=True)
    print(f"cost of this run (sum of the cost fields returned by the API, USD): {cost:.4f}")

    bad = 0
    with open(out / "predictions.jsonl", "w", encoding="utf-8") as f:
        for name, rows in rows_by_set.items():
            hit = tot = 0
            for r in rows:
                resp = done.get((name, r["id"]))
                if not resp:
                    continue
                _, keys = question_for(r)
                p = probs_from(r, (resp.get("answers") or {}).get("q") or {}, keys)
                if p is None:
                    bad += 1
                    continue
                k = max(range(len(p)), key=p.__getitem__)
                tot += 1
                hit += LETTERS[k] == r["label"]
                f.write(json.dumps({"dataset": name, "id": r["id"], "source": r["source"], "kind": r["kind"], "n": r["n"],
                                    "label": r["label"], "pred": LETTERS[k], "p_max": round(p[k], 4)}, ensure_ascii=False) + "\n")
            if tot:
                print(f"  {name:<16} {tot:>4} items  acc {hit / tot:.3f}")
    if bad:
        print(f"{bad} responses had an unrecognized format (not written to predictions); check the raw responses in {raw_path}")
    print(f"wrote {out / 'predictions.jsonl'}")


if __name__ == "__main__":
    main()
