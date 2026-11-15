"""Readout: one forward pass, take the logits at the last position, keep only the option letters of this item, divide by T and softmax.

The trained model generates no text. evaluate.py, calibrate.py and serve.py only call the functions here.
Two backends, chosen automatically from the files in the adapter directory:
  torch   transformers + PEFT (Linux + CUDA; with torch installed a Mac can also run CPU inference)
  mlx     mlx-lm (Mac; only it can read adapters saved by train_mlx.py, and only torch can read PEFT adapters)

    load_model(cfg, adapter_dir=None)      returns (model, tok); without adapter_dir, work_dir/adapters is used, and if that is missing, the untrained base
    option_logits(model, tok, prompt, n)   float32 numpy vector of length n; index i corresponds to LETTERS[i]
    decide(logits, T=1.0)                  softmax(logits / T), returns a list of probabilities
    confidence(p)                          the official definition (p_max - 1/n) / (1 - 1/n): 1 when all mass is on one option, 0 when uniform
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from .template import LETTERS

_LETTER_IDS: dict[int, list[int]] = {}   # id(tok) -> token ids of the 16 letters


def adapter_backend(adapter_dir: Path | None) -> str:
    """Check whose files are in the adapter directory: mlx-lm saves adapters.safetensors, PEFT saves adapter_model.safetensors."""
    if adapter_dir is None:
        return "torch"
    if (adapter_dir / "adapters.safetensors").exists():
        return "mlx"
    if (adapter_dir / "adapter_model.safetensors").exists() or (adapter_dir / "adapter_model.bin").exists():
        return "torch"
    raise FileNotFoundError(f"{adapter_dir} contains neither adapters.safetensors (mlx) nor adapter_model.safetensors (PEFT)")


def resolve_adapter(cfg: dict, adapter_dir) -> Path | None:
    """None -> cfg["adapter_dir"] (CLI --adapter-dir) or work_dir/adapters (only if it exists); False / "" -> force no adapter (untrained base)."""
    if adapter_dir is False or adapter_dir == "":
        return None
    if adapter_dir is None:
        adapter_dir = cfg.get("adapter_dir")
    if adapter_dir is None:
        p = Path(cfg["work_dir"]) / "adapters"
        return p if p.exists() else None
    p = Path(adapter_dir)
    if not p.is_absolute():
        p = Path(cfg["_root"]) / p
    return p


def load_model(cfg: dict, adapter_dir=None, backend: str | None = None):
    base = str(cfg["base_model"])
    adir = resolve_adapter(cfg, adapter_dir)
    if backend is None:
        # with an adapter, go by its file format; without one (untrained baseline), mlx on Mac and torch elsewhere
        backend = adapter_backend(adir) if adir else ("mlx" if sys.platform == "darwin" else "torch")
    print(f"[readout] base {base}  adapter {adir or 'none (untrained)'}  backend {backend}")

    if backend == "mlx":
        from mlx_lm import load

        model, tok = load(base, adapter_path=str(adir) if adir else None)
        return model, tok

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16
    else:
        device, dtype = "cpu", torch.float32      # MPS lacks some bf16 ops; for fast inference on Mac use the mlx backend
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype)
    if adir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adir))
    model.to(device).eval()
    return model, tok


def _is_mlx(model) -> bool:
    return type(model).__module__.startswith("mlx")


def letter_ids(tok) -> list[int]:
    """Token ids of " A".." P"; each must be a single token."""
    key = id(tok)
    if key not in _LETTER_IDS:
        ids = []
        for c in LETTERS:
            enc = tok.encode(" " + c, add_special_tokens=False)
            assert len(enc) == 1, f"' {c}' is split into {len(enc)} tokens: {enc}; use another tokenizer or change the template"
            ids.append(enc[0])
        _LETTER_IDS[key] = ids
    return _LETTER_IDS[key]


def last_token_logits(model, enc):
    """Compute vocabulary logits at the last position only; returns [batch, vocab].

    model(**enc).logits runs the output layer at every position, giving a batch x length x 150K-vocabulary tensor,
    while only the last position is used; with 32 items per batch that tensor does not fit in 8 GB of VRAM. Here
    the hidden state of the last position is taken first and only that goes through the output layer.
    LoRA is injected into the layers of the base model, so calling the base model inside a PeftModel gives the same result.
    """
    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    body, head = getattr(core, "model", None), getattr(core, "lm_head", None)
    if body is None or head is None:                       # unknown architecture: fall back to the plain forward pass
        return model(**enc).logits[:, -1]
    return head(body(**enc).last_hidden_state[:, -1])


def option_logits(model, tok, prompt: str, n: int) -> np.ndarray:
    ids = letter_ids(tok)[:n]
    if _is_mlx(model):
        import mlx.core as mx

        toks = mx.array([tok.encode(prompt, add_special_tokens=False)])
        last = model(toks)[0, -1].astype(mx.float32)
        return np.array(last[mx.array(ids)], dtype=np.float32)

    import torch

    with torch.no_grad():
        enc = tok(prompt, return_tensors="pt").to(model.device)
        last = last_token_logits(model, enc)[0]
        return last[ids].float().cpu().numpy()


def decide(logits, T: float = 1.0) -> list[float]:
    z = np.asarray(logits, dtype=np.float64) / float(T)
    z = z - z.max()
    p = np.exp(z)
    p /= p.sum()
    return p.tolist()


def confidence(p) -> float:
    p = np.asarray(p, dtype=np.float64)
    n = len(p)
    return float((p.max() - 1.0 / n) / (1.0 - 1.0 / n))


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def collect_logits(model, tok, rows: list[dict], desc: str = "") -> tuple[np.ndarray, np.ndarray, list[float]]:
    """One forward pass per row. Returns L [N, 16] (unused positions filled with -inf), correct indices y [N], and the time per row (seconds)."""
    L = np.full((len(rows), len(LETTERS)), -np.inf, dtype=np.float32)
    y = np.zeros(len(rows), dtype=np.int64)
    times = []
    for i, r in enumerate(rows):
        t0 = time.perf_counter()
        L[i, :r["n"]] = option_logits(model, tok, r["prompt"], r["n"])
        times.append(time.perf_counter() - t0)
        y[i] = LETTERS.index(r["label"])
        if desc and (i + 1) % 100 == 0:
            print(f"  [{desc}] {i + 1}/{len(rows)}", flush=True)
    return L, y, times


def check_tokens(tok, prompt: str, completion: str = " A") -> None:
    """Debugging aid: confirm that the last token of prompt + completion is the id of the letter."""
    full = tok.encode(prompt + completion, add_special_tokens=False)
    p = tok.encode(prompt, add_special_tokens=False)
    assert full[:len(p)] == p and len(full) == len(p) + 1, "completion is not a single token appended to prompt"
    assert full[-1] == letter_ids(tok)[LETTERS.index(completion.strip())], "the last token is not the id of the letter"
    print("[readout] token check passed:", full[-1])
