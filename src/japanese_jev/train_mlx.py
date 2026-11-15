"""Training on Mac (mlx-lm Python API + LoRA).

The `mlx_lm.lora` command line is not used: it wraps prompt/completion in Qwen3's chat template (with an empty
<think> block), which does not match the template used at inference. Here a custom Dataset feeds the raw text of
template.render(), and the loss covers only the one letter.
mlx-lm has no alpha; scale = alpha / r is used. Outputs go to cfg["work_dir"]/adapters as in train_trl.py:
    adapter_config.json   fine_tune_type / num_layers / lora_parameters; mlx_lm.load(adapter_path=) relies on it
    adapters.safetensors  final weights; a 0000200_adapters.safetensors is saved every save_steps during training

The APIs used (mlx_lm.tuner.train / TrainingArgs / linear_to_lora_layers) are the ones the mlx_lm.lora command
line uses itself; if a new version breaks them, follow train_model in mlx_lm/lora.py.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .template import LETTERS

# PEFT module names -> submodule paths of Qwen3 in mlx-lm (mlx-lm "keys" need the parent module name)
_PARENT = {"q_proj": "self_attn", "k_proj": "self_attn", "v_proj": "self_attn", "o_proj": "self_attn",
           "gate_proj": "mlp", "up_proj": "mlp", "down_proj": "mlp"}


class LetterDataset:
    """Each sample = (token sequence, position from which the loss is computed). Raw prompt + " A", no chat template."""

    def __init__(self, path: Path, tok, max_len: int):
        self.items: list[tuple[list[int], int]] = []
        skipped = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                p = tok.encode(d["prompt"], add_special_tokens=False)
                full = tok.encode(d["prompt"] + d["completion"], add_special_tokens=False)
                assert full[:len(p)] == p and len(full) == len(p) + 1, "the letter must be a single token appended to prompt"
                if len(full) > max_len:
                    skipped += 1           # truncation would cut off the final letter, so such samples are dropped
                    continue
                self.items.append((full, len(p)))
        if skipped:
            print(f"[train_mlx] {path.name}: {skipped} rows longer than {max_len} tokens, dropped")

    def __getitem__(self, i):
        return self.items[i]

    def __len__(self):
        return len(self.items)


def letter_loss(model, batch, lengths):
    """Cross-entropy at the letter position only. lengths[:, 0] is the prompt length, lengths[:, 1] the full length.

    The mask of mlx-lm's default_loss is offset <= step <= length, which counts one pad position; here it is offset <= step < length.
    """
    import mlx.core as mx
    import mlx.nn as nn

    logits = model(batch[:, :-1])
    targets = batch[:, 1:]
    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps < lengths[:, 1:])
    ce = nn.losses.cross_entropy(logits, targets) * mask
    ntoks = mask.sum()
    return ce.astype(mx.float32).sum() / ntoks, ntoks


def main(cfg: dict, max_steps: int | None = None) -> None:
    import mlx.optimizers as optim
    from mlx_lm import load
    from mlx_lm.tuner import TrainingArgs, linear_to_lora_layers, train
    from mlx_lm.utils import save_config

    data_dir = Path(cfg["data_dir"])
    out = Path(cfg["work_dir"]) / "adapters"
    out.mkdir(parents=True, exist_ok=True)
    lora, tr = cfg["lora"], cfg["train"]
    max_len = int(tr["max_len"])

    model, tok = load(str(cfg["base_model"]))
    model.freeze()
    lora_params = {
        "rank": int(lora["r"]),
        "scale": float(lora["alpha"]) / float(lora["r"]),    # scale = PEFT alpha / r
        "dropout": float(lora["dropout"]),
        "keys": sorted({f"{_PARENT[m]}.{m}" for m in lora["target_modules"] if m in _PARENT}),
    }
    num_layers = -1                                           # -1 = LoRA on all layers
    linear_to_lora_layers(model, num_layers, lora_params)

    train_ds = LetterDataset(data_dir / "train.jsonl", tok, max_len)
    valid_ds = LetterDataset(data_dir / "valid.jsonl", tok, max_len)
    batch = int(tr["batch_size"])
    accum = int(tr.get("accumulation", 1))
    iters = int(tr["epochs"]) * math.ceil(len(train_ds) / (batch * accum))
    if max_steps:
        iters = min(iters, int(max_steps))
    print(f"[train_mlx] train {len(train_ds)} rows  valid {len(valid_ds)} rows  iters {iters}  → {out}")

    save_config({"fine_tune_type": "lora", "num_layers": num_layers, "lora_parameters": lora_params},
                out / "adapter_config.json")
    args = TrainingArgs(
        batch_size=batch,
        iters=iters,
        steps_per_report=int(tr.get("logging_steps", 20)),
        steps_per_eval=int(tr.get("eval_steps", 200)),
        steps_per_save=int(tr.get("save_steps", 200)),
        max_seq_length=max_len,
        adapter_file=str(out / "adapters.safetensors"),
        grad_accumulation_steps=accum,
    )
    lr = float(tr["lr"])
    warm = max(1, int(iters * float(tr.get("warmup_ratio", 0.03))))
    schedule = optim.schedulers.join_schedules(
        [optim.schedulers.linear_schedule(0.0, lr, warm), optim.schedulers.cosine_decay(lr, max(1, iters - warm))],
        [warm + 1],
    )
    train(model, optim.AdamW(learning_rate=schedule), train_ds, valid_ds, args, loss=letter_loss)
    print(f"[train_mlx] done, adapter in {out} (alphabet {LETTERS[:4]}...; single-token check done when the dataset was loaded)")
