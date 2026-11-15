"""Distillation: the small model learns the teacher's probability for every option, not just the one correct letter.

When the config has "distill": {"alpha": 0.5}, `japanese-jev train` comes here instead of train_trl.py.
The data is data_distill/train.jsonl written by scripts/distill_label.py: each row has the usual fields plus
"soft" (the teacher's probabilities, in the same order as options).

The only difference from train_trl.py (SFT) is the loss:
  SFT      looks at the correct letter only:           loss = -log p(correct letter)
  distill  also matches the teacher's distribution:    loss = alpha * KL(teacher || student) + (1 - alpha) * (-log p(correct letter))
  p is a softmax at the last position over the option letters of this item only, exactly how readout.py reads answers.
Rows without soft (e.g. valid) only get the second term; rows without a label only get the first. So eval_loss is
the cross-entropy over option letters on valid, which is smaller than the eval_loss of train_trl.py (computed over
the whole vocabulary, plus an EOS token); the two are not comparable.

Everything else (LoRA hyperparameters, learning rate, checkpoints, resuming, picking the checkpoint with the lowest
eval_loss, continuing from init_adapter) comes from cfg as in train_trl.py.
"""

from __future__ import annotations

import json
from pathlib import Path

from .template import LETTERS, render

NEG = -1e4      # mask out option letters this item does not have; -inf is avoided because 0 * inf = nan


def _read(path: Path) -> list[dict]:
    rows = []
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if "prompt" not in r:                 # new items with only a question (datasets/distill/) have no prompt; render it here
                r["prompt"] = render(r["kind"], r["state"], r["question"], [tuple(o) for o in r["options"]])
            rows.append(r)
    return rows


def main(cfg: dict, max_steps: int | None = None) -> None:
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    from .readout import letter_ids

    data_dir = Path(cfg["data_dir"])
    out_dir = Path(cfg["work_dir"]) / "adapters"
    out_dir.mkdir(parents=True, exist_ok=True)
    lora, tr = cfg["lora"], cfg["train"]
    alpha = float(cfg["distill"].get("alpha", 0.5))
    max_len = int(tr["max_len"])

    train_rows, valid_rows = _read(data_dir / "train.jsonl"), _read(data_dir / "valid.jsonl")
    n_soft = sum("soft" in r for r in train_rows)
    print(f"[train_distill] train {len(train_rows)} rows (with teacher probabilities: {n_soft})  valid {len(valid_rows)} rows  alpha {alpha}  → {out_dir}")
    if n_soft == 0:
        raise ValueError(f"{data_dir / 'train.jsonl'} has no row with soft; run scripts/distill_label.py first")

    bf16 = tr.get("precision", "bf16") == "bf16" and torch.cuda.is_available()
    dtype = torch.bfloat16 if bf16 else torch.float32
    tok = AutoTokenizer.from_pretrained(str(cfg["base_model"]))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side, tok.truncation_side = "right", "left"      # truncate from the start when too long; [decision]: must stay at the end
    ids = letter_ids(tok)

    base = AutoModelForCausalLM.from_pretrained(str(cfg["base_model"]), dtype=dtype)
    init = cfg.get("init_adapter")
    if init:
        init = Path(init) if Path(init).is_absolute() else Path(cfg["_root"]) / init
        if not (init / "adapter_config.json").exists():
            raise FileNotFoundError(f"init_adapter {init} has no adapter_config.json")
        print(f"[train_distill] continuing from an existing adapter: {init}")
        model = PeftModel.from_pretrained(base, str(init), is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(
            r=int(lora["r"]), lora_alpha=int(lora["alpha"]), lora_dropout=float(lora["dropout"]),
            task_type="CAUSAL_LM", target_modules=list(lora["target_modules"])))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False

    def collate(batch: list[dict]) -> dict:
        enc = tok([r["prompt"] for r in batch], return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len, add_special_tokens=False)
        B, K = len(batch), len(LETTERS)
        soft = torch.zeros(B, K)
        mask = torch.zeros(B, K, dtype=torch.bool)
        gold = torch.full((B,), -1, dtype=torch.long)
        has_soft = torch.zeros(B, dtype=torch.bool)
        for i, r in enumerate(batch):
            n = int(r["n"])
            mask[i, :n] = True
            if r.get("label"):
                gold[i] = LETTERS.index(r["label"])
            if r.get("soft") and len(r["soft"]) == n:
                soft[i, :n] = torch.tensor(r["soft"], dtype=torch.float32)
                has_soft[i] = True
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "soft": soft, "opt_mask": mask, "gold": gold, "has_soft": has_soft}

    letter_t = torch.tensor(ids)

    class DistillTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            soft, mask = inputs["soft"], inputs["opt_mask"]
            gold, has_soft = inputs["gold"], inputs["has_soft"]
            core = model.get_base_model() if hasattr(model, "get_base_model") else model
            h = core.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).last_hidden_state
            last = inputs["attention_mask"].sum(1) - 1                          # position of the last real token in each row
            h = h[torch.arange(h.size(0), device=h.device), last]
            z = core.lm_head(h)[:, letter_t.to(h.device)].float()               # [B, 16]: last position only, letters only
            logp = F.log_softmax(z.masked_fill(~mask, NEG), dim=-1)
            kl = (soft * (torch.log(soft.clamp_min(1e-8)) - logp)).masked_fill(~mask, 0).sum(-1)
            has_gold = gold >= 0
            ce = -logp.gather(1, gold.clamp_min(0).unsqueeze(1)).squeeze(1)
            w_kl = torch.where(has_soft, torch.where(has_gold, alpha, 1.0), 0.0)
            w_ce = torch.where(has_gold, torch.where(has_soft, 1 - alpha, 1.0), 0.0)
            loss = (w_kl * kl + w_ce * ce).mean()
            return (loss, {"logits": logp}) if return_outputs else loss

    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=int(tr["batch_size"]),
        per_device_eval_batch_size=int(tr["batch_size"]),
        gradient_accumulation_steps=int(tr.get("accumulation", 1)),
        learning_rate=float(tr["lr"]),
        num_train_epochs=float(tr["epochs"]),
        max_steps=int(max_steps) if max_steps else -1,
        lr_scheduler_type=tr.get("lr_scheduler", "cosine"),
        warmup_steps=float(tr.get("warmup_ratio", 0.03)),
        bf16=bf16,
        logging_steps=int(tr.get("logging_steps", 20)),
        eval_strategy="steps",
        eval_steps=int(tr.get("eval_steps", 200)),
        save_steps=int(tr.get("save_steps", 200)),
        save_total_limit=2,
        load_best_model_at_end=not cfg.get("init_adapter"),      # continued training keeps the final weights; see train_trl.py for the reason
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        prediction_loss_only=True,
        label_names=["soft", "opt_mask", "gold", "has_soft"],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=int(cfg.get("seed", 42)),
        report_to="none",
    )
    trainer = DistillTrainer(model=model, args=args, data_collator=collate,
                             train_dataset=train_rows, eval_dataset=valid_rows)
    ckpts = sorted(out_dir.glob("checkpoint-*"), key=lambda d: int(d.name.split("-")[1]))
    if ckpts:
        print(f"[train_distill] found a checkpoint, resuming from {ckpts[-1].name}")
    trainer.train(resume_from_checkpoint=str(ckpts[-1]) if ckpts else None)
    model.save_pretrained(str(out_dir))
    print(f"[train_distill] done, adapter in {out_dir}")
