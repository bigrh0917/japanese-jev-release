"""Training on a CUDA GPU (PyTorch + TRL SFTTrainer + PEFT LoRA).

Reads cfg["data_dir"]/train.jsonl and valid.jsonl, keeps only the prompt / completion columns, computes the loss
on the completion only (the one letter + the EOS that TRL appends), and saves the LoRA weights to
cfg["work_dir"]/adapters. All hyperparameters come from cfg["lora"] and cfg["train"]; none are hard-coded.

TRL does not apply a chat template to prompt-completion data; it concatenates prompt + completion, like train_mlx.py.
Argument names have changed across TRL versions (max_seq_length -> max_length; completion_only_loss appeared in
0.17); on an error, check SFTConfig of the installed version.
"""

from __future__ import annotations

from pathlib import Path


def main(cfg: dict, max_steps: int | None = None) -> None:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    data_dir = Path(cfg["data_dir"])
    out_dir = Path(cfg["work_dir"]) / "adapters"
    out_dir.mkdir(parents=True, exist_ok=True)
    lora, tr = cfg["lora"], cfg["train"]

    for f in ("train.jsonl", "valid.jsonl"):
        if not (data_dir / f).exists():
            raise FileNotFoundError(f"{data_dir / f} does not exist; run japanese-jev build first")

    ds = load_dataset("json", data_files={"train": str(data_dir / "train.jsonl"),
                                          "valid": str(data_dir / "valid.jsonl")})
    ds = ds.select_columns(["prompt", "completion"])   # keep only these two columns so TRL treats the data as prompt-completion
    print(f"[train_trl] train {len(ds['train'])} rows  valid {len(ds['valid'])} rows  → {out_dir}")

    bf16 = tr.get("precision", "bf16") == "bf16" and torch.cuda.is_available()
    if not torch.cuda.is_available():
        print("[train_trl] warning: CUDA not detected, training will run on CPU and be very slow; under WSL2, check the driver with nvidia-smi")

    args = SFTConfig(
        output_dir=str(out_dir),
        model_init_kwargs={"dtype": torch.bfloat16 if bf16 else torch.float32},   # passing the model as a string loads fp32 by default, which does not fit in 8 GB
        per_device_train_batch_size=int(tr["batch_size"]),
        per_device_eval_batch_size=int(tr["batch_size"]),
        gradient_accumulation_steps=int(tr.get("accumulation", 1)),
        learning_rate=float(tr["lr"]),
        num_train_epochs=float(tr["epochs"]),
        max_steps=int(max_steps) if max_steps else -1,
        lr_scheduler_type=tr.get("lr_scheduler", "cosine"),
        warmup_steps=float(tr.get("warmup_ratio", 0.03)),   # transformers 5 removed warmup_ratio; a warmup_steps value between 0 and 1 is a ratio
        bf16=bf16,
        max_length=int(tr["max_len"]),
        completion_only_loss=True,                 # already the default for prompt-completion data; spelled out for clarity
        gradient_checkpointing=True,
        logging_steps=int(tr.get("logging_steps", 20)),
        eval_strategy="steps",
        eval_steps=int(tr.get("eval_steps", 200)),
        save_steps=int(tr.get("save_steps", 200)),
        save_total_limit=2,
        # From scratch: after training, the checkpoint with the lowest eval_loss becomes the final adapter, guarding against overfitting in the second epoch.
        # Continued training (init_adapter): keep the final weights. valid only contains the old tasks, and feeding new data slightly raises its loss, so picking by lowest loss
        # would select a checkpoint from the very start, where the new data has barely been learned.
        load_best_model_at_end=not cfg.get("init_adapter"),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=int(cfg.get("seed", 42)),
        report_to="none",
    )
    peft = LoraConfig(
        r=int(lora["r"]), lora_alpha=int(lora["alpha"]), lora_dropout=float(lora["dropout"]),
        task_type="CAUSAL_LM",
        target_modules=list(lora["target_modules"]),   # without this, PEFT only adapts q/v on Qwen3 by default
    )
    init = cfg.get("init_adapter")
    if init:
        # Continue from an already trained LoRA (new data + part of the old data) instead of starting over.
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        init = Path(init) if Path(init).is_absolute() else Path(cfg["_root"]) / init
        if not (init / "adapter_config.json").exists():
            raise FileNotFoundError(f"init_adapter {init} has no adapter_config.json")
        print(f"[train_trl] continuing from an existing adapter: {init}")
        tok = AutoTokenizer.from_pretrained(str(cfg["base_model"]))
        base = AutoModelForCausalLM.from_pretrained(str(cfg["base_model"]), dtype=torch.bfloat16 if bf16 else torch.float32)
        model = PeftModel.from_pretrained(base, str(init), is_trainable=True)
        model.enable_input_require_grads()            # gradient checkpointing with a frozen base model needs this, otherwise there are no gradients
        trainer = SFTTrainer(model=model, args=args, processing_class=tok,
                             train_dataset=ds["train"], eval_dataset=ds["valid"])
    else:
        trainer = SFTTrainer(model=str(cfg["base_model"]), args=args, peft_config=peft,
                             train_dataset=ds["train"], eval_dataset=ds["valid"])
    ckpts = sorted(out_dir.glob("checkpoint-*"), key=lambda d: int(d.name.split("-")[1]))
    if ckpts:      # resume when the directory has a checkpoint; use another work_dir to start from scratch
        print(f"[train_trl] found a checkpoint, resuming from {ckpts[-1].name}")
    trainer.train(resume_from_checkpoint=str(ckpts[-1]) if ckpts else None)
    trainer.save_model(str(out_dir))     # adapter_model.safetensors + adapter_config.json; readout uses the tokenizer of the base model
    print(f"[train_trl] done, adapter in {out_dir}")
