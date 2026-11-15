"""Command-line entry point: `japanese-jev <stage> --config configs/xxx.json` (or `python -m japanese_jev ...`).

    japanese-jev build     --config configs/qwen3-0.6b-ja.json   # public data -> data/*.jsonl (once per dataset)
    japanese-jev train     --config ...                          # MLX on Mac, TRL elsewhere
    japanese-jev calibrate --config ...                          # fit temperatures on calib
    japanese-jev evaluate  --config ...                          # tables on valid and each test_<source>
    japanese-jev serve     --config ...                          # FastAPI /v1/systemone
    japanese-jev run       --config ...                          # train -> calibrate -> evaluate in one go

Re-evaluate the same adapter with another calibration set (no retraining):
    japanese-jev calibrate --config ... --work-dir runs/x_sib --adapter-dir runs/x/adapters --calib-split calib_sib200
    japanese-jev evaluate  --config ... --work-dir runs/x_sib --adapter-dir runs/x/adapters
"""

from __future__ import annotations

import argparse
import sys

from . import config as cfgmod

STAGES = ("build", "train", "calibrate", "evaluate", "serve", "run", "bench")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="japanese-jev", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)
    for name in STAGES:
        sp = sub.add_parser(name)
        sp.add_argument("--config", required=True, help="a JSON file under configs/")
        sp.add_argument("--work-dir", default=None, help="override work_dir from the config (used by scripts/train.sh)")
        if name in ("train", "run"):
            sp.add_argument("--max-steps", type=int, default=None, help="train only this many steps (smoke test)")
        if name in ("calibrate", "evaluate", "serve", "bench"):
            sp.add_argument("--adapter-dir", default=None, help="use the adapter in another directory (default: work_dir/adapters)")
        if name == "evaluate":
            sp.add_argument("--final", action="store_true", help="evaluate the sealed sets final_*.jsonl (run once, right before reporting numbers); writes evaluation_final.json")
            sp.add_argument("--blank-state", action="store_true", help="blank out [state] in every item (question and options only). Accuracy well above chance means the set can be guessed from the options; writes evaluation_blank.json")
            sp.add_argument("--perturb", default=None, choices=["hiragana", "noise", "zenkaku", "hankaku"],
                            help="robustness test: rewrite [state] in every item before evaluating (hiragana: all hiragana / noise: append an unrelated sentence / zenkaku: full-width / hankaku: half-width); writes evaluation_<name>.json")
            sp.add_argument("--shots", type=int, default=0, help="put N examples from calib before every item (a fair baseline for the untrained base model); writes evaluation_shotsN.json")
        if name == "calibrate":
            sp.add_argument("--calib-split", default=None, help="use another calibration set, e.g. calib_sib200 (default: calibration.split in the config)")
    return p


def _train(cfg, max_steps):
    if cfg.get("distill"):                 # a config with a distill section trains by distillation (learning the teacher's probabilities); see train_distill.py
        from . import train_distill as mod
    elif sys.platform == "darwin":
        from . import train_mlx as mod
    else:
        from . import train_trl as mod
    mod.main(cfg, max_steps=max_steps)


def main(argv=None) -> None:
    args = _parser().parse_args(argv)
    cfg = cfgmod.load(args.config, overrides={"work_dir": args.work_dir})
    if getattr(args, "adapter_dir", None):
        cfg["adapter_dir"] = args.adapter_dir
    if getattr(args, "blank_state", False):
        cfg["blank_state"] = True
    if getattr(args, "perturb", None):
        cfg["perturb"] = args.perturb
    if getattr(args, "final", False):
        cfg["final"] = True
    if getattr(args, "shots", 0):
        cfg["shots"] = args.shots
    if getattr(args, "calib_split", None):
        cfg.setdefault("calibration", {})["split"] = args.calib_split

    if args.stage == "build":
        from .data_pipeline import build
        build.main(cfg)
    elif args.stage == "train":
        _train(cfg, args.max_steps)
    elif args.stage == "calibrate":
        from . import calibrate
        calibrate.main(cfg)
    elif args.stage == "evaluate":
        from . import evaluate
        evaluate.main(cfg)
    elif args.stage == "bench":
        from . import bench
        bench.main(cfg)
    elif args.stage == "serve":
        from . import serve
        serve.main(cfg)
    elif args.stage == "run":
        from . import calibrate, evaluate
        _train(cfg, args.max_steps)
        calibrate.main(cfg)
        evaluate.main(cfg)


if __name__ == "__main__":
    main()
