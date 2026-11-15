# Usage

Commands assume Linux + CUDA. Results and conclusions are in the README at the repository root.

## Pipeline

```
build ──> data/*.jsonl ──> train ──> <work_dir>/adapters ──> calibrate ──> temperature.json ──> evaluate ──> evaluation.json
```

| Path | Contents |
| --- | --- |
| `src/japanese_jev/` | `template.py` three-primitive template, `readout.py` logit readout, `train_trl.py` (CUDA) / `train_mlx.py` (Mac), `calibrate.py`, `evaluate.py`, `serve.py`; the entry point is `cli.py` |
| `src/japanese_jev/data_pipeline/` | public datasets -> three-primitive jsonl: one file per dataset in `adapters/`; `build.py` merges, deduplicates and splits; `augment.py` does Japanese augmentation |
| `configs/` | one JSON file drives every stage; see `template.json` for the format |
| `scripts/train.sh` | train -> calibrate -> evaluate, one `runs/<run_name>_<timestamp>/` per run |
| `tests/` | `python tests/test_pipeline.py`; no pytest, model or data needed |
| `data/` `models/` `runs/` | generated jsonl, base models, training outputs; not in git |

## Training from scratch

```bash
uv sync
source .venv/bin/activate
python tests/test_pipeline.py
hf download Qwen/Qwen3-1.7B-Base --local-dir models/Qwen3-1.7B-Base
japanese-jev build     --config configs/qwen3-1.7b-ja.json            # public data -> data/*.jsonl, once per dataset
python scripts/check_data.py                                          # answer distribution, option shuffling, evaluation texts leaking into training
bash scripts/train.sh  --config configs/qwen3-1.7b-ja.json --smoke    # 200 steps first, to check the pipeline
bash scripts/train.sh  --config configs/qwen3-1.7b-ja.json            # full run
python scripts/report.py runs/<dir>/evaluation.json                   # result table (with the WRIME two-class average); per-item predictions are in runs/<dir>/predictions.jsonl
```

For 0.6B, use `configs/qwen3-0.6b-ja.json`. Every stage can also be run on its own:
`japanese-jev {build,train,calibrate,evaluate,serve,bench} --config ... [--work-dir ...] [--max-steps N]`.
`train` uses `train_mlx.py` on Mac and `train_trl.py` elsewhere; `readout.py` picks the backend from the files in the adapter directory.

## Continued training with synthetic data

```bash
python scripts/make_continue_set.py 5000 5 data_ft                    # synthetic data x 5 + 5,000 original training rows -> data_ft/
# set init_adapter in configs/qwen3-1.7b-ja-ft.json to runs/<dir>/adapters of the previous run
bash scripts/train.sh --config configs/qwen3-1.7b-ja-ft.json
```

Continued training keeps the weights from the end of training; training from scratch keeps the checkpoint with the lowest loss on valid.
The two files in `datasets/synthetic/` are already in the repository; `scripts/make_synthetic_v1.py` / `make_synthetic_v2.py` regenerate them.
`build` reads every `datasets/synthetic/train*.jsonl`; the released weights used only `train.jsonl` (v1).

## Baselines and checks

```bash
# untrained base model: point --work-dir at a directory without adapters/
japanese-jev calibrate --config configs/qwen3-1.7b-ja.json --work-dir runs/baseline-1.7b
japanese-jev evaluate  --config configs/qwen3-1.7b-ja.json --work-dir runs/baseline-1.7b            # 0-shot
japanese-jev evaluate  --config configs/qwen3-1.7b-ja.json --work-dir runs/baseline-1.7b --shots 4  # 4 examples before every item

japanese-jev evaluate --config ... --work-dir runs/<dir> --blank-state        # hide the text, keep question and options; well above chance = guessable from the options
japanese-jev evaluate --config ... --work-dir runs/<dir> --perturb hiragana   # rewrite the text in hiragana before evaluating (also noise / zenkaku / hankaku)
japanese-jev evaluate --config ... --work-dir runs/<dir> --final              # sealed sets final_*.jsonl
japanese-jev bench    --config ... --work-dir runs/<dir>                      # latency (warm-up, merged LoRA) -> bench.json

python scripts/audit.py lora=runs/<dir>/predictions.jsonl shot4=runs/baseline-1.7b/predictions_shots4.jsonl
python scripts/breakdown.py runs/<dir>/predictions.jsonl                      # self-built sets by phenomenon / scene / primitive
```

`audit.py` gives a 95% confidence interval for every evaluation set, two baselines (chance, always the same letter) and paired per-item comparisons between models.
`final_*.jsonl` is a second batch that does not overlap test; run it once, right before reporting numbers.

### Using another calibration set

By default the temperature is fitted on `calib.jsonl`, cut from the training sources. SIB-200 in `calib_sources` is neither trained on nor evaluated; build writes `data/calib_sib200.jsonl`,
which can be used to refit the temperature for the same adapter:

```bash
japanese-jev calibrate --config ... --work-dir runs/<dir>_sib --adapter-dir runs/<dir>/adapters --calib-split calib_sib200
japanese-jev evaluate  --config ... --work-dir runs/<dir>_sib --adapter-dir runs/<dir>/adapters
```

SIB-200 only has choice items; the noul / score temperatures are taken from `calib.jsonl` (`calibration.fallback_split`).

## Comparing with the Jev API, and distillation

You need your own API key, kept in an environment variable only.

```bash
export OPENROUTER_API_KEY=...
python scripts/eval_jev_api.py --via openrouter --limit 2     # try 2 items first
python scripts/eval_jev_api.py --via openrouter               # results in runs/jev-api/
python scripts/cascade.py runs/<dir>/predictions.jsonl        # accuracy when confident items are answered locally and the rest by Jev

python scripts/distill_label.py --max 20                      # label training items with teacher probabilities -> data_distill/ (can be interrupted and resumed)
bash scripts/train.sh --config configs/qwen3-1.7b-ja-distill.json
```

## Serving

```bash
japanese-jev serve --config configs/qwen3-1.7b-ja.json --work-dir runs/<dir>

curl -s -X POST http://127.0.0.1:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "ベルリン行きの航空券が二重に請求されました。電話も繋がりません。返金してください。",
  "questions": {
    "refund_requested": {"type": "noul", "instructions": "顧客は返金を明確に求めていますか。"},
    "request_type": {"type": "choice", "instructions": "主な要望は何ですか。",
      "criteria": {"返金": "お金を返してほしい", "予約変更": "別の便に変えてほしい", "情報": "情報だけ欲しい"}},
    "frustration": {"type": "score", "instructions": "顧客はどのくらい不満ですか。",
      "criteria": ["落ち着いている", "不満だが協力的", "怒っている・最終通告"]}
  }}'
```

## Config keys

| Key | Meaning |
| --- | --- |
| `source_limits` | maximum rows taken from a training source |
| `max_per_kind` | maximum rows per primitive, so that choice does not swamp noul / score |
| `split_sizes` | number of valid / calib rows cut from the training data |
| `eval_items_per_source` | items taken from each evaluation set (fixed seed) |
| `augment_ratio` | share of the training set added by Japanese augmentation (hiragana, full / half width, unrelated sentence, option permutation); training set only |
| `calibration.per_n` | fit temperatures per option-count bucket (2 / 3-5 / 6-16) in addition to per primitive |
| `calib_sources` | datasets used only to fit temperatures; must not overlap training or evaluation sources |
| `init_adapter` | directory of the LoRA to continue training from |
| `distill.alpha` | weight of the teacher probabilities in the distillation loss |

## Data rules

1. Putting an adapter with `EVAL_ONLY=True` into `train_sources` raises an error (`build.check_sources`).
2. Each evaluation set has its own `test_<source>.jsonl`; neither training nor calibration reads them.
3. Data whose license does not allow training (livedoor: CC BY-ND; WRIME: research only) is used for evaluation only.
4. Calibration sets are neither trained on nor evaluated; `check_sources` checks for overlap.
