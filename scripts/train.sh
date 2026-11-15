#!/usr/bin/env bash
# One command for train -> calibrate -> evaluate. Every run gets its own directory runs/<run_name>_<timestamp>/, so runs never overwrite each other.
#   bash scripts/train.sh --config configs/qwen3-0.6b-ja.json
#   bash scripts/train.sh --config configs/qwen3-0.6b-ja.json --smoke    # smoke test: train 200 steps only
#   bash scripts/train.sh --config configs/qwen3-0.6b-ja.json --resume runs/<dir>   # resume from the last checkpoint in that directory
# Run source .venv/bin/activate first (or prefix with uv run). The same script is used on Mac and WSL2.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG=""
SMOKE=0
RESUME=""
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --smoke)  SMOKE=1; shift ;;
    --resume) RESUME="${2%/}"; shift 2 ;;
    -h|--help) sed -n '2,5p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

if [[ -z "$CONFIG" || ! -f "$CONFIG" ]]; then
  echo "error: --config <file> is required (see configs/)" >&2
  exit 2
fi

NAME="$(python -c "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['run_name'])" "$CONFIG")"
RUN_DIR="${RESUME:-runs/${NAME}_$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$RUN_DIR"

ARGS=(run --config "$CONFIG" --work-dir "$RUN_DIR")
[[ $SMOKE -eq 1 ]] && ARGS+=(--max-steps 200)
if [[ ${#EXTRA[@]} -gt 0 ]]; then
  ARGS+=("${EXTRA[@]}")
fi

export PYTHONUNBUFFERED=1
echo "run dir : $RUN_DIR"
echo "config  : $CONFIG"
python -m japanese_jev "${ARGS[@]}" 2>&1 | tee -a "$RUN_DIR/train.log"
echo "run complete -> $RUN_DIR"
