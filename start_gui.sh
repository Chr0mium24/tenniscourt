#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CHECKPOINT="${CHECKPOINT:-runs/v6/best.pt}"
REMOTE_CHECKPOINT="${REMOTE_CHECKPOINT:-anilam@10.31.151.120:/home/anilam/Codes/tenniscourt/runs/heatmap-10000-v6-low-lr-hard/best.pt}"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "checkpoint not found: $CHECKPOINT"
  echo "fetching from: $REMOTE_CHECKPOINT"
  mkdir -p "$(dirname "$CHECKPOINT")"
  rsync -av "$REMOTE_CHECKPOINT" "$CHECKPOINT"
fi

cmd=(
  uv run --extra viewer --extra train tenniscourt-viewer
  --width "${WIDTH:-640}"
  --height "${HEIGHT:-360}"
  --supersample "${SUPERSAMPLE:-1}"
  --checkpoint "$CHECKPOINT"
  --selection-score "${SELECTION_SCORE:-combined}"
  --score-threshold "${SCORE_THRESHOLD:-0.5}"
  --short-labels
)

if [[ "${SHOW_ALL:-0}" == "1" ]]; then
  cmd+=(--show-all-keypoints)
fi

if [[ "${RELOAD_CHECKPOINT:-0}" == "1" ]]; then
  cmd+=(--reload-checkpoint)
fi

if [[ "${HEADLESS:-0}" == "1" ]]; then
  cmd+=(--headless --max-frames "${MAX_FRAMES:-1}")
  if [[ -n "${SAVE_FRAME:-}" ]]; then
    cmd+=(--save-frame "$SAVE_FRAME")
  fi
fi

echo "starting GUI with checkpoint: $CHECKPOINT"
exec "${cmd[@]}" "$@"
