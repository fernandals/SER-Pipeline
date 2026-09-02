#!/bin/bash
# INFERENCE ONLY - run an already-trained checkpoint on the RAVDESS test split.
# Does NOT train. Run from the repo root:
#   bash src/recipes/ravdess_eval.sh [checkpoint-or-dir] [fold] [gpu]
#
# checkpoint-or-dir: a .ckpt file, or a directory (the best val_f1 ckpt in it is
#                    picked automatically). Default: the newest checkpoints/ dir.
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PY="$REPO_ROOT/.venv/bin/python"; [ -x "$PY" ] || PY=python

CONFIG="$REPO_ROOT/config/audio-wavlm-ravdess.yaml"
FOLD="${2:-1}"
GPU="${3:-0}"

if [ -n "$1" ]; then
    CKPT="$(realpath "$1")"
else
    CKPT="$REPO_ROOT/$(ls -td checkpoints/*/ | head -1)"   # newest run directory
fi
echo "Checkpoint: $CKPT"

"$PY" src/eval/inference.py -c "$CONFIG" -ck "$CKPT" \
    -t "$REPO_ROOT/data/ravdess/test.csv" \
    -o "$REPO_ROOT/results/ravdess_fold${FOLD}_test.csv" \
    -g "$GPU"
