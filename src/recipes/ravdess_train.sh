#!/bin/bash
# Prepare the RAVDESS CSVs for a fold and TRAIN a model on them.
# Run from the repo root:  bash src/recipes/ravdess_train.sh [fold] [gpu]
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PY="$REPO_ROOT/.venv/bin/python"; [ -x "$PY" ] || PY=python

FOLD="${1:-1}"
GPU="${2:-0}"
CONFIG="$REPO_ROOT/config/audio-wavlm-ravdess.yaml"

"$PY" scripts/prepare_ravdess.py --emobox-dir ../EmoBox --audio-dir ../RAVDESS --fold "$FOLD"

"$PY" src/main.py -c "$CONFIG" -g "$GPU"

echo
echo "Training done. Evaluate with:  bash src/recipes/ravdess_eval.sh"
