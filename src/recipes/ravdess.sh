#!/bin/bash
# FULL end-to-end RAVDESS run: prepare CSVs -> TRAIN -> evaluate on test actors.
# This WILL train a model (takes a while). For inference only on an existing
# checkpoint, use  src/recipes/ravdess_eval.sh  instead.
#
# Run from the repo root:  bash src/recipes/ravdess.sh [fold] [gpu]
set -e

FOLD="${1:-1}"
GPU="${2:-0}"

bash "$(dirname "${BASH_SOURCE[0]}")/ravdess_train.sh" "$FOLD" "$GPU"
bash "$(dirname "${BASH_SOURCE[0]}")/ravdess_eval.sh" "" "$FOLD" "$GPU"
