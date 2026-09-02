#!/bin/bash
# Run from the src/ directory: bash recipes/predict.sh path/to/audio.wav
set -e

CONFIG_PATH="../config/audio-wavlm.yaml"
CHECKPOINT_PATH="../checkpoints/last.ckpt"
AUDIO_PATH="${1:?Usage: bash recipes/predict.sh <audio_path>}"

python predict.py -c "$CONFIG_PATH" -ck "$CHECKPOINT_PATH" -a "$AUDIO_PATH" --probs
