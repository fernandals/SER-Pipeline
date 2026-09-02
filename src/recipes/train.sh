#!/bin/bash
# Run from the src/ directory: bash recipes/train.sh
set -e

GPU=0
CONFIG_PATH="../config/audio-wavlm.yaml"

python main.py -c "$CONFIG_PATH" -g "$GPU"
