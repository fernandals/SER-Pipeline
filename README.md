# Speech Emotion Recognition with WavLM on MSP-Podcast

Training uses 16 kHz mono audio and keeps WavLM Base+ frozen. It learns how to combine the encoder layers and updates the MLP. Masked mean pooling over time feeds the combined encoder output into the MLP.

From the repository root, with the dependencies in `requirements.txt` installed in the virtual environment, run:

```bash
.venv/bin/python scripts/prepare_msp.py
.venv/bin/python src/main.py
```

The first command builds the training and validation manifests from `data/msp_podcast/`. The second saves checkpoints to `checkpoints/<title>/` and logs metrics to `lightning_logs/<title>/`. Checkpoint selection and early stopping both use validation macro-F1.

Edit [config/audio-wavlm-msp.yaml](config/audio-wavlm-msp.yaml) to change paths, GPU settings, model settings, and hyperparameters. To use another YAML file, pass `-c path/to/config.yaml` to the commands. Relative paths are resolved from the repository root.

## Inference

After training, run inference on a single audio file:

```bash
.venv/bin/python src/predict.py -a /path/to/audio.wav --probs
```

By default, the command uses the `last.ckpt` specified in the YAML file. To use a different checkpoint, such as the one selected by validation macro-F1, pass `-ck path/to/model.ckpt`. If you trained with a different YAML file, pass that same file with `-c`.

## Testing on MSP-Podcast

Generate the Test1 manifest and evaluate a checkpoint:

```bash
.venv/bin/python scripts/prepare_msp.py --test
.venv/bin/python src/test.py
```

Evaluation uses `last.ckpt` by default. Pass `-ck` to evaluate the checkpoint selected during validation and `-c` if you trained with a different YAML file. Set `test.output_dir` to save evaluation results to different directories.

- `metrics.json` reports unweighted average recall (UAR) and macro-F1. Both averages are computed over the classes present in the manifest.
- `per_emotion.csv` reports F1, precision, recall, and the number of examples for each emotion.
- `confusion_matrix.csv` reports counts with ground-truth classes in rows and predicted classes in columns.

The YAML file excludes `MSP-PODCAST_1909_1017.wav` from Test1 because the WAV file is missing from the local dataset copy. If you recover the file, remove it from `test.exclude_files` and regenerate the manifest.
