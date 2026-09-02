# SER-Audio

A minimal **speech emotion recognition** pipeline: one audio file in, one of 8
emotions out (Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness,
Surprise).

It is a stripped-down sibling of
[InterSpeech-SER-2025](../InterSpeech-SER-2025) — same project structure and
conventions (OmegaConf configs, a PyTorch-Lightning `PLWrapper`, a
`create_ser_model` factory, `Dataset`/`Collate` pairs), but **audio-only**: no
ASR/text branch, no F0 branch, no graph fusion, no ensembling.

## Architecture

```
waveform (16 kHz mono)
   │
   ▼
SSL backbone  (microsoft/wavlm-large, frozen by default)
   │  hidden states of every layer
   ▼
layer aggregation   "last"  |  learned "weighted_sum" (softmax over layers)
   │
   ▼
temporal pooling    masked "mean"  |  "attpool" (attentive statistics)
   │
   ▼
MLP head  →  8 emotion logits
```

Everything is configurable from [config/audio-wavlm.yaml](config/audio-wavlm.yaml).
Only the aggregation weights, pooling and MLP are trained when
`model.freeze_backbone: true` — so training is fast and fits on a small GPU.

## Install

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

## Data

See [data/README.md](data/README.md). You need a `train.csv` / `dev.csv` with
`FileName,EmoClass` columns and a folder of `.wav` files.

### RAVDESS (via EmoBox partitioning)

[`scripts/prepare_ravdess.py`](scripts/prepare_ravdess.py) turns EmoBox's
speaker-independent RAVDESS folds into the CSVs this repo expects. Clone
[EmoBox](https://github.com/emo-box/EmoBox) next to this repo and unpack the
RAVDESS audio (the `Actor_01/ … Actor_24/` folders), then:

```bash
python scripts/prepare_ravdess.py --emobox-dir ../EmoBox --audio-dir ../RAVDESS --fold 1
# writes data/ravdess/{train,dev,test}.csv  (test = EmoBox fold-1 held-out actors,
# dev = 2 more actors held out of the train split -> still speaker-independent)
```

Then use the recipe scripts (run from the repo root):

```bash
bash src/recipes/ravdess_train.sh 1 0   # prepare fold 1 + TRAIN on gpu 0
bash src/recipes/ravdess_eval.sh        # INFERENCE ONLY on the newest checkpoint
bash src/recipes/ravdess.sh   1 0       # both, end to end
```

`ravdess_eval.sh` never trains. Its first argument is a `.ckpt` file **or** a
directory (best `val_f1` checkpoint inside is auto-selected); with no argument it
uses the newest `checkpoints/` directory:

```bash
bash src/recipes/ravdess_eval.sh "checkpoints/<title>"          # best in dir
bash src/recipes/ravdess_eval.sh path/to/epoch=09-val_f1=0.64.ckpt
```

Use fold `2..6` for full 6-fold cross-validation (each call regenerates the CSVs).

## Train

```bash
cd src
python main.py -c ../config/audio-wavlm.yaml -g 0
```

Checkpoints go to `<repo>/checkpoints/<title>/`, best model selected on
`val_f1` (macro-F1); frozen-backbone checkpoints are ~3 MB (the backbone is
stripped on save and reloaded from HF on load). Metrics log to `lightning_logs/`
(CSV) by default; set `use_wandb: true` in the config for Weights & Biases.

Training stops early when `val_f1` plateaus (`early_stopping:` block in the
config, `patience: 5` validations). Comment the block out to always run the full
`max_epochs`.

## Predict a single file

```bash
cd src
python predict.py -c ../config/audio-wavlm.yaml \
    -ck ../checkpoints/<title>/last.ckpt \
    -a /path/to/audio.wav --probs
```

```python
from predict import predict
label, probs = predict("audio.wav", "../config/audio-wavlm.yaml",
                       "../checkpoints/.../last.ckpt", return_probs=True)
```

## Batch inference / evaluation

```bash
cd src
python eval/inference.py -c ../config/audio-wavlm.yaml \
    -ck ../checkpoints/<title>/last.ckpt \
    -t ../data/dev.csv -o ../results/dev_predictions.csv
```

Prints accuracy, macro-F1 and a per-class report when the CSV has labels.

## Layout

```
config/         experiment YAMLs
scripts/        dataset preparation (prepare_ravdess.py)
src/
├── main.py            training entry point
├── predict.py         single-file prediction
├── models/
│   ├── factory.py     model_type -> class
│   ├── modules.py     AudioSERModel, pooling, MLP head
│   └── pl_wrapper.py  LightningModule (data, loss, optim, metrics)
├── utils/
│   ├── dataloader.py  AudioDataset + AudioCollate + build_datasets
│   ├── schedulers.py  CosineWarmupLR
│   └── utils.py       class weights, id2label
├── eval/inference.py  batch inference + metrics
└── recipes/           train.sh / predict.sh
```

## Extending to multimodal

`create_ser_model` in [src/models/factory.py](src/models/factory.py) is the seam:
add a new `model_type` and model class in `modules.py`, a matching
`Dataset`/`Collate` in `dataloader.py`, and branch on `model_type` in
`PLWrapper`. That is the pattern InterSpeech-SER-2025 uses for its ~25 variants.
