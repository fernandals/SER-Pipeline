# Data layout

Put your audio and metadata here (both are git-ignored).

```
data/
├── audio/            # all .wav files (any subfolder structure is fine)
├── train.csv
└── dev.csv
```

## Metadata CSV format

One row per utterance. Column names are configurable in `config/audio-wavlm.yaml`
(`filename_column`, `target_column`); the defaults are:

| Column     | Meaning                                                        |
|------------|---------------------------------------------------------------|
| `FileName` | path to the audio file, **relative to `audio_base_dir`**      |
| `EmoClass` | emotion label, one of the keys in `data.label2id` (A/C/D/F/H/N/S/U) |

Example:

```csv
FileName,EmoClass
spk001_0007.wav,H
spk001_0012.wav,N
spk042_0103.wav,A
```

For unlabeled inference sets, `EmoClass` may be missing or contain placeholder
values — `src/eval/inference.py` will still produce predictions and just skip
the metrics.
