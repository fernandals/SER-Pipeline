# MSP-Podcast Data

```
data/
├── msp_podcast/
│   ├── Audios/
│   └── msp_podcast_complete.csv
└── msp_splits/
    ├── train.csv
    └── dev.csv
```

`scripts/prepare_msp.py` reads the official `Train` and `Development` splits and writes manifests with the columns `FileName,EmoClass`. File names are relative to `Audios/`. Configure the paths in [audio-wavlm-msp.yaml](../config/audio-wavlm-msp.yaml).
