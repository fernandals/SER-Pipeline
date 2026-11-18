import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
from models.pl_wrapper import PLWrapper
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Train WavLM on MSP-Podcast.")
    parser.add_argument("-c", "--config", type=Path, default=ROOT / "config/audio-wavlm-msp.yaml")
    config = OmegaConf.load(parser.parse_args().config)

    for dataset in config.datasets.values():
        dataset.metadata_path = str(ROOT / dataset.metadata_path)
        dataset.audio_base_dir = str(ROOT / dataset.audio_base_dir)

    # Set CUBLAS_WORKSPACE_CONFIG for deterministic/reproducible CUDA results
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    pl.seed_everything(config.seed, workers=True)
    checkpoint_dir = ROOT / config.checkpoint_dir / config.title
    callbacks = [
        ModelCheckpoint(dirpath=checkpoint_dir, **config.model_checkpoint),
        LearningRateMonitor(logging_interval="step"),
    ]
    if config.get("early_stopping"):
        callbacks.append(EarlyStopping(**config.early_stopping))

    logger = CSVLogger(ROOT / config.log_dir, name=config.title)
    trainer = pl.Trainer(callbacks=callbacks, logger=logger, **config.trainer)
    trainer.fit(PLWrapper(config))


if __name__ == "__main__":
    main()
