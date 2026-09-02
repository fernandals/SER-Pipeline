import os
import sys
import argparse
from pprint import pprint

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from models.pl_wrapper import PLWrapper

# Repo root (parent of src/) so outputs land there regardless of the cwd.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_logger(config):
    """CSV logger by default; W&B when ``use_wandb`` is set in the config."""
    if config.get("use_wandb", False):
        return WandbLogger(
            project=config.get("wandb_project", "SER-Audio"),
            entity=config.get("wandb_entity", None),
            name=config.title,
            tags=list(config.get("tags", [])),
            config=OmegaConf.to_container(config, resolve=True),
        )
    return CSVLogger(save_dir=os.path.join(REPO_ROOT, "lightning_logs"), name=config.title)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", required=True, type=str,
                        help="YAML file with configurations")
    parser.add_argument("-g", "--gpu", default=0, type=int,
                        help="GPU index to train on")
    parser.add_argument("-ck", "--checkpoint_dir", default=None, type=str,
                        help="where to save checkpoints (default: <repo>/checkpoints)")
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    seed_everything(config.get("seed", 42), workers=True)

    logger = build_logger(config)

    checkpoint_dir = args.checkpoint_dir or os.path.join(REPO_ROOT, "checkpoints")
    ckpt_kwargs = OmegaConf.to_container(config.model_checkpoint, resolve=True)
    ckpt_kwargs.pop("dirpath", None)
    ckpt_kwargs["dirpath"] = os.path.join(checkpoint_dir, config.title)
    print(f"Checkpoints -> {ckpt_kwargs['dirpath']}")
    callbacks = [
        ModelCheckpoint(**ckpt_kwargs),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Optional early stopping: add an `early_stopping:` block to the config.
    if config.get("early_stopping", None):
        es_kwargs = OmegaConf.to_container(config.early_stopping, resolve=True)
        callbacks.append(EarlyStopping(**es_kwargs))
        print(f"Early stopping -> monitor={es_kwargs.get('monitor')} "
              f"patience={es_kwargs.get('patience')}")

    model = PLWrapper(config)
    pprint(model)

    trainer = pl.Trainer(
        **config.trainer,
        logger=logger,
        callbacks=callbacks,
        devices=[args.gpu],
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
