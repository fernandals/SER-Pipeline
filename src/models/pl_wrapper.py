import numpy as np
import pytorch_lightning as pl
import torch
from models.factory import create_ser_model
from omegaconf import DictConfig
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor
from utils.dataloader import AudioCollate, build_datasets
from utils.schedulers import CosineWarmupLR
from utils.utils import get_classes_weights


class PLWrapper(pl.LightningModule):
    """LightningModule tying together the model, data, loss and metrics."""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        self.model = create_ser_model(
            model_type=config.model.model_type,
            model_name=config.model.model_name,
            num_classes=config.data.num_classes,
            freeze_backbone=config.model.freeze_backbone,
            layer_aggregation=config.model.layer_aggregation,
            pooling_strategy=config.model.pooling_strategy,
            mlp_hidden_dim=config.model.mlp_hidden_dim,
            mlp_num_layers=config.model.mlp_num_layers,
            mlp_dropout=config.model.mlp_dropout,
            mlp_activation_func=config.model.mlp_activation_func,
        )

        self.processor = AutoFeatureExtractor.from_pretrained(config.model.model_name)

        self.criterion = torch.nn.CrossEntropyLoss()

        self._train_preds, self._train_targets = [], []
        self._val_preds, self._val_targets = [], []

    # ------------------------------------------------------------------ setup
    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset, self.val_dataset = build_datasets(self.config)

            weights = None
            if self.config.loss.get("use_weighted_loss", False):
                weights = get_classes_weights(self.config).to(self.device)
                print(f"Using weighted CrossEntropyLoss | weights: {weights.tolist()}")

            self.criterion = torch.nn.CrossEntropyLoss(
                weight=weights,
                label_smoothing=self.config.loss.get("label_smoothing", 0.0),
            )

    # ------------------------------------------------------------ dataloaders
    def _collate_fn(self) -> AudioCollate:
        return AudioCollate(processor=self.processor, target_sr=self.config.data.target_sr)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=self.config.train.shuffle,
            num_workers=self.config.train.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=False,
            num_workers=self.config.train.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn(),
        )

    # -------------------------------------------------------------- optimizers
    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        opt_params = self.config.optimizer.params

        if self.config.optimizer.name.lower() == "adam":
            optimizer = Adam(
                params,
                lr=opt_params.learning_rate,
                eps=opt_params.eps,
                betas=tuple(opt_params.betas),
                weight_decay=opt_params.weight_decay,
            )
        elif self.config.optimizer.name.lower() == "adamw":
            optimizer = AdamW(
                params,
                lr=opt_params.learning_rate,
                eps=opt_params.eps,
                betas=tuple(opt_params.betas),
                weight_decay=opt_params.weight_decay,
            )
        else:
            raise ValueError(f"Invalid optimizer: {self.config.optimizer.name}")

        if not self.config.get("scheduler", None):
            return optimizer

        # estimated_stepping_batches already accounts for devices, max_epochs,
        # accumulate_grad_batches and limit_train_batches.
        total_steps = max(int(self.trainer.estimated_stepping_batches), 1)
        warmup = int(self.config.scheduler.params.get("warmup_ratio", 0.05) * total_steps)

        if self.config.scheduler.name.lower() == "cosinewarmuplr":
            scheduler = CosineWarmupLR(
                optimizer,
                lr_min=opt_params.get("min_learning_rate", 1e-6),
                lr_max=opt_params.learning_rate,
                warmup=warmup,
                T_max=total_steps,
            )
        else:
            raise ValueError(f"Invalid scheduler: {self.config.scheduler.name}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    # --------------------------------------------------------------- fwd/steps
    def forward(self, features: dict) -> torch.Tensor:
        return self.model(**features)

    def _shared_step(self, batch, preds_buf, targets_buf, split: str):
        features, targets = batch
        logits = self.forward(features)
        loss = self.criterion(logits, targets)

        preds = torch.argmax(logits, dim=1)
        preds_buf.append(preds.detach().cpu().numpy())
        targets_buf.append(targets.detach().cpu().numpy())

        self.log(f"{split}/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, self._train_preds, self._train_targets, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, self._val_preds, self._val_targets, "val")

    # ------------------------------------------------------------- epoch hooks
    def _log_epoch_metrics(self, preds_buf, targets_buf, split: str):
        if not preds_buf:
            return
        preds = np.concatenate(preds_buf)
        targets = np.concatenate(targets_buf)

        accuracy = accuracy_score(targets, preds)
        macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)

        self.log(f"{split}/accuracy", accuracy, prog_bar=True)
        self.log(
            f"{split}/precision",
            precision_score(targets, preds, average="macro", zero_division=0),
        )
        self.log(
            f"{split}/recall",
            recall_score(targets, preds, average="macro", zero_division=0),
        )
        self.log(f"{split}/f1-score", macro_f1, prog_bar=True)

        # Slash-free aliases for ModelCheckpoint monitor / filename templates
        # ("val/f1-score" in a filename would create a "val/" sub-directory).
        self.log(f"{split}_acc", accuracy)
        self.log(f"{split}_f1", macro_f1)

        if split == "val":
            print(f"\n[{split}] confusion matrix (rows=true, cols=pred):")
            print(confusion_matrix(targets, preds))

        preds_buf.clear()
        targets_buf.clear()

    def on_train_epoch_end(self):
        self._log_epoch_metrics(self._train_preds, self._train_targets, "train")

    def on_validation_epoch_end(self):
        self._log_epoch_metrics(self._val_preds, self._val_targets, "val")

    # ------------------------------------------------------------- checkpoint io
    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Drop the frozen backbone from the checkpoint.

        When ``model.freeze_backbone`` is set, the backbone weights never change
        and ``__init__`` reloads them from HuggingFace anyway — so storing them
        just bloats every checkpoint by ~350 MB. Only the trainable head /
        layer-aggregation weights are kept. Loaders must use ``strict=False``.
        """
        if getattr(self.model, "freeze_backbone", False):
            state_dict = checkpoint["state_dict"]
            for key in list(state_dict):
                if key.startswith("model.backbone."):
                    del state_dict[key]
