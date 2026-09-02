import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import re
import glob
import argparse

import torch
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor
from sklearn.metrics import accuracy_score, f1_score, classification_report

from models.pl_wrapper import PLWrapper
from utils.dataloader import AudioDataset, AudioCollate, read_metadata
from utils.utils import id2label_from_config


def resolve_checkpoint(path: str) -> str:
    """Accept a .ckpt file or a directory.

    For a directory, pick the ``epoch=NN-val_f1=X.ckpt`` with the highest
    ``val_f1``, falling back to ``last.ckpt``.
    """
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        scored = []
        for ckpt in glob.glob(os.path.join(path, "*.ckpt")):
            match = re.search(r"val_f1=([0-9]*\.?[0-9]+)", os.path.basename(ckpt))
            if match:
                scored.append((float(match.group(1)), ckpt))
        if scored:
            best = max(scored)[1]
            print(f"Selected best checkpoint: {best}")
            return best
        fallback = os.path.join(path, "last.ckpt")
        if os.path.isfile(fallback):
            print(f"No scored checkpoints; using {fallback}")
            return fallback
    raise FileNotFoundError(f"No checkpoint found at {path}")


@torch.inference_mode()
def run_inference(
    config_path: str,
    checkpoint_path: str,
    metadata_path: str,
    output_path: str,
    gpu: int = 0,
    batch_size: int = 8,
    num_workers: int = 8,
) -> pd.DataFrame:
    """Run a trained model over a metadata CSV and write per-file predictions.

    If the CSV has a valid target column, macro-F1 / accuracy are printed too.
    """
    config = OmegaConf.load(config_path)
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    id2label = id2label_from_config(config)
    label2id = dict(config.data.label2id)

    checkpoint_path = resolve_checkpoint(checkpoint_path)

    # strict=False: the CE-loss `weight` buffer is training-only and frozen
    # backbone weights are stripped on save (reloaded from HF in __init__).
    model = PLWrapper.load_from_checkpoint(
        checkpoint_path, config=config, inference_mode=True,
        map_location=device, strict=False,
    )
    model.eval().to(device)

    df = pd.read_csv(metadata_path)
    target_column = config.datasets.val.target_column
    filename_column = config.datasets.val.filename_column
    has_targets = target_column in df.columns and df[target_column].isin(label2id).all()

    if has_targets:
        df = read_metadata(metadata_path, target_column, label2id)
    else:
        df["target"] = -1

    dataset = AudioDataset(
        data=df,
        filename_column=filename_column,
        target_column="target",
        audio_base_dir=config.datasets.val.audio_base_dir,
        target_sr=config.data.target_sr,
        max_duration=config.data.get("max_duration", None),
    )
    processor = AutoFeatureExtractor.from_pretrained(config.model.model_name)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=AudioCollate(processor=processor, target_sr=config.data.target_sr),
    )

    all_preds = []
    for features, _ in tqdm(loader, desc="inference"):
        features = {k: v.to(device) for k, v in features.items()}
        logits = model(features)
        all_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())

    out = pd.DataFrame({
        filename_column: df[filename_column].values,
        "pred": all_preds,
        "EmoClass": [id2label[p] for p in all_preds],
    })
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Wrote {len(out)} predictions to {output_path}")

    if has_targets:
        y_true = df["target"].values
        print(f"\nAccuracy : {accuracy_score(y_true, all_preds):.4f}")
        print(f"Macro-F1 : {f1_score(y_true, all_preds, average='macro', zero_division=0):.4f}")
        print(classification_report(
            y_true, all_preds,
            labels=list(range(config.data.num_classes)),
            target_names=[id2label[i] for i in range(config.data.num_classes)],
            zero_division=0,
        ))

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", required=True, type=str)
    parser.add_argument("-ck", "--checkpoint_path", required=True, type=str)
    parser.add_argument("-t", "--metadata_path", required=True, type=str)
    parser.add_argument("-o", "--output_path", default="results/predictions.csv", type=str)
    parser.add_argument("-g", "--gpu", default=0, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--num-workers", default=8, type=int)
    args = parser.parse_args()

    run_inference(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        metadata_path=args.metadata_path,
        output_path=args.output_path,
        gpu=args.gpu,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
