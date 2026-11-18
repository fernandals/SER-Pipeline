"""Evaluate a trained checkpoint on a test set."""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf
from predict import ROOT, load_model
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from utils.dataloader import AudioCollate, build_dataset


@torch.inference_mode()
def evaluate(config, checkpoint, device):
    settings = config.datasets.test
    settings.metadata_path = str(ROOT / settings.metadata_path)
    settings.audio_base_dir = str(ROOT / settings.audio_base_dir)
    dataset = build_dataset(settings, config.data)
    if not len(dataset):
        raise ValueError("The test manifest is empty.")

    model = load_model(checkpoint, config, device)
    loader = DataLoader(
        dataset, batch_size=config.test.batch_size, num_workers=config.test.num_workers,
        collate_fn=AudioCollate(model.processor, config.data.target_sr),
    )
    targets, predictions = [], []
    for features, labels in loader:
        features = {key: value.to(device) for key, value in features.items()}
        predictions.extend(model(features).argmax(dim=-1).cpu().tolist())
        targets.extend(labels.tolist())

    label_names = sorted(config.data.label2id, key=config.data.label2id.get)
    ids = [config.data.label2id[name] for name in label_names]
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=ids, zero_division=0,
    )
    present = support > 0
    summary = {
        "uar": float(recall[present].mean()),
        "macro_f1": float(f1[present].mean()),
        "num_examples": len(targets),
        "checkpoint": str(checkpoint.resolve()),
        "manifest": settings.metadata_path,
    }
    per_emotion = pd.DataFrame({
        "emotion": label_names, "precision": precision, "recall": recall,
        "f1": f1, "support": support,
    })
    matrix = pd.DataFrame(confusion_matrix(targets, predictions, labels=ids),
                          index=label_names, columns=label_names)

    output = ROOT / config.test.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    per_emotion.to_csv(output / "per_emotion.csv", index=False)
    matrix.to_csv(output / "confusion_matrix.csv", index_label="true_label")
    print(f"UAR: {summary['uar']:.4f}")
    print(f"Macro-F1: {summary['macro_f1']:.4f}")
    print(f"Reports: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", type=Path, default=ROOT / "config/audio-wavlm-msp.yaml")
    parser.add_argument("-ck", "--checkpoint", type=Path, help="Defaults to last.ckpt for the YAML's run.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    checkpoint = args.checkpoint or ROOT / config.checkpoint_dir / config.title / "last.ckpt"
    if not checkpoint.is_file():
        parser.error(f"Checkpoint not found: {checkpoint}")
    evaluate(config, checkpoint, args.device)


if __name__ == "__main__":
    main()
