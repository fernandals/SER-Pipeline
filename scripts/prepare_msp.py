"""Prepare fixed MSP-Podcast manifests from the supplied composite CSV."""

import argparse
import os
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", type=Path, default=ROOT / "config/audio-wavlm-msp.yaml")
    parser.add_argument("--test", action="store_true", help="Prepare only the official Test1 manifest.")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    metadata = ROOT / config.msp.metadata

    columns = ["FileName", "EmoClass", "Split_Set"]
    df = pd.read_csv(metadata, usecols=columns)
    if args.test:
        splits = {ROOT / config.datasets.test.metadata_path: "Test1"}
        df = df[df.Split_Set == "Test1"]
        excluded = df.FileName.isin(config.test.get("exclude_files", []))
        if excluded.any():
            print(f"Excluded from Test1: {', '.join(df.loc[excluded, 'FileName'])}")
        df = df.loc[~excluded]
    else:
        splits = {
            ROOT / config.datasets.train.metadata_path: "Train",
            ROOT / config.datasets.val.metadata_path: "Development",
        }
        df = df[df.Split_Set.isin(splits.values())]
    if df.FileName.isna().any() or df.FileName.duplicated().any():
        raise ValueError("Selected splits contain missing or duplicate filenames.")

    audio_dir = ROOT / (config.datasets.test.audio_base_dir if args.test else config.msp.audio_dir)
    with os.scandir(audio_dir) as entries:
        available = {entry.name for entry in entries if entry.is_file()}
    manifests = {}
    for path, name in splits.items():
        split = df.loc[df.Split_Set == name, ["FileName", "EmoClass"]]
        if split.empty:
            raise ValueError(f"{name} split is empty. Check Split_Set in {metadata}.")
        missing = sorted(set(split.FileName) - available)
        if missing:
            raise FileNotFoundError(f"{name}: {len(missing)} missing audio files in {audio_dir}: {missing[:5]}")

        manifests[path] = split

    for path, split in manifests.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        split.to_csv(path, index=False)
        print(f"{path.name}: {len(split)} clips")


if __name__ == "__main__":
    main()
