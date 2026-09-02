"""Prepare the RAVDESS dataset for SER-Pipeline using EmoBox's partitioning.

EmoBox (https://github.com/emo-box/EmoBox) already ships speaker-independent
k-fold metadata for RAVDESS under ``EmoBox/data/ravdess/`` (6 folds, 4 held-out
actors per test fold) plus a ``label_map.json`` that normalises the raw RAVDESS
emotion names.

This script is the bridge between that metadata and the CSV format
``src/utils/dataloader.py`` expects:

    FileName,EmoClass
    Actor_18/03-01-05-01-02-01-18.wav,Angry
    ...

For a chosen fold it writes ``train.csv`` / ``dev.csv`` / ``test.csv`` where:
  * ``test``  = EmoBox's official test split for that fold (unseen actors)
  * ``dev``   = a few actors held out of EmoBox's train split (still speaker-independent)
  * ``train`` = the remaining actors of EmoBox's train split

It also prints the ``label2id`` block to drop into the config.

Example:
    python scripts/prepare_ravdess.py \
        --emobox-dir ../EmoBox \
        --audio-dir  ../RAVDESS \
        --fold 1
"""

import os
import csv
import json
import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


def build_audio_index(audio_dir: Path) -> Dict[str, Path]:
    """Map every ``*.wav`` basename under ``audio_dir`` to its path.

    RAVDESS archives often contain the files twice (a flat ``Actor_XX/`` tree and
    a nested ``audio_speech_actors_01-24/Actor_XX/`` copy); we prefer the shallow
    path so the ``FileName`` column stays short.
    """
    index: Dict[str, Path] = {}
    for path in audio_dir.rglob("*.wav"):
        name = path.name
        if name not in index or len(path.parts) < len(index[name].parts):
            index[name] = path
    return index


def parse_actor(key: str) -> str:
    """``ravdess-03-01-05-01-02-01-18`` -> ``18`` (the actor id)."""
    return key.split("-")[-1]


def load_fold(meta_dir: Path, fold: int, split: str) -> List[dict]:
    """Load one EmoBox fold jsonl (``split`` is ``train`` or ``test``)."""
    path = meta_dir / f"fold_{fold}" / f"ravdess_{split}_fold_{fold}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"EmoBox metadata not found: {path}")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def to_rows(
    entries: List[dict],
    label_map: Dict[str, str],
    audio_index: Dict[str, Path],
    audio_dir: Path,
) -> List[dict]:
    """Turn EmoBox jsonl entries into CSV rows, resolving real audio paths."""
    rows, missing = [], []
    for entry in entries:
        key = entry["key"]
        wav_name = os.path.basename(entry["wav"])
        emo_class = label_map[entry["emo"]]

        resolved = audio_index.get(wav_name)
        if resolved is None:
            missing.append(wav_name)
            continue

        rows.append(
            {
                "FileName": os.path.relpath(resolved, audio_dir),
                "EmoClass": emo_class,
                "key": key,
                "actor": parse_actor(key),
                "raw_emotion": entry["emo"],
            }
        )

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} audio files from EmoBox metadata were not found under "
            f"{audio_dir} (e.g. {missing[:3]}). Check --audio-dir."
        )
    return rows


def split_train_dev(
    train_rows: List[dict], dev_actors: int, seed: int
) -> Tuple[List[dict], List[dict], List[str]]:
    """Hold out ``dev_actors`` whole actors from the train split for validation."""
    actors = sorted({row["actor"] for row in train_rows})
    rng = random.Random(seed)
    rng.shuffle(actors)
    dev_set = set(actors[:dev_actors])

    train = [r for r in train_rows if r["actor"] not in dev_set]
    dev = [r for r in train_rows if r["actor"] in dev_set]
    return train, dev, sorted(dev_set)


def write_csv(rows: List[dict], path: Path) -> None:
    fields = ["FileName", "EmoClass", "key", "actor", "raw_emotion"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarise(name: str, rows: List[dict]) -> None:
    dist = Counter(r["EmoClass"] for r in rows)
    actors = sorted({r["actor"] for r in rows})
    print(f"  {name:5s}: {len(rows):4d} utts | {len(actors):2d} actors | {dict(sorted(dist.items()))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--emobox-dir", default="../EmoBox", type=str,
                        help="Path to the cloned EmoBox repository")
    parser.add_argument("--audio-dir", default="../RAVDESS", type=str,
                        help="Directory that contains the RAVDESS Actor_XX/ folders")
    parser.add_argument("--fold", default=1, type=int, help="EmoBox fold (1-6)")
    parser.add_argument("--dev-actors", default=2, type=int,
                        help="How many train-split actors to hold out as the dev set")
    parser.add_argument("--output-dir", default="data/ravdess", type=str)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    emobox_dir = Path(args.emobox_dir).resolve()
    audio_dir = Path(args.audio_dir).resolve()
    meta_dir = emobox_dir / "data" / "ravdess"
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    label_map = json.loads((meta_dir / "label_map.json").read_text())
    label2id = {lab: i for i, lab in enumerate(sorted(set(label_map.values())))}

    print(f"EmoBox metadata : {meta_dir}")
    print(f"Audio directory : {audio_dir}")
    print(f"Fold            : {args.fold}\n")

    audio_index = build_audio_index(audio_dir)
    print(f"Indexed {len(audio_index)} unique .wav files\n")

    train_full = to_rows(load_fold(meta_dir, args.fold, "train"), label_map, audio_index, audio_dir)
    test_rows = to_rows(load_fold(meta_dir, args.fold, "test"), label_map, audio_index, audio_dir)
    train_rows, dev_rows, dev_actor_ids = split_train_dev(train_full, args.dev_actors, args.seed)

    write_csv(train_rows, output_dir / "train.csv")
    write_csv(dev_rows, output_dir / "dev.csv")
    write_csv(test_rows, output_dir / "test.csv")

    print("Splits (speaker-independent):")
    summarise("train", train_rows)
    summarise("dev", dev_rows)
    summarise("test", test_rows)
    print(f"\n  dev held-out actors: {dev_actor_ids}")
    print(f"  CSVs written to     : {output_dir}\n")

    print("label2id for the config:")
    print("    label2id:")
    for lab, idx in label2id.items():
        print(f"        {lab}: {idx}")


if __name__ == "__main__":
    main()
