import os
from typing import List, Optional, Tuple

import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset


def load_audio(path: str, target_sr: int, max_duration: Optional[float] = None) -> torch.Tensor:
    """Load an audio file as a mono waveform at ``target_sr``.

    Params:
        path (str): Path to the audio file.
        target_sr (int): Sampling rate the waveform is resampled to.
        max_duration (Optional[float]): If set, the waveform is center-truncated
            to at most this many seconds.

    Returns:
        torch.Tensor: 1-D waveform of shape [num_samples].
    """
    waveform, sr = torchaudio.load(path)

    # Downmix to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    waveform = waveform.squeeze(0)

    if max_duration is not None:
        max_samples = int(max_duration * target_sr)
        if waveform.shape[0] > max_samples:
            start = (waveform.shape[0] - max_samples) // 2
            waveform = waveform[start:start + max_samples]

    return waveform


class AudioDataset(Dataset):
    """Reads a metadata CSV and yields (waveform, target) pairs."""

    def __init__(
        self,
        data: pd.DataFrame,
        filename_column: str,
        target_column: str,
        audio_base_dir: str,
        target_sr: int = 16000,
        max_duration: Optional[float] = None,
    ):
        """Initialization.

        Params:
            data (pd.DataFrame): Metadata with a filename column and an integer
                ``target_column`` (already mapped through ``label2id``).
            filename_column (str): Column holding the audio file name.
            target_column (str): Column holding the integer class id.
            audio_base_dir (str): Directory the file names are relative to.
            target_sr (int): Target sampling rate.
            max_duration (Optional[float]): Max clip length in seconds.
        """
        self.data = data.reset_index(drop=True)
        self.filenames = self.data[filename_column].values
        self.targets = self.data[target_column].values

        self.audio_base_dir = audio_base_dir
        self.target_sr = target_sr
        self.max_duration = max_duration

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        filepath = os.path.join(self.audio_base_dir, str(self.filenames[index]))
        waveform = load_audio(filepath, self.target_sr, self.max_duration)
        target = int(self.targets[index])
        return waveform, target


class AudioCollate:
    """Pads a batch of waveforms with a HF feature extractor.

    Produces the tensors the SSL backbone expects: ``input_values`` and
    ``attention_mask``.
    """

    def __init__(self, processor, target_sr: int = 16000):
        """Initialization.

        Params:
            processor: A ``transformers`` feature extractor
                (e.g. ``AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")``).
            target_sr (int): Sampling rate passed to the processor.
        """
        self.processor = processor
        self.target_sr = target_sr

    def __call__(
        self, batch: List[Tuple[torch.Tensor, int]]
    ) -> Tuple[dict, torch.Tensor]:
        waveforms, targets = zip(*batch)

        features = self.processor(
            [w.numpy() for w in waveforms],
            sampling_rate=self.target_sr,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )

        targets = torch.tensor(targets, dtype=torch.long)
        return dict(features), targets


def read_metadata(metadata_path: str, target_column: str, label2id: dict) -> pd.DataFrame:
    """Load a metadata CSV and add an integer ``target`` column."""
    df = pd.read_csv(metadata_path)
    df["target"] = df[target_column].map(label2id)
    if df["target"].isna().any():
        unknown = sorted(df.loc[df["target"].isna(), target_column].unique().tolist())
        raise ValueError(f"Labels in '{metadata_path}' missing from label2id: {unknown}")
    df["target"] = df["target"].astype(int)
    return df


def build_datasets(config) -> Tuple[AudioDataset, AudioDataset]:
    """Build the train and validation datasets from a config."""
    label2id = dict(config.data.label2id)

    train_df = read_metadata(
        config.datasets.train.metadata_path,
        config.datasets.train.target_column,
        label2id,
    )
    val_df = read_metadata(
        config.datasets.val.metadata_path,
        config.datasets.val.target_column,
        label2id,
    )

    train_dataset = AudioDataset(
        data=train_df,
        filename_column=config.datasets.train.filename_column,
        target_column="target",
        audio_base_dir=config.datasets.train.audio_base_dir,
        target_sr=config.data.target_sr,
        max_duration=config.data.get("max_duration", None),
    )
    val_dataset = AudioDataset(
        data=val_df,
        filename_column=config.datasets.val.filename_column,
        target_column="target",
        audio_base_dir=config.datasets.val.audio_base_dir,
        target_sr=config.data.target_sr,
        max_duration=config.data.get("max_duration", None),
    )

    return train_dataset, val_dataset
