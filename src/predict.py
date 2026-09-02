import os
import sys
import argparse

sys.path.append(os.path.dirname(__file__))

import torch
from omegaconf import OmegaConf
from transformers import AutoFeatureExtractor

from models.pl_wrapper import PLWrapper
from utils.dataloader import load_audio
from utils.utils import id2label_from_config


@torch.inference_mode()
def predict(
    audio_path: str,
    config_path: str,
    checkpoint_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    return_probs: bool = False,
):
    """Predict the emotion of a single audio file.

    Params:
        audio_path (str): Path to a .wav (or any torchaudio-readable) file.
        config_path (str): The YAML config the checkpoint was trained with.
        checkpoint_path (str): Path to the Lightning ``.ckpt`` file.
        device (str): "cuda" or "cpu".
        return_probs (bool): Also return the per-class softmax probabilities.

    Returns:
        str | tuple[str, dict]: Predicted label, optionally with a
            {label: probability} dict.
    """
    config = OmegaConf.load(config_path)
    id2label = id2label_from_config(config)

    # strict=False: the CE-loss `weight` buffer is only created during training
    # setup(), and frozen backbone weights are stripped on save (reloaded from HF
    # in __init__), so both are legitimately absent from the checkpoint.
    model = PLWrapper.load_from_checkpoint(
        checkpoint_path, config=config, inference_mode=True,
        map_location=device, strict=False,
    )
    model.eval().to(device)

    processor = AutoFeatureExtractor.from_pretrained(config.model.model_name)
    waveform = load_audio(
        audio_path, config.data.target_sr, config.data.get("max_duration", None)
    )
    features = processor(
        waveform.numpy(),
        sampling_rate=config.data.target_sr,
        return_tensors="pt",
        return_attention_mask=True,
    )
    features = {k: v.to(device) for k, v in features.items()}

    logits = model(features)
    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu()
    pred_id = int(torch.argmax(probs))
    label = id2label[pred_id]

    if return_probs:
        return label, {id2label[i]: float(p) for i, p in enumerate(probs)}
    return label


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--audio_path", required=True, type=str)
    parser.add_argument("-c", "--config_path", required=True, type=str)
    parser.add_argument("-ck", "--checkpoint_path", required=True, type=str)
    parser.add_argument("--probs", action="store_true", help="print per-class probabilities")
    args = parser.parse_args()

    result = predict(
        args.audio_path, args.config_path, args.checkpoint_path, return_probs=args.probs
    )

    if args.probs:
        label, probs = result
        print(f"Predicted emotion: {label}")
        for name, p in sorted(probs.items(), key=lambda x: -x[1]):
            print(f"  {name}: {p:.4f}")
    else:
        print(f"Predicted emotion: {result}")


if __name__ == "__main__":
    main()
