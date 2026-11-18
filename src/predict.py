"""Predict the emotion of a single audio file with a trained checkpoint."""

import argparse
from pathlib import Path

import torch
from models.pl_wrapper import PLWrapper
from omegaconf import OmegaConf
from utils.dataloader import AudioCollate, load_audio

ROOT = Path(__file__).resolve().parent.parent


def load_model(checkpoint, config, device):
    # The frozen encoder is loaded from pretrained weights. The checkpoint
    # stores the trained head and can also contain training-only loss weights.
    return PLWrapper.load_from_checkpoint(
        checkpoint, config=config, map_location="cpu", strict=False,
    ).eval().to(device)


@torch.inference_mode()
def predict(audio, checkpoint, config, device):
    model = load_model(checkpoint, config, device)
    sample_rate = config.data.target_sr
    waveform = load_audio(audio, sample_rate, config.data.get("max_duration"))
    features, _ = AudioCollate(model.processor, sample_rate)([(waveform, -1)])
    features = {key: value.to(device) for key, value in features.items()}

    scores = model(features).softmax(dim=-1)[0].cpu().tolist()
    probabilities = {label: scores[index] for label, index in config.data.label2id.items()}
    return max(probabilities, key=probabilities.get), probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-a", "--audio", type=Path, required=True)
    parser.add_argument("-c", "--config", type=Path, default=ROOT / "config/audio-wavlm-msp.yaml")
    parser.add_argument("-ck", "--checkpoint", type=Path, help="Defaults to last.ckpt for the YAML's run.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--probs", action="store_true", help="Show probabilities for all classes.")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    checkpoint = args.checkpoint or ROOT / config.checkpoint_dir / config.title / "last.ckpt"
    if not checkpoint.is_file():
        parser.error(f"Checkpoint not found: {checkpoint}")
    if not args.audio.is_file():
        parser.error(f"Audio not found: {args.audio}")

    label, probabilities = predict(args.audio, checkpoint, config, args.device)
    print(label)
    if args.probs:
        for emotion, probability in sorted(probabilities.items(), key=lambda item: -item[1]):
            print(f"{emotion}\t{probability:.4f}")


if __name__ == "__main__":
    main()
