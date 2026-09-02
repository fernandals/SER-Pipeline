import torch

from utils.dataloader import read_metadata


def get_classes_weights(config) -> torch.Tensor:
    """Inverse-frequency class weights computed from the train split.

    weight[c] = total_samples / count[c]

    Params:
        config (DictConfig): Experiment config.

    Returns:
        torch.Tensor: Float tensor of shape [num_classes], ordered by class id.
    """
    num_classes = config.data.num_classes
    label2id = dict(config.data.label2id)

    train_df = read_metadata(
        config.datasets.train.metadata_path,
        config.datasets.train.target_column,
        label2id,
    )

    counts = train_df["target"].value_counts().to_dict()
    total = len(train_df)

    weights = [total / counts.get(c, total) for c in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float)


def id2label_from_config(config) -> dict:
    """Invert ``config.data.label2id`` into an {id: label} mapping."""
    return {int(v): k for k, v in dict(config.data.label2id).items()}
