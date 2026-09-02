from models.modules import AudioSERModel


def create_ser_model(model_type: str, **kwargs):
    """Factory that maps a ``model_type`` string to a model class.

    Kept as a factory so new architectures (e.g. audio+text fusion) can be
    added without touching the training loop.

    Params:
        model_type (str): Currently only ``"audio"``.
        **kwargs: Forwarded to the model constructor.

    Returns:
        nn.Module: The instantiated model.
    """
    if model_type.lower() == "audio":
        return AudioSERModel(**kwargs)

    raise ValueError(f"Unknown model_type: {model_type}. Must be one of: 'audio'.")
