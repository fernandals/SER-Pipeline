from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from transformers import AutoConfig, AutoModel

ACTIVATIONS_FUNCS = {
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
    "leaky_relu": nn.LeakyReLU(),
}


class MLPBase(nn.Module):
    """Simple configurable MLP classification head."""

    def __init__(
        self,
        input_size: int = 1024,
        hidden_dim: int = 1024,
        num_layers: int = 1,
        output_size: int = 8,
        dropout: float = 0.1,
        activation_func: str = "relu",
    ) -> None:
        super().__init__()

        activation_func = activation_func.lower()
        if activation_func not in ACTIVATIONS_FUNCS:
            raise ValueError(
                f"Unsupported activation function: {activation_func}. "
                f"Supported: {list(ACTIVATIONS_FUNCS.keys())}"
            )
        self.activation_func = activation_func
        activation_layer = ACTIVATIONS_FUNCS[self.activation_func]

        layers: List[nn.Module] = []
        current_dim = input_size
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(activation_layer)
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_size))

        self.layers = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming init for linear layers, zeros for biases."""
        for module in self.layers:
            if isinstance(module, nn.Linear):
                if self.activation_func in ["relu", "gelu"]:
                    init.kaiming_uniform_(module.weight, a=0, nonlinearity="relu")
                elif self.activation_func == "leaky_relu":
                    init.kaiming_uniform_(module.weight, a=0.01, nonlinearity="leaky_relu")
                if module.bias is not None:
                    init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MaskedMeanPooling(nn.Module):
    """Mean over the time axis, ignoring padded frames."""

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Params:
            x (torch.Tensor): [B, T, F] frame features.
            mask (Optional[torch.Tensor]): [B, T] with 1 for valid frames.

        Returns:
            torch.Tensor: [B, F] pooled features.
        """
        if mask is None:
            return x.mean(dim=1)
        mask = mask.unsqueeze(-1).to(x.dtype)
        return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)


class AttentiveStatisticsPooling(nn.Module):
    """Attentive Statistics Pooling (Okabe et al., 2018).

    Input:  x [B, T, F]  (+ optional [B, T] mask)
    Output: [B, 2F]  (attention-weighted mean concatenated with std)
    """

    def __init__(self, input_size: int):
        super().__init__()
        self.sap_linear = nn.Linear(input_size, input_size)
        self.attention = nn.Parameter(torch.FloatTensor(input_size, 1))
        torch.nn.init.normal_(self.attention, mean=0.0, std=1.0)

    def forward(self, xs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = torch.tanh(self.sap_linear(xs))
        w = torch.matmul(h, self.attention).squeeze(dim=2)  # [B, T]

        if mask is not None:
            w = w.masked_fill(mask == 0, float("-inf"))

        w = F.softmax(w, dim=1).unsqueeze(2)  # [B, T, 1]

        mu = torch.sum(xs * w, dim=1)
        rh = torch.sqrt((torch.sum((xs ** 2) * w, dim=1) - mu ** 2).clamp(min=1e-5))
        return torch.cat((mu, rh), dim=1)


class AudioSERModel(nn.Module):
    """Speech-only emotion classifier.

    Pipeline:
        waveform -> SSL backbone -> hidden states
        -> layer aggregation ("last" | learned "weighted_sum")
        -> temporal pooling ("mean" | "attpool")
        -> MLP -> emotion logits
    """

    def __init__(
        self,
        model_name: str = "microsoft/wavlm-large",
        num_classes: int = 8,
        freeze_backbone: bool = True,
        layer_aggregation: str = "weighted_sum",
        pooling_strategy: str = "mean",
        mlp_hidden_dim: int = 1024,
        mlp_num_layers: int = 1,
        mlp_dropout: float = 0.1,
        mlp_activation_func: str = "relu",
    ):
        super().__init__()

        config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.backbone = AutoModel.from_pretrained(model_name, config=config)

        # Whisper is encoder-decoder; only the encoder is needed here.
        if "whisper" in model_name.lower():
            self.backbone = self.backbone.encoder

        self.hidden_size = config.hidden_size
        self.num_feature_layers = config.num_hidden_layers + 1  # + embedding output

        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        if layer_aggregation not in ("last", "weighted_sum"):
            raise ValueError(f"Invalid layer_aggregation: {layer_aggregation}")
        self.layer_aggregation = layer_aggregation
        if layer_aggregation == "weighted_sum":
            self.layer_weights = nn.Parameter(torch.zeros(self.num_feature_layers))

        if pooling_strategy == "mean":
            self.pooling = MaskedMeanPooling()
            mlp_input_dim = self.hidden_size
        elif pooling_strategy == "attpool":
            self.pooling = AttentiveStatisticsPooling(input_size=self.hidden_size)
            mlp_input_dim = self.hidden_size * 2
        else:
            raise ValueError(f"Invalid pooling_strategy: {pooling_strategy}")
        self.pooling_strategy = pooling_strategy

        self.mlp = MLPBase(
            input_size=mlp_input_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            output_size=num_classes,
            dropout=mlp_dropout,
            activation_func=mlp_activation_func,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _feature_attention_mask(
        self, attention_mask: torch.Tensor, feat_len: int
    ) -> torch.Tensor:
        """Downsample the waveform-level mask to the backbone's frame rate."""
        output_lengths = self.backbone._get_feat_extract_output_lengths(
            attention_mask.sum(-1)
        ).long()
        mask = torch.zeros(
            (attention_mask.shape[0], feat_len),
            device=attention_mask.device,
            dtype=torch.long,
        )
        for i, length in enumerate(output_lengths):
            mask[i, : min(int(length), feat_len)] = 1
        return mask

    def _aggregate_layers(self, hidden_states) -> torch.Tensor:
        if self.layer_aggregation == "last":
            return hidden_states[-1]
        stacked = torch.stack(hidden_states, dim=0)  # [L, B, T, F]
        weights = torch.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
        return (stacked * weights).sum(dim=0)

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.backbone(
                    input_values=input_values,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
        else:
            outputs = self.backbone(
                input_values=input_values,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        hidden = self._aggregate_layers(outputs.hidden_states)  # [B, T, F]

        feat_mask = None
        if attention_mask is not None:
            feat_mask = self._feature_attention_mask(attention_mask, hidden.shape[1])

        pooled = self.pooling(hidden, feat_mask)
        logits = self.mlp(pooled)
        return logits
