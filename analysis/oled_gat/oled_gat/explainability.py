from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor
from torch_geometric.nn import global_mean_pool

from .graph_data import LAYER_NODE, MATERIAL_NODE, ROOT_NODE
from .model import OLEDGATQuantile


@dataclass(frozen=True)
class AttentionBlock:
    edge_index: Tensor
    coefficients: Tensor


@dataclass(frozen=True)
class AttentionForward:
    output: Tensor
    attention: tuple[AttentionBlock, ...]
    hidden: Tensor


def load_frozen_model(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[OLEDGATQuantile, dict[str, Any]]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    vocabulary = checkpoint["vocabulary"]
    model_config = checkpoint["model_config"]
    model = OLEDGATQuantile(
        vocabulary,
        hidden_dim=int(model_config["hidden_dim"]),
        attention_heads=int(model_config["attention_heads"]),
        message_passing_layers=int(model_config["message_passing_layers"]),
        edge_dim=int(model_config["edge_dim"]),
        dropout=float(model_config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def forward_with_attention(
    model: OLEDGATQuantile,
    batch: Any,
    *,
    retain_attention_gradients: bool = False,
) -> AttentionForward:
    hidden = model.node_encoder(batch)
    edge_features = model.edge_embedding(batch.edge_type)
    attention: list[AttentionBlock] = []
    for convolution, normalization in zip(
        model.convolutions,
        model.normalizations,
        strict=True,
    ):
        update, (edge_index, coefficients) = convolution(
            hidden,
            batch.edge_index,
            edge_attr=edge_features,
            return_attention_weights=True,
        )
        if retain_attention_gradients:
            coefficients.retain_grad()
        attention.append(
            AttentionBlock(
                edge_index=edge_index,
                coefficients=coefficients,
            )
        )
        hidden = normalization(
            hidden
            + functional.dropout(
                functional.silu(update),
                p=model.dropout,
                training=model.training,
            )
        )

    root = hidden[batch.node_type.eq(ROOT_NODE)]
    layer_mask = batch.node_type.eq(LAYER_NODE)
    material_mask = batch.node_type.eq(MATERIAL_NODE)
    eml_layer_mask = layer_mask & batch.node_numeric[:, 9].gt(0.5)
    eml_material_mask = material_mask & batch.node_numeric[:, 9].gt(0.5)

    def pool(mask: Tensor) -> Tensor:
        return global_mean_pool(
            hidden[mask],
            batch.batch[mask],
            size=batch.num_graphs,
        )

    context = torch.cat(
        [
            model.mechanism(batch.mechanism_id.view(-1)),
            model.color(batch.color_id.view(-1)),
            model.process(batch.process_id.view(-1)),
            model.device_type(batch.device_type_id.view(-1)),
            model.context_numeric(batch.context_numeric),
        ],
        dim=-1,
    )
    raw = model.readout(
        torch.cat(
            [
                root,
                pool(layer_mask),
                pool(material_mask),
                pool(eml_layer_mask),
                pool(eml_material_mask),
                context,
            ],
            dim=-1,
        )
    )
    mean = raw[:, 0]
    median = raw[:, 1]
    lower = median - functional.softplus(raw[:, 2])
    upper = median + functional.softplus(raw[:, 3])
    output = torch.stack([mean, lower, median, upper], dim=-1)
    return AttentionForward(
        output=output,
        attention=tuple(attention),
        hidden=hidden,
    )


def denormalize_prediction(
    normalized: Tensor,
    vocabulary: dict[str, Any],
) -> Tensor:
    target_stats = vocabulary["numeric_stats"]
    return (
        normalized * float(target_stats["target_std"])
        + float(target_stats["target_mean"])
    )
