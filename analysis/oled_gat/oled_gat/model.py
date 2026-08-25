from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as functional
from torch import nn
from torch_geometric.nn import GATv2Conv, global_mean_pool

from .graph_data import LAYER_NODE, MATERIAL_NODE, ROOT_NODE


class NodeEncoder(nn.Module):
    def __init__(self, vocabulary: dict[str, Any], hidden_dim: int) -> None:
        super().__init__()
        self.node_type = nn.Embedding(4, 12)
        self.layer_role = nn.Embedding(len(vocabulary["layer_roles"]), 24)
        self.component_role = nn.Embedding(
            len(vocabulary["component_roles"]),
            24,
        )
        self.material = nn.Embedding(len(vocabulary["materials"]), 48)
        self.atomic_number = nn.Embedding(119, 32)
        self.degree = nn.Embedding(9, 8)
        self.formal_charge = nn.Embedding(9, 8)
        self.hybridization = nn.Embedding(8, 8)
        self.chirality = nn.Embedding(5, 8)
        self.numeric = nn.Sequential(
            nn.Linear(10, 32),
            nn.SiLU(),
            nn.LayerNorm(32),
        )
        molecular_width = (
            len(vocabulary["numeric_stats"]["molecular_descriptor_mean"])
            + int(vocabulary.get("molecular_fingerprint_size", 128))
        )
        self.molecular = nn.Sequential(
            nn.Linear(molecular_width, 64),
            nn.SiLU(),
            nn.LayerNorm(64),
        )
        width = 12 + 24 + 24 + 48 + 32 + 8 + 8 + 8 + 8 + 32 + 64
        self.projection = nn.Sequential(
            nn.Linear(width, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

    def forward(self, batch: Any) -> torch.Tensor:
        encoded = torch.cat(
            [
                self.node_type(batch.node_type),
                self.layer_role(batch.layer_role),
                self.component_role(batch.component_role),
                self.material(batch.material_id),
                self.atomic_number(batch.atomic_number),
                self.degree(batch.degree),
                self.formal_charge(batch.formal_charge),
                self.hybridization(batch.hybridization),
                self.chirality(batch.chirality),
                self.numeric(batch.node_numeric),
                self.molecular(batch.molecular_features),
            ],
            dim=-1,
        )
        return self.projection(encoded)


class OLEDGATQuantile(nn.Module):
    def __init__(
        self,
        vocabulary: dict[str, Any],
        *,
        hidden_dim: int,
        attention_heads: int,
        message_passing_layers: int,
        edge_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_dim % attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.dropout = dropout
        self.node_encoder = NodeEncoder(vocabulary, hidden_dim)
        self.edge_embedding = nn.Embedding(len(vocabulary["edge_types"]), edge_dim)
        self.convolutions = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        for _ in range(message_passing_layers):
            self.convolutions.append(
                GATv2Conv(
                    hidden_dim,
                    hidden_dim // attention_heads,
                    heads=attention_heads,
                    concat=True,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    add_self_loops=True,
                    share_weights=False,
                )
            )
            self.normalizations.append(nn.LayerNorm(hidden_dim))

        self.mechanism = nn.Embedding(len(vocabulary["mechanisms"]), 20)
        self.color = nn.Embedding(len(vocabulary["colors"]), 12)
        self.process = nn.Embedding(len(vocabulary["processes"]), 12)
        self.device_type = nn.Embedding(len(vocabulary["device_types"]), 12)
        self.context_numeric = nn.Sequential(
            nn.Linear(2, 20),
            nn.SiLU(),
            nn.LayerNorm(20),
        )
        context_width = 20 + 12 + 12 + 12 + 20
        graph_width = hidden_dim * 5
        self.readout = nn.Sequential(
            nn.Linear(graph_width + context_width, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, batch: Any) -> torch.Tensor:
        hidden = self.node_encoder(batch)
        edge_features = self.edge_embedding(batch.edge_type)
        for convolution, normalization in zip(
            self.convolutions,
            self.normalizations,
            strict=True,
        ):
            update = convolution(
                hidden,
                batch.edge_index,
                edge_attr=edge_features,
            )
            hidden = normalization(
                hidden
                + functional.dropout(
                    functional.silu(update),
                    p=self.dropout,
                    training=self.training,
                )
            )

        root = hidden[batch.node_type.eq(ROOT_NODE)]
        layer_mask = batch.node_type.eq(LAYER_NODE)
        material_mask = batch.node_type.eq(MATERIAL_NODE)
        eml_layer_mask = layer_mask & batch.node_numeric[:, 9].gt(0.5)
        eml_material_mask = material_mask & batch.node_numeric[:, 9].gt(0.5)
        layer_pool = global_mean_pool(
            hidden[layer_mask],
            batch.batch[layer_mask],
            size=batch.num_graphs,
        )
        material_pool = global_mean_pool(
            hidden[material_mask],
            batch.batch[material_mask],
            size=batch.num_graphs,
        )
        eml_layer_pool = global_mean_pool(
            hidden[eml_layer_mask],
            batch.batch[eml_layer_mask],
            size=batch.num_graphs,
        )
        eml_material_pool = global_mean_pool(
            hidden[eml_material_mask],
            batch.batch[eml_material_mask],
            size=batch.num_graphs,
        )
        context = torch.cat(
            [
                self.mechanism(batch.mechanism_id.view(-1)),
                self.color(batch.color_id.view(-1)),
                self.process(batch.process_id.view(-1)),
                self.device_type(batch.device_type_id.view(-1)),
                self.context_numeric(batch.context_numeric),
            ],
            dim=-1,
        )
        raw = self.readout(
            torch.cat(
                [
                    root,
                    layer_pool,
                    material_pool,
                    eml_layer_pool,
                    eml_material_pool,
                    context,
                ],
                dim=-1,
            )
        )
        mean = raw[:, 0]
        median = raw[:, 1]
        lower = median - functional.softplus(raw[:, 2])
        upper = median + functional.softplus(raw[:, 3])
        return torch.stack([mean, lower, median, upper], dim=-1)


def build_oled_gat(
    vocabulary: dict[str, Any],
    model_config: dict[str, Any],
) -> nn.Module:
    architecture = str(model_config.get("architecture", "homogeneous_gat_v1"))
    common = {
        "vocabulary": vocabulary,
        "hidden_dim": int(model_config["hidden_dim"]),
        "attention_heads": int(model_config["attention_heads"]),
        "edge_dim": int(model_config["edge_dim"]),
        "dropout": float(model_config["dropout"]),
    }
    if architecture == "homogeneous_gat_v1":
        return OLEDGATQuantile(
            **common,
            message_passing_layers=int(
                model_config["message_passing_layers"]
            ),
        )
    if architecture == "hierarchical_device_gat_v2":
        from .hierarchical_model import HierarchicalOLEDGAT

        return HierarchicalOLEDGAT(
            **common,
            atom_message_passing_layers=int(
                model_config["atom_message_passing_layers"]
            ),
            device_message_passing_layers=int(
                model_config["device_message_passing_layers"]
            ),
        )
    raise ValueError(f"Unsupported OLED-GAT architecture: {architecture}")


def quantile_hybrid_loss(
    predictions: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor,
    *,
    quantiles: tuple[float, float, float],
    mean_mse_weight: float,
    quantile_loss_weight: float,
) -> torch.Tensor:
    target = target.view(-1, 1)
    quantile_predictions = predictions[:, 1:]
    residual = target - quantile_predictions
    quantile_tensor = predictions.new_tensor(quantiles).view(1, -1)
    pinball = torch.maximum(
        quantile_tensor * residual,
        (quantile_tensor - 1.0) * residual,
    ).mean(dim=1)
    mean_mse = functional.mse_loss(
        predictions[:, 0],
        target[:, 0],
        reduction="none",
    )
    per_sample = mean_mse_weight * mean_mse + quantile_loss_weight * pinball
    weights = sample_weight.view(-1)
    weights = weights / weights.mean().clamp_min(1e-8)
    return (per_sample * weights).mean()
