from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn.functional as functional
from torch import nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import scatter, softmax

from .features import EMITTER_ROLES, HOST_ROLES, SENSITIZER_ROLES
from .graph_data import (
    ATOM_NODE,
    EDGE_TYPE_TO_ID,
    LAYER_NODE,
    MATERIAL_NODE,
    ROOT_NODE,
)
from .model import NodeEncoder


def _ids(values: list[str], names: Iterable[str]) -> tuple[int, ...]:
    lookup = {value.casefold(): index for index, value in enumerate(values)}
    return tuple(
        lookup[name.casefold()]
        for name in names
        if name.casefold() in lookup
    )


def _mask_for_ids(values: torch.Tensor, identifiers: tuple[int, ...]) -> torch.Tensor:
    if not identifiers:
        return torch.zeros_like(values, dtype=torch.bool)
    mask = values.eq(identifiers[0])
    for identifier in identifiers[1:]:
        mask |= values.eq(identifier)
    return mask


class ResidualGATBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        attention_heads: int,
        edge_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.normalization = nn.LayerNorm(hidden_dim)
        self.convolution = GATv2Conv(
            hidden_dim,
            hidden_dim // attention_heads,
            heads=attention_heads,
            concat=True,
            dropout=dropout,
            edge_dim=edge_dim,
            add_self_loops=True,
            share_weights=False,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        update = self.convolution(
            hidden,
            edge_index,
            edge_attr=edge_features,
        )
        return self.normalization(
            hidden
            + functional.dropout(
                functional.silu(update),
                p=self.dropout,
                training=self.training,
            )
        )


class HierarchicalOLEDGAT(nn.Module):
    """Stage molecular, material, layer, interface, and device message passing."""

    def __init__(
        self,
        vocabulary: dict[str, Any],
        *,
        hidden_dim: int,
        attention_heads: int,
        atom_message_passing_layers: int,
        device_message_passing_layers: int,
        edge_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_dim % attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.node_encoder = NodeEncoder(vocabulary, hidden_dim)
        self.atom_edge_embedding = nn.Embedding(
            len(vocabulary["edge_types"]),
            edge_dim,
        )
        self.coarse_edge_embedding = nn.Embedding(
            len(vocabulary["edge_types"]),
            edge_dim,
        )
        self.atom_blocks = nn.ModuleList(
            ResidualGATBlock(
                hidden_dim,
                attention_heads,
                edge_dim,
                dropout,
            )
            for _ in range(atom_message_passing_layers)
        )
        self.coarse_blocks = nn.ModuleList(
            ResidualGATBlock(
                hidden_dim,
                attention_heads,
                edge_dim,
                dropout,
            )
            for _ in range(device_message_passing_layers)
        )

        self.molecular_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.material_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.layer_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.mechanism = nn.Embedding(len(vocabulary["mechanisms"]), 24)
        self.color = nn.Embedding(len(vocabulary["colors"]), 16)
        self.process = nn.Embedding(len(vocabulary["processes"]), 16)
        self.device_type = nn.Embedding(len(vocabulary["device_types"]), 16)
        self.context_numeric = nn.Sequential(
            nn.Linear(2, 24),
            nn.SiLU(),
            nn.LayerNorm(24),
        )
        context_width = 24 + 16 + 16 + 16 + 24
        self.context_projection = nn.Sequential(
            nn.Linear(context_width, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.root_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.layer_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.interface_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.interface_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        layer_roles = vocabulary["layer_roles"]
        component_roles = vocabulary["component_roles"]
        self.layer_role_groups = {
            "eml": _ids(layer_roles, ("eml",)),
            "htl": _ids(layer_roles, ("htl",)),
            "etl": _ids(layer_roles, ("etl",)),
            "blocking": _ids(layer_roles, ("ebl", "hbl")),
        }
        self.component_role_groups = {
            "host": _ids(component_roles, HOST_ROLES),
            "emitter": _ids(component_roles, EMITTER_ROLES),
            "sensitizer": _ids(component_roles, SENSITIZER_ROLES),
        }

        # root, attention/mean/max layers, four layer-role pools, interface,
        # and three EML component-role pools.
        graph_width = hidden_dim * 12
        self.readout = nn.Sequential(
            nn.Linear(graph_width + context_width, hidden_dim * 3),
            nn.LayerNorm(hidden_dim * 3),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim, 4),
        )

    @staticmethod
    def _mean_pool(
        hidden: torch.Tensor,
        groups: torch.Tensor,
        size: int,
    ) -> torch.Tensor:
        return scatter(hidden, groups, dim=0, dim_size=size, reduce="mean")

    @staticmethod
    def _max_pool(
        hidden: torch.Tensor,
        groups: torch.Tensor,
        size: int,
    ) -> torch.Tensor:
        return scatter(hidden, groups, dim=0, dim_size=size, reduce="max")

    @staticmethod
    def _attention_pool(
        hidden: torch.Tensor,
        groups: torch.Tensor,
        gate: nn.Module,
        size: int,
    ) -> torch.Tensor:
        if hidden.numel() == 0:
            return hidden.new_zeros((size, hidden.shape[-1]))
        weights = softmax(gate(hidden).view(-1), groups, num_nodes=size)
        return scatter(
            hidden * weights.unsqueeze(-1),
            groups,
            dim=0,
            dim_size=size,
            reduce="sum",
        )

    def _role_pool(
        self,
        hidden: torch.Tensor,
        groups: torch.Tensor,
        role_values: torch.Tensor,
        identifiers: tuple[int, ...],
        size: int,
    ) -> torch.Tensor:
        mask = _mask_for_ids(role_values, identifiers)
        if not mask.any():
            return hidden.new_zeros((size, self.hidden_dim))
        return self._mean_pool(hidden[mask], groups[mask], size)

    def _encode_atoms(
        self,
        hidden: torch.Tensor,
        batch: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        atom_global = batch.node_type.eq(ATOM_NODE).nonzero(as_tuple=False).view(-1)
        if atom_global.numel() == 0:
            return (
                hidden.new_zeros((0, self.hidden_dim)),
                atom_global,
            )
        global_to_atom = torch.full(
            (hidden.shape[0],),
            -1,
            dtype=torch.long,
            device=hidden.device,
        )
        global_to_atom[atom_global] = torch.arange(
            atom_global.numel(),
            device=hidden.device,
        )
        bond_mask = batch.edge_type.ge(EDGE_TYPE_TO_ID["bond_single"])
        bond_edges = batch.edge_index[:, bond_mask]
        atom_edges = global_to_atom[bond_edges]
        valid = atom_edges.ge(0).all(dim=0)
        atom_edges = atom_edges[:, valid]
        atom_edge_types = batch.edge_type[bond_mask][valid]
        atom_hidden = hidden[atom_global]
        atom_edge_features = self.atom_edge_embedding(atom_edge_types)
        for block in self.atom_blocks:
            atom_hidden = block(
                atom_hidden,
                atom_edges,
                atom_edge_features,
            )
        return atom_hidden, atom_global

    def _fuse_materials(
        self,
        hidden: torch.Tensor,
        atom_hidden: torch.Tensor,
        atom_global: torch.Tensor,
        batch: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        material_global = (
            batch.node_type.eq(MATERIAL_NODE).nonzero(as_tuple=False).view(-1)
        )
        atom_parent_mask = batch.edge_type.eq(
            EDGE_TYPE_TO_ID["atom_to_material"]
        )
        atom_sources = batch.edge_index[0, atom_parent_mask]
        atom_parents = batch.edge_index[1, atom_parent_mask]
        global_to_atom = torch.full(
            (hidden.shape[0],),
            -1,
            dtype=torch.long,
            device=hidden.device,
        )
        global_to_atom[atom_global] = torch.arange(
            atom_global.numel(),
            device=hidden.device,
        )
        atom_rows = global_to_atom[atom_sources]
        valid = atom_rows.ge(0)
        molecular_pool = scatter(
            atom_hidden[atom_rows[valid]],
            atom_parents[valid],
            dim=0,
            dim_size=hidden.shape[0],
            reduce="mean",
        )
        atom_count = scatter(
            torch.ones(
                int(valid.sum()),
                device=hidden.device,
                dtype=hidden.dtype,
            ),
            atom_parents[valid],
            dim=0,
            dim_size=hidden.shape[0],
            reduce="sum",
        )
        base = hidden[material_global]
        update = self.molecular_fusion(
            torch.cat(
                [
                    base,
                    molecular_pool[material_global],
                    atom_count[material_global].gt(0).float().unsqueeze(-1),
                ],
                dim=-1,
            )
        )
        return base + update, material_global

    def _fuse_layers(
        self,
        hidden: torch.Tensor,
        material_hidden: torch.Tensor,
        material_global: torch.Tensor,
        batch: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer_global = (
            batch.node_type.eq(LAYER_NODE).nonzero(as_tuple=False).view(-1)
        )
        global_to_material = torch.full(
            (hidden.shape[0],),
            -1,
            dtype=torch.long,
            device=hidden.device,
        )
        global_to_material[material_global] = torch.arange(
            material_global.numel(),
            device=hidden.device,
        )
        material_parent_mask = batch.edge_type.eq(
            EDGE_TYPE_TO_ID["material_to_layer"]
        )
        material_sources = batch.edge_index[0, material_parent_mask]
        layer_parents = batch.edge_index[1, material_parent_mask]
        material_rows = global_to_material[material_sources]
        valid = material_rows.ge(0)
        source_hidden = material_hidden[material_rows[valid]]
        parent_global = layer_parents[valid]
        base = hidden[layer_global]
        attention_scores = self.material_attention(
            torch.cat(
                [source_hidden, hidden[parent_global]],
                dim=-1,
            )
        ).view(-1)
        attention_weights = softmax(
            attention_scores,
            parent_global,
            num_nodes=hidden.shape[0],
        )
        attentive = scatter(
            source_hidden * attention_weights.unsqueeze(-1),
            parent_global,
            dim=0,
            dim_size=hidden.shape[0],
            reduce="sum",
        )
        mean = scatter(
            source_hidden,
            parent_global,
            dim=0,
            dim_size=hidden.shape[0],
            reduce="mean",
        )
        update = self.layer_fusion(
            torch.cat(
                [
                    base,
                    attentive[layer_global],
                    mean[layer_global],
                ],
                dim=-1,
            )
        )
        return base + update, layer_global

    def forward(self, batch: Any) -> torch.Tensor:
        hidden = self.node_encoder(batch)
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
        context_hidden = self.context_projection(context)

        atom_hidden, atom_global = self._encode_atoms(hidden, batch)
        material_hidden, material_global = self._fuse_materials(
            hidden,
            atom_hidden,
            atom_global,
            batch,
        )
        hidden = hidden.clone()
        hidden[material_global] = material_hidden
        layer_hidden, layer_global = self._fuse_layers(
            hidden,
            material_hidden,
            material_global,
            batch,
        )
        hidden[layer_global] = layer_hidden

        root_global = (
            batch.node_type.eq(ROOT_NODE).nonzero(as_tuple=False).view(-1)
        )
        hidden[root_global] = self.root_fusion(
            torch.cat([hidden[root_global], context_hidden], dim=-1)
        )
        coarse_global = (
            batch.node_type.eq(ROOT_NODE) | batch.node_type.eq(LAYER_NODE)
        ).nonzero(as_tuple=False).view(-1)
        global_to_coarse = torch.full(
            (hidden.shape[0],),
            -1,
            dtype=torch.long,
            device=hidden.device,
        )
        global_to_coarse[coarse_global] = torch.arange(
            coarse_global.numel(),
            device=hidden.device,
        )
        coarse_edge_mask = batch.edge_type.le(
            EDGE_TYPE_TO_ID["previous_layer"]
        )
        coarse_edges = global_to_coarse[batch.edge_index[:, coarse_edge_mask]]
        valid_coarse_edges = coarse_edges.ge(0).all(dim=0)
        coarse_edges = coarse_edges[:, valid_coarse_edges]
        coarse_edge_types = batch.edge_type[coarse_edge_mask][valid_coarse_edges]
        coarse_hidden = hidden[coarse_global]
        coarse_edge_features = self.coarse_edge_embedding(coarse_edge_types)
        for block in self.coarse_blocks:
            coarse_hidden = block(
                coarse_hidden,
                coarse_edges,
                coarse_edge_features,
            )

        coarse_types = batch.node_type[coarse_global]
        root = coarse_hidden[coarse_types.eq(ROOT_NODE)]
        coarse_layers = coarse_hidden[coarse_types.eq(LAYER_NODE)]
        layer_batch = batch.batch[layer_global]
        layer_roles = batch.layer_role[layer_global]
        size = int(batch.num_graphs)
        layer_attention = self._attention_pool(
            coarse_layers,
            layer_batch,
            self.layer_gate,
            size,
        )
        layer_mean = self._mean_pool(coarse_layers, layer_batch, size)
        layer_max = self._max_pool(coarse_layers, layer_batch, size)
        layer_role_pools = [
            self._role_pool(
                coarse_layers,
                layer_batch,
                layer_roles,
                identifiers,
                size,
            )
            for identifiers in self.layer_role_groups.values()
        ]

        next_layer_mask = batch.edge_type.eq(EDGE_TYPE_TO_ID["next_layer"])
        next_edges_global = batch.edge_index[:, next_layer_mask]
        next_edges = global_to_coarse[next_edges_global]
        valid_next = next_edges.ge(0).all(dim=0)
        next_edges = next_edges[:, valid_next]
        if next_edges.shape[1]:
            previous_hidden = coarse_hidden[next_edges[0]]
            next_hidden = coarse_hidden[next_edges[1]]
            interface_hidden = self.interface_encoder(
                torch.cat(
                    [
                        previous_hidden,
                        next_hidden,
                        next_hidden - previous_hidden,
                    ],
                    dim=-1,
                )
            )
            interface_batch = batch.batch[
                next_edges_global[0, valid_next]
            ]
            interface_pool = self._attention_pool(
                interface_hidden,
                interface_batch,
                self.interface_gate,
                size,
            )
        else:
            interface_pool = hidden.new_zeros((size, self.hidden_dim))

        material_batch = batch.batch[material_global]
        component_roles = batch.component_role[material_global]
        material_role_pools = [
            self._role_pool(
                material_hidden,
                material_batch,
                component_roles,
                identifiers,
                size,
            )
            for identifiers in self.component_role_groups.values()
        ]
        graph_embedding = torch.cat(
            [
                root,
                layer_attention,
                layer_mean,
                layer_max,
                *layer_role_pools,
                interface_pool,
                *material_role_pools,
                context,
            ],
            dim=-1,
        )
        raw = self.readout(graph_embedding)
        mean = raw[:, 0]
        median = raw[:, 1]
        lower = median - functional.softplus(raw[:, 2])
        upper = median + functional.softplus(raw[:, 3])
        return torch.stack([mean, lower, median, upper], dim=-1)
