from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

from .evaluation import conformal_quantile_offset
from .graph_data import (
    DeviceGraphBuilder,
    graph_vocabulary_from_dict,
    link_components,
)
from .model import OLEDGATQuantile


def _input_frames(
    payload: dict[str, Any],
    *,
    target_mean: float,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    device_id = str(payload.get("id") or "prediction-device")
    layers = pd.DataFrame(payload.get("layers") or [])
    components = pd.DataFrame(payload.get("components") or [])
    materials = pd.DataFrame(payload.get("materials") or [])
    if layers.empty or components.empty or materials.empty:
        raise ValueError("layers, components, and materials must all be non-empty")

    required_layers = {"layer_index", "layer_role"}
    required_components = {
        "layer_index",
        "layer_role",
        "paper_material_id",
        "component_role",
    }
    required_materials = {"paper_material_id", "canonical_smiles"}
    for name, frame, required in (
        ("layers", layers, required_layers),
        ("components", components, required_components),
        ("materials", materials, required_materials),
    ):
        missing = required - set(frame)
        if missing:
            raise ValueError(f"{name} is missing fields: {sorted(missing)}")

    layers["device_id"] = device_id
    if "thickness_value" not in layers:
        layers["thickness_value"] = np.nan
    components["device_id"] = device_id
    for column in ("material_key", "material_mention", "ratio"):
        if column not in components:
            components[column] = None
    components["ratio"] = components["ratio"].map(
        lambda value: json.dumps(value) if isinstance(value, dict) else value
    )
    materials["device_id"] = device_id
    for column in (
        "global_material_id",
        "material_key",
        "normalized_name",
        "abbreviation",
    ):
        if column not in materials:
            materials[column] = None

    linked = link_components(components, materials)
    eml = linked.loc[linked["layer_role"].astype(str).str.casefold().eq("eml")]
    if eml.empty:
        raise ValueError("At least one EML component is required")
    if eml["canonical_smiles"].isna().any():
        missing_ids = eml.loc[
            eml["canonical_smiles"].isna(),
            "paper_material_id",
        ].tolist()
        raise ValueError(f"Every EML component needs SMILES: {missing_ids}")

    device_payload = dict(payload.get("device") or {})
    device_payload.update(
        {
            "id": device_id,
            "eqe_max": target_mean,
            "scope_device_count": 1,
            "layer_count": len(layers),
            "material_count": components["paper_material_id"].nunique(),
        }
    )
    for column, default in (
        ("primary_mechanism", "unknown"),
        ("color_group", "unknown"),
        ("fabrication_method", "unknown"),
        ("device_type", "unknown"),
    ):
        device_payload.setdefault(column, default)
    return pd.Series(device_payload), layers, linked


def predict_device(
    payload: dict[str, Any],
    *,
    checkpoint_path: Path,
    config: dict[str, Any],
    validation_predictions_path: Path | None = None,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for OLED-GAT inference")
    device = torch.device("cuda")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    vocabulary_payload = checkpoint["vocabulary"]
    vocabulary = graph_vocabulary_from_dict(vocabulary_payload)
    target_stats = vocabulary_payload["numeric_stats"]
    device_row, layers, linked = _input_frames(
        payload,
        target_mean=float(target_stats["target_mean"]),
    )
    builder = DeviceGraphBuilder(
        vocabulary,
        maximum_atoms_per_molecule=int(
            config["graph"]["maximum_atoms_per_molecule"]
        ),
        molecular_fingerprint_size=int(
            config["graph"]["molecular_fingerprint_size"]
        ),
    )
    graph = builder.build(
        device_row,
        layers,
        linked,
        sample_row=0,
    )
    model_config = checkpoint["model_config"]
    model = OLEDGATQuantile(
        vocabulary_payload,
        hidden_dim=int(model_config["hidden_dim"]),
        attention_heads=int(model_config["attention_heads"]),
        message_passing_layers=int(model_config["message_passing_layers"]),
        edge_dim=int(model_config["edge_dim"]),
        dropout=float(model_config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    batch = Batch.from_data_list([graph]).to(device)
    with torch.no_grad():
        normalized = model(batch).float().cpu().numpy()[0]
    raw = np.clip(
        normalized * float(target_stats["target_std"])
        + float(target_stats["target_mean"]),
        0.0,
        60.0,
    )
    result: dict[str, Any] = {
        "device_id": str(device_row["id"]),
        "mean_eqe_pct": float(raw[0]),
        "q10_eqe_pct": float(raw[1]),
        "q50_eqe_pct": float(raw[2]),
        "q90_eqe_pct": float(raw[3]),
        "graph_nodes": int(graph.num_nodes),
        "graph_edges": int(graph.num_edges),
    }
    if validation_predictions_path and validation_predictions_path.exists():
        validation = pd.read_csv(validation_predictions_path)
        offset = conformal_quantile_offset(
            validation["eqe_max"].to_numpy(),
            validation["q10"].to_numpy(),
            validation["q90"].to_numpy(),
            alpha=0.2,
        )
        result.update(
            {
                "calibrated_q10_eqe_pct": float(max(0.0, raw[1] - offset)),
                "calibrated_q90_eqe_pct": float(min(60.0, raw[3] + offset)),
                "conformal_offset": offset,
                "nominal_coverage": 0.8,
            }
        )
    return result
