from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import spearmanr
from torch_geometric.data import Batch, Data

from oled_gat.explainability import (
    denormalize_prediction,
    forward_with_attention,
    load_frozen_model,
)
from oled_gat.graph_data import (
    EDGE_TYPE_TO_ID,
    EDGE_TYPES,
    LAYER_NODE,
    load_graph_cache,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {
    "ink": "#18323a",
    "muted": "#627980",
    "teal": "#188a91",
    "blue": "#3478b4",
    "green": "#3d9159",
    "amber": "#dc9512",
    "coral": "#cf603d",
    "grid": "#dbe5e7",
    "white": "#ffffff",
}

PRIMARY_LAYER_ROLES = [
    "anode",
    "HIL",
    "HTL",
    "EBL",
    "EML",
    "HBL",
    "ETL",
    "EIL",
    "cathode",
    "interlayer",
    "buffer_layer",
    "spacer_layer",
]


@dataclass(frozen=True)
class VariantTask:
    record_id: int
    variant: str
    graph: Data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explain a frozen OLED-GAT model with attention and interventions."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("analysis/oled_gat/outputs_device_random"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate figures from existing CSV and JSON outputs.",
    )
    return parser.parse_args()


def configure_plotting() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "axes.titlecolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "text.color": COLORS["ink"],
            "axes.linewidth": 1.1,
            "figure.dpi": 160,
            "savefig.dpi": 220,
        }
    )


def save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path.with_suffix(".png"), bbox_inches="tight")
    figure.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    x_masked = x[mask]
    y_masked = y[mask]
    if np.unique(x_masked).size < 2 or np.unique(y_masked).size < 2:
        return float("nan")
    return float(spearmanr(x_masked, y_masked).statistic)


def _normalized_entropy(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0]
    if len(probabilities) < 2 or positive.size == 0:
        return float("nan")
    entropy = -float(np.sum(positive * np.log(positive)))
    return entropy / math.log(len(probabilities))


def _head_pair_correlation(shares: np.ndarray) -> float:
    correlations: list[float] = []
    for first in range(shares.shape[1]):
        for second in range(first + 1, shares.shape[1]):
            value = _safe_spearman(shares[:, first], shares[:, second])
            if np.isfinite(value):
                correlations.append(value)
    return float(np.mean(correlations)) if correlations else float("nan")


def _layer_rows_for_device(
    layer_groups: dict[str, pd.DataFrame],
    device_id: str,
    layer_node_count: int,
) -> pd.DataFrame:
    rows = layer_groups[device_id].sort_values("layer_index").reset_index(drop=True)
    if len(rows) != layer_node_count:
        raise RuntimeError(
            f"{device_id}: graph has {layer_node_count} layer nodes but "
            f"prepared table has {len(rows)} rows"
        )
    return rows


def extract_test_attributions(
    model: torch.nn.Module,
    graphs: list[Data],
    vocabulary: dict[str, Any],
    manifest: pd.DataFrame,
    layers: pd.DataFrame,
    *,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    layer_groups = {
        str(device_id): group.copy()
        for device_id, group in layers.groupby("device_id", sort=False)
    }
    role_vocabulary = vocabulary["layer_roles"]
    thickness_std = float(vocabulary["numeric_stats"]["thickness_std"])
    layer_records: list[dict[str, Any]] = []
    edge_records: list[dict[str, Any]] = []
    interface_records: list[dict[str, Any]] = []
    stability_records: list[dict[str, Any]] = []

    test_rows = manifest.index[manifest["split"].eq("test")].tolist()
    for sequence, graph_row in enumerate(test_rows, start=1):
        graph = graphs[graph_row]
        sample_row = int(graph.sample_row.view(-1)[0])
        if sample_row != graph_row:
            raise RuntimeError(
                f"Graph cache row mismatch: expected {graph_row}, got {sample_row}"
            )
        device_row = manifest.iloc[graph_row]
        device_id = str(device_row["id"])
        cpu_layer_nodes = (
            graph.node_type.eq(LAYER_NODE).nonzero(as_tuple=False).view(-1)
        )
        device_layers = _layer_rows_for_device(
            layer_groups,
            device_id,
            len(cpu_layer_nodes),
        )

        batch = Batch.from_data_list([graph]).to(device)
        batch.node_numeric = (
            batch.node_numeric.detach().clone().requires_grad_(True)
        )
        model.zero_grad(set_to_none=True)
        explained = forward_with_attention(
            model,
            batch,
            retain_attention_gradients=True,
        )
        raw_prediction = denormalize_prediction(
            explained.output[:, 0],
            vocabulary,
        )
        raw_prediction.sum().backward()
        prediction = float(raw_prediction.detach().cpu()[0])
        numeric_gradient = batch.node_numeric.grad.detach().cpu()
        original_edge_count = int(graph.edge_index.shape[1])
        edge_type = graph.edge_type.numpy()
        source_nodes = graph.edge_index[0].numpy()
        target_nodes = graph.edge_index[1].numpy()
        layer_node_values = cpu_layer_nodes.numpy()
        node_to_layer_position = {
            int(node): position
            for position, node in enumerate(layer_node_values)
        }

        device_layer_records: list[dict[str, Any]] = []
        for position, (node_id, layer) in enumerate(
            zip(layer_node_values, device_layers.to_dict("records"), strict=True)
        ):
            role_id = int(graph.layer_role[node_id])
            role = role_vocabulary[role_id]
            table_role = str(layer["layer_role"])
            if role.casefold() != table_role.casefold():
                raise RuntimeError(
                    f"{device_id}: graph role {role!r} does not match "
                    f"prepared role {table_role!r}"
                )
            thickness = pd.to_numeric(
                pd.Series([layer["thickness_value"]]),
                errors="coerce",
            ).iloc[0]
            thickness_value = (
                float(thickness) if pd.notna(thickness) else float("nan")
            )
            grad_z = float(numeric_gradient[node_id, 1])
            gradient_per_nm = (
                grad_z / (thickness_std * (1.0 + thickness_value))
                if np.isfinite(thickness_value)
                else float("nan")
            )
            record = {
                "record_id": len(layer_records) + len(device_layer_records),
                "graph_row": graph_row,
                "node_id": int(node_id),
                "device_id": device_id,
                "paper_id": str(device_row["paper_id"]),
                "doi": str(device_row["doi"]),
                "title": str(device_row["title"]),
                "journal": str(device_row["journal"]),
                "device_label": str(device_row["device_label"]),
                "primary_mechanism": str(device_row["primary_mechanism"]),
                "color_group": str(device_row["color_group"]),
                "fabrication_method": str(device_row["fabrication_method"]),
                "observed_eqe": float(device_row["eqe_max"]),
                "predicted_eqe": prediction,
                "layer_position": position,
                "layer_index": layer["layer_index"],
                "layer_name": layer["layer_name"],
                "layer_role": table_role,
                "thickness_nm": thickness_value,
                "thickness_gradient_z": grad_z,
                "d_predicted_eqe_d_thickness_nm": gradient_per_nm,
                "linearized_delta_eqe_plus_10pct": (
                    gradient_per_nm * thickness_value * 0.1
                    if np.isfinite(gradient_per_nm)
                    else float("nan")
                ),
            }
            device_layer_records.append(record)

        for block_index, block in enumerate(explained.attention, start=1):
            returned_edges = block.edge_index.detach().cpu()
            if not torch.equal(
                returned_edges[:, :original_edge_count],
                graph.edge_index,
            ):
                raise RuntimeError(
                    f"{device_id}: GATv2 returned reordered original edges"
                )
            coefficients = (
                block.coefficients[:original_edge_count].detach().cpu().numpy()
            )
            coefficient_gradient_tensor = block.coefficients.grad
            if coefficient_gradient_tensor is None:
                raise RuntimeError("Attention gradients were not retained")
            coefficient_gradients = (
                coefficient_gradient_tensor[:original_edge_count]
                .detach()
                .cpu()
                .numpy()
            )

            layer_to_root = np.flatnonzero(
                edge_type == EDGE_TYPE_TO_ID["layer_to_root"]
            )
            layer_alpha = coefficients[layer_to_root]
            layer_attention_gradient = np.abs(
                coefficients[layer_to_root]
                * coefficient_gradients[layer_to_root]
            )
            alpha_denominator = np.maximum(
                layer_alpha.sum(axis=0, keepdims=True),
                1e-12,
            )
            attention_share = layer_alpha / alpha_denominator
            gradient_denominator = np.maximum(
                layer_attention_gradient.sum(axis=0, keepdims=True),
                1e-12,
            )
            attention_gradient_share = (
                layer_attention_gradient / gradient_denominator
            )
            signed_attention_gradient = (
                coefficients[layer_to_root]
                * coefficient_gradients[layer_to_root]
            )
            source_to_edge = {
                int(source_nodes[edge_position]): local_position
                for local_position, edge_position in enumerate(layer_to_root)
            }
            for node_id, record in zip(
                layer_node_values,
                device_layer_records,
                strict=True,
            ):
                local_position = source_to_edge[int(node_id)]
                record[f"attention_share_block_{block_index}"] = float(
                    attention_share[local_position].mean()
                )
                record[
                    f"attention_gradient_share_block_{block_index}"
                ] = float(attention_gradient_share[local_position].mean())
                record[
                    f"signed_attention_gradient_block_{block_index}"
                ] = float(signed_attention_gradient[local_position].mean())

            stability_records.append(
                {
                    "device_id": device_id,
                    "paper_id": str(device_row["paper_id"]),
                    "block": block_index,
                    "layer_count": len(layer_to_root),
                    "head_pair_spearman": _head_pair_correlation(
                        attention_share
                    ),
                    "normalized_attention_entropy": float(
                        np.mean(
                            [
                                _normalized_entropy(attention_share[:, head])
                                for head in range(attention_share.shape[1])
                            ]
                        )
                    ),
                }
            )

            for edge_type_id, edge_name in enumerate(EDGE_TYPES):
                positions = np.flatnonzero(edge_type == edge_type_id)
                if positions.size == 0:
                    continue
                edge_records.append(
                    {
                        "device_id": device_id,
                        "paper_id": str(device_row["paper_id"]),
                        "block": block_index,
                        "edge_type": edge_name,
                        "edge_count": int(positions.size),
                        "mean_attention": float(
                            coefficients[positions].mean()
                        ),
                        "mean_abs_attention_gradient": float(
                            np.abs(
                                coefficients[positions]
                                * coefficient_gradients[positions]
                            ).mean()
                        ),
                    }
                )

            interface_positions = np.flatnonzero(
                np.isin(
                    edge_type,
                    [
                        EDGE_TYPE_TO_ID["next_layer"],
                        EDGE_TYPE_TO_ID["previous_layer"],
                    ],
                )
            )
            for edge_position in interface_positions:
                source = int(source_nodes[edge_position])
                target = int(target_nodes[edge_position])
                if (
                    source not in node_to_layer_position
                    or target not in node_to_layer_position
                ):
                    continue
                source_role = device_layer_records[
                    node_to_layer_position[source]
                ]["layer_role"]
                target_role = device_layer_records[
                    node_to_layer_position[target]
                ]["layer_role"]
                interface_records.append(
                    {
                        "device_id": device_id,
                        "paper_id": str(device_row["paper_id"]),
                        "block": block_index,
                        "source_role": source_role,
                        "target_role": target_role,
                        "edge_type": EDGE_TYPES[edge_type[edge_position]],
                        "attention": float(
                            coefficients[edge_position].mean()
                        ),
                        "abs_attention_gradient": float(
                            np.abs(
                                coefficients[edge_position]
                                * coefficient_gradients[edge_position]
                            ).mean()
                        ),
                    }
                )

        for record in device_layer_records:
            attention_columns = [
                record[f"attention_share_block_{block}"]
                for block in range(1, len(explained.attention) + 1)
            ]
            gradient_columns = [
                record[f"attention_gradient_share_block_{block}"]
                for block in range(1, len(explained.attention) + 1)
            ]
            signed_columns = [
                record[f"signed_attention_gradient_block_{block}"]
                for block in range(1, len(explained.attention) + 1)
            ]
            record["attention_share_mean"] = float(np.mean(attention_columns))
            record["attention_gradient_share_mean"] = float(
                np.mean(gradient_columns)
            )
            record["signed_attention_gradient_mean"] = float(
                np.mean(signed_columns)
            )
        layer_records.extend(device_layer_records)
        if sequence == 1 or sequence % 50 == 0 or sequence == len(test_rows):
            print(
                f"attention {sequence}/{len(test_rows)} "
                f"device={device_id} layers={len(device_layer_records)}"
            )

    return (
        pd.DataFrame(layer_records),
        pd.DataFrame(edge_records),
        pd.DataFrame(interface_records),
        pd.DataFrame(stability_records),
    )


@torch.no_grad()
def predict_variant_tasks(
    model: torch.nn.Module,
    vocabulary: dict[str, Any],
    tasks: Iterable[VariantTask],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[tuple[int, str], float]:
    predictions: dict[tuple[int, str], float] = {}
    buffer: list[VariantTask] = []

    def flush() -> None:
        if not buffer:
            return
        batch = Batch.from_data_list([task.graph for task in buffer]).to(device)
        raw = denormalize_prediction(model(batch)[:, 0], vocabulary)
        values = raw.detach().cpu().numpy()
        for task, value in zip(buffer, values, strict=True):
            predictions[(task.record_id, task.variant)] = float(value)
        buffer.clear()

    for task in tasks:
        buffer.append(task)
        if len(buffer) >= batch_size:
            flush()
    flush()
    return predictions


def _training_thickness_bounds(
    manifest: pd.DataFrame,
    layers: pd.DataFrame,
) -> dict[str, tuple[float, float, int]]:
    train_ids = set(
        manifest.loc[manifest["split"].eq("train"), "id"].astype(str)
    )
    values = layers.loc[layers["device_id"].isin(train_ids)].copy()
    values["thickness_value"] = pd.to_numeric(
        values["thickness_value"],
        errors="coerce",
    )
    values = values.loc[
        values["thickness_value"].notna()
        & values["thickness_value"].gt(0)
    ]
    bounds: dict[str, tuple[float, float, int]] = {}
    for role, group in values.groupby("layer_role"):
        role_values = group["thickness_value"].to_numpy(dtype=float)
        bounds[str(role)] = (
            float(np.quantile(role_values, 0.01)),
            float(np.quantile(role_values, 0.99)),
            int(len(role_values)),
        )
    return bounds


def _thickness_variants(
    layer_attributions: pd.DataFrame,
    graphs: list[Data],
    vocabulary: dict[str, Any],
    bounds: dict[str, tuple[float, float, int]],
) -> Iterator[VariantTask]:
    stats = vocabulary["numeric_stats"]
    thickness_mean = float(stats["thickness_mean"])
    thickness_std = float(stats["thickness_std"])
    for row in layer_attributions.itertuples(index=False):
        thickness = float(row.thickness_nm)
        if not np.isfinite(thickness) or thickness <= 0:
            continue
        role_bounds = bounds.get(str(row.layer_role))
        if role_bounds is None or role_bounds[2] < 20:
            continue
        lower_bound, upper_bound, _ = role_bounds
        low = thickness * 0.9
        high = thickness * 1.1
        if low < lower_bound or high > upper_bound:
            continue
        for variant, value in (("thickness_minus_10pct", low), ("thickness_plus_10pct", high)):
            graph = graphs[int(row.graph_row)].clone()
            graph.node_numeric[int(row.node_id), 1] = (
                math.log1p(value) - thickness_mean
            ) / thickness_std
            yield VariantTask(
                record_id=int(row.record_id),
                variant=variant,
                graph=graph,
            )


def _root_link_ablation_variants(
    layer_attributions: pd.DataFrame,
    graphs: list[Data],
) -> Iterator[VariantTask]:
    for row in layer_attributions.itertuples(index=False):
        graph = graphs[int(row.graph_row)].clone()
        node_id = int(row.node_id)
        source = graph.edge_index[0]
        target = graph.edge_index[1]
        root_link = ((source == 0) & (target == node_id)) | (
            (source == node_id) & (target == 0)
        )
        graph.edge_index = graph.edge_index[:, ~root_link]
        graph.edge_type = graph.edge_type[~root_link]
        yield VariantTask(
            record_id=int(row.record_id),
            variant="root_link_ablation",
            graph=graph,
        )


def add_counterfactuals(
    layer_attributions: pd.DataFrame,
    model: torch.nn.Module,
    graphs: list[Data],
    vocabulary: dict[str, Any],
    manifest: pd.DataFrame,
    layers: pd.DataFrame,
    *,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    bounds = _training_thickness_bounds(manifest, layers)
    thickness_predictions = predict_variant_tasks(
        model,
        vocabulary,
        _thickness_variants(
            layer_attributions,
            graphs,
            vocabulary,
            bounds,
        ),
        device=device,
        batch_size=batch_size,
    )
    print(f"thickness counterfactual predictions={len(thickness_predictions)}")
    root_predictions = predict_variant_tasks(
        model,
        vocabulary,
        _root_link_ablation_variants(layer_attributions, graphs),
        device=device,
        batch_size=batch_size,
    )
    print(f"root-link ablation predictions={len(root_predictions)}")

    result = layer_attributions.copy()
    minus_values: list[float] = []
    plus_values: list[float] = []
    root_values: list[float] = []
    for row in result.itertuples(index=False):
        record_id = int(row.record_id)
        minus_values.append(
            thickness_predictions.get(
                (record_id, "thickness_minus_10pct"),
                float("nan"),
            )
        )
        plus_values.append(
            thickness_predictions.get(
                (record_id, "thickness_plus_10pct"),
                float("nan"),
            )
        )
        root_values.append(
            root_predictions[(record_id, "root_link_ablation")]
        )
    result["predicted_eqe_thickness_minus_10pct"] = minus_values
    result["predicted_eqe_thickness_plus_10pct"] = plus_values
    result["delta_eqe_thickness_minus_10pct"] = (
        result["predicted_eqe_thickness_minus_10pct"]
        - result["predicted_eqe"]
    )
    result["delta_eqe_thickness_plus_10pct"] = (
        result["predicted_eqe_thickness_plus_10pct"]
        - result["predicted_eqe"]
    )
    result["central_delta_eqe_per_10pct_thickness"] = (
        result["predicted_eqe_thickness_plus_10pct"]
        - result["predicted_eqe_thickness_minus_10pct"]
    ) / 2.0
    result["absolute_delta_eqe_per_10pct_thickness"] = result[
        "central_delta_eqe_per_10pct_thickness"
    ].abs()
    result["predicted_eqe_root_link_ablation"] = root_values
    result["delta_eqe_root_link_ablation"] = (
        result["predicted_eqe_root_link_ablation"]
        - result["predicted_eqe"]
    )
    result["absolute_delta_eqe_root_link_ablation"] = result[
        "delta_eqe_root_link_ablation"
    ].abs()
    return result


def _cluster_bootstrap_interval(
    frame: pd.DataFrame,
    value_column: str,
    *,
    repeats: int,
    seed: int,
) -> tuple[float, float, float]:
    paper_values = (
        frame.groupby("paper_id", sort=False)[value_column]
        .mean()
        .dropna()
        .to_numpy(dtype=float)
    )
    if paper_values.size == 0:
        return float("nan"), float("nan"), float("nan")
    estimate = float(np.median(paper_values))
    if paper_values.size < 2:
        return estimate, float("nan"), float("nan")
    generator = np.random.default_rng(seed)
    sampled = generator.choice(
        paper_values,
        size=(repeats, len(paper_values)),
        replace=True,
    )
    bootstrap = np.median(sampled, axis=1)
    return (
        estimate,
        float(np.quantile(bootstrap, 0.025)),
        float(np.quantile(bootstrap, 0.975)),
    )


def summarize_roles(
    layer_attributions: pd.DataFrame,
    *,
    bootstrap_repeats: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for role, group in layer_attributions.groupby("layer_role", sort=False):
        row: dict[str, Any] = {
            "layer_role": role,
            "layer_count": int(len(group)),
            "device_count": int(group["device_id"].nunique()),
            "paper_count": int(group["paper_id"].nunique()),
            "known_thickness_count": int(group["thickness_nm"].notna().sum()),
            "counterfactual_thickness_count": int(
                group["absolute_delta_eqe_per_10pct_thickness"].notna().sum()
            ),
            "mean_attention_share": float(group["attention_share_mean"].mean()),
            "median_attention_share": float(
                group["attention_share_mean"].median()
            ),
            "mean_attention_gradient_share": float(
                group["attention_gradient_share_mean"].mean()
            ),
            "median_attention_gradient_share": float(
                group["attention_gradient_share_mean"].median()
            ),
            "median_abs_root_link_ablation_eqe": float(
                group["absolute_delta_eqe_root_link_ablation"].median()
            ),
            "median_signed_thickness_delta_10pct_eqe": float(
                group["central_delta_eqe_per_10pct_thickness"].median()
            ),
            "median_abs_thickness_delta_10pct_eqe": float(
                group["absolute_delta_eqe_per_10pct_thickness"].median()
            ),
        }
        for value_column, prefix in (
            ("attention_share_mean", "attention_share"),
            ("attention_gradient_share_mean", "attention_gradient_share"),
            (
                "absolute_delta_eqe_per_10pct_thickness",
                "abs_thickness_delta_10pct_eqe",
            ),
            (
                "central_delta_eqe_per_10pct_thickness",
                "signed_thickness_delta_10pct_eqe",
            ),
        ):
            estimate, lower, upper = _cluster_bootstrap_interval(
                group,
                value_column,
                repeats=bootstrap_repeats,
                seed=seed,
            )
            row[f"{prefix}_paper_median"] = estimate
            row[f"{prefix}_ci_lower"] = lower
            row[f"{prefix}_ci_upper"] = upper
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["layer_count", "layer_role"],
        ascending=[False, True],
    )


def summarize_edge_types(edge_attributions: pd.DataFrame) -> pd.DataFrame:
    return (
        edge_attributions.groupby(["block", "edge_type"], as_index=False)
        .agg(
            device_count=("device_id", "nunique"),
            paper_count=("paper_id", "nunique"),
            edge_count=("edge_count", "sum"),
            mean_attention=("mean_attention", "mean"),
            mean_abs_attention_gradient=(
                "mean_abs_attention_gradient",
                "mean",
            ),
        )
        .sort_values(["block", "mean_abs_attention_gradient"], ascending=[True, False])
    )


def summarize_interfaces(interface_attributions: pd.DataFrame) -> pd.DataFrame:
    return (
        interface_attributions.groupby(
            ["block", "source_role", "target_role"],
            as_index=False,
        )
        .agg(
            device_count=("device_id", "nunique"),
            paper_count=("paper_id", "nunique"),
            edge_count=("attention", "size"),
            mean_attention=("attention", "mean"),
            mean_abs_attention_gradient=("abs_attention_gradient", "mean"),
        )
        .sort_values(
            ["block", "mean_abs_attention_gradient"],
            ascending=[True, False],
        )
    )


def calculate_faithfulness(
    layer_attributions: pd.DataFrame,
    stability: pd.DataFrame,
) -> dict[str, Any]:
    attention = layer_attributions["attention_share_mean"].to_numpy()
    attention_gradient = layer_attributions[
        "attention_gradient_share_mean"
    ].to_numpy()
    root_effect = layer_attributions[
        "absolute_delta_eqe_root_link_ablation"
    ].to_numpy()
    within_device_attention: list[float] = []
    within_device_attention_gradient: list[float] = []
    for _, group in layer_attributions.groupby("device_id", sort=False):
        within_device_attention.append(
            _safe_spearman(
                group["attention_share_mean"].to_numpy(),
                group["absolute_delta_eqe_root_link_ablation"].to_numpy(),
            )
        )
        within_device_attention_gradient.append(
            _safe_spearman(
                group["attention_gradient_share_mean"].to_numpy(),
                group["absolute_delta_eqe_root_link_ablation"].to_numpy(),
            )
        )
    within_attention = np.asarray(within_device_attention)
    within_attention_gradient = np.asarray(within_device_attention_gradient)
    finite_attention = within_attention[np.isfinite(within_attention)]
    finite_attention_gradient = within_attention_gradient[
        np.isfinite(within_attention_gradient)
    ]

    thickness = layer_attributions.dropna(
        subset=[
            "linearized_delta_eqe_plus_10pct",
            "central_delta_eqe_per_10pct_thickness",
        ]
    )
    stability_summary = (
        stability.groupby("block")
        .agg(
            devices=("device_id", "nunique"),
            median_head_pair_spearman=("head_pair_spearman", "median"),
            q25_head_pair_spearman=(
                "head_pair_spearman",
                lambda values: values.quantile(0.25),
            ),
            q75_head_pair_spearman=(
                "head_pair_spearman",
                lambda values: values.quantile(0.75),
            ),
            median_normalized_entropy=(
                "normalized_attention_entropy",
                "median",
            ),
        )
        .reset_index()
        .to_dict("records")
    )
    return {
        "pooled_layer_level": {
            "attention_vs_abs_root_link_ablation_spearman": _safe_spearman(
                attention,
                root_effect,
            ),
            "attention_gradient_vs_abs_root_link_ablation_spearman": _safe_spearman(
                attention_gradient,
                root_effect,
            ),
        },
        "within_device": {
            "attention_vs_abs_root_link_ablation": {
                "device_count": int(finite_attention.size),
                "median_spearman": (
                    float(np.median(finite_attention))
                    if finite_attention.size
                    else float("nan")
                ),
                "q25": (
                    float(np.quantile(finite_attention, 0.25))
                    if finite_attention.size
                    else float("nan")
                ),
                "q75": (
                    float(np.quantile(finite_attention, 0.75))
                    if finite_attention.size
                    else float("nan")
                ),
            },
            "attention_gradient_vs_abs_root_link_ablation": {
                "device_count": int(finite_attention_gradient.size),
                "median_spearman": (
                    float(np.median(finite_attention_gradient))
                    if finite_attention_gradient.size
                    else float("nan")
                ),
                "q25": (
                    float(np.quantile(finite_attention_gradient, 0.25))
                    if finite_attention_gradient.size
                    else float("nan")
                ),
                "q75": (
                    float(np.quantile(finite_attention_gradient, 0.75))
                    if finite_attention_gradient.size
                    else float("nan")
                ),
            },
        },
        "gradient_counterfactual_consistency": {
            "layer_count": int(len(thickness)),
            "spearman": _safe_spearman(
                thickness["linearized_delta_eqe_plus_10pct"].to_numpy(),
                thickness[
                    "central_delta_eqe_per_10pct_thickness"
                ].to_numpy(),
            ),
            "median_absolute_error_eqe": float(
                (
                    thickness["linearized_delta_eqe_plus_10pct"]
                    - thickness["central_delta_eqe_per_10pct_thickness"]
                )
                .abs()
                .median()
            ),
        },
        "attention_head_stability": stability_summary,
    }


def plot_attention_heatmap(
    layer_attributions: pd.DataFrame,
    output_dir: Path,
) -> None:
    common_roles = (
        layer_attributions["layer_role"].value_counts().loc[lambda values: values >= 10]
    )
    role_order = [
        role for role in PRIMARY_LAYER_ROLES if role in common_roles.index
    ]
    role_order.extend(
        role
        for role in common_roles.index
        if role not in role_order
    )
    block_columns = [
        column
        for column in layer_attributions.columns
        if column.startswith("attention_gradient_share_block_")
    ]
    aggregated = (
        layer_attributions.loc[
            layer_attributions["layer_role"].isin(role_order)
        ]
        .groupby("layer_role")[block_columns]
        .median()
        .reindex(role_order)
    )
    aggregated.columns = [
        f"Block {column.rsplit('_', 1)[-1]}" for column in block_columns
    ]
    figure, axis = plt.subplots(
        figsize=(8.4, max(5.0, 0.42 * len(role_order) + 1.6))
    )
    heatmap_axis = sns.heatmap(
        aggregated * 100.0,
        cmap="YlGnBu",
        annot=True,
        fmt=".1f",
        linewidths=0.6,
        linecolor=COLORS["white"],
        cbar_kws={"label": "Median attention-gradient share (%)"},
        ax=axis,
    )
    axis.set(
        title="Output-conditioned layer attention across GATv2 blocks",
        xlabel="Message-passing block",
        ylabel="Layer role",
    )
    colorbar = heatmap_axis.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=9)
    colorbar.set_label(
        "Median attention-gradient share (%)",
        fontsize=10,
    )
    figure.tight_layout()
    save_figure(figure, output_dir / "attention_gradient_by_role_and_block")
    plt.close(figure)


def plot_role_summary(
    role_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    common = role_summary.loc[role_summary["layer_count"].ge(20)].copy()
    order = (
        common.sort_values(
            "attention_gradient_share_paper_median",
            ascending=True,
        )["layer_role"]
        .tolist()
    )
    common = common.set_index("layer_role").loc[order].reset_index()
    y = np.arange(len(common))
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.4, max(6.2, 0.48 * len(common) + 1.8)),
    )
    attention = common["attention_gradient_share_paper_median"] * 100.0
    attention_lower = (
        common["attention_gradient_share_paper_median"]
        - common["attention_gradient_share_ci_lower"]
    ) * 100.0
    attention_upper = (
        common["attention_gradient_share_ci_upper"]
        - common["attention_gradient_share_paper_median"]
    ) * 100.0
    axes[0].barh(y, attention, color=COLORS["teal"], alpha=0.9)
    axes[0].errorbar(
        attention,
        y,
        xerr=np.vstack([attention_lower, attention_upper]),
        fmt="none",
        ecolor=COLORS["ink"],
        capsize=3,
        linewidth=1,
    )
    axes[0].set(
        yticks=y,
        yticklabels=common["layer_role"],
        xlabel="Output-conditioned attention share (%)",
        title="Where the model routes information",
    )

    sensitivity = (
        common["abs_thickness_delta_10pct_eqe_paper_median"] * 1000.0
    )
    sensitivity_lower = (
        common["abs_thickness_delta_10pct_eqe_paper_median"]
        - common["abs_thickness_delta_10pct_eqe_ci_lower"]
    ) * 1000.0
    sensitivity_upper = (
        common["abs_thickness_delta_10pct_eqe_ci_upper"]
        - common["abs_thickness_delta_10pct_eqe_paper_median"]
    ) * 1000.0
    axes[1].barh(y, sensitivity, color=COLORS["amber"], alpha=0.9)
    axes[1].errorbar(
        sensitivity,
        y,
        xerr=np.vstack([sensitivity_lower, sensitivity_upper]),
        fmt="none",
        ecolor=COLORS["ink"],
        capsize=3,
        linewidth=1,
    )
    axes[1].set(
        yticks=y,
        yticklabels=[],
        xlabel="Median |delta EQE| for +/-10% thickness (x 1e-3)",
        title="How sensitive the prediction is",
    )
    for axis in axes:
        axis.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
        axis.grid(axis="y", visible=False)
        axis.tick_params(labelsize=10)
        axis.xaxis.label.set_size(11)
        axis.title.set_size(13)
    figure.suptitle(
        "Layer-role importance requires attention and intervention",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(figure, output_dir / "layer_role_attention_and_thickness")
    plt.close(figure)


def plot_attention_faithfulness(
    layer_attributions: pd.DataFrame,
    faithfulness: dict[str, Any],
    output_dir: Path,
) -> None:
    frame = layer_attributions.copy()
    role_counts = frame["layer_role"].value_counts()
    common_roles = role_counts.loc[lambda values: values >= 20].index
    frame["plot_role"] = frame["layer_role"].where(
        frame["layer_role"].isin(common_roles),
        "other",
    )
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.5))
    sns.scatterplot(
        data=frame,
        x="attention_share_mean",
        y="absolute_delta_eqe_root_link_ablation",
        hue="plot_role",
        s=25,
        alpha=0.55,
        linewidth=0,
        ax=axes[0],
    )
    sns.scatterplot(
        data=frame,
        x="attention_gradient_share_mean",
        y="absolute_delta_eqe_root_link_ablation",
        hue="plot_role",
        s=25,
        alpha=0.55,
        linewidth=0,
        legend=False,
        ax=axes[1],
    )
    pooled = faithfulness["pooled_layer_level"]
    axes[0].set_title(
        "Raw attention "
        f"(Spearman r={pooled['attention_vs_abs_root_link_ablation_spearman']:.2f})"
    )
    axes[1].set_title(
        "Attention x gradient "
        f"(Spearman r={pooled['attention_gradient_vs_abs_root_link_ablation_spearman']:.2f})"
    )
    axes[0].set_xlabel("Mean layer-to-root attention share")
    axes[1].set_xlabel("Mean output-conditioned attention share")
    axes[0].set_ylabel("|delta predicted EQE| after root-link ablation")
    axes[1].set_ylabel("")
    axes[0].legend(
        title="Layer role",
        fontsize=7.5,
        title_fontsize=8.5,
        ncol=2,
        frameon=False,
    )
    for axis in axes:
        axis.set_yscale("log")
        axis.grid(color=COLORS["grid"], linewidth=0.7)
        axis.tick_params(labelsize=10)
        axis.xaxis.label.set_size(11)
        axis.yaxis.label.set_size(11)
        axis.title.set_size(13)
    figure.suptitle(
        "Attention faithfulness check",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(figure, output_dir / "attention_faithfulness")
    plt.close(figure)


def plot_signed_thickness_effects(
    layer_attributions: pd.DataFrame,
    output_dir: Path,
) -> None:
    frame = layer_attributions.dropna(
        subset=["central_delta_eqe_per_10pct_thickness"]
    ).copy()
    counts = frame["layer_role"].value_counts()
    roles = counts.loc[lambda values: values >= 20].index
    frame = frame.loc[frame["layer_role"].isin(roles)]
    role_medians = (
        frame.groupby("layer_role")["central_delta_eqe_per_10pct_thickness"]
        .median()
        .sort_values()
    )
    order = role_medians.index.tolist()
    figure, axis = plt.subplots(
        figsize=(9.4, max(5.6, 0.46 * len(order) + 1.8))
    )
    sns.boxplot(
        data=frame,
        x="central_delta_eqe_per_10pct_thickness",
        y="layer_role",
        order=order,
        color="#dcecee",
        showfliers=False,
        linewidth=1.0,
        ax=axis,
    )
    axis.axvline(0.0, color=COLORS["coral"], linestyle="--", linewidth=1.4)
    axis.set(
        title="Context-dependent signed thickness sensitivity",
        xlabel="Predicted EQE change for a local +10% thickness perturbation",
        ylabel="Layer role",
    )
    axis.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
    axis.grid(axis="y", visible=False)
    figure.tight_layout()
    save_figure(figure, output_dir / "signed_thickness_effect_by_role")
    plt.close(figure)


def plot_interface_summary(
    interface_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    common = interface_summary.loc[interface_summary["edge_count"].ge(30)].copy()
    common["interface"] = (
        common["source_role"] + " -> " + common["target_role"]
    )
    ranked = (
        common.groupby("interface")["mean_abs_attention_gradient"]
        .mean()
        .nlargest(12)
        .index
    )
    selected = common.loc[common["interface"].isin(ranked)]
    matrix = selected.pivot_table(
        index="interface",
        columns="block",
        values="mean_abs_attention_gradient",
        aggfunc="mean",
    )
    matrix = matrix.loc[
        matrix.mean(axis=1).sort_values(ascending=False).index
    ]
    matrix.columns = [f"Block {int(block)}" for block in matrix.columns]
    figure, axis = plt.subplots(
        figsize=(8.8, max(5.8, 0.43 * len(matrix) + 1.7))
    )
    heatmap_axis = sns.heatmap(
        matrix * 1000.0,
        cmap="mako",
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        linecolor=COLORS["white"],
        cbar_kws={"label": "Mean |attention x gradient| (x 1e-3 EQE)"},
        ax=axis,
    )
    axis.set(
        title="Most influential directed layer-interface messages",
        xlabel="Message-passing block",
        ylabel="Message direction",
    )
    colorbar = heatmap_axis.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=9)
    colorbar.set_label(
        "Mean |attention x gradient| (x 1e-3 EQE)",
        fontsize=10,
    )
    figure.tight_layout()
    save_figure(figure, output_dir / "interface_attention_by_block")
    plt.close(figure)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    checkpoint_path = (
        args.checkpoint.resolve()
        if args.checkpoint is not None
        else output_root / "gat" / "best_model.pt"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else output_root / "explainability"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    configure_plotting()

    if args.plots_only:
        layer_attributions = pd.read_csv(
            output_dir / "layer_attributions.csv"
        )
        role_summary = pd.read_csv(output_dir / "layer_role_summary.csv")
        summary = json.loads(
            (output_dir / "explainability_summary.json").read_text(
                encoding="utf-8"
            )
        )
        plot_attention_heatmap(layer_attributions, output_dir)
        plot_role_summary(role_summary, output_dir)
        plot_attention_faithfulness(
            layer_attributions,
            summary["faithfulness"],
            output_dir,
        )
        plot_signed_thickness_effects(layer_attributions, output_dir)
        interface_summary = pd.read_csv(
            output_dir / "interface_summary.csv"
        )
        plot_interface_summary(interface_summary, output_dir)
        print(
            json.dumps(
                {
                    "output_dir": str(output_dir),
                    "mode": "plots_only",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    manifest = pd.read_parquet(output_root / "prepared" / "manifest.parquet")
    layers = pd.read_parquet(output_root / "prepared" / "layers.parquet")
    graphs, vocabulary = load_graph_cache(output_root)
    model, checkpoint = load_frozen_model(
        checkpoint_path,
        device=device,
    )
    layer_attributions, edge_attributions, interfaces, stability = (
        extract_test_attributions(
            model,
            graphs,
            vocabulary,
            manifest,
            layers,
            device=device,
        )
    )
    layer_attributions = add_counterfactuals(
        layer_attributions,
        model,
        graphs,
        vocabulary,
        manifest,
        layers,
        device=device,
        batch_size=args.batch_size,
    )
    role_summary = summarize_roles(
        layer_attributions,
        bootstrap_repeats=args.bootstrap_repeats,
        seed=args.seed,
    )
    edge_summary = summarize_edge_types(edge_attributions)
    interface_summary = summarize_interfaces(interfaces)
    faithfulness = calculate_faithfulness(layer_attributions, stability)
    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "test_device_count": int(
            manifest.loc[manifest["split"].eq("test"), "id"].nunique()
        ),
        "test_paper_count": int(
            manifest.loc[manifest["split"].eq("test"), "paper_id"].nunique()
        ),
        "layer_record_count": int(len(layer_attributions)),
        "known_thickness_count": int(
            layer_attributions["thickness_nm"].notna().sum()
        ),
        "counterfactual_thickness_count": int(
            layer_attributions[
                "central_delta_eqe_per_10pct_thickness"
            ].notna().sum()
        ),
        "model_config": checkpoint["model_config"],
        "faithfulness": faithfulness,
    }

    layer_attributions.to_csv(
        output_dir / "layer_attributions.csv",
        index=False,
    )
    edge_attributions.to_csv(
        output_dir / "edge_attributions.csv",
        index=False,
    )
    interfaces.to_csv(
        output_dir / "interface_attributions.csv",
        index=False,
    )
    stability.to_csv(
        output_dir / "attention_head_stability.csv",
        index=False,
    )
    role_summary.to_csv(
        output_dir / "layer_role_summary.csv",
        index=False,
    )
    edge_summary.to_csv(
        output_dir / "edge_type_summary.csv",
        index=False,
    )
    interface_summary.to_csv(
        output_dir / "interface_summary.csv",
        index=False,
    )
    write_json(output_dir / "explainability_summary.json", summary)

    plot_attention_heatmap(layer_attributions, output_dir)
    plot_role_summary(role_summary, output_dir)
    plot_attention_faithfulness(
        layer_attributions,
        faithfulness,
        output_dir,
    )
    plot_signed_thickness_effects(layer_attributions, output_dir)
    plot_interface_summary(interface_summary, output_dir)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "test_devices": summary["test_device_count"],
                "layer_records": summary["layer_record_count"],
                "thickness_counterfactuals": summary[
                    "counterfactual_thickness_count"
                ],
                "faithfulness": faithfulness,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=True,
        )
    )


if __name__ == "__main__":
    main()
