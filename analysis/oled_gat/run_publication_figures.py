from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PROJECT_DIR.parent.parent

COLORS = {
    "ink": "#17323A",
    "muted": "#5F747A",
    "grid": "#DCE6E7",
    "teal": "#167D86",
    "teal_light": "#D9EEF0",
    "blue": "#3973A5",
    "blue_light": "#DCEAF5",
    "coral": "#C95D3A",
    "coral_light": "#F7E1D9",
    "gold": "#D49A25",
    "gold_light": "#F8EDCF",
    "green": "#428A5B",
    "green_light": "#DDEEDF",
    "gray_light": "#EEF2F2",
    "white": "#FFFFFF",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw publication figures for the frozen OLED-GAT model."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PROJECT_DIR
        / "outputs_device_random"
        / "final"
        / "test_predictions.csv",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=PROJECT_DIR
        / "outputs_device_random"
        / "final"
        / "test_metrics.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "outputs_device_random" / "final" / "figures",
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "axes.edgecolor": COLORS["ink"],
            "axes.linewidth": 1.0,
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "text.color": COLORS["ink"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def save_figure(figure: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")


def _density(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    values = np.vstack([observed, predicted])
    try:
        return gaussian_kde(values)(values)
    except np.linalg.LinAlgError:
        return np.ones_like(observed)


def _regression_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    return {
        "r2": float(r2_score(observed, predicted)),
        "mae": float(mean_absolute_error(observed, predicted)),
        "rmse": float(root_mean_squared_error(observed, predicted)),
    }


def draw_regression_figure(
    predictions: pd.DataFrame,
    output_dir: Path,
) -> dict[str, dict[str, float]]:
    observed = predictions["eqe_max"].to_numpy(dtype=float)
    panels = [
        ("OLED-GAT", predictions["mean_gat"].to_numpy(dtype=float)),
        (
            "OLED-GAT + CatBoost ensemble",
            predictions["mean_ensemble"].to_numpy(dtype=float),
        ),
    ]
    densities = [_density(observed, predicted) for _, predicted in panels]
    normalization = Normalize(
        vmin=min(float(values.min()) for values in densities),
        vmax=max(float(values.max()) for values in densities),
    )
    maximum = max(
        float(observed.max()),
        *(float(predicted.max()) for _, predicted in panels),
    )
    axis_maximum = max(45.0, math.ceil(maximum / 5.0) * 5.0)

    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.55), sharex=True, sharey=True)
    results: dict[str, dict[str, float]] = {}
    scatter = None
    for axis, (name, predicted), density in zip(
        axes,
        panels,
        densities,
        strict=True,
    ):
        order = np.argsort(density)
        scatter = axis.scatter(
            observed[order],
            predicted[order],
            c=density[order],
            cmap="viridis",
            norm=normalization,
            s=31,
            alpha=0.82,
            edgecolors=COLORS["white"],
            linewidths=0.25,
            zorder=3,
        )
        axis.plot(
            [0, axis_maximum],
            [0, axis_maximum],
            color=COLORS["coral"],
            linestyle=(0, (5, 3)),
            linewidth=1.8,
            label="Identity line",
            zorder=2,
        )
        slope, intercept = np.polyfit(observed, predicted, 1)
        fit_x = np.asarray([0.0, axis_maximum])
        axis.plot(
            fit_x,
            slope * fit_x + intercept,
            color=COLORS["blue"],
            linewidth=1.8,
            label="Least-squares fit",
            zorder=2,
        )
        metrics = _regression_metrics(observed, predicted)
        results[name] = metrics
        metric_text = (
            f"$R^2$ = {metrics['r2']:.3f}\n"
            f"MAE = {metrics['mae']:.3f}\n"
            f"RMSE = {metrics['rmse']:.3f}\n"
            f"$n$ = {len(observed)}"
        )
        axis.text(
            0.055,
            0.94,
            metric_text,
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=11,
            linespacing=1.45,
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": COLORS["white"],
                "edgecolor": COLORS["grid"],
                "alpha": 0.94,
            },
        )
        axis.set(
            title=name,
            xlabel="Observed maximum EQE (%)",
            xlim=(0, axis_maximum),
            ylim=(0, axis_maximum),
            aspect="equal",
        )
        axis.grid(color=COLORS["grid"], linewidth=0.7, alpha=0.8)
        axis.set_axisbelow(True)
        axis.legend(loc="upper right", frameon=False, fontsize=8.2)
    axes[0].set_ylabel("Predicted maximum EQE (%)")
    figure.suptitle(
        "Frozen test-set regression",
        fontsize=17,
        fontweight="bold",
        color=COLORS["ink"],
        y=1.01,
    )
    figure.text(
        0.5,
        -0.015,
        "Device-random split; sibling devices from the same paper may occur across splits.",
        ha="center",
        va="top",
        fontsize=9,
        color=COLORS["muted"],
    )
    if scatter is not None:
        colorbar = figure.colorbar(
            scatter,
            ax=axes,
            fraction=0.018,
            pad=0.025,
        )
        colorbar.set_label("Local point density", color=COLORS["muted"])
        colorbar.set_ticks([])
    figure.subplots_adjust(left=0.07, right=0.90, bottom=0.13, top=0.88, wspace=0.18)
    save_figure(figure, output_dir / "frozen_test_regression")
    plt.close(figure)

    gat_metrics = results["OLED-GAT"]
    single, axis = plt.subplots(figsize=(6.45, 5.9))
    gat_prediction = panels[0][1]
    gat_density = densities[0]
    order = np.argsort(gat_density)
    scatter = axis.scatter(
        observed[order],
        gat_prediction[order],
        c=gat_density[order],
        cmap="viridis",
        s=35,
        alpha=0.84,
        edgecolors=COLORS["white"],
        linewidths=0.25,
    )
    axis.plot(
        [0, axis_maximum],
        [0, axis_maximum],
        color=COLORS["coral"],
        linestyle=(0, (5, 3)),
        linewidth=1.9,
    )
    slope, intercept = np.polyfit(observed, gat_prediction, 1)
    fit_x = np.asarray([0.0, axis_maximum])
    axis.plot(
        fit_x,
        slope * fit_x + intercept,
        color=COLORS["blue"],
        linewidth=1.9,
    )
    axis.text(
        0.055,
        0.945,
        (
            f"$R^2$ = {gat_metrics['r2']:.3f}\n"
            f"MAE = {gat_metrics['mae']:.3f}\n"
            f"RMSE = {gat_metrics['rmse']:.3f}\n"
            f"$n$ = {len(observed)}"
        ),
        transform=axis.transAxes,
        va="top",
        fontsize=11.5,
        linespacing=1.5,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": COLORS["white"],
            "edgecolor": COLORS["grid"],
            "alpha": 0.94,
        },
    )
    axis.set(
        title="OLED-GAT: frozen test-set regression",
        xlabel="Observed maximum EQE (%)",
        ylabel="Predicted maximum EQE (%)",
        xlim=(0, axis_maximum),
        ylim=(0, axis_maximum),
        aspect="equal",
    )
    axis.grid(color=COLORS["grid"], linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    colorbar = single.colorbar(scatter, ax=axis, fraction=0.045, pad=0.035)
    colorbar.set_label("Local point density", color=COLORS["muted"])
    colorbar.set_ticks([])
    single.tight_layout()
    save_figure(single, output_dir / "oled_gat_frozen_test_regression")
    plt.close(single)
    return results


def _box(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    *,
    facecolor: str,
    edgecolor: str,
    fontsize: float = 9.0,
    fontweight: str = "normal",
    radius: float = 0.12,
    zorder: int = 3,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.06,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.25,
        zorder=zorder,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        color=COLORS["ink"],
        zorder=zorder + 1,
        linespacing=1.25,
    )
    return patch


def _arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["muted"],
    width: float = 1.35,
    style: str = "-|>",
    connectionstyle: str = "arc3",
    zorder: int = 2,
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=12,
            linewidth=width,
            color=color,
            connectionstyle=connectionstyle,
            zorder=zorder,
        )
    )


def _node(
    axis: plt.Axes,
    x: float,
    y: float,
    radius: float,
    color: str,
    label: str | None = None,
    *,
    fontsize: float = 7.5,
) -> None:
    circle = Circle(
        (x, y),
        radius,
        facecolor=color,
        edgecolor=COLORS["white"],
        linewidth=1.0,
        zorder=5,
    )
    axis.add_patch(circle)
    if label:
        axis.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=COLORS["white"],
            fontweight="bold",
            zorder=6,
        )


def _stage_label(axis: plt.Axes, x: float, width: float, number: str, label: str) -> None:
    axis.text(
        x,
        8.47,
        number,
        fontsize=9.5,
        fontweight="bold",
        color=COLORS["white"],
        ha="center",
        va="center",
        bbox={
            "boxstyle": "circle,pad=0.35",
            "facecolor": COLORS["teal"],
            "edgecolor": COLORS["teal"],
        },
    )
    axis.text(
        x + 0.34,
        8.47,
        label,
        fontsize=11,
        fontweight="bold",
        color=COLORS["ink"],
        ha="left",
        va="center",
    )
    axis.plot(
        [x - 0.18, x + width],
        [8.18, 8.18],
        color=COLORS["grid"],
        linewidth=1.1,
    )


def draw_architecture_figure(output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(16, 8.8))
    axis.set_xlim(0, 16)
    axis.set_ylim(0, 9)
    axis.axis("off")
    figure.patch.set_facecolor(COLORS["white"])
    axis.set_facecolor(COLORS["white"])

    axis.text(
        0.35,
        8.87,
        "OLED-GAT architecture",
        fontsize=20,
        fontweight="bold",
        color=COLORS["ink"],
        ha="left",
        va="top",
    )
    axis.text(
        15.65,
        8.87,
        "Hierarchical device graph + EML molecular graph + quantile regression",
        fontsize=10,
        color=COLORS["muted"],
        ha="right",
        va="top",
    )

    _stage_label(axis, 0.55, 3.0, "1", "Device inputs")
    _stage_label(axis, 4.05, 3.4, "2", "Hierarchical graph")
    _stage_label(axis, 8.05, 3.4, "3", "GATv2 encoder")
    _stage_label(axis, 12.05, 3.45, "4", "Prediction and interval")

    layer_names = ["Cathode", "EIL", "ETL", "HBL", "EML", "HTL", "Anode"]
    layer_colors = [
        COLORS["gray_light"],
        COLORS["blue_light"],
        COLORS["blue_light"],
        COLORS["gold_light"],
        COLORS["coral_light"],
        COLORS["teal_light"],
        COLORS["gray_light"],
    ]
    stack_x = 0.48
    stack_y = 3.25
    stack_width = 1.55
    layer_height = 0.44
    for index, (name, color) in enumerate(zip(layer_names, layer_colors, strict=True)):
        y = stack_y + index * layer_height
        _box(
            axis,
            stack_x,
            y,
            stack_width,
            layer_height - 0.04,
            name,
            facecolor=color,
            edgecolor=COLORS["white"],
            fontsize=7.5,
            radius=0.03,
        )
    axis.text(
        stack_x + stack_width / 2,
        stack_y - 0.33,
        "Ordered multilayer stack",
        ha="center",
        fontsize=8.5,
        color=COLORS["muted"],
    )

    molecule_center = (2.85, 4.72)
    molecule_nodes = [
        (-0.55, 0.00),
        (-0.28, 0.48),
        (0.27, 0.48),
        (0.55, 0.00),
        (0.27, -0.48),
        (-0.28, -0.48),
        (0.92, 0.34),
        (1.23, 0.00),
    ]
    molecule_edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 0),
        (3, 6),
        (6, 7),
    ]
    for source, target in molecule_edges:
        x1, y1 = molecule_nodes[source]
        x2, y2 = molecule_nodes[target]
        axis.plot(
            [molecule_center[0] + x1, molecule_center[0] + x2],
            [molecule_center[1] + y1, molecule_center[1] + y2],
            color=COLORS["ink"],
            linewidth=1.5,
            zorder=2,
        )
    for index, (dx, dy) in enumerate(molecule_nodes):
        color = COLORS["coral"] if index == 7 else COLORS["teal"]
        _node(
            axis,
            molecule_center[0] + dx,
            molecule_center[1] + dy,
            0.115,
            color,
            "N" if index == 7 else None,
        )
    axis.text(
        2.92,
        3.86,
        "EML SMILES\natom/bond graph",
        ha="center",
        va="top",
        fontsize=8.5,
        color=COLORS["muted"],
        linespacing=1.2,
    )
    _box(
        axis,
        0.55,
        1.47,
        3.0,
        0.72,
        "Context: mechanism · color · process · geometry",
        facecolor=COLORS["green_light"],
        edgecolor=COLORS["green"],
        fontsize=8.4,
    )
    _box(
        axis,
        0.55,
        0.53,
        3.0,
        0.64,
        "Thickness · composition ratio · material identity",
        facecolor=COLORS["gray_light"],
        edgecolor=COLORS["muted"],
        fontsize=8.2,
    )

    graph_root = (4.72, 6.75)
    _node(axis, *graph_root, 0.25, COLORS["ink"], "D", fontsize=8.5)
    layer_y = 5.75
    graph_layers_x = [4.25, 4.85, 5.45, 6.05, 6.65]
    for index, x in enumerate(graph_layers_x):
        _arrow(
            axis,
            graph_root,
            (x, layer_y + 0.15),
            color=COLORS["grid"],
            width=1.0,
            style="-",
        )
        color = COLORS["coral"] if index == 2 else COLORS["blue"]
        _node(axis, x, layer_y, 0.19, color, "L")
        if index:
            _arrow(
                axis,
                (graph_layers_x[index - 1] + 0.19, layer_y),
                (x - 0.19, layer_y),
                color=COLORS["blue"],
                width=1.15,
                style="-|>",
            )
    material_positions = [(4.2, 4.75), (5.2, 4.75), (5.72, 4.75), (6.62, 4.75)]
    parent_indices = [0, 2, 2, 4]
    for (x, y), parent_index in zip(material_positions, parent_indices, strict=True):
        _arrow(
            axis,
            (graph_layers_x[parent_index], layer_y - 0.18),
            (x, y + 0.16),
            color=COLORS["muted"],
            width=1.0,
            style="-",
        )
        _node(axis, x, y, 0.17, COLORS["gold"], "M")
    atom_center = (5.46, 3.62)
    atom_offsets = [(-0.52, 0), (-0.25, 0.42), (0.25, 0.42), (0.52, 0), (0.25, -0.42), (-0.25, -0.42)]
    _arrow(
        axis,
        (5.46, 4.58),
        (5.46, 4.07),
        color=COLORS["coral"],
        width=1.2,
    )
    for index, (dx, dy) in enumerate(atom_offsets):
        next_dx, next_dy = atom_offsets[(index + 1) % len(atom_offsets)]
        axis.plot(
            [atom_center[0] + dx, atom_center[0] + next_dx],
            [atom_center[1] + dy, atom_center[1] + next_dy],
            color=COLORS["teal"],
            linewidth=1.25,
        )
        _node(
            axis,
            atom_center[0] + dx,
            atom_center[1] + dy,
            0.105,
            COLORS["teal"],
        )
    axis.text(
        5.48,
        2.75,
        "EML material nodes expand into molecular atom/bond subgraphs",
        ha="center",
        fontsize=8.3,
        color=COLORS["muted"],
    )
    _box(
        axis,
        4.05,
        1.44,
        3.35,
        0.72,
        "Node features\nrole · identity · atom type · RDKit · Morgan FP",
        facecolor=COLORS["teal_light"],
        edgecolor=COLORS["teal"],
        fontsize=8.5,
    )
    _box(
        axis,
        4.05,
        0.50,
        3.35,
        0.62,
        "Typed edges: order · contains · bond type",
        facecolor=COLORS["gray_light"],
        edgecolor=COLORS["muted"],
        fontsize=8.3,
    )

    _box(
        axis,
        8.08,
        6.40,
        3.22,
        0.76,
        "Categorical + numeric node encoder",
        facecolor=COLORS["blue_light"],
        edgecolor=COLORS["blue"],
        fontsize=9.2,
        fontweight="bold",
    )
    _arrow(axis, (9.69, 6.36), (9.69, 5.95), color=COLORS["blue"])
    for index in range(4):
        y = 5.20 - index * 0.72
        _box(
            axis,
            8.35,
            y,
            2.68,
            0.52,
            f"GATv2Conv {index + 1}  +  residual  +  LayerNorm",
            facecolor=COLORS["teal_light"] if index % 2 == 0 else COLORS["green_light"],
            edgecolor=COLORS["teal"] if index % 2 == 0 else COLORS["green"],
            fontsize=8.2,
        )
        if index < 3:
            _arrow(
                axis,
                (9.69, y - 0.04),
                (9.69, y - 0.19),
                color=COLORS["muted"],
                width=1.0,
            )
    pooling_y = 1.19
    pooling_labels = ["Root", "All layers", "All materials", "EML layer", "EML materials"]
    for index, label in enumerate(pooling_labels):
        x = 7.86 + index * 0.75
        _box(
            axis,
            x,
            pooling_y,
            0.67,
            0.65,
            label,
            facecolor=COLORS["gold_light"],
            edgecolor=COLORS["gold"],
            fontsize=6.8,
            radius=0.07,
        )
    axis.text(
        9.69,
        0.80,
        "Separate graph-level mean pooling",
        ha="center",
        fontsize=8.4,
        color=COLORS["muted"],
    )
    _arrow(axis, (9.69, 2.49), (9.69, 1.92), color=COLORS["gold"])

    _box(
        axis,
        12.08,
        6.38,
        3.15,
        0.82,
        "Concatenate graph pools\n+ context embeddings",
        facecolor=COLORS["green_light"],
        edgecolor=COLORS["green"],
        fontsize=9.0,
        fontweight="bold",
    )
    _arrow(axis, (13.65, 6.33), (13.65, 5.87), color=COLORS["green"])
    _box(
        axis,
        12.38,
        5.05,
        2.55,
        0.78,
        "Shared MLP readout",
        facecolor=COLORS["blue_light"],
        edgecolor=COLORS["blue"],
        fontsize=9.3,
        fontweight="bold",
    )
    _arrow(axis, (13.65, 5.00), (13.65, 4.57), color=COLORS["blue"])
    _box(
        axis,
        12.02,
        3.66,
        1.46,
        0.78,
        "Mean head\nEQE point estimate",
        facecolor=COLORS["coral_light"],
        edgecolor=COLORS["coral"],
        fontsize=8.1,
    )
    _box(
        axis,
        13.82,
        3.66,
        1.72,
        0.78,
        "Quantile head\nq10 ≤ q50 ≤ q90",
        facecolor=COLORS["gold_light"],
        edgecolor=COLORS["gold"],
        fontsize=8.1,
    )
    _arrow(
        axis,
        (13.65, 4.57),
        (12.75, 4.47),
        color=COLORS["coral"],
        connectionstyle="arc3,rad=0.12",
    )
    _arrow(
        axis,
        (13.65, 4.57),
        (14.68, 4.47),
        color=COLORS["gold"],
        connectionstyle="arc3,rad=-0.12",
    )
    _box(
        axis,
        12.08,
        2.34,
        3.34,
        0.78,
        "Conformal calibration\nvalidation residual quantile",
        facecolor=COLORS["teal_light"],
        edgecolor=COLORS["teal"],
        fontsize=8.7,
    )
    _arrow(axis, (14.68, 3.61), (13.75, 3.15), color=COLORS["teal"])
    _box(
        axis,
        12.08,
        1.00,
        3.34,
        0.82,
        "Final output\nmean EQE + calibrated 80% interval",
        facecolor=COLORS["green_light"],
        edgecolor=COLORS["green"],
        fontsize=9.2,
        fontweight="bold",
    )
    _arrow(axis, (13.75, 2.28), (13.75, 1.87), color=COLORS["green"])

    _arrow(axis, (3.62, 4.82), (4.00, 4.82), color=COLORS["ink"], width=1.7)
    _arrow(axis, (7.46, 4.55), (8.00, 4.55), color=COLORS["ink"], width=1.7)
    _arrow(axis, (11.37, 4.55), (11.94, 4.55), color=COLORS["ink"], width=1.7)
    _arrow(
        axis,
        (3.58, 1.82),
        (12.00, 6.72),
        color=COLORS["green"],
        width=1.1,
        connectionstyle="arc3,rad=-0.16",
    )

    legend_y = 0.13
    legend_items = [
        ("D", COLORS["ink"], "device root"),
        ("L", COLORS["blue"], "layer"),
        ("M", COLORS["gold"], "material"),
        ("", COLORS["teal"], "atom"),
    ]
    for index, (label, color, description) in enumerate(legend_items):
        x = 0.65 + index * 1.34
        _node(axis, x, legend_y + 0.03, 0.105, color, label, fontsize=6.3)
        axis.text(
            x + 0.17,
            legend_y + 0.03,
            description,
            fontsize=7.6,
            va="center",
            color=COLORS["muted"],
        )
    axis.text(
        15.42,
        0.14,
        "Training loss = MSE(mean) + pinball(q10, q50, q90)",
        fontsize=8.1,
        color=COLORS["muted"],
        ha="right",
        va="center",
    )

    figure.tight_layout(pad=0.4)
    save_figure(figure, output_dir / "oled_gat_model_architecture")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    predictions_path = (
        args.predictions
        if args.predictions.is_absolute()
        else REPOSITORY_DIR / args.predictions
    )
    metrics_path = (
        args.metrics if args.metrics.is_absolute() else REPOSITORY_DIR / args.metrics
    )
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else REPOSITORY_DIR / args.output_dir
    )
    configure_matplotlib()
    predictions = pd.read_csv(predictions_path)
    frozen_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    regression_metrics = draw_regression_figure(predictions, output_dir)
    draw_architecture_figure(output_dir)
    summary = {
        "source_predictions": str(predictions_path),
        "source_metrics": str(metrics_path),
        "regression_metrics_recomputed": regression_metrics,
        "frozen_test_metrics": {
            "oled_gat": frozen_metrics["component_models"]["oled_gat"]["mean_head"],
            "ensemble": frozen_metrics["test"]["mean_head"],
        },
        "output_dir": str(output_dir),
    }
    (output_dir / "figure_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
