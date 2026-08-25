from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


PROJECT_DIR = Path(__file__).resolve().parent
COLORS = {
    "ink": "#18343B",
    "muted": "#60767B",
    "grid": "#DCE6E7",
    "teal": "#167D86",
    "coral": "#C95D3A",
    "blue": "#3973A5",
    "white": "#FFFFFF",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw the pure-GNN OLED campaign regression figure."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PROJECT_DIR
        / "outputs_campaign_gat"
        / "gnn_ensemble_refit"
        / "test_predictions.csv",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=PROJECT_DIR
        / "outputs_campaign_gat"
        / "gnn_ensemble_refit"
        / "metrics.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR
        / "outputs_campaign_gat"
        / "gnn_ensemble_refit"
        / "figures",
    )
    return parser.parse_args()


def _save(figure: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")


def _cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    iterations: int = 3000,
    seed: int = 20260727,
) -> dict[str, list[float]]:
    groups = [
        group.index.to_numpy()
        for _, group in frame.groupby("paper_id", sort=True)
    ]
    observed = frame["eqe_max"].to_numpy(dtype=float)
    predicted = frame["mean_ensemble"].to_numpy(dtype=float)
    generator = np.random.default_rng(seed)
    values: list[tuple[float, float, float]] = []
    for _ in range(iterations):
        selected = generator.integers(0, len(groups), len(groups))
        rows = np.concatenate([groups[index] for index in selected])
        target = observed[rows]
        estimate = predicted[rows]
        values.append(
            (
                float(r2_score(target, estimate)),
                float(mean_absolute_error(target, estimate)),
                float(root_mean_squared_error(target, estimate)),
            )
        )
    matrix = np.asarray(values)
    return {
        name: np.quantile(matrix[:, index], [0.025, 0.5, 0.975]).tolist()
        for index, name in enumerate(("r2", "mae", "rmse"))
    }


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.predictions)
    metrics = json.loads(args.metrics.read_text())
    observed = frame["eqe_max"].to_numpy(dtype=float)
    predicted = frame["mean_ensemble"].to_numpy(dtype=float)
    residual = predicted - observed
    values = np.vstack([observed, predicted])
    density = gaussian_kde(values)(values)
    order = np.argsort(density)
    limit = max(45.0, float(np.ceil(max(observed.max(), predicted.max()) / 5) * 5))

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.edgecolor": COLORS["ink"],
            "text.color": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 5.25))
    scatter = axes[0].scatter(
        observed[order],
        predicted[order],
        c=density[order],
        cmap="viridis",
        s=34,
        alpha=0.84,
        edgecolors=COLORS["white"],
        linewidths=0.25,
    )
    axes[0].plot(
        [0, limit],
        [0, limit],
        color=COLORS["coral"],
        linestyle=(0, (5, 3)),
        linewidth=1.8,
    )
    slope, intercept = np.polyfit(observed, predicted, 1)
    fit_x = np.asarray([0.0, limit])
    axes[0].plot(
        fit_x,
        slope * fit_x + intercept,
        color=COLORS["blue"],
        linewidth=1.6,
    )
    test = metrics["test"]
    axes[0].text(
        0.055,
        0.94,
        (
            f"$R^2$ = {test['r2']:.3f}\n"
            f"MAE = {test['mae']:.3f}\n"
            f"RMSE = {test['rmse']:.3f}\n"
            f"Spearman = {test['spearman']:.3f}\n"
            f"$n$ = {test['device_count']}"
        ),
        transform=axes[0].transAxes,
        va="top",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": COLORS["white"],
            "edgecolor": COLORS["grid"],
            "alpha": 0.95,
        },
    )
    axes[0].set(
        title="Frozen campaign test set",
        xlabel="Observed maximum EQE (%)",
        ylabel="Predicted maximum EQE (%)",
        xlim=(0, limit),
        ylim=(0, limit),
        aspect="equal",
    )
    figure.colorbar(scatter, ax=axes[0], label="Point density", shrink=0.82)

    axes[1].scatter(
        predicted,
        residual,
        c=COLORS["teal"],
        s=31,
        alpha=0.72,
        edgecolors=COLORS["white"],
        linewidths=0.25,
    )
    axes[1].axhline(0.0, color=COLORS["coral"], linewidth=1.7)
    axes[1].axhline(
        float(np.median(residual)),
        color=COLORS["blue"],
        linestyle=(0, (5, 3)),
        linewidth=1.5,
    )
    axes[1].set(
        title="Residual diagnostics",
        xlabel="Predicted maximum EQE (%)",
        ylabel="Prediction - observation (EQE points)",
    )
    for axis in axes:
        axis.grid(color=COLORS["grid"], linewidth=0.7, alpha=0.85)
        axis.set_axisbelow(True)
    figure.suptitle(
        "Pure OLED-GAT ensemble: multilayer device-graph regression",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )
    figure.text(
        0.5,
        -0.01,
        "Weights selected on validation data; final models refit on train + validation.",
        ha="center",
        color=COLORS["muted"],
        fontsize=9.5,
    )
    _save(figure, args.output_dir / "campaign_frozen_test_regression")
    plt.close(figure)

    bootstrap = _cluster_bootstrap(frame)
    metrics["paper_cluster_bootstrap_95_ci"] = bootstrap
    args.metrics.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "figure_metrics.json").write_text(
        json.dumps(
            {
                "test": test,
                "paper_cluster_bootstrap_95_ci": bootstrap,
                "residual_median": float(np.median(residual)),
                "regression_slope": float(slope),
                "regression_intercept": float(intercept),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
