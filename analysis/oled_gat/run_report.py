from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PROJECT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from oled_gat.data import load_config  # noqa: E402
from oled_gat.metrics import regression_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the OLED-GAT evaluation report.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs" / "device_random.yaml",
    )
    return parser.parse_args()


def _metric(value: float) -> str:
    return f"{value:.3f}"


def _plot_overview(frame: pd.DataFrame, output_path: Path) -> None:
    observed = frame["eqe_max"].to_numpy()
    predicted = frame["mean_ensemble"].to_numpy()
    residual = observed - predicted
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].scatter(
        observed,
        predicted,
        c=np.abs(residual),
        cmap="viridis",
        alpha=0.75,
        s=24,
        edgecolors="none",
    )
    axes[0].plot([0, 60], [0, 60], color="#b42318", linestyle="--", linewidth=1.5)
    axes[0].set(
        xlabel="Observed maximum EQE (%)",
        ylabel="Predicted maximum EQE (%)",
        xlim=(0, 60),
        ylim=(0, 60),
        title="Frozen test predictions",
    )
    axes[1].hist(residual, bins=30, color="#2f6f78", alpha=0.85)
    axes[1].axvline(0, color="#b42318", linestyle="--", linewidth=1.5)
    axes[1].set(
        xlabel="Observed - predicted EQE (percentage points)",
        ylabel="Device count",
        title="Residual distribution",
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_intervals(frame: pd.DataFrame, output_path: Path) -> None:
    ordered = frame.sort_values("mean_ensemble").reset_index(drop=True)
    positions = np.arange(len(ordered))
    figure, axis = plt.subplots(figsize=(11, 4.8))
    axis.fill_between(
        positions,
        ordered["q10_calibrated"],
        ordered["q90_calibrated"],
        color="#76a9a5",
        alpha=0.35,
        label="Calibrated 10-90% interval",
    )
    axis.plot(
        positions,
        ordered["mean_ensemble"],
        color="#154c57",
        linewidth=1.3,
        label="Predicted mean",
    )
    axis.scatter(
        positions,
        ordered["eqe_max"],
        s=8,
        color="#bd5d38",
        alpha=0.65,
        label="Observed EQE",
    )
    axis.set(
        xlabel="Test devices ordered by predicted EQE",
        ylabel="Maximum EQE (%)",
        ylim=(0, 60),
        title="Conformalized prediction intervals",
    )
    axis.legend(frameon=False, ncol=3, loc="upper left")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _subgroup_table(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    merged = predictions.merge(
        manifest[["id", "primary_mechanism", "fabrication_method", "color_group"]],
        on="id",
        how="left",
        validate="one_to_one",
    )
    rows: list[dict[str, object]] = []
    for column in ("primary_mechanism", "fabrication_method", "color_group"):
        for value, group in merged.groupby(column, dropna=False):
            if len(group) < 10:
                continue
            metrics = regression_metrics(
                group["eqe_max"].to_numpy(),
                group["mean_ensemble"].to_numpy(),
            )
            rows.append(
                {
                    "dimension": column,
                    "group": str(value),
                    "devices": len(group),
                    "r2": metrics["r2"],
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                }
            )
    return pd.DataFrame(rows).sort_values(["dimension", "devices"], ascending=[True, False])


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = (REPOSITORY_DIR / config["output_dir"]).resolve()
    final_dir = output_dir / "final"
    metrics = json.loads((final_dir / "test_metrics.json").read_text())
    metadata = json.loads(
        (output_dir / "prepared" / "dataset_metadata.json").read_text()
    )
    predictions = pd.read_csv(final_dir / "test_predictions.csv")
    manifest = pd.read_parquet(output_dir / "prepared" / "manifest.parquet")

    _plot_overview(predictions, final_dir / "test_prediction_overview.png")
    _plot_intervals(predictions, final_dir / "test_prediction_intervals.png")
    subgroup = _subgroup_table(predictions, manifest)
    subgroup.to_csv(final_dir / "test_subgroup_metrics.csv", index=False)

    train_papers = set(manifest.loc[manifest["split"].eq("train"), "paper_id"])
    test_papers = set(manifest.loc[manifest["split"].eq("test"), "paper_id"])
    paper_overlap = len(train_papers & test_papers)
    test = metrics["test"]
    gat = metrics["component_models"]["oled_gat"]["mean_head"]
    catboost = metrics["component_models"]["catboost"]["mean_head"]
    ensemble = test["mean_head"]
    interval = test["calibrated_interval_10_90"]

    subgroup_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(row.dimension))}</td>"
        f"<td>{html.escape(str(row.group))}</td>"
        f"<td>{int(row.devices)}</td>"
        f"<td>{_metric(row.r2)}</td>"
        f"<td>{_metric(row.rmse)}</td>"
        "</tr>"
        for row in subgroup.itertuples()
    )
    report = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OLED-GAT frozen evaluation</title>
<style>
body{{font-family:Inter,Arial,sans-serif;color:#172328;margin:0;background:#f4f7f7}}
main{{max-width:1120px;margin:auto;background:white;padding:34px 44px 60px}}
h1,h2{{color:#154c57}} h1{{font-size:32px;margin-bottom:6px}}
.lede{{font-size:17px;color:#46585d;max-width:900px}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}}
.metric{{border:1px solid #cad7d8;padding:15px;border-radius:6px;background:#fbfdfd}}
.metric strong{{display:block;font-size:26px;color:#154c57;margin-top:5px}}
.warning{{border-left:5px solid #bd5d38;background:#fff6f1;padding:14px 18px;margin:24px 0}}
table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{border-bottom:1px solid #dbe4e4;padding:8px;text-align:left}}
th{{background:#eef4f4}} img{{width:100%;margin:12px 0 24px}}
code{{background:#eef4f4;padding:2px 5px}} .small{{font-size:13px;color:#607176}}
</style>
</head>
<body><main>
<h1>OLED-GAT frozen evaluation</h1>
<p class="lede">Maximum-EQE prediction from ordered OLED layer graphs and EML
molecular graphs, with non-crossing quantile regression and conformal interval
calibration.</p>
<div class="grid">
  <div class="metric">OLED-GAT test R²<strong>{_metric(gat["r2"])}</strong></div>
  <div class="metric">Ensemble test R²<strong>{_metric(ensemble["r2"])}</strong></div>
  <div class="metric">Ensemble RMSE<strong>{_metric(ensemble["rmse"])}</strong></div>
  <div class="metric">80% interval coverage<strong>{interval["coverage"]:.1%}</strong></div>
</div>
<div class="warning"><strong>Evaluation boundary.</strong> This achieved result
uses a frozen device-random split. {paper_overlap} test papers also have sibling
devices in training. It measures interpolation among mined literature devices,
not generalization to entirely unseen papers. The DOI-grouped benchmark remains
the primary target for publication-grade claims.</div>
<h2>Dataset and protocol</h2>
<ul>
  <li>{metadata["device_count"]:,} devices from {metadata["paper_count"]:,} papers.</li>
  <li>Organic small-molecule emitters; valid EQE and complete EML SMILES.</li>
  <li>Single-junction, non-white devices without explicit outcoupling layers.</li>
  <li>Frozen split fingerprint: <code>{metadata["dataset_fingerprint"]}</code>.</li>
  <li>Train/validation/test: 3,382 / 423 / 423 devices.</li>
</ul>
<h2>Model comparison</h2>
<table><thead><tr><th>Model</th><th>R²</th><th>MAE</th><th>RMSE</th><th>Spearman</th></tr></thead>
<tbody>
<tr><td>OLED-GAT</td><td>{_metric(gat["r2"])}</td><td>{_metric(gat["mae"])}</td><td>{_metric(gat["rmse"])}</td><td>{_metric(gat["spearman"])}</td></tr>
<tr><td>CatBoost baseline</td><td>{_metric(catboost["r2"])}</td><td>{_metric(catboost["mae"])}</td><td>{_metric(catboost["rmse"])}</td><td>{_metric(catboost["spearman"])}</td></tr>
<tr><td>Fixed 0.62/0.38 ensemble</td><td>{_metric(ensemble["r2"])}</td><td>{_metric(ensemble["mae"])}</td><td>{_metric(ensemble["rmse"])}</td><td>{_metric(ensemble["spearman"])}</td></tr>
</tbody></table>
<img src="test_prediction_overview.png" alt="Prediction scatter and residuals">
<h2>Prediction intervals</h2>
<p>The GAT jointly predicts mean, q10, q50, and q90. Quantiles are constrained
to be non-crossing. A conformal offset fitted on validation data gives
{interval["coverage"]:.1%} empirical test coverage at a nominal 80% level,
with mean width {_metric(interval["mean_width"])} EQE percentage points.</p>
<img src="test_prediction_intervals.png" alt="Calibrated prediction intervals">
<h2>Subgroup audit</h2>
<table><thead><tr><th>Dimension</th><th>Group</th><th>Devices</th><th>R²</th><th>RMSE</th></tr></thead>
<tbody>{subgroup_rows}</tbody></table>
<h2>Limitations</h2>
<ul>
  <li>Most labels and architectures are automatically mined, not fully human-audited.</li>
  <li>SMILES and layer stacks omit PLQY, dipole orientation, energy levels,
  charge mobility, and interfacial morphology.</li>
  <li>The model must not be described as achieving R² &gt; 0.7 on unseen papers.</li>
  <li>An external, fully reviewed DOI-disjoint test set is required before a
  publication-grade generalization claim.</li>
</ul>
<p class="small">Generated from immutable prediction files; test evaluation UTC:
{html.escape(metrics["evaluated_at_utc"])}.</p>
</main></body></html>"""
    (final_dir / "report.html").write_text(report, encoding="utf-8")
    print(final_dir / "report.html")


if __name__ == "__main__":
    main()
