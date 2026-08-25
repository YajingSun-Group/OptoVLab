from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PROJECT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from oled_gat.data import load_config  # noqa: E402
from oled_gat.evaluation import (  # noqa: E402
    evaluate_saved_gat,
    select_convex_blend_weight,
)
from oled_gat.graph_data import load_graph_cache  # noqa: E402
from oled_gat.metrics import regression_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a validation-weighted, pure OLED-GAT ensemble."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs" / "campaign_gat.yaml",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=["gat", "gat_seed_20260727"],
    )
    parser.add_argument(
        "--weight-step",
        type=float,
        default=0.01,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.runs) != 2:
        raise ValueError("The current evaluator requires exactly two GAT runs")
    config = load_config(args.config)
    output_dir = (REPOSITORY_DIR / config["output_dir"]).resolve()
    manifest = pd.read_parquet(output_dir / "prepared" / "manifest.parquet")
    graphs, _ = load_graph_cache(output_dir)

    validation_frames = [
        pd.read_csv(output_dir / run / "validation_predictions.csv")
        .sort_values("id")
        .reset_index(drop=True)
        for run in args.runs
    ]
    if not validation_frames[0]["id"].equals(validation_frames[1]["id"]):
        raise RuntimeError("Validation prediction rows differ between GAT runs")
    observed_validation = validation_frames[0]["eqe_max"].to_numpy()
    first_weight, validation_metrics = select_convex_blend_weight(
        observed_validation,
        validation_frames[0]["mean"].to_numpy(),
        validation_frames[1]["mean"].to_numpy(),
        step=args.weight_step,
    )
    second_weight = 1.0 - first_weight
    validation = validation_frames[0][
        ["id", "paper_id", "doi", "eqe_max", "split"]
    ].copy()
    validation[f"mean_{args.runs[0]}"] = validation_frames[0]["mean"]
    validation[f"mean_{args.runs[1]}"] = validation_frames[1]["mean"]
    validation["mean_ensemble"] = (
        first_weight * validation_frames[0]["mean"]
        + second_weight * validation_frames[1]["mean"]
    )

    test_frames: list[pd.DataFrame] = []
    component_metrics: dict[str, object] = {}
    for run in args.runs:
        frame, metrics = evaluate_saved_gat(
            graphs,
            manifest,
            checkpoint_path=output_dir / run / "best_model.pt",
            batch_size=int(config["training"]["batch_size"]) * 2,
            num_workers=int(config["training"]["num_workers"]),
        )
        test_frames.append(frame.sort_values("id").reset_index(drop=True))
        component_metrics[run] = metrics["mean_head"]
    if not test_frames[0]["id"].equals(test_frames[1]["id"]):
        raise RuntimeError("Test prediction rows differ between GAT runs")
    test = test_frames[0][
        ["id", "paper_id", "doi", "eqe_max", "split"]
    ].copy()
    test[f"mean_{args.runs[0]}"] = test_frames[0]["mean"]
    test[f"mean_{args.runs[1]}"] = test_frames[1]["mean"]
    test["mean_ensemble"] = (
        first_weight * test_frames[0]["mean"]
        + second_weight * test_frames[1]["mean"]
    )
    test_metrics = regression_metrics(
        test["eqe_max"].to_numpy(),
        test["mean_ensemble"].to_numpy(),
    )

    result_dir = output_dir / "gnn_ensemble"
    result_dir.mkdir(parents=True, exist_ok=True)
    validation.to_csv(result_dir / "validation_predictions.csv", index=False)
    test.to_csv(result_dir / "test_predictions.csv", index=False)
    dataset_metadata = json.loads(
        (output_dir / "prepared" / "dataset_metadata.json").read_text()
    )
    payload = {
        "model_family": "pure_oled_gat_ensemble",
        "runs": args.runs,
        "validation_selected_weights": {
            args.runs[0]: first_weight,
            args.runs[1]: second_weight,
        },
        "validation": validation_metrics,
        "test": {
            "device_count": int(len(test)),
            "paper_count": int(test["paper_id"].nunique()),
            **test_metrics,
        },
        "component_test_metrics": component_metrics,
        "dataset_fingerprint": dataset_metadata["dataset_fingerprint"],
        "split_mode": dataset_metadata["split_mode"],
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
    }
    (result_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
