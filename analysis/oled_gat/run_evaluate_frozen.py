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
    build_frozen_test_result,
    evaluate_saved_catboost,
    evaluate_saved_gat,
    write_json,
)
from oled_gat.graph_data import load_graph_cache  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen OLED-GAT test data exactly once."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs" / "device_random.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = (REPOSITORY_DIR / config["output_dir"]).resolve()
    manifest = pd.read_parquet(output_dir / "prepared" / "manifest.parquet")
    metadata = json.loads(
        (output_dir / "prepared" / "dataset_metadata.json").read_text()
    )
    features = pd.read_parquet(output_dir / "features" / "tabular_features.parquet")
    feature_metadata = json.loads(
        (output_dir / "features" / "feature_metadata.json").read_text()
    )
    graphs, _ = load_graph_cache(output_dir)

    gat_test, gat_metrics = evaluate_saved_gat(
        graphs,
        manifest,
        checkpoint_path=output_dir / "gat" / "best_model.pt",
        batch_size=int(config["training"]["batch_size"]) * 2,
        num_workers=int(config["training"]["num_workers"]),
    )
    catboost_test, catboost_metrics = evaluate_saved_catboost(
        features,
        numeric_columns=feature_metadata["numeric_columns"],
        categorical_columns=feature_metadata["categorical_columns"],
        model_dir=output_dir / "baseline",
    )
    catboost_validation = pd.read_csv(
        output_dir / "baseline" / "validation_predictions.csv"
    )
    gat_validation = pd.read_csv(
        output_dir / "gat" / "validation_predictions.csv"
    )
    final_predictions, final_metrics = build_frozen_test_result(
        catboost_validation,
        gat_validation,
        catboost_test,
        gat_test,
    )
    final_metrics.update(
        {
            "evaluated_at_utc": datetime.now(UTC).isoformat(),
            "dataset_fingerprint": metadata["dataset_fingerprint"],
            "split_mode": metadata["split_mode"],
            "doi_disjoint": metadata["doi_disjoint"],
            "component_models": {
                "oled_gat": gat_metrics,
                "catboost": catboost_metrics,
            },
        }
    )
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_predictions.to_csv(final_dir / "test_predictions.csv", index=False)
    write_json(final_dir / "test_metrics.json", final_metrics)
    print(json.dumps(final_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
