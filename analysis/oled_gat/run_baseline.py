from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PROJECT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from oled_gat.baseline import train_catboost_baseline  # noqa: E402
from oled_gat.data import load_config  # noqa: E402
from oled_gat.features import build_features, load_prepared_frames  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the OLED-GAT tabular baseline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs" / "default.yaml",
    )
    parser.add_argument("--evaluate-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = (REPOSITORY_DIR / config["output_dir"]).resolve()
    manifest, layers, components, materials = load_prepared_frames(output_dir)
    features = build_features(
        manifest,
        layers,
        components,
        materials,
        material_vocabulary_size=int(
            config["graph"]["material_vocabulary_size"]
        ),
        minimum_material_frequency=int(
            config["graph"]["minimum_material_frequency"]
        ),
    )
    feature_dir = output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    features.frame.to_parquet(feature_dir / "tabular_features.parquet", index=False)
    (feature_dir / "feature_metadata.json").write_text(
        json.dumps(features.metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    result = train_catboost_baseline(
        features.frame,
        numeric_columns=features.numeric_columns,
        categorical_columns=features.categorical_columns,
        output_dir=output_dir / "baseline",
        seed=int(config["seed"]),
        evaluate_test=args.evaluate_test,
    )
    print(json.dumps(result.metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
