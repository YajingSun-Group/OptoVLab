from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PROJECT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from oled_gat.data import load_config  # noqa: E402
from oled_gat.graph_data import load_graph_cache  # noqa: E402
from oled_gat.training import train_oled_gat  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the quantile OLED-GAT.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs" / "default.yaml",
    )
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument(
        "--run-name",
        default="gat",
        help="Subdirectory used for checkpoints and metrics.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override only the model-training seed; the prepared split is unchanged.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = copy.deepcopy(load_config(args.config))
    if args.seed is not None:
        config["seed"] = args.seed
    output_dir = (REPOSITORY_DIR / config["output_dir"]).resolve()
    manifest = pd.read_parquet(output_dir / "prepared" / "manifest.parquet")
    graphs, vocabulary = load_graph_cache(output_dir)
    result = train_oled_gat(
        graphs,
        manifest,
        vocabulary,
        config,
        output_dir=output_dir / args.run_name,
        evaluate_test=args.evaluate_test,
    )
    print(json.dumps(result.metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
