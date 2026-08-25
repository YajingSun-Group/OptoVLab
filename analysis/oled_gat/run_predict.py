from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PROJECT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from oled_gat.data import load_config  # noqa: E402
from oled_gat.inference import predict_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict EQE for one OLED device.")
    parser.add_argument("input_json", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs" / "device_random.yaml",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = (REPOSITORY_DIR / config["output_dir"]).resolve()
    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    result = predict_device(
        payload,
        checkpoint_path=output_dir / "gat" / "best_model.pt",
        config=config,
        validation_predictions_path=output_dir
        / "gat"
        / "validation_predictions.csv",
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
