from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PROJECT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from oled_gat.data import (  # noqa: E402
    load_config,
    load_source_tables,
    prepare_data,
    write_prepared_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the frozen OLED-GAT dataset.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs" / "default.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    input_dir = (REPOSITORY_DIR / config["input_dir"]).resolve()
    output_dir = (REPOSITORY_DIR / config["output_dir"]).resolve()
    data = prepare_data(load_source_tables(input_dir), config)
    write_prepared_data(data, output_dir)
    print(json.dumps(data.metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
