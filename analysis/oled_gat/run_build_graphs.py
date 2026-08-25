from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PROJECT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from oled_gat.data import load_config  # noqa: E402
from oled_gat.features import load_prepared_frames  # noqa: E402
from oled_gat.graph_data import build_graphs, save_graph_cache  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cached OLED device graphs.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs" / "default.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = (REPOSITORY_DIR / config["output_dir"]).resolve()
    manifest, layers, components, materials = load_prepared_frames(output_dir)
    graphs, vocabulary, metadata = build_graphs(
        manifest,
        layers,
        components,
        materials,
        material_vocabulary_size=int(
            config["graph"]["material_vocabulary_size"]
        ),
        maximum_atoms_per_molecule=int(
            config["graph"]["maximum_atoms_per_molecule"]
        ),
        molecular_fingerprint_size=int(
            config["graph"]["molecular_fingerprint_size"]
        ),
    )
    save_graph_cache(graphs, vocabulary, metadata, output_dir)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
