from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import torch

PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PROJECT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from oled_gat.data import load_config  # noqa: E402
from oled_gat.graph_data import build_graphs  # noqa: E402
from oled_gat.metrics import regression_metrics  # noqa: E402
from oled_gat.training import fit_fixed_oled_gat  # noqa: E402


def select_simplex_weights(
    observed: pd.Series,
    predictions: list[pd.Series],
    *,
    units: int = 100,
) -> tuple[list[float], dict[str, float]]:
    best_weights: list[float] | None = None
    best_metrics: dict[str, float] | None = None
    model_count = len(predictions)
    for cuts in itertools.combinations_with_replacement(
        range(units + 1),
        model_count - 1,
    ):
        boundaries = (0, *cuts, units)
        integer_weights = [
            boundaries[index + 1] - boundaries[index]
            for index in range(model_count)
        ]
        weights = [value / units for value in integer_weights]
        blended = sum(
            weight * prediction
            for weight, prediction in zip(
                weights,
                predictions,
                strict=True,
            )
        )
        metrics = regression_metrics(observed.to_numpy(), blended.to_numpy())
        if best_metrics is None or metrics["rmse"] < best_metrics["rmse"]:
            best_weights = weights
            best_metrics = metrics
    if best_weights is None or best_metrics is None:
        raise RuntimeError("No simplex weights were evaluated")
    return best_weights, best_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refit the selected pure GAT ensemble on train plus validation."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs" / "campaign_gat.yaml",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=["gat", "gat_seed_20260727", "gat_seed_20260728"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = (REPOSITORY_DIR / config["output_dir"]).resolve()
    prepared_dir = output_dir / "prepared"
    manifest = pd.read_parquet(prepared_dir / "manifest.parquet")
    refit_manifest = manifest.copy()
    refit_manifest.loc[
        refit_manifest["split"].eq("validation"),
        "split",
    ] = "train"
    layers = pd.read_parquet(prepared_dir / "layers.parquet")
    components = pd.read_parquet(prepared_dir / "components.parquet")
    materials = pd.read_parquet(prepared_dir / "materials.parquet")
    graph_config = config["graph"]
    graphs, vocabulary, graph_metadata = build_graphs(
        refit_manifest,
        layers,
        components,
        materials,
        material_vocabulary_size=int(
            graph_config["material_vocabulary_size"]
        ),
        maximum_atoms_per_molecule=int(
            graph_config["maximum_atoms_per_molecule"]
        ),
        molecular_fingerprint_size=int(
            graph_config["molecular_fingerprint_size"]
        ),
    )
    vocabulary_payload = asdict(vocabulary)

    validation_frames = [
        pd.read_csv(output_dir / run / "validation_predictions.csv")
        .sort_values("id")
        .reset_index(drop=True)
        for run in args.runs
    ]
    if any(
        not validation_frames[0]["id"].equals(frame["id"])
        for frame in validation_frames[1:]
    ):
        raise RuntimeError("Validation prediction rows differ between GAT runs")
    selected_weights, validation_metrics = select_simplex_weights(
        validation_frames[0]["eqe_max"],
        [frame["mean"] for frame in validation_frames],
    )
    weights = dict(zip(args.runs, selected_weights, strict=True))
    test_frames: list[pd.DataFrame] = []
    component_metrics: dict[str, object] = {}
    refit_specs: list[dict[str, object]] = []
    for run in args.runs:
        checkpoint = torch.load(
            output_dir / run / "best_model.pt",
            map_location="cpu",
            weights_only=False,
        )
        run_config = copy.deepcopy(config)
        run_config["seed"] = int(checkpoint["seed"])
        epochs = int(checkpoint["epoch"])
        refit_name = f"refit_{run}"
        refit_dir = output_dir / refit_name
        if (
            (refit_dir / "best_model.pt").exists()
            and (refit_dir / "test_predictions.csv").exists()
            and (refit_dir / "test_metrics.json").exists()
        ):
            predictions = pd.read_csv(refit_dir / "test_predictions.csv")
            metrics = json.loads((refit_dir / "test_metrics.json").read_text())
        else:
            result = fit_fixed_oled_gat(
                graphs,
                refit_manifest,
                vocabulary_payload,
                run_config,
                output_dir=refit_dir,
                epochs=epochs,
            )
            predictions = result.predictions
            metrics = result.metrics
        test_frames.append(predictions.sort_values("id").reset_index(drop=True))
        component_metrics[refit_name] = metrics["mean_head"]
        refit_specs.append(
            {
                "source_run": run,
                "refit_run": refit_name,
                "seed": int(checkpoint["seed"]),
                "epochs": epochs,
                "weight": float(weights[run]),
            }
        )

    if any(
        not test_frames[0]["id"].equals(frame["id"])
        for frame in test_frames[1:]
    ):
        raise RuntimeError("Refit test prediction rows differ")
    test = test_frames[0][
        ["id", "paper_id", "doi", "eqe_max", "split"]
    ].copy()
    for index, (run, frame) in enumerate(
        zip(args.runs, test_frames, strict=True)
    ):
        test[f"mean_refit_{index}_{run}"] = frame["mean"]
    test["mean_ensemble"] = sum(
        float(weights[run]) * frame["mean"]
        for run, frame in zip(args.runs, test_frames, strict=True)
    )
    test_metrics = regression_metrics(
        test["eqe_max"].to_numpy(),
        test["mean_ensemble"].to_numpy(),
    )
    dataset_metadata = json.loads(
        (prepared_dir / "dataset_metadata.json").read_text()
    )
    payload = {
        "model_family": "pure_oled_gat_ensemble",
        "training_protocol": "validation_selected_then_train_plus_validation_refit",
        "refit_specs": refit_specs,
        "graph_metadata": graph_metadata,
        "validation_selected_weights": weights,
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
    result_dir = output_dir / "gnn_ensemble_refit"
    result_dir.mkdir(parents=True, exist_ok=True)
    test.to_csv(result_dir / "test_predictions.csv", index=False)
    (result_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
