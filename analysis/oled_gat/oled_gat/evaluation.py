from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor

from .baseline import _matrix
from .metrics import interval_metrics, pinball_loss, regression_metrics
from .model import build_oled_gat
from .training import _loader, predict


def select_convex_blend_weight(
    observed: np.ndarray,
    primary: np.ndarray,
    secondary: np.ndarray,
    *,
    step: float = 0.01,
) -> tuple[float, dict[str, float]]:
    best_weight = 0.0
    best_metrics: dict[str, float] | None = None
    for weight in np.arange(0.0, 1.0 + step / 2, step):
        prediction = weight * primary + (1.0 - weight) * secondary
        metrics = regression_metrics(observed, prediction)
        if best_metrics is None or metrics["rmse"] < best_metrics["rmse"]:
            best_weight = float(weight)
            best_metrics = metrics
    if best_metrics is None:
        raise RuntimeError("Blend search did not evaluate any weights")
    return best_weight, best_metrics


def conformal_quantile_offset(
    observed: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    alpha: float,
) -> float:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    scores = np.maximum(lower - observed, observed - upper)
    rank = min(
        len(scores),
        math.ceil((len(scores) + 1) * (1.0 - alpha)),
    )
    return float(np.sort(scores)[rank - 1])


def _quantile_metrics(
    observed: np.ndarray,
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "mean_head": regression_metrics(observed, predictions["mean"].to_numpy()),
        "median_head": regression_metrics(observed, predictions["q50"].to_numpy()),
        "interval_10_90": interval_metrics(
            observed,
            predictions["q10"].to_numpy(),
            predictions["q90"].to_numpy(),
        ),
        "pinball": {
            "q10": pinball_loss(observed, predictions["q10"].to_numpy(), 0.1),
            "q50": pinball_loss(observed, predictions["q50"].to_numpy(), 0.5),
            "q90": pinball_loss(observed, predictions["q90"].to_numpy(), 0.9),
        },
    }


def evaluate_saved_gat(
    graphs: list[Any],
    manifest: pd.DataFrame,
    *,
    checkpoint_path: Path,
    batch_size: int,
    num_workers: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for OLED-GAT evaluation")
    device = torch.device("cuda")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    vocabulary = checkpoint["vocabulary"]
    model_config = checkpoint["model_config"]
    model = build_oled_gat(vocabulary, model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    test_rows = np.flatnonzero(manifest["split"].eq("test").to_numpy()).tolist()
    loader = _loader(
        graphs,
        test_rows,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    target_stats = vocabulary["numeric_stats"]
    sample_rows, observed, outputs = predict(
        model,
        loader,
        device=device,
        target_mean=float(target_stats["target_mean"]),
        target_std=float(target_stats["target_std"]),
    )
    predictions = manifest.iloc[sample_rows][
        ["id", "paper_id", "doi", "eqe_max", "split"]
    ].copy()
    predictions[["mean", "q10", "q50", "q90"]] = outputs
    return predictions, _quantile_metrics(observed, predictions)


def evaluate_saved_catboost(
    features: pd.DataFrame,
    *,
    numeric_columns: list[str],
    categorical_columns: list[str],
    model_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    matrix = _matrix(features, numeric_columns, categorical_columns)
    test_mask = features["split"].eq("test")
    predictions = features.loc[
        test_mask,
        ["id", "paper_id", "doi", "eqe_max", "split"],
    ].copy()
    for name in ("mean", "q10", "q50", "q90"):
        model = CatBoostRegressor()
        model.load_model(model_dir / f"catboost_{name}.cbm")
        predictions[name] = model.predict(matrix.loc[test_mask])
    quantiles = np.sort(predictions[["q10", "q50", "q90"]].to_numpy(), axis=1)
    predictions[["q10", "q50", "q90"]] = quantiles
    return predictions, _quantile_metrics(
        predictions["eqe_max"].to_numpy(),
        predictions,
    )


def build_frozen_test_result(
    catboost_validation: pd.DataFrame,
    gat_validation: pd.DataFrame,
    catboost_test: pd.DataFrame,
    gat_test: pd.DataFrame,
    *,
    alpha: float = 0.2,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    validation = catboost_validation.merge(
        gat_validation[["id", "mean", "q10", "q50", "q90"]],
        on="id",
        suffixes=("_catboost", "_gat"),
        validate="one_to_one",
    )
    observed_validation = validation["eqe_max"].to_numpy()
    catboost_weight, validation_point_metrics = select_convex_blend_weight(
        observed_validation,
        validation["mean_catboost"].to_numpy(),
        validation["mean_gat"].to_numpy(),
    )
    gat_weight = 1.0 - catboost_weight
    validation_lower = (
        catboost_weight * validation["q10_catboost"]
        + gat_weight * validation["q10_gat"]
    ).to_numpy()
    validation_upper = (
        catboost_weight * validation["q90_catboost"]
        + gat_weight * validation["q90_gat"]
    ).to_numpy()
    conformal_offset = conformal_quantile_offset(
        observed_validation,
        validation_lower,
        validation_upper,
        alpha=alpha,
    )

    test = catboost_test.merge(
        gat_test[["id", "mean", "q10", "q50", "q90"]],
        on="id",
        suffixes=("_catboost", "_gat"),
        validate="one_to_one",
    )
    for output in ("mean", "q10", "q50", "q90"):
        test[f"{output}_ensemble"] = (
            catboost_weight * test[f"{output}_catboost"]
            + gat_weight * test[f"{output}_gat"]
        )
    test["q10_calibrated"] = np.clip(
        test["q10_ensemble"] - conformal_offset,
        0.0,
        60.0,
    )
    test["q90_calibrated"] = np.clip(
        test["q90_ensemble"] + conformal_offset,
        0.0,
        60.0,
    )
    observed_test = test["eqe_max"].to_numpy()
    metrics = {
        "blend": {
            "catboost_weight": catboost_weight,
            "gat_weight": gat_weight,
            "validation_mean_head": validation_point_metrics,
        },
        "conformal": {
            "alpha": alpha,
            "nominal_coverage": 1.0 - alpha,
            "validation_offset": conformal_offset,
        },
        "test": {
            "device_count": int(len(test)),
            "paper_count": int(test["paper_id"].nunique()),
            "mean_head": regression_metrics(
                observed_test,
                test["mean_ensemble"].to_numpy(),
            ),
            "median_head": regression_metrics(
                observed_test,
                test["q50_ensemble"].to_numpy(),
            ),
            "raw_interval_10_90": interval_metrics(
                observed_test,
                test["q10_ensemble"].to_numpy(),
                test["q90_ensemble"].to_numpy(),
            ),
            "calibrated_interval_10_90": interval_metrics(
                observed_test,
                test["q10_calibrated"].to_numpy(),
                test["q90_calibrated"].to_numpy(),
            ),
        },
    }
    return test, metrics


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
