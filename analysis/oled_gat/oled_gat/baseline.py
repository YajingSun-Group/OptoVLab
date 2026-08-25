from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from .metrics import interval_metrics, pinball_loss, regression_metrics


@dataclass(frozen=True)
class BaselineResult:
    metrics: dict[str, Any]
    predictions: pd.DataFrame


def _matrix(
    frame: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
) -> pd.DataFrame:
    matrix = frame[numeric + categorical].copy()
    for column in numeric:
        matrix[column] = pd.to_numeric(matrix[column], errors="coerce").fillna(
            frame.loc[frame["split"].eq("train"), column].median()
        )
    for column in categorical:
        matrix[column] = matrix[column].fillna("unknown").astype(str)
    return matrix


def train_catboost_baseline(
    frame: pd.DataFrame,
    *,
    numeric_columns: list[str],
    categorical_columns: list[str],
    output_dir: Path,
    seed: int,
    iterations: int = 1800,
    depth: int = 8,
    learning_rate: float = 0.035,
    evaluate_test: bool = False,
) -> BaselineResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = _matrix(frame, numeric_columns, categorical_columns)
    train_mask = frame["split"].eq("train")
    validation_mask = frame["split"].eq("validation")
    evaluation_mask = frame["split"].eq("test" if evaluate_test else "validation")
    categorical_indices = [
        matrix.columns.get_loc(column) for column in categorical_columns
    ]
    common: dict[str, Any] = {
        "iterations": iterations,
        "depth": depth,
        "learning_rate": learning_rate,
        "random_seed": seed,
        "task_type": "GPU",
        "devices": "0",
        "verbose": 100,
        "allow_writing_files": False,
        "l2_leaf_reg": 5.0,
        "random_strength": 0.5,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 0.5,
    }
    models: dict[str, CatBoostRegressor] = {}
    losses = {
        "q10": "Quantile:alpha=0.1",
        "q50": "Quantile:alpha=0.5",
        "q90": "Quantile:alpha=0.9",
        "mean": "RMSE",
    }
    for name, loss in losses.items():
        model = CatBoostRegressor(loss_function=loss, **common)
        model.fit(
            matrix.loc[train_mask],
            frame.loc[train_mask, "eqe_max"],
            cat_features=categorical_indices,
            eval_set=(
                matrix.loc[validation_mask],
                frame.loc[validation_mask, "eqe_max"],
            ),
            early_stopping_rounds=150,
        )
        model.save_model(output_dir / f"catboost_{name}.cbm")
        models[name] = model

    predictions = frame.loc[
        evaluation_mask,
        ["id", "paper_id", "doi", "eqe_max", "split"],
    ].copy()
    evaluation_matrix = matrix.loc[evaluation_mask]
    for name, model in models.items():
        predictions[name] = model.predict(evaluation_matrix)
    quantiles = np.sort(predictions[["q10", "q50", "q90"]].to_numpy(), axis=1)
    predictions[["q10", "q50", "q90"]] = quantiles

    observed = predictions["eqe_max"].to_numpy()
    metrics = {
        "evaluation_split": "test" if evaluate_test else "validation",
        "device_count": int(len(predictions)),
        "paper_count": int(predictions["paper_id"].nunique()),
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
        "best_iterations": {
            name: int(model.get_best_iteration()) for name, model in models.items()
        },
    }
    predictions.to_csv(
        output_dir
        / ("test_predictions.csv" if evaluate_test else "validation_predictions.csv"),
        index=False,
    )
    (output_dir / ("test_metrics.json" if evaluate_test else "validation_metrics.json")).write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return BaselineResult(metrics=metrics, predictions=predictions)
