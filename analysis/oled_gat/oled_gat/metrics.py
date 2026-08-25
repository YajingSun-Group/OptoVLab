from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def regression_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    correlation = (
        float("nan")
        if np.ptp(observed) == 0 or np.ptp(predicted) == 0
        else float(spearmanr(observed, predicted).statistic)
    )
    return {
        "mae": float(mean_absolute_error(observed, predicted)),
        "rmse": float(root_mean_squared_error(observed, predicted)),
        "r2": float(r2_score(observed, predicted)),
        "spearman": correlation,
    }


def pinball_loss(
    observed: np.ndarray,
    predicted: np.ndarray,
    quantile: float,
) -> float:
    residual = observed - predicted
    return float(np.mean(np.maximum(quantile * residual, (quantile - 1) * residual)))


def interval_metrics(
    observed: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, Any]:
    covered = (observed >= lower) & (observed <= upper)
    return {
        "coverage": float(covered.mean()),
        "mean_width": float(np.mean(upper - lower)),
        "median_width": float(np.median(upper - lower)),
        "crossing_count": int(np.sum(lower > upper)),
    }
