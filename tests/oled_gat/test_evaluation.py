from __future__ import annotations

import numpy as np

from oled_gat.evaluation import (
    conformal_quantile_offset,
    select_convex_blend_weight,
)


def test_convex_blend_selects_lower_rmse_weight() -> None:
    observed = np.asarray([0.0, 1.0, 2.0, 3.0])
    primary = np.asarray([0.0, 1.0, 2.0, 3.0])
    secondary = np.asarray([3.0, 2.0, 1.0, 0.0])

    weight, metrics = select_convex_blend_weight(
        observed,
        primary,
        secondary,
    )

    assert weight == 1.0
    assert metrics["rmse"] == 0.0


def test_conformal_offset_uses_finite_sample_rank() -> None:
    observed = np.asarray([1.0, 2.0, 3.0, 10.0])
    lower = np.asarray([0.0, 1.0, 2.0, 4.0])
    upper = np.asarray([2.0, 3.0, 4.0, 6.0])

    offset = conformal_quantile_offset(
        observed,
        lower,
        upper,
        alpha=0.25,
    )

    assert offset == 4.0
