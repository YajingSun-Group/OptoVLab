from __future__ import annotations

import numpy as np

from oled_gat.features import MOLECULAR_DESCRIPTOR_NAMES, _parse_ratio, molecular_vector
from oled_gat.metrics import interval_metrics, pinball_loss


def test_molecular_vector_is_deterministic() -> None:
    first = molecular_vector("c1ccccc1", fingerprint_size=64)
    second = molecular_vector("c1ccccc1", fingerprint_size=64)

    assert first.shape == (len(MOLECULAR_DESCRIPTOR_NAMES) + 64,)
    np.testing.assert_array_equal(first, second)
    assert first.sum() > 0


def test_ratio_and_interval_metrics() -> None:
    assert _parse_ratio('{"value": 12, "unit": "wt%"}') == 12
    assert np.isnan(_parse_ratio('{"value": 30, "unit": "nm"}'))
    observed = np.asarray([1.0, 2.0, 3.0])
    lower = np.asarray([0.0, 1.5, 4.0])
    upper = np.asarray([2.0, 2.5, 5.0])

    metrics = interval_metrics(observed, lower, upper)
    assert metrics["coverage"] == 2 / 3
    assert metrics["crossing_count"] == 0
    assert pinball_loss(observed, observed, 0.5) == 0.0
