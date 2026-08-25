from __future__ import annotations

import pandas as pd

from oled_gat.data import (
    _campaign_assignments,
    _dataset_fingerprint,
    _device_folds,
    _paper_folds,
)


def test_paper_folds_are_deterministic_and_disjoint() -> None:
    rows = []
    for paper_index in range(56):
        for device_index in range(2):
            rows.append(
                {
                    "id": f"D{paper_index}_{device_index}",
                    "paper_id": f"P{paper_index}",
                    "eqe_max": float(paper_index + device_index / 10),
                }
            )
    manifest = pd.DataFrame(rows)
    first = _paper_folds(manifest, folds=7, target_bins=4, seed=20260726)
    second = _paper_folds(manifest, folds=7, target_bins=4, seed=20260726)

    pd.testing.assert_frame_equal(first, second)
    assert first["paper_id"].is_unique
    assert set(first["fold"]) == set(range(7))


def test_device_folds_are_deterministic_and_stratified() -> None:
    manifest = pd.DataFrame(
        {
            "id": [f"D{index:03d}" for index in range(100)],
            "eqe_max": [float(index) for index in range(100)],
        }
    )
    first = _device_folds(
        manifest,
        folds=10,
        target_bins=5,
        seed=20260726,
    )
    second = _device_folds(
        manifest,
        folds=10,
        target_bins=5,
        seed=20260726,
    )

    pd.testing.assert_frame_equal(first, second)
    assert first["id"].is_unique
    assert first.groupby("fold").size().eq(10).all()


def test_campaign_assignments_hold_out_every_eligible_paper() -> None:
    manifest = pd.DataFrame(
        [
            {
                "id": f"{paper}-{index}",
                "paper_id": paper,
                "eqe_max": float(index + 1),
            }
            for paper in ("P1", "P2")
            for index in range(6)
        ]
    )
    first = _campaign_assignments(
        manifest,
        validation_fraction=0.15,
        test_fraction=0.15,
        seed=19,
    )
    second = _campaign_assignments(
        manifest,
        validation_fraction=0.15,
        test_fraction=0.15,
        seed=19,
    )

    pd.testing.assert_frame_equal(first, second)
    joined = first.merge(manifest[["id", "paper_id"]], on="id")
    assert set(joined.groupby("split")["paper_id"].nunique()) == {2}
    assert joined.groupby(["paper_id", "split"]).size().ge(1).all()


def test_dataset_fingerprint_changes_with_split() -> None:
    frame = pd.DataFrame(
        [
            {"id": "D1", "paper_id": "P1", "eqe_max": 10.0, "split": "train"},
            {"id": "D2", "paper_id": "P2", "eqe_max": 20.0, "split": "test"},
        ]
    )
    original = _dataset_fingerprint(frame)
    frame.loc[frame["id"].eq("D2"), "split"] = "validation"

    assert original != _dataset_fingerprint(frame)
