from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem
from sklearn.model_selection import StratifiedKFold


@dataclass(frozen=True)
class SourceTables:
    devices: pd.DataFrame
    layers: pd.DataFrame
    components: pd.DataFrame
    materials: pd.DataFrame


@dataclass(frozen=True)
class PreparedData:
    manifest: pd.DataFrame
    layers: pd.DataFrame
    components: pd.DataFrame
    materials: pd.DataFrame
    metadata: dict[str, Any]


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return payload


def load_source_tables(input_dir: Path) -> SourceTables:
    return SourceTables(
        devices=pd.read_parquet(input_dir / "primary_scope_devices.parquet"),
        layers=pd.read_parquet(input_dir / "normalized_layers.parquet"),
        components=pd.read_parquet(
            input_dir / "normalized_layer_components.parquet"
        ),
        materials=pd.read_parquet(input_dir / "normalized_materials.parquet"),
    )


def _valid_smiles(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and Chem.MolFromSmiles(value) is not None


def preferred_materials(materials: pd.DataFrame) -> pd.DataFrame:
    frame = materials.copy()
    frame["_confirmed"] = frame["structure_review_status"].eq("confirmed").astype(int)
    frame["_valid_smiles"] = frame["canonical_smiles"].map(_valid_smiles).astype(int)
    frame["_confidence"] = frame["match_confidence"].fillna(-1)
    return (
        frame.sort_values(
            ["_confirmed", "_valid_smiles", "_confidence"],
            ascending=False,
        )
        .drop_duplicates(["device_id", "paper_material_id"])
        .drop(columns=["_confirmed", "_valid_smiles", "_confidence"])
    )


def _paper_folds(
    manifest: pd.DataFrame,
    *,
    folds: int,
    target_bins: int,
    seed: int,
) -> pd.DataFrame:
    papers = (
        manifest.groupby("paper_id", as_index=False)
        .agg(
            paper_median_eqe=("eqe_max", "median"),
            scope_device_count=("id", "size"),
        )
        .sort_values("paper_id")
        .reset_index(drop=True)
    )
    papers["target_bin"] = pd.qcut(
        papers["paper_median_eqe"],
        q=target_bins,
        labels=False,
        duplicates="drop",
    ).astype(int)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    papers["fold"] = -1
    for fold, (_, holdout) in enumerate(
        splitter.split(papers, papers["target_bin"])
    ):
        papers.loc[holdout, "fold"] = fold
    if papers["fold"].lt(0).any():
        raise RuntimeError("Not every paper received a fold")
    return papers[
        [
            "paper_id",
            "paper_median_eqe",
            "scope_device_count",
            "target_bin",
            "fold",
        ]
    ]


def _device_folds(
    manifest: pd.DataFrame,
    *,
    folds: int,
    target_bins: int,
    seed: int,
) -> pd.DataFrame:
    devices = manifest[["id", "eqe_max"]].sort_values("id").reset_index(drop=True)
    devices["target_bin"] = pd.qcut(
        devices["eqe_max"],
        q=target_bins,
        labels=False,
        duplicates="drop",
    ).astype(int)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    devices["fold"] = -1
    for fold, (_, holdout) in enumerate(
        splitter.split(devices, devices["target_bin"])
    ):
        devices.loc[holdout, "fold"] = fold
    if devices["fold"].lt(0).any():
        raise RuntimeError("Not every device received a fold")
    return devices[["id", "target_bin", "fold"]]


def _campaign_assignments(
    manifest: pd.DataFrame,
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> pd.DataFrame:
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between zero and 0.5")
    if not 0.0 < test_fraction < 0.5:
        raise ValueError("test_fraction must be between zero and 0.5")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("campaign validation and test fractions must sum below one")
    assignments: list[dict[str, Any]] = []
    for paper_id, group in manifest.groupby("paper_id", sort=True):
        identifiers = group["id"].astype(str).sort_values().to_numpy()
        paper_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{paper_id}".encode()).digest()[:8],
            byteorder="little",
        )
        generator = np.random.default_rng(paper_seed)
        generator.shuffle(identifiers)
        count = len(identifiers)
        test_count = max(1, int(round(count * test_fraction)))
        validation_count = max(1, int(round(count * validation_fraction)))
        if test_count + validation_count >= count:
            validation_count = 1
            test_count = 1
        for position, identifier in enumerate(identifiers):
            if position < test_count:
                split = "test"
                fold = 0
            elif position < test_count + validation_count:
                split = "validation"
                fold = 1
            else:
                split = "train"
                fold = 2
            assignments.append(
                {
                    "id": identifier,
                    "fold": fold,
                    "split": split,
                }
            )
    frame = pd.DataFrame(assignments)
    target_by_id = manifest.set_index("id")["eqe_max"]
    frame["target_bin"] = pd.qcut(
        frame["id"].map(target_by_id),
        q=5,
        labels=False,
        duplicates="drop",
    ).astype(int)
    return frame


def _dataset_fingerprint(manifest: pd.DataFrame) -> str:
    stable = manifest[["id", "paper_id", "eqe_max", "split"]].sort_values("id")
    payload = stable.to_csv(index=False, float_format="%.8g").encode()
    return hashlib.sha256(payload).hexdigest()


def prepare_data(tables: SourceTables, config: dict[str, Any]) -> PreparedData:
    scope = config["scope"]
    split_config = config["split"]
    devices = tables.devices.copy()

    valid = (
        devices["eqe_max"].notna()
        & devices["eqe_max_plausible"]
        & devices["eqe_max"].between(
            float(scope["eqe_min"]),
            float(scope["eqe_max"]),
            inclusive="both",
        )
        & devices["final_emitter_smiles"].map(_valid_smiles)
    )
    if scope.get("exclude_tandem", True):
        valid &= ~devices["is_tandem"].fillna(False)
    if scope.get("exclude_white", True):
        valid &= ~devices["is_white_oled"].fillna(False)

    manifest = devices.loc[valid].copy()
    device_ids = set(manifest["id"].astype(str))
    layers = tables.layers.loc[tables.layers["device_id"].isin(device_ids)].copy()
    components = tables.components.loc[
        tables.components["device_id"].isin(device_ids)
    ].copy()
    materials = preferred_materials(
        tables.materials.loc[tables.materials["device_id"].isin(device_ids)]
    )

    if scope.get("exclude_outcoupling_layers", True):
        roles = set(scope["outcoupling_roles"])
        excluded_ids = set(
            layers.loc[layers["layer_role"].isin(roles), "device_id"].astype(str)
        )
        manifest = manifest.loc[~manifest["id"].isin(excluded_ids)].copy()

    devices_with_layers = set(layers["device_id"].astype(str))
    manifest = manifest.loc[manifest["id"].isin(devices_with_layers)].copy()

    eml = components.loc[components["layer_role"].eq("EML")].merge(
        materials[
            [
                "device_id",
                "paper_material_id",
                "canonical_smiles",
            ]
        ],
        on=["device_id", "paper_material_id"],
        how="left",
    )
    eml_status = (
        eml.groupby("device_id")
        .agg(
            eml_component_count=("paper_material_id", "size"),
            eml_smiles_count=(
                "canonical_smiles",
                lambda values: int(values.map(_valid_smiles).sum()),
            ),
        )
        .reset_index()
    )
    eml_status["all_eml_smiles"] = (
        eml_status["eml_component_count"].gt(0)
        & eml_status["eml_component_count"].eq(eml_status["eml_smiles_count"])
    )
    manifest = manifest.merge(
        eml_status,
        left_on="id",
        right_on="device_id",
        how="left",
    ).drop(columns="device_id")
    manifest["all_eml_smiles"] = manifest["all_eml_smiles"].fillna(False)
    if scope.get("require_all_eml_smiles", True):
        manifest = manifest.loc[manifest["all_eml_smiles"]].copy()

    manifest = (
        manifest.sort_values(["paper_id", "device_index", "id"])
        .drop_duplicates("id")
        .reset_index(drop=True)
    )
    paper_stats = (
        manifest.groupby("paper_id", as_index=False)
        .agg(
            paper_median_eqe=("eqe_max", "median"),
            scope_device_count=("id", "size"),
        )
    )
    split_mode = str(split_config.get("mode", "paper_grouped"))
    if split_mode == "paper_grouped":
        assignments = _paper_folds(
            manifest,
            folds=int(split_config["folds"]),
            target_bins=int(split_config["target_bins"]),
            seed=int(config["seed"]),
        )
        manifest = manifest.merge(assignments, on="paper_id", how="left")
    elif split_mode == "device_random":
        assignments = _device_folds(
            manifest,
            folds=int(split_config["folds"]),
            target_bins=int(split_config["target_bins"]),
            seed=int(config["seed"]),
        )
        manifest = manifest.merge(paper_stats, on="paper_id", how="left")
        manifest = manifest.merge(assignments, on="id", how="left")
    elif split_mode == "within_paper_campaign":
        minimum_devices = int(split_config.get("minimum_devices_per_paper", 5))
        eligible_papers = set(
            paper_stats.loc[
                paper_stats["scope_device_count"].ge(minimum_devices),
                "paper_id",
            ].astype(str)
        )
        manifest = manifest.loc[
            manifest["paper_id"].astype(str).isin(eligible_papers)
        ].copy()
        paper_stats = (
            manifest.groupby("paper_id", as_index=False)
            .agg(
                paper_median_eqe=("eqe_max", "median"),
                scope_device_count=("id", "size"),
            )
        )
        assignments = _campaign_assignments(
            manifest,
            validation_fraction=float(
                split_config.get("validation_fraction", 0.15)
            ),
            test_fraction=float(split_config.get("test_fraction", 0.15)),
            seed=int(config["seed"]),
        )
        manifest = manifest.merge(paper_stats, on="paper_id", how="left")
        manifest = manifest.merge(assignments, on="id", how="left")
    else:
        raise ValueError(f"Unsupported split mode: {split_mode}")
    if split_mode != "within_paper_campaign":
        manifest["split"] = np.select(
            [
                manifest["fold"].eq(int(split_config["test_fold"])),
                manifest["fold"].eq(int(split_config["validation_fold"])),
            ],
            ["test", "validation"],
            default="train",
        )

    kept_ids = set(manifest["id"].astype(str))
    layers = layers.loc[layers["device_id"].isin(kept_ids)].copy()
    components = components.loc[components["device_id"].isin(kept_ids)].copy()
    materials = materials.loc[materials["device_id"].isin(kept_ids)].copy()

    split_papers = {
        split: set(group["paper_id"].astype(str))
        for split, group in manifest.groupby("split")
    }
    doi_disjoint = split_mode == "paper_grouped"
    if doi_disjoint:
        if split_papers["train"] & split_papers["validation"]:
            raise RuntimeError("Train/validation paper leakage")
        if split_papers["train"] & split_papers["test"]:
            raise RuntimeError("Train/test paper leakage")
        if split_papers["validation"] & split_papers["test"]:
            raise RuntimeError("Validation/test paper leakage")
    elif split_mode == "within_paper_campaign":
        expected_papers = set(manifest["paper_id"].astype(str))
        for split in ("train", "validation", "test"):
            if split_papers[split] != expected_papers:
                raise RuntimeError(
                    f"Campaign split {split} does not cover every eligible paper"
                )

    split_summary = (
        manifest.groupby("split")
        .agg(
            devices=("id", "size"),
            papers=("paper_id", "nunique"),
            eqe_median=("eqe_max", "median"),
            eqe_mean=("eqe_max", "mean"),
            eqe_std=("eqe_max", "std"),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    metadata = {
        "dataset_fingerprint": _dataset_fingerprint(manifest),
        "seed": int(config["seed"]),
        "device_count": int(len(manifest)),
        "paper_count": int(manifest["paper_id"].nunique()),
        "split_summary": split_summary,
        "scope": scope,
        "split_mode": split_mode,
        "doi_disjoint": doi_disjoint,
    }
    return PreparedData(
        manifest=manifest,
        layers=layers,
        components=components,
        materials=materials,
        metadata=metadata,
    )


def write_prepared_data(data: PreparedData, output_dir: Path) -> None:
    prepared_dir = output_dir / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    data.manifest.to_parquet(prepared_dir / "manifest.parquet", index=False)
    data.manifest.to_csv(prepared_dir / "manifest.csv", index=False)
    data.layers.to_parquet(prepared_dir / "layers.parquet", index=False)
    data.components.to_parquet(prepared_dir / "components.parquet", index=False)
    data.materials.to_parquet(prepared_dir / "materials.parquet", index=False)
    (prepared_dir / "dataset_metadata.json").write_text(
        json.dumps(data.metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
