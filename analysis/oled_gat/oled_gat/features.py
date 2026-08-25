from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator


HOST_ROLES = {"host", "co_host", "host_donor", "host_acceptor"}
EMITTER_ROLES = {
    "emitter",
    "final_emitter",
    "emitter_dopant",
    "dopant",
}
SENSITIZER_ROLES = {"sensitizer", "sensitizer_dopant", "assistant_dopant"}
MOLECULAR_DESCRIPTOR_NAMES = [
    "mol_weight",
    "logp",
    "tpsa",
    "hbd",
    "hba",
    "rotatable_bonds",
    "ring_count",
    "fraction_csp3",
    "heavy_atoms",
    "hetero_atoms",
    "aromatic_fraction",
    "formal_charge",
]


@dataclass(frozen=True)
class FeatureBundle:
    frame: pd.DataFrame
    numeric_columns: list[str]
    categorical_columns: list[str]
    metadata: dict[str, Any]


def _text(value: Any) -> str:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return "unknown"
    text = str(value).strip()
    return text if text else "unknown"


def _feature_key(value: Any) -> str:
    return (
        _text(value)
        .casefold()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )


def _material_token(row: pd.Series) -> str:
    for column in ("global_material_id", "material_key", "normalized_name", "abbreviation"):
        value = row.get(column)
        if isinstance(value, str) and value.strip():
            return value.strip().casefold()
    return "unknown"


@lru_cache(maxsize=8192)
def molecular_vector(smiles: str, fingerprint_size: int = 256) -> np.ndarray:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return np.zeros(len(MOLECULAR_DESCRIPTOR_NAMES) + fingerprint_size, dtype=np.float32)
    atom_count = max(1, molecule.GetNumAtoms())
    descriptors = np.asarray(
        [
            Descriptors.MolWt(molecule),
            Crippen.MolLogP(molecule),
            Descriptors.TPSA(molecule),
            Lipinski.NumHDonors(molecule),
            Lipinski.NumHAcceptors(molecule),
            Lipinski.NumRotatableBonds(molecule),
            Lipinski.RingCount(molecule),
            Descriptors.FractionCSP3(molecule),
            Lipinski.HeavyAtomCount(molecule),
            Lipinski.NumHeteroatoms(molecule),
            sum(atom.GetIsAromatic() for atom in molecule.GetAtoms()) / atom_count,
            Chem.GetFormalCharge(molecule),
        ],
        dtype=np.float32,
    )
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=fingerprint_size,
        includeChirality=True,
    )
    fingerprint = generator.GetFingerprint(molecule)
    bits = np.zeros(fingerprint_size, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fingerprint, bits)
    return np.concatenate([descriptors, bits])


def _parse_ratio(value: Any) -> float:
    if not isinstance(value, str) or not value.strip():
        return np.nan
    try:
        payload = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return np.nan
    number = payload.get("value")
    unit = str(payload.get("unit") or "").casefold()
    raw = str(payload.get("raw") or "").casefold()
    if not isinstance(number, (int, float)):
        return np.nan
    if "%" not in unit and "%" not in raw and "wt" not in unit and "vol" not in unit:
        return np.nan
    return float(number)


def _fit_material_vocabulary(
    components: pd.DataFrame,
    train_ids: set[str],
    *,
    maximum_size: int,
    minimum_frequency: int,
) -> list[str]:
    counter: Counter[str] = Counter()
    train = components.loc[components["device_id"].isin(train_ids)]
    for device_id, group in train.groupby("device_id"):
        del device_id
        counter.update(set(group["material_token"].dropna().astype(str)))
    return [
        token
        for token, count in counter.most_common(maximum_size)
        if token != "unknown" and count >= minimum_frequency
    ]


def _aggregate_molecular_group(
    linked_components: pd.DataFrame,
    device_ids: list[str],
    roles: set[str] | None,
    prefix: str,
    fingerprint_size: int,
) -> pd.DataFrame:
    frame = linked_components.loc[linked_components["layer_role"].eq("EML")].copy()
    if roles is not None:
        frame = frame.loc[frame["component_role"].isin(roles)]
    vectors: dict[str, np.ndarray] = {}
    width = len(MOLECULAR_DESCRIPTOR_NAMES) + fingerprint_size
    for device_id, group in frame.groupby("device_id"):
        smiles = list(dict.fromkeys(group["canonical_smiles"].dropna().astype(str)))
        if not smiles:
            continue
        matrix = np.stack([molecular_vector(value, fingerprint_size) for value in smiles])
        descriptor_mean = matrix[:, : len(MOLECULAR_DESCRIPTOR_NAMES)].mean(axis=0)
        fingerprint_or = matrix[:, len(MOLECULAR_DESCRIPTOR_NAMES) :].max(axis=0)
        vectors[str(device_id)] = np.concatenate([descriptor_mean, fingerprint_or])
    columns = [
        *(f"{prefix}_{name}" for name in MOLECULAR_DESCRIPTOR_NAMES),
        *(f"{prefix}_morgan_{index:03d}" for index in range(fingerprint_size)),
    ]
    rows = [
        vectors.get(device_id, np.zeros(width, dtype=np.float32))
        for device_id in device_ids
    ]
    return pd.DataFrame(rows, columns=columns)


def build_features(
    manifest: pd.DataFrame,
    layers: pd.DataFrame,
    components: pd.DataFrame,
    materials: pd.DataFrame,
    *,
    material_vocabulary_size: int,
    minimum_material_frequency: int,
    fingerprint_size: int = 256,
) -> FeatureBundle:
    frame = manifest.reset_index(drop=True).copy()
    frame["device_id"] = frame["id"].astype(str)
    device_ids = frame["device_id"].tolist()
    train_ids = set(frame.loc[frame["split"].eq("train"), "device_id"])

    material_lookup = materials[
        [
            "device_id",
            "paper_material_id",
            "global_material_id",
            "material_key",
            "normalized_name",
            "abbreviation",
            "canonical_smiles",
        ]
    ].copy()
    material_lookup["material_token"] = material_lookup.apply(_material_token, axis=1)
    linked = components.merge(
        material_lookup,
        on=["device_id", "paper_material_id"],
        how="left",
        suffixes=("", "_resolved"),
    )
    linked["material_token"] = linked["material_token"].fillna(
        linked["material_key"].map(_text)
    )
    linked["ratio_pct"] = linked["ratio"].map(_parse_ratio)
    linked["layer_index_numeric"] = pd.to_numeric(
        linked["layer_index"],
        errors="coerce",
    ).fillna(10_000)

    vocabulary = _fit_material_vocabulary(
        linked,
        train_ids,
        maximum_size=material_vocabulary_size,
        minimum_frequency=minimum_material_frequency,
    )
    material_sets = (
        linked.groupby("device_id")["material_token"].agg(lambda values: set(values))
    )
    material_features = {
        f"material_{index:04d}": [
            float(token in material_sets.get(device_id, set()))
            for device_id in device_ids
        ]
        for index, token in enumerate(vocabulary)
    }
    frame = pd.concat(
        [frame, pd.DataFrame(material_features, index=frame.index)],
        axis=1,
    )

    train_linked = linked.loc[linked["device_id"].isin(train_ids)]
    role_identity_columns: list[str] = []
    train_component_roles = (
        train_linked.groupby("layer_role")["device_id"]
        .nunique()
        .loc[lambda values: values.ge(10)]
        .index.astype(str)
        .tolist()
    )

    def material_signature(group: pd.DataFrame) -> str:
        ordered = group.sort_values(
            ["layer_index_numeric", "component_role", "material_token"]
        )
        return "|".join(
            dict.fromkeys(
                f"{_text(row.component_role).casefold()}={_text(row.material_token).casefold()}"
                for row in ordered.itertuples()
            )
        )

    for role in sorted(train_component_roles):
        column = f"role_materials_{_feature_key(role)}"
        role_identity_columns.append(column)
        identities = (
            linked.loc[linked["layer_role"].eq(role)]
            .groupby("device_id")
            .apply(material_signature, include_groups=False)
        )
        frame[column] = frame["device_id"].map(identities).fillna("unknown")

    def architecture_signature(group: pd.DataFrame) -> str:
        layers_in_device: list[str] = []
        for _, layer_group in group.sort_values("layer_index_numeric").groupby(
            "layer_index_numeric",
            sort=True,
        ):
            role = _text(layer_group["layer_role"].iloc[0]).casefold()
            materials_in_layer = material_signature(layer_group)
            layers_in_device.append(f"{role}[{materials_in_layer}]")
        return ">".join(layers_in_device)

    architecture_signatures = linked.groupby("device_id").apply(
        architecture_signature,
        include_groups=False,
    )
    frame["architecture_material_sequence"] = (
        frame["device_id"].map(architecture_signatures).fillna("unknown")
    )
    frame["eml_composition_identity"] = frame["device_id"].map(
        linked.loc[linked["layer_role"].eq("EML")]
        .groupby("device_id")
        .apply(material_signature, include_groups=False)
    ).fillna("unknown")

    role_vocabulary = sorted(
        layers.loc[layers["device_id"].isin(train_ids), "layer_role"]
        .dropna()
        .astype(str)
        .unique()
    )
    layer_subset = layers.loc[layers["device_id"].isin(set(device_ids))].copy()
    for role in role_vocabulary:
        role_key = role.casefold().replace(" ", "_").replace("/", "_")
        role_layers = layer_subset.loc[layer_subset["layer_role"].eq(role)]
        count_map = role_layers.groupby("device_id").size()
        sum_map = role_layers.groupby("device_id")["thickness_value"].sum(min_count=1)
        mean_map = role_layers.groupby("device_id")["thickness_value"].mean()
        frame[f"layer_count_{role_key}"] = frame["device_id"].map(count_map).fillna(0)
        frame[f"thickness_sum_{role_key}"] = frame["device_id"].map(sum_map)
        frame[f"thickness_mean_{role_key}"] = frame["device_id"].map(mean_map)
    total_thickness = layer_subset.groupby("device_id")["thickness_value"].sum(min_count=1)
    frame["total_reported_thickness_nm"] = frame["device_id"].map(total_thickness)

    ratio_summary = (
        linked.loc[linked["layer_role"].eq("EML")]
        .groupby("device_id")["ratio_pct"]
        .agg(["min", "max", "mean", "count"])
    )
    for statistic in ("min", "max", "mean", "count"):
        frame[f"eml_ratio_{statistic}"] = frame["device_id"].map(
            ratio_summary[statistic]
        )

    train_ratio_roles = (
        train_linked.loc[
            train_linked["layer_role"].eq("EML")
            & train_linked["ratio_pct"].notna()
        ]
        .groupby("component_role")["device_id"]
        .nunique()
        .loc[lambda values: values.ge(10)]
        .index.astype(str)
        .tolist()
    )
    for component_role in sorted(train_ratio_roles):
        role_key = _feature_key(component_role)
        summary = (
            linked.loc[
                linked["layer_role"].eq("EML")
                & linked["component_role"].eq(component_role)
            ]
            .groupby("device_id")["ratio_pct"]
            .agg(["min", "max", "mean", "count"])
        )
        for statistic in ("min", "max", "mean", "count"):
            frame[f"eml_ratio_{role_key}_{statistic}"] = frame["device_id"].map(
                summary[statistic]
            )

    for roles, prefix in (
        (None, "eml"),
        (HOST_ROLES, "host"),
        (EMITTER_ROLES, "emitter"),
        (SENSITIZER_ROLES, "sensitizer"),
    ):
        molecular = _aggregate_molecular_group(
            linked,
            device_ids,
            roles,
            prefix,
            fingerprint_size,
        )
        frame = pd.concat([frame, molecular], axis=1)

    for roles, column in (
        (HOST_ROLES, "host_identity"),
        (EMITTER_ROLES, "emitter_identity"),
        (SENSITIZER_ROLES, "sensitizer_identity"),
    ):
        identities = (
            linked.loc[linked["layer_role"].eq("EML") & linked["component_role"].isin(roles)]
            .groupby("device_id")["material_token"]
            .agg(lambda values: "|".join(sorted(set(values.astype(str)))))
        )
        frame[column] = frame["device_id"].map(identities).fillna("unknown")

    categorical_columns = [
        "primary_mechanism",
        "color_group",
        "fabrication_method",
        "device_type",
        "layer_sequence",
        "host_identity",
        "emitter_identity",
        "sensitizer_identity",
        "architecture_material_sequence",
        "eml_composition_identity",
        *role_identity_columns,
    ]
    for column in categorical_columns:
        frame[column] = frame[column].map(_text)

    excluded_numeric = {
        "eqe_max",
        "year",
        "paper_median_eqe",
        "scope_device_count",
        "fold",
        "device_index",
        "eml_component_count",
        "eml_smiles_count",
    }
    candidate_numeric = [
        "layer_count",
        "material_count",
        "is_tandem",
        "is_white_oled",
        *material_features.keys(),
        *[
            column
            for column in frame.columns
            if column.startswith(
                (
                    "layer_count_",
                    "thickness_",
                    "total_reported_",
                    "eml_ratio_",
                    "eml_",
                    "host_",
                    "emitter_",
                    "sensitizer_",
                )
            )
            and column not in categorical_columns
        ],
    ]
    numeric_columns = list(
        dict.fromkeys(
            column
            for column in candidate_numeric
            if column in frame
            and column not in excluded_numeric
            and pd.api.types.is_numeric_dtype(frame[column])
        )
    )
    metadata = {
        "material_vocabulary": vocabulary,
        "material_feature_labels": {
            f"material_{index:04d}": token
            for index, token in enumerate(vocabulary)
        },
        "layer_role_vocabulary": role_vocabulary,
        "role_identity_columns": role_identity_columns,
        "ratio_component_roles": train_ratio_roles,
        "fingerprint_size": fingerprint_size,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
    }
    return FeatureBundle(
        frame=frame,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        metadata=metadata,
    )


def load_prepared_frames(output_dir: Path) -> tuple[pd.DataFrame, ...]:
    prepared = output_dir / "prepared"
    return (
        pd.read_parquet(prepared / "manifest.parquet"),
        pd.read_parquet(prepared / "layers.parquet"),
        pd.read_parquet(prepared / "components.parquet"),
        pd.read_parquet(prepared / "materials.parquet"),
    )
