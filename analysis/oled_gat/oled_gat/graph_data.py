from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data

from .features import MOLECULAR_DESCRIPTOR_NAMES, _parse_ratio, molecular_vector


ROOT_NODE = 0
LAYER_NODE = 1
MATERIAL_NODE = 2
ATOM_NODE = 3

EDGE_TYPES = [
    "root_to_layer",
    "layer_to_root",
    "next_layer",
    "previous_layer",
    "layer_to_material",
    "material_to_layer",
    "material_to_atom",
    "atom_to_material",
    "bond_single",
    "bond_double",
    "bond_triple",
    "bond_aromatic",
    "bond_other",
]
EDGE_TYPE_TO_ID = {name: index for index, name in enumerate(EDGE_TYPES)}

HYBRIDIZATION_TO_ID = {
    Chem.HybridizationType.UNSPECIFIED: 1,
    Chem.HybridizationType.S: 2,
    Chem.HybridizationType.SP: 3,
    Chem.HybridizationType.SP2: 4,
    Chem.HybridizationType.SP3: 5,
    Chem.HybridizationType.SP3D: 6,
    Chem.HybridizationType.SP3D2: 7,
}
CHIRALITY_TO_ID = {
    Chem.ChiralType.CHI_UNSPECIFIED: 1,
    Chem.ChiralType.CHI_TETRAHEDRAL_CW: 2,
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW: 3,
    Chem.ChiralType.CHI_OTHER: 4,
}


@dataclass(frozen=True)
class NumericStats:
    thickness_mean: float
    thickness_std: float
    ratio_mean: float
    ratio_std: float
    layer_count_mean: float
    layer_count_std: float
    material_count_mean: float
    material_count_std: float
    target_mean: float
    target_std: float
    molecular_descriptor_mean: list[float]
    molecular_descriptor_std: list[float]


@dataclass(frozen=True)
class GraphVocabulary:
    layer_roles: list[str]
    component_roles: list[str]
    materials: list[str]
    mechanisms: list[str]
    colors: list[str]
    processes: list[str]
    device_types: list[str]
    edge_types: list[str]
    molecular_fingerprint_size: int
    numeric_stats: NumericStats


def graph_vocabulary_from_dict(payload: dict[str, Any]) -> GraphVocabulary:
    values = dict(payload)
    values["numeric_stats"] = NumericStats(**values["numeric_stats"])
    return GraphVocabulary(**values)


def _safe_std(values: pd.Series) -> float:
    value = float(values.std())
    return value if np.isfinite(value) and value > 1e-8 else 1.0


def _text(value: Any) -> str:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return "unknown"
    text = str(value).strip().casefold()
    return text if text else "unknown"


def _with_reserved(values: pd.Series, maximum_size: int | None = None) -> list[str]:
    counts = values.dropna().map(_text).value_counts()
    selected = counts.index.tolist()
    if maximum_size is not None:
        selected = selected[: max(0, maximum_size - 2)]
    return ["<pad>", "<oov>", *selected]


def _material_token(row: pd.Series) -> str:
    for column in (
        "global_material_id",
        "material_key_resolved",
        "material_key",
        "normalized_name",
        "abbreviation",
    ):
        value = row.get(column)
        if isinstance(value, str) and value.strip():
            return value.strip().casefold()
    return "unknown"


def link_components(
    components: pd.DataFrame,
    materials: pd.DataFrame,
) -> pd.DataFrame:
    material_columns = [
        "device_id",
        "paper_material_id",
        "global_material_id",
        "material_key",
        "normalized_name",
        "abbreviation",
        "canonical_smiles",
    ]
    linked = components.merge(
        materials[material_columns],
        on=["device_id", "paper_material_id"],
        how="left",
        suffixes=("", "_resolved"),
    )
    linked["material_token"] = linked.apply(_material_token, axis=1)
    linked["ratio_pct"] = linked["ratio"].map(_parse_ratio)
    return linked


def fit_graph_vocabulary(
    manifest: pd.DataFrame,
    layers: pd.DataFrame,
    linked_components: pd.DataFrame,
    *,
    material_vocabulary_size: int,
    molecular_fingerprint_size: int,
) -> GraphVocabulary:
    train = manifest.loc[manifest["split"].eq("train")].copy()
    train_ids = set(train["id"].astype(str))
    train_layers = layers.loc[layers["device_id"].isin(train_ids)]
    train_components = linked_components.loc[
        linked_components["device_id"].isin(train_ids)
    ]

    thickness_log = np.log1p(
        pd.to_numeric(train_layers["thickness_value"], errors="coerce").dropna()
    )
    ratios = pd.to_numeric(train_components["ratio_pct"], errors="coerce").dropna()
    train_eml_smiles = (
        train_components.loc[train_components["layer_role"].eq("EML"), "canonical_smiles"]
        .dropna()
        .astype(str)
        .drop_duplicates()
    )
    molecular_descriptors = np.stack(
        [
            molecular_vector(smiles, molecular_fingerprint_size)[
                : len(MOLECULAR_DESCRIPTOR_NAMES)
            ]
            for smiles in train_eml_smiles
        ]
    )
    descriptor_mean = molecular_descriptors.mean(axis=0)
    descriptor_std = molecular_descriptors.std(axis=0)
    descriptor_std[descriptor_std < 1e-8] = 1.0
    stats = NumericStats(
        thickness_mean=float(thickness_log.mean()),
        thickness_std=_safe_std(thickness_log),
        ratio_mean=float(ratios.mean()) if not ratios.empty else 0.0,
        ratio_std=_safe_std(ratios) if not ratios.empty else 1.0,
        layer_count_mean=float(train["layer_count"].mean()),
        layer_count_std=_safe_std(train["layer_count"]),
        material_count_mean=float(train["material_count"].mean()),
        material_count_std=_safe_std(train["material_count"]),
        target_mean=float(train["eqe_max"].mean()),
        target_std=_safe_std(train["eqe_max"]),
        molecular_descriptor_mean=descriptor_mean.astype(float).tolist(),
        molecular_descriptor_std=descriptor_std.astype(float).tolist(),
    )
    return GraphVocabulary(
        layer_roles=_with_reserved(train_layers["layer_role"]),
        component_roles=_with_reserved(train_components["component_role"]),
        materials=_with_reserved(
            train_components["material_token"],
            maximum_size=material_vocabulary_size,
        ),
        mechanisms=_with_reserved(train["primary_mechanism"]),
        colors=_with_reserved(train["color_group"]),
        processes=_with_reserved(train["fabrication_method"]),
        device_types=_with_reserved(train["device_type"]),
        edge_types=EDGE_TYPES,
        molecular_fingerprint_size=molecular_fingerprint_size,
        numeric_stats=stats,
    )


def _lookup(values: list[str]) -> dict[str, int]:
    return {value: index for index, value in enumerate(values)}


def _category(value: Any, lookup: dict[str, int]) -> int:
    return lookup.get(_text(value), 1)


def _zscore(value: float, mean: float, std: float) -> float:
    return (value - mean) / std


def _bond_edge_type(bond: Chem.Bond) -> int:
    bond_type = bond.GetBondType()
    if bond_type == Chem.BondType.SINGLE:
        return EDGE_TYPE_TO_ID["bond_single"]
    if bond_type == Chem.BondType.DOUBLE:
        return EDGE_TYPE_TO_ID["bond_double"]
    if bond_type == Chem.BondType.TRIPLE:
        return EDGE_TYPE_TO_ID["bond_triple"]
    if bond_type == Chem.BondType.AROMATIC:
        return EDGE_TYPE_TO_ID["bond_aromatic"]
    return EDGE_TYPE_TO_ID["bond_other"]


class DeviceGraphBuilder:
    def __init__(
        self,
        vocabulary: GraphVocabulary,
        *,
        maximum_atoms_per_molecule: int,
        molecular_fingerprint_size: int,
    ) -> None:
        self.vocabulary = vocabulary
        self.maximum_atoms_per_molecule = maximum_atoms_per_molecule
        self.molecular_fingerprint_size = molecular_fingerprint_size
        self.molecular_feature_size = (
            len(MOLECULAR_DESCRIPTOR_NAMES) + molecular_fingerprint_size
        )
        self.layer_lookup = _lookup(vocabulary.layer_roles)
        self.component_lookup = _lookup(vocabulary.component_roles)
        self.material_lookup = _lookup(vocabulary.materials)
        self.mechanism_lookup = _lookup(vocabulary.mechanisms)
        self.color_lookup = _lookup(vocabulary.colors)
        self.process_lookup = _lookup(vocabulary.processes)
        self.device_type_lookup = _lookup(vocabulary.device_types)

    def build(
        self,
        device: pd.Series,
        layers: pd.DataFrame,
        components: pd.DataFrame,
        *,
        sample_row: int,
    ) -> Data:
        node_type: list[int] = [ROOT_NODE]
        layer_role: list[int] = [0]
        component_role: list[int] = [0]
        material_id: list[int] = [0]
        atomic_number: list[int] = [0]
        degree: list[int] = [0]
        formal_charge: list[int] = [4]
        hybridization: list[int] = [0]
        chirality: list[int] = [0]
        node_numeric: list[list[float]] = [[0.0] * 10]
        molecular_features: list[list[float]] = [
            [0.0] * self.molecular_feature_size
        ]
        edge_source: list[int] = []
        edge_target: list[int] = []
        edge_type: list[int] = []
        stats = self.vocabulary.numeric_stats

        def add_node(
            *,
            kind: int,
            layer_role_id: int = 0,
            component_role_id: int = 0,
            material: int = 0,
            atom_number: int = 0,
            atom_degree: int = 0,
            charge: int = 4,
            hybrid: int = 0,
            chiral: int = 0,
            numeric: list[float] | None = None,
            molecular: list[float] | None = None,
        ) -> int:
            index = len(node_type)
            node_type.append(kind)
            layer_role.append(layer_role_id)
            component_role.append(component_role_id)
            material_id.append(material)
            atomic_number.append(atom_number)
            degree.append(atom_degree)
            formal_charge.append(charge)
            hybridization.append(hybrid)
            chirality.append(chiral)
            node_numeric.append(numeric or [0.0] * 10)
            molecular_features.append(
                molecular or [0.0] * self.molecular_feature_size
            )
            return index

        def add_edge(source: int, target: int, kind: str | int) -> None:
            edge_source.append(source)
            edge_target.append(target)
            edge_type.append(
                EDGE_TYPE_TO_ID[kind] if isinstance(kind, str) else kind
            )

        ordered_layers = layers.sort_values("layer_index").reset_index(drop=True)
        layer_nodes: dict[Any, int] = {}
        denominator = max(1, len(ordered_layers) - 1)
        for position, layer in ordered_layers.iterrows():
            thickness = pd.to_numeric(
                pd.Series([layer["thickness_value"]]),
                errors="coerce",
            ).iloc[0]
            missing = float(pd.isna(thickness))
            thickness_z = (
                0.0
                if missing
                else _zscore(math.log1p(float(thickness)), stats.thickness_mean, stats.thickness_std)
            )
            role_id = _category(layer["layer_role"], self.layer_lookup)
            is_eml = float(_text(layer["layer_role"]) == "eml")
            layer_node = add_node(
                kind=LAYER_NODE,
                layer_role_id=role_id,
                numeric=[
                    position / denominator,
                    thickness_z,
                    missing,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    is_eml,
                ],
            )
            layer_nodes[layer["layer_index"]] = layer_node
            add_edge(0, layer_node, "root_to_layer")
            add_edge(layer_node, 0, "layer_to_root")
            if position:
                previous = layer_nodes[ordered_layers.iloc[position - 1]["layer_index"]]
                add_edge(previous, layer_node, "next_layer")
                add_edge(layer_node, previous, "previous_layer")

        for _, component in components.iterrows():
            parent = layer_nodes.get(component["layer_index"])
            if parent is None:
                continue
            ratio = component["ratio_pct"]
            ratio_missing = float(pd.isna(ratio))
            ratio_z = (
                0.0
                if ratio_missing
                else _zscore(float(ratio), stats.ratio_mean, stats.ratio_std)
            )
            smiles = component["canonical_smiles"]
            molecular_feature: list[float] | None = None
            if (
                _text(component["layer_role"]) == "eml"
                and isinstance(smiles, str)
            ):
                vector = molecular_vector(
                    smiles,
                    self.molecular_fingerprint_size,
                ).copy()
                descriptor_count = len(MOLECULAR_DESCRIPTOR_NAMES)
                vector[:descriptor_count] = (
                    vector[:descriptor_count]
                    - np.asarray(stats.molecular_descriptor_mean)
                ) / np.asarray(stats.molecular_descriptor_std)
                molecular_feature = vector.astype(float).tolist()
            material_node = add_node(
                kind=MATERIAL_NODE,
                component_role_id=_category(
                    component["component_role"],
                    self.component_lookup,
                ),
                material=_category(
                    component["material_token"],
                    self.material_lookup,
                ),
                numeric=[
                    0.0,
                    0.0,
                    1.0,
                    ratio_z,
                    ratio_missing,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    float(_text(component["layer_role"]) == "eml"),
                ],
                molecular=molecular_feature,
            )
            add_edge(parent, material_node, "layer_to_material")
            add_edge(material_node, parent, "material_to_layer")

            if _text(component["layer_role"]) != "eml" or not isinstance(smiles, str):
                continue
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None or molecule.GetNumAtoms() > self.maximum_atoms_per_molecule:
                continue
            atom_nodes: list[int] = []
            for atom in molecule.GetAtoms():
                charge = int(np.clip(atom.GetFormalCharge(), -4, 4)) + 4
                atom_node = add_node(
                    kind=ATOM_NODE,
                    atom_number=min(118, atom.GetAtomicNum()),
                    atom_degree=min(8, atom.GetDegree()),
                    charge=charge,
                    hybrid=HYBRIDIZATION_TO_ID.get(atom.GetHybridization(), 0),
                    chiral=CHIRALITY_TO_ID.get(atom.GetChiralTag(), 0),
                    numeric=[
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        1.0,
                        atom.GetMass() / 200.0,
                        float(atom.GetIsAromatic()),
                        float(atom.IsInRing()),
                        atom.GetDegree() / 8.0,
                        1.0,
                    ],
                )
                atom_nodes.append(atom_node)
                add_edge(material_node, atom_node, "material_to_atom")
                add_edge(atom_node, material_node, "atom_to_material")
            for bond in molecule.GetBonds():
                source = atom_nodes[bond.GetBeginAtomIdx()]
                target = atom_nodes[bond.GetEndAtomIdx()]
                kind = _bond_edge_type(bond)
                add_edge(source, target, kind)
                add_edge(target, source, kind)

        paper_device_count = max(1, int(device["scope_device_count"]))
        sample_weight = 1.0 / math.sqrt(paper_device_count)
        target = float(device["eqe_max"])
        target_normalized = _zscore(target, stats.target_mean, stats.target_std)
        context_numeric = [
            _zscore(float(device["layer_count"]), stats.layer_count_mean, stats.layer_count_std),
            _zscore(
                float(device["material_count"]),
                stats.material_count_mean,
                stats.material_count_std,
            ),
        ]
        return Data(
            node_type=torch.tensor(node_type, dtype=torch.long),
            layer_role=torch.tensor(layer_role, dtype=torch.long),
            component_role=torch.tensor(component_role, dtype=torch.long),
            material_id=torch.tensor(material_id, dtype=torch.long),
            atomic_number=torch.tensor(atomic_number, dtype=torch.long),
            degree=torch.tensor(degree, dtype=torch.long),
            formal_charge=torch.tensor(formal_charge, dtype=torch.long),
            hybridization=torch.tensor(hybridization, dtype=torch.long),
            chirality=torch.tensor(chirality, dtype=torch.long),
            node_numeric=torch.tensor(node_numeric, dtype=torch.float32),
            molecular_features=torch.tensor(
                molecular_features,
                dtype=torch.float32,
            ),
            edge_index=torch.tensor(
                [edge_source, edge_target],
                dtype=torch.long,
            ),
            edge_type=torch.tensor(edge_type, dtype=torch.long),
            mechanism_id=torch.tensor(
                [_category(device["primary_mechanism"], self.mechanism_lookup)],
                dtype=torch.long,
            ),
            color_id=torch.tensor(
                [_category(device["color_group"], self.color_lookup)],
                dtype=torch.long,
            ),
            process_id=torch.tensor(
                [_category(device["fabrication_method"], self.process_lookup)],
                dtype=torch.long,
            ),
            device_type_id=torch.tensor(
                [_category(device["device_type"], self.device_type_lookup)],
                dtype=torch.long,
            ),
            context_numeric=torch.tensor(
                [context_numeric],
                dtype=torch.float32,
            ),
            y=torch.tensor([target_normalized], dtype=torch.float32),
            y_raw=torch.tensor([target], dtype=torch.float32),
            sample_weight=torch.tensor([sample_weight], dtype=torch.float32),
            # Avoid names containing "index": PyG offsets such attributes while batching.
            sample_row=torch.tensor([sample_row], dtype=torch.long),
        )


def build_graphs(
    manifest: pd.DataFrame,
    layers: pd.DataFrame,
    components: pd.DataFrame,
    materials: pd.DataFrame,
    *,
    material_vocabulary_size: int,
    maximum_atoms_per_molecule: int,
    molecular_fingerprint_size: int,
) -> tuple[list[Data], GraphVocabulary, dict[str, Any]]:
    linked = link_components(components, materials)
    vocabulary = fit_graph_vocabulary(
        manifest,
        layers,
        linked,
        material_vocabulary_size=material_vocabulary_size,
        molecular_fingerprint_size=molecular_fingerprint_size,
    )
    builder = DeviceGraphBuilder(
        vocabulary,
        maximum_atoms_per_molecule=maximum_atoms_per_molecule,
        molecular_fingerprint_size=molecular_fingerprint_size,
    )
    layer_groups = {key: group for key, group in layers.groupby("device_id")}
    component_groups = {
        key: group for key, group in linked.groupby("device_id")
    }
    graphs: list[Data] = []
    empty_components = linked.iloc[0:0]
    for sample_row, device in manifest.reset_index(drop=True).iterrows():
        device_id = str(device["id"])
        graphs.append(
            builder.build(
                device,
                layer_groups[device_id],
                component_groups.get(device_id, empty_components),
                sample_row=sample_row,
            )
        )
    metadata = {
        "graph_count": len(graphs),
        "node_count": int(sum(graph.num_nodes for graph in graphs)),
        "edge_count": int(sum(graph.num_edges for graph in graphs)),
        "mean_nodes_per_graph": float(
            np.mean([graph.num_nodes for graph in graphs])
        ),
        "maximum_nodes_per_graph": int(
            max(graph.num_nodes for graph in graphs)
        ),
        "split_graph_counts": manifest["split"].value_counts().to_dict(),
    }
    return graphs, vocabulary, metadata


def save_graph_cache(
    graphs: list[Data],
    vocabulary: GraphVocabulary,
    metadata: dict[str, Any],
    output_dir: Path,
) -> None:
    graph_dir = output_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    torch.save(graphs, graph_dir / "device_graphs.pt")
    vocabulary_payload = asdict(vocabulary)
    (graph_dir / "vocabulary.json").write_text(
        json.dumps(vocabulary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (graph_dir / "graph_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_graph_cache(output_dir: Path) -> tuple[list[Data], dict[str, Any]]:
    graph_dir = output_dir / "graphs"
    graphs = torch.load(
        graph_dir / "device_graphs.pt",
        map_location="cpu",
        weights_only=False,
    )
    vocabulary = json.loads((graph_dir / "vocabulary.json").read_text(encoding="utf-8"))
    return graphs, vocabulary
