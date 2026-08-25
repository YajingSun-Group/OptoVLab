from __future__ import annotations

import pandas as pd

from oled_gat.graph_data import (
    ATOM_NODE,
    LAYER_NODE,
    MATERIAL_NODE,
    ROOT_NODE,
    build_graphs,
)


def test_hierarchical_graph_contains_device_layers_material_and_atoms() -> None:
    manifest = pd.DataFrame(
        [
            {
                "id": "D1",
                "paper_id": "P1",
                "split": "train",
                "eqe_max": 20.0,
                "layer_count": 2,
                "material_count": 2,
                "scope_device_count": 1,
                "primary_mechanism": "TADF",
                "color_group": "green",
                "fabrication_method": "vacuum_deposition",
                "device_type": "bottom_emission",
            }
        ]
    )
    layers = pd.DataFrame(
        [
            {
                "device_id": "D1",
                "layer_index": 0,
                "layer_role": "anode",
                "thickness_value": 100.0,
            },
            {
                "device_id": "D1",
                "layer_index": 1,
                "layer_role": "EML",
                "thickness_value": 20.0,
            },
        ]
    )
    components = pd.DataFrame(
        [
            {
                "device_id": "D1",
                "layer_index": 0,
                "layer_role": "anode",
                "paper_material_id": "M1",
                "material_key": "ito",
                "material_mention": "ITO",
                "component_role": "electrode_material",
                "ratio": None,
            },
            {
                "device_id": "D1",
                "layer_index": 1,
                "layer_role": "EML",
                "paper_material_id": "M2",
                "material_key": "emitter",
                "material_mention": "Emitter",
                "component_role": "final_emitter",
                "ratio": '{"value": 10, "unit": "wt%"}',
            },
        ]
    )
    materials = pd.DataFrame(
        [
            {
                "device_id": "D1",
                "paper_material_id": "M1",
                "global_material_id": "G1",
                "material_key": "ito",
                "normalized_name": "ITO",
                "abbreviation": "ITO",
                "canonical_smiles": None,
            },
            {
                "device_id": "D1",
                "paper_material_id": "M2",
                "global_material_id": "G2",
                "material_key": "emitter",
                "normalized_name": "Emitter",
                "abbreviation": "E",
                "canonical_smiles": "c1ccccc1",
            },
        ]
    )

    graphs, _, metadata = build_graphs(
        manifest,
        layers,
        components,
        materials,
        material_vocabulary_size=16,
        maximum_atoms_per_molecule=32,
        molecular_fingerprint_size=32,
    )

    graph = graphs[0]
    assert (graph.node_type == ROOT_NODE).sum().item() == 1
    assert (graph.node_type == LAYER_NODE).sum().item() == 2
    assert (graph.node_type == MATERIAL_NODE).sum().item() == 2
    assert (graph.node_type == ATOM_NODE).sum().item() == 6
    assert graph.num_edges > 20
    assert graph.molecular_features.shape[1] == 44
    assert graph.molecular_features.abs().sum().item() > 0
    assert metadata["graph_count"] == 1
