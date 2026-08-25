from __future__ import annotations

from evolab_local.database_site.exporter import (
    _flatten_oled_device,
    _normalize_doi,
    _set_by_path,
    _string_list,
)


def test_normalize_doi_removes_resolver_prefix() -> None:
    assert _normalize_doi("https://doi.org/10.1002/test.1") == "10.1002/test.1"
    assert _normalize_doi("doi:10.1002/test.2") == "10.1002/test.2"


def test_set_by_path_updates_nested_array_value() -> None:
    payload = {"devices": [{"layers": [{"layer_name": "before"}]}]}

    _set_by_path(payload, "devices[0].layers[0].layer_name", "after")

    assert payload["devices"][0]["layers"][0]["layer_name"] == "after"


def test_string_list_repairs_json_encoded_array() -> None:
    assert _string_list('["TADF", "MR_TADF"]') == ["TADF", "MR_TADF"]


def test_flatten_oled_device_preserves_variable_layers_and_performance() -> None:
    record = _flatten_oled_device(
        paper={
            "paper_id": "10.1002%2Ftest",
            "doi": "10.1002/test",
            "title": "Test OLED",
            "journal": "Test Journal",
            "publisher": "Test Publisher",
            "year": 2026,
        },
        device={
            "device_label": "D1",
            "architecture_text": "ITO/EML/Al",
            "emission_mechanism": ["TADF"],
            "final_emitter": {
                "mention": "Emitter",
                "paper_material_id": "M1",
                "material_class": "small_molecule_organic",
            },
            "layers": [
                {
                    "layer_index": 1,
                    "layer_role": "EML",
                    "components": [
                        {
                            "paper_material_id": "M1",
                            "material_mention": "Emitter",
                        }
                    ],
                }
            ],
            "performance": [
                {
                    "metric_family": "EQE",
                    "statistic": "max",
                    "normalized_value": 25.2,
                }
            ],
        },
        paper_materials={
            "M1": {
                "paper_material_id": "M1",
                "mention": "Emitter",
                "canonical_smiles": "c1ccccc1",
            }
        },
        candidate_run_id="run-1",
        device_index=0,
        quality_tier="human_finalized",
        finalized_at="2026-07-25T00:00:00+00:00",
    )

    assert record["layer_count"] == 1
    assert record["eqe_max"] == 25.2
    assert record["final_emitter_smiles"] == "c1ccccc1"
    assert record["materials"][0]["paper_material_id"] == "M1"
