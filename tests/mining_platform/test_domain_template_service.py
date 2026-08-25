from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from evolab_local.mining_platform.domain_template_service import (
    DomainTemplateLoader,
    DomainTemplateValidationError,
    load_domain_template,
)


def test_load_oled_device_template_from_repo() -> None:
    template = load_domain_template(
        Path("config/mining_platform/domains/oled_device_v1.yaml")
    )

    assert template.template_id == "oled_device_v1"
    assert template.domain == "oled"
    assert template.paper_metadata_policy.llm_should_extract is False
    assert template.required_output_keys == ["evidence", "materials", "devices"]
    assert template.field_by_path("materials[].paper_material_id") is not None
    assert template.field_by_path("devices[].performance[].metric_family").enum_ref == (
        "metric_family"
    )


def test_template_loader_lists_templates(tmp_path: Path) -> None:
    template_dir = tmp_path / "domains"
    template_dir.mkdir()
    _write_template(template_dir / "test_template.yaml", _valid_template_payload())

    templates = DomainTemplateLoader(template_dir).list_templates()

    assert [template.template_id for template in templates] == ["test_template"]


def test_template_loader_rejects_unknown_enum_ref(tmp_path: Path) -> None:
    payload = _valid_template_payload()
    payload["fields"][-1]["enum_ref"] = "missing_vocabulary"
    path = tmp_path / "bad_enum.yaml"
    _write_template(path, payload)

    with pytest.raises(DomainTemplateValidationError, match="unknown enum_ref"):
        load_domain_template(path)


def test_template_loader_rejects_duplicate_field_paths(tmp_path: Path) -> None:
    payload = _valid_template_payload()
    payload["fields"].append(dict(payload["fields"][0]))
    path = tmp_path / "duplicate.yaml"
    _write_template(path, payload)

    with pytest.raises(DomainTemplateValidationError, match="duplicate field_path"):
        load_domain_template(path)


def test_template_loader_rejects_field_entity_path_mismatch(tmp_path: Path) -> None:
    payload = _valid_template_payload()
    payload["fields"][1]["entity"] = "device"
    path = tmp_path / "entity_mismatch.yaml"
    _write_template(path, payload)

    with pytest.raises(DomainTemplateValidationError, match="does not match entity path"):
        load_domain_template(path)


def test_template_loader_rejects_missing_required_example_key(tmp_path: Path) -> None:
    payload = _valid_template_payload()
    payload["example_output"].pop("materials")
    path = tmp_path / "missing_example_key.yaml"
    _write_template(path, payload)

    with pytest.raises(DomainTemplateValidationError, match="example_output is missing"):
        load_domain_template(path)


def _write_template(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _valid_template_payload() -> dict[str, Any]:
    return {
        "template_id": "test_template",
        "domain": "oled",
        "template_name": "Test Template",
        "version": "v1",
        "status": "draft",
        "paper_metadata_policy": {
            "source": "existing_papers_table",
            "llm_should_extract": False,
        },
        "entities": {
            "evidence": {"path": "evidence[]", "label": "Evidence"},
            "materials": {"path": "materials[]", "label": "Materials"},
            "devices": {"path": "devices[]", "label": "Devices"},
        },
        "vocabularies": {
            "device_type": ["bottom_emission", "unknown"],
        },
        "llm_output_schema": {
            "root": "object",
            "required_keys": ["evidence", "materials", "devices"],
            "excluded_keys": ["paper"],
        },
        "fields": [
            {
                "field_path": "evidence[].evidence_id",
                "label": "Evidence ID",
                "data_type": "string",
                "required": True,
                "entity": "evidence",
            },
            {
                "field_path": "materials[].paper_material_id",
                "label": "Paper material ID",
                "data_type": "string",
                "required": True,
                "entity": "material",
            },
            {
                "field_path": "devices[].device_label",
                "label": "Device label",
                "data_type": "string",
                "required": True,
                "entity": "device",
            },
            {
                "field_path": "devices[].device_type",
                "label": "Device type",
                "data_type": "enum",
                "enum_ref": "device_type",
                "required": False,
                "entity": "device",
            },
        ],
        "example_output": {
            "evidence": [],
            "materials": [],
            "devices": [],
        },
    }
