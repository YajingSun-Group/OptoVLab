from __future__ import annotations

import copy
import json
from pathlib import Path

from typer.testing import CliRunner

from evolab_local.mining_platform.cli import app
from evolab_local.mining_platform.domain_template_service import load_domain_template
from evolab_local.mining_platform.mining_result_validator import MiningResultValidator


def test_validator_accepts_template_example_output() -> None:
    template = _template()
    report = MiningResultValidator(template).validate(copy.deepcopy(template.example_output))

    assert report.valid is True
    assert report.errors == []
    assert report.metadata["evidence_count"] == 3
    assert report.metadata["material_count"] == 2
    assert report.metadata["device_count"] == 1


def test_validator_rejects_missing_top_level_key() -> None:
    result = _valid_result()
    result.pop("materials")

    report = _validate(result)

    assert report.valid is False
    assert _has_code(report, "missing_top_level_key")


def test_validator_warns_excluded_paper_metadata() -> None:
    result = _valid_result()
    result["paper"] = {"doi": "10.1000/example"}

    report = _validate(result)

    assert report.valid is True
    assert _has_code(report, "excluded_top_level_key", "warnings")
    assert report.warning_count == 1


def test_validator_warns_invalid_enum_value() -> None:
    result = _valid_result()
    result["devices"][0]["layers"][3]["layer_role"] = "emitting_layer"

    report = _validate(result)

    assert report.valid is True
    assert _has_code(report, "invalid_field_value", "warnings")
    assert report.warning_count == 1


def test_validator_rejects_unknown_evidence_ref() -> None:
    result = _valid_result()
    result["devices"][0]["performance"][0]["evidence_refs"] = ["E999"]

    report = _validate(result)

    assert report.valid is False
    assert _has_code(report, "unknown_evidence_ref")


def test_validator_rejects_unknown_paper_material_ref() -> None:
    result = _valid_result()
    result["devices"][0]["layers"][3]["components"][0]["paper_material_id"] = "M999"

    report = _validate(result)

    assert report.valid is False
    assert _has_code(report, "unknown_paper_material_id")


def test_validator_rejects_missing_required_field() -> None:
    result = _valid_result()
    result["devices"][0].pop("device_label")

    report = _validate(result)

    assert report.valid is False
    assert _has_code(report, "required_field_missing")


def test_validator_allows_empty_nested_collections() -> None:
    result = _valid_result()
    result["devices"][0]["layers"] = []
    result["devices"][0]["performance"] = []

    report = _validate(result)

    assert report.valid is True
    assert not _has_code(report, "required_field_missing")


def test_validator_still_rejects_missing_field_in_existing_nested_item() -> None:
    result = _valid_result()
    result["devices"][0]["layers"][0].pop("layer_index")

    report = _validate(result)

    assert report.valid is False
    assert _has_code(report, "required_field_missing")


def test_validator_marks_optional_shape_issue_repairable() -> None:
    result = _valid_result()
    result["devices"][0]["layers"][0]["thickness"] = [{"value": 40, "unit": "nm"}]

    report = _validate(result)

    assert report.valid is True
    assert _has_code(report, "expected_object", "repairable")
    assert report.repairable_count >= 1


def test_validator_warns_value_outside_validation_range() -> None:
    result = _valid_result()
    result["materials"][0]["structure_confidence"] = 1.5

    report = _validate(result)

    assert report.valid is True
    assert _has_code(report, "value_above_max", "warnings")
    assert report.warning_count == 1


def test_validate_mining_result_cli(tmp_path: Path, mining_config_path: Path) -> None:
    input_path = tmp_path / "mock_result.json"
    input_path.write_text(json.dumps(_valid_result(), ensure_ascii=False), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "validate-mining-result",
            "--config",
            str(mining_config_path),
            "--template-id",
            str(_template().source_path.resolve()),
            "--input",
            str(input_path),
        ],
    )

    assert result.exit_code == 0
    assert "Mining result is valid" in result.output


def _validate(result: dict[str, object]):
    return MiningResultValidator(_template()).validate(result)


def _valid_result() -> dict[str, object]:
    return copy.deepcopy(_template().example_output)


def _template():
    return load_domain_template(Path("config/mining_platform/domains/oled_device_v1.yaml"))


def _has_code(report, code: str, bucket: str = "errors") -> bool:
    return any(issue.code == code for issue in getattr(report, bucket))
