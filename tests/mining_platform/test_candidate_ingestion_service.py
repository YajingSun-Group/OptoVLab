from __future__ import annotations

import copy
import json
from pathlib import Path

from typer.testing import CliRunner

from evolab_local.mining_platform.candidate_ingestion_service import CandidateIngestionService
from evolab_local.mining_platform.cli import app
from evolab_local.mining_platform.core.config import load_config
from evolab_local.mining_platform.domain_template_service import load_domain_template
from evolab_local.mining_platform.library.paper_service import PaperService
from evolab_local.mining_platform.storage.database import Database
from evolab_local.mining_platform.storage.repositories import (
    CandidateIngestionRepository,
    PaperRepository,
)


def test_candidate_ingestion_writes_valid_mock_result(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper(config, mining_config_path, fake_pdf_factory, manifest_writer)
    payload = _valid_result()

    result = CandidateIngestionService(config).ingest_mining_result(
        paper_id="10.1000/example",
        template_id=str(_template_path().resolve()),
        payload=payload,
        source_name="mock",
        source_version="v0",
    )

    assert result is not None
    assert result.run.status == "completed"
    assert result.validation_report.valid is True
    assert result.evidence_anchor_count == 3
    assert result.entity_count == 21
    assert result.value_count > 40

    repository = CandidateIngestionRepository(Database(config.paths.sqlite_path))
    runs = repository.list_runs_by_paper("10.1000%2Fexample")
    assert [run.candidate_run_id for run in runs] == [result.run.candidate_run_id]

    entities = repository.list_entities_by_run(result.run.candidate_run_id)
    values = repository.list_values_by_run(result.run.candidate_run_id)
    assert {entity.entity_type for entity in entities} == {
        "components",
        "devices",
        "layers",
        "materials",
        "performance",
    }
    assert any(entity.entity_path == "devices[0].layers[3]" for entity in entities)
    assert any(
        value.template_field_path == "devices[].layers[].components[].paper_material_id"
        and value.value_json == "M001"
        and value.evidence_anchor_ids
        for value in values
    )
    assert any(
        value.template_field_path == "devices[].performance[].normalized_value"
        and value.value_json == 31.2
        for value in values
    )

    paper = PaperRepository(Database(config.paths.sqlite_path)).get("10.1000%2Fexample")
    assert paper is not None
    assert paper.mining_status == "completed"
    assert paper.review_status == "needs_review"


def test_candidate_ingestion_persists_failed_validation_without_values(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper(config, mining_config_path, fake_pdf_factory, manifest_writer)
    payload = _valid_result()
    payload["devices"][0]["layers"][3]["components"][0]["paper_material_id"] = "M999"

    result = CandidateIngestionService(config).ingest_mining_result(
        paper_id="10.1000%2Fexample",
        template_id=str(_template_path().resolve()),
        payload=payload,
        source_name="mock",
    )

    assert result is not None
    assert result.run.status == "failed"
    assert result.validation_report.valid is False
    assert result.entity_count == 0
    assert result.value_count == 0
    assert result.evidence_anchor_count == 0
    assert result.run.error_message

    repository = CandidateIngestionRepository(Database(config.paths.sqlite_path))
    assert repository.list_entities_by_run(result.run.candidate_run_id) == []
    assert repository.list_values_by_run(result.run.candidate_run_id) == []

    paper = PaperRepository(Database(config.paths.sqlite_path)).get("10.1000%2Fexample")
    assert paper is not None
    assert paper.mining_status == "failed"
    assert paper.review_status == "confirmed"
    assert paper.review_reason == "device_data_validation_failed"


def test_ingest_mining_result_returns_none_for_missing_paper(mining_config_path: Path) -> None:
    config = load_config(mining_config_path)

    result = CandidateIngestionService(config).ingest_mining_result(
        paper_id="10.1000/missing",
        template_id=str(_template_path().resolve()),
        payload=_valid_result(),
    )

    assert result is None


def test_ingest_mining_result_cli(
    tmp_path: Path,
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper(config, mining_config_path, fake_pdf_factory, manifest_writer)
    input_path = tmp_path / "mock_result.json"
    input_path.write_text(json.dumps(_valid_result(), ensure_ascii=False), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "ingest-mining-result",
            "--config",
            str(mining_config_path),
            "--paper-id",
            "10.1000/example",
            "--template-id",
            str(_template_path().resolve()),
            "--input",
            str(input_path),
            "--source-name",
            "mock",
        ],
    )

    assert result.exit_code == 0
    assert "Ingestion run" in result.output
    assert "completed" in result.output


def test_confirm_review_v2_cli(
    tmp_path: Path,
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper(config, mining_config_path, fake_pdf_factory, manifest_writer)
    input_path = tmp_path / "mock_result.json"
    input_path.write_text(json.dumps(_valid_result(), ensure_ascii=False), encoding="utf-8")
    ingest_result = CliRunner().invoke(
        app,
        [
            "ingest-mining-result",
            "--config",
            str(mining_config_path),
            "--paper-id",
            "10.1000/example",
            "--template-id",
            str(_template_path().resolve()),
            "--input",
            str(input_path),
        ],
    )
    assert ingest_result.exit_code == 0

    confirm_result = CliRunner().invoke(
        app,
        [
            "confirm-review-v2",
            "--config",
            str(mining_config_path),
            "--paper-id",
            "10.1000/example",
            "--actor",
            "tester",
        ],
    )

    assert confirm_result.exit_code == 0
    assert "Confirmed 10.1000%2Fexample" in confirm_result.output


def _seed_paper(config, mining_config_path: Path, fake_pdf_factory, manifest_writer) -> None:
    pdf_path = fake_pdf_factory(mining_config_path)
    manifest_writer(
        mining_config_path,
        [
            {
                "doi": "10.1000/example",
                "status": "completed",
                "journal": "Journal",
                "publisher": "Publisher",
                "pdf_path": pdf_path.relative_to(config.project_root).as_posix(),
            }
        ],
    )
    PaperService(config).ingest_from_pdf_downloader(domain="oled")


def _valid_result() -> dict[str, object]:
    return copy.deepcopy(load_domain_template(_template_path()).example_output)


def _template_path() -> Path:
    return Path("config/mining_platform/domains/oled_device_v1.yaml")
