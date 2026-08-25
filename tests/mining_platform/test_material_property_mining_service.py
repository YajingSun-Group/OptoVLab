from __future__ import annotations

import copy
import json
from pathlib import Path

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from evolab_local.mining_platform.api.app import create_app
from evolab_local.mining_platform.candidate_ingestion_service import CandidateIngestionService
from evolab_local.mining_platform.cli import app
from evolab_local.mining_platform.core.config import load_config
from evolab_local.mining_platform.domain_template_service import load_domain_template
from evolab_local.mining_platform.external.openai_compatible_client import (
    LLMResponse,
    StaticJSONLLMClient,
)
from evolab_local.mining_platform.library.paper_service import PaperService
from evolab_local.mining_platform.material_property_mining_service import MaterialPropertyMiningService
from evolab_local.mining_platform.schemas.document import DocumentBlock
from evolab_local.mining_platform.storage.database import Database
from evolab_local.mining_platform.storage.repositories import (
    DocumentBlockRepository,
    MaterialPropertyCandidateRepository,
)


def test_material_property_mining_service_mock_result_enters_candidate_table(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper_and_candidate(config, mining_config_path, fake_pdf_factory, manifest_writer)
    _seed_property_blocks(config)

    result = MaterialPropertyMiningService(
        config,
        llm_client=StaticJSONLLMClient(_property_payload()),
    ).mine_paper_properties("10.1000/example", provider="deepseek")

    assert result is not None
    assert result.paper_id == "10.1000%2Fexample"
    assert result.candidate_run_id
    assert result.source_count == 1
    assert len(result.candidates) == 1
    assert result.skipped == [
        {"index": 1, "reason": "invalid_paper_material_id", "value": "M999"}
    ]
    candidate = result.candidates[0]
    assert candidate.paper_material_id == "M001"
    assert candidate.property_name == "PLQY"
    assert candidate.property_category == "photophysical"
    assert candidate.value_numeric == 83.0
    assert candidate.unit == "%"
    assert candidate.normalized_value_numeric == 0.83
    assert candidate.normalized_unit == "fraction"
    assert candidate.condition["host"] == "mCBP"
    assert candidate.llm_evidence_text == "The doped film of BN-1 in mCBP showed a PLQY of 83%."
    assert candidate.source_block_text is not None
    assert candidate.source_block_text.startswith("Table 1 Photophysical properties.")
    assert candidate.evidence_anchor["block_id"] == "p1_b0"
    assert candidate.evidence_anchor["evidence_match_method"] == "exact_substring"
    assert candidate.provider == "deepseek"
    assert candidate.model == "deepseek-v4-flash"

    stored = MaterialPropertyCandidateRepository(Database(config.paths.sqlite_path)).list_by_run(
        result.candidate_run_id
    )
    assert [item.property_candidate_id for item in stored] == [candidate.property_candidate_id]


def test_material_property_mining_service_accepts_high_similarity_evidence(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper_and_candidate(config, mining_config_path, fake_pdf_factory, manifest_writer)
    _seed_property_blocks(config)
    payload = {
        "properties": [
            {
                "paper_material_id": "M001",
                "property_name": "PLQY",
                "property_category": "photophysical",
                "value_numeric": 83,
                "value_raw": "83%",
                "unit": "%",
                "condition": {"sample_form": "doped_film", "host": "mCBP"},
                "method": "integrating_sphere",
                "source_type": "table",
                "evidence_text": "The BN-1 doped film in mCBP showed a PLQY of 83%.",
                "evidence_block_id": "p1_b0",
                "confidence": 0.8,
            }
        ]
    }

    result = MaterialPropertyMiningService(
        config,
        llm_client=StaticJSONLLMClient(payload),
    ).mine_paper_properties("10.1000/example", provider="deepseek")

    assert result is not None
    assert result.skipped == []
    candidate = result.candidates[0]
    assert candidate.llm_evidence_text == "The BN-1 doped film in mCBP showed a PLQY of 83%."
    assert candidate.source_block_text is not None
    assert candidate.evidence_text in candidate.source_block_text
    assert candidate.evidence_anchor["evidence_match_method"] == "fuzzy_source_fragment"
    assert candidate.evidence_anchor["evidence_match_score"] >= 0.82


def test_material_property_mining_service_repairs_invalid_json_response(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper_and_candidate(config, mining_config_path, fake_pdf_factory, manifest_writer)
    _seed_property_blocks(config)

    class RepairingClient:
        def __init__(self) -> None:
            self.calls = 0

        def generate_json(self, messages, *, model=None, temperature=None, max_tokens=None):
            del messages, model, temperature, max_tokens
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(
                    content='{"properties":[{"paper_material_id":"M001","evidence_text":"broken}',
                    parsed_json=None,
                    parse_error="Unterminated string",
                )
            return LLMResponse(
                content=json.dumps(_property_payload(), ensure_ascii=False),
                parsed_json=_property_payload(),
            )

    client = RepairingClient()
    result = MaterialPropertyMiningService(config, llm_client=client).mine_paper_properties(
        "10.1000/example",
        provider="deepseek",
    )

    assert result is not None
    assert client.calls == 2
    assert len(result.candidates) == 1
    assert result.candidates[0].evidence_anchor["evidence_match_method"] == "exact_substring"


def test_material_property_mining_service_rejects_low_similarity_evidence(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper_and_candidate(config, mining_config_path, fake_pdf_factory, manifest_writer)
    _seed_property_blocks(config)
    payload = {
        "properties": [
            {
                "paper_material_id": "M001",
                "property_name": "PLQY",
                "value_numeric": 99,
                "value_raw": "99%",
                "unit": "%",
                "evidence_text": "A completely unrelated sentence about another experiment.",
                "evidence_block_id": "p1_b0",
                "confidence": 0.8,
            }
        ]
    }

    result = MaterialPropertyMiningService(
        config,
        llm_client=StaticJSONLLMClient(payload),
    ).mine_paper_properties("10.1000/example", provider="deepseek")

    assert result is not None
    assert result.candidates == []
    assert result.skipped == [
        {
            "index": 0,
            "reason": "evidence_text_below_similarity_threshold",
            "threshold": 0.82,
        }
    ]


def test_material_property_cli_supports_mock_input(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper_and_candidate(config, mining_config_path, fake_pdf_factory, manifest_writer)
    _seed_property_blocks(config)
    mock_path = mining_config_path.parent / "mock_material_properties.json"
    mock_path.write_text(json.dumps(_property_payload(), ensure_ascii=False), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "mine-material-properties",
            "--paper-id",
            "10.1000/example",
            "--mock-input",
            str(mock_path),
            "--config",
            str(mining_config_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "stored=1" in result.output
    assert "skipped=1" in result.output
    assert "PLQY" in result.output


def test_material_property_review_api_updates_bundle_and_events(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper_and_candidate(config, mining_config_path, fake_pdf_factory, manifest_writer)
    _seed_property_blocks(config)
    mined = MaterialPropertyMiningService(
        config,
        llm_client=StaticJSONLLMClient(_property_payload()),
    ).mine_paper_properties("10.1000/example", provider="deepseek")
    assert mined is not None
    property_candidate_id = mined.candidates[0].property_candidate_id
    client = TestClient(create_app(config=config))

    bundle_response = client.get("/api/papers/10.1000%2Fexample/material-structures")
    assert bundle_response.status_code == 200
    assert len(bundle_response.json()["property_candidates"]) == 1

    accept_response = client.post(
        f"/api/material-property-candidates/{property_candidate_id}/accept",
        json={"actor": "tester", "message": "Looks correct."},
    )
    assert accept_response.status_code == 200
    accepted = accept_response.json()
    assert accepted["property_candidates"][0]["status"] == "accepted"
    assert accepted["property_reviews"][0]["decision"] == "accept"
    assert accepted["property_review_events"][0]["event_type"] == "accept"

    edit_response = client.put(
        f"/api/material-property-candidates/{property_candidate_id}",
        json={
            "actor": "tester",
            "reviewed_property_name": "PLQY",
            "reviewed_value_numeric": 85,
            "reviewed_value_raw": "85%",
            "reviewed_unit": "%",
            "reviewed_condition": {"sample_form": "doped_film", "host": "mCBP"},
            "message": "Corrected from table.",
        },
    )
    assert edit_response.status_code == 200
    edited = edit_response.json()
    edited_candidate = edited["property_candidates"][0]
    assert edited_candidate["status"] == "edited_accepted"
    assert edited_candidate["value_numeric"] == 85
    assert edited_candidate["normalized_value_numeric"] == 0.85
    assert edited["property_reviews"][0]["decision"] == "edit_accept"

    manual_response = client.post(
        "/api/papers/10.1000%2Fexample/manual-material-property?paper_material_id=M001",
        json={
            "actor": "tester",
            "property_name": "HOMO",
            "value_numeric": -5.52,
            "value_raw": "-5.52 eV",
            "unit": "eV",
            "source_type": "text",
            "evidence_text": "The HOMO level of BN-1 was -5.52 eV.",
            "evidence_anchor": {"block_id": "p1_b0", "page_id": 1},
            "message": "Added missed HOMO.",
        },
    )
    assert manual_response.status_code == 200
    manual = manual_response.json()
    assert len(manual["property_candidates"]) == 2
    manual_candidate = next(
        item for item in manual["property_candidates"] if item["provider"] == "manual_input"
    )
    assert manual_candidate["status"] == "manual_added"
    assert manual["property_review_events"][-1]["event_type"] == "manual_add"

    reject_response = client.post(
        f"/api/material-property-candidates/{property_candidate_id}/reject",
        json={"actor": "tester", "message": "Rejected after re-check."},
    )
    assert reject_response.status_code == 200
    rejected = reject_response.json()
    rejected_candidate = next(
        item
        for item in rejected["property_candidates"]
        if item["property_candidate_id"] == property_candidate_id
    )
    assert rejected_candidate["status"] == "rejected"
    assert rejected["property_review_events"][-1]["event_type"] == "reject"


def _seed_paper_and_candidate(
    config,
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
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
    CandidateIngestionService(config).ingest_mining_result(
        paper_id="10.1000/example",
        template_id=str(_template_path().resolve()),
        payload=_valid_result(),
        source_name="mock",
    )


def _seed_property_blocks(config) -> None:
    text = (
        "Table 1 Photophysical properties. "
        "The doped film of BN-1 in mCBP showed a PLQY of 83%. "
        "The HOMO level of BN-1 was -5.52 eV."
    )
    DocumentBlockRepository(Database(config.paths.sqlite_path)).replace_for_paper(
        "10.1000%2Fexample",
        [
            DocumentBlock(
                paper_id="10.1000%2Fexample",
                block_id="p1_b0",
                page_id=1,
                block_index=0,
                block_type="table",
                text=text,
                bbox=[10.0, 20.0, 200.0, 260.0],
                source="test",
            )
        ],
    )


def _property_payload() -> dict[str, object]:
    return {
        "properties": [
            {
                "paper_material_id": "M001",
                "property_name": "photoluminescence quantum yield",
                "property_category": "photophysical",
                "value_numeric": 83,
                "value_raw": "83%",
                "unit": "%",
                "condition": {"sample_form": "doped_film", "host": "mCBP"},
                "method": "integrating_sphere",
                "source_type": "table",
                "evidence_text": "The doped film of BN-1 in mCBP showed a PLQY of 83%.",
                "evidence_block_id": "p1_b0",
                "confidence": 0.91,
            },
            {
                "paper_material_id": "M999",
                "property_name": "HOMO",
                "value_numeric": -5.52,
                "unit": "eV",
                "evidence_text": "The HOMO level of BN-1 was -5.52 eV.",
                "evidence_block_id": "p1_b0",
                "confidence": 0.88,
            },
        ]
    }


def _valid_result() -> dict[str, object]:
    return copy.deepcopy(load_domain_template(_template_path()).example_output)


def _template_path() -> Path:
    return Path("config/mining_platform/domains/oled_device_v1.yaml")
