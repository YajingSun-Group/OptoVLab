from __future__ import annotations

import copy
import shutil
from pathlib import Path

from evolab_local.mining_platform.candidate_ingestion_service import CandidateIngestionService
from evolab_local.mining_platform.core.config import load_config
from evolab_local.mining_platform.domain_template_service import load_domain_template
from evolab_local.mining_platform.external.mineru_client import MinerUParsedDocument
from evolab_local.mining_platform.external.openai_compatible_client import StaticJSONLLMClient
from evolab_local.mining_platform.library.paper_service import PaperService
from evolab_local.mining_platform.mining.llm_mining_service import LLMMiningService, sanitize_llm_mining_result
from evolab_local.mining_platform.mining.mineru_parse_service import MinerUParseService
from evolab_local.mining_platform.schemas.document import DocumentBlock
from evolab_local.mining_platform.storage.database import Database
from evolab_local.mining_platform.storage.repositories import (
    DocumentBlockRepository,
    PaperRepository,
)


def test_mineru_parse_service_writes_outputs_and_document_blocks(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
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

    service = MinerUParseService(config, client=_FakeMinerUClient())
    run = service.parse_paper("10.1000/example")

    assert run is not None
    assert run.status == "completed"
    assert run.content_item_count == 2
    assert run.content_list_path is not None
    assert Path(run.content_list_path).exists()
    blocks = DocumentBlockRepository(Database(config.paths.sqlite_path)).list_by_paper(
        "10.1000%2Fexample"
    )
    assert [block.block_type for block in blocks] == ["text", "table"]


def test_llm_mining_service_mock_result_enters_candidate_v2(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _copy_template_to_config(mining_config_path)
    _seed_paper(config, mining_config_path, fake_pdf_factory, manifest_writer)
    DocumentBlockRepository(Database(config.paths.sqlite_path)).replace_for_paper(
        "10.1000%2Fexample",
        [
            DocumentBlock(
                paper_id="10.1000%2Fexample",
                block_id="p1_b0",
                page_id=1,
                block_index=0,
                block_type="table",
                text="D1: ITO/HATCN/TAPC/mCBP:3 wt% BN-1/TPBi/LiF/Al; EQEmax = 31.2%.",
                bbox=[],
                source="test",
            )
        ],
    )

    result = LLMMiningService(config, llm_client=StaticJSONLLMClient(_valid_result())).mine_paper(
        paper_id="10.1000/example",
        template_id=str(_template_path().resolve()),
        provider="deepseek",
    )

    assert result is not None
    assert result.run.status == "completed"
    assert result.candidate_run_id
    bundle = CandidateIngestionService(config).get_review_bundle("10.1000/example")
    assert bundle is not None
    assert bundle.values


def test_llm_mining_service_validation_failure_keeps_failed_run(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _copy_template_to_config(mining_config_path)
    _seed_paper(config, mining_config_path, fake_pdf_factory, manifest_writer)
    DocumentBlockRepository(Database(config.paths.sqlite_path)).replace_for_paper(
        "10.1000%2Fexample",
        [
            DocumentBlock(
                paper_id="10.1000%2Fexample",
                block_id="p1_b0",
                page_id=1,
                block_index=0,
                block_type="text",
                text="OLED device data are discussed.",
                bbox=[],
                source="test",
            )
        ],
    )

    result = LLMMiningService(config, llm_client=StaticJSONLLMClient({"evidence": []})).mine_paper(
        paper_id="10.1000/example",
        template_id=str(_template_path().resolve()),
    )

    assert result is not None
    assert result.run.status == "failed"
    bundle = CandidateIngestionService(config).get_review_bundle("10.1000/example")
    assert bundle is not None
    assert bundle.values == []
    paper = PaperRepository(Database(config.paths.sqlite_path)).get("10.1000%2Fexample")
    assert paper is not None
    assert paper.review_status == "confirmed"
    assert paper.review_reason == "device_data_validation_failed"


def test_sanitize_llm_mining_result_drops_list_thickness_for_composite_layers() -> None:
    payload = {
        "devices": [
            {
                "layers": [
                    {
                        "layer_name": "ITO/Ag/ITO",
                        "thickness": [
                            {"value": 40, "unit": "nm"},
                            {"value": 140, "unit": "nm"},
                        ],
                    }
                ]
            }
        ]
    }

    repaired = sanitize_llm_mining_result(payload)

    assert repaired["devices"][0]["layers"][0]["thickness"] is None
    assert payload["devices"][0]["layers"][0]["thickness"] != repaired["devices"][0]["layers"][0]["thickness"]


def test_sanitize_llm_mining_result_repairs_layer_component_mentions() -> None:
    payload = {
        "materials": [
            {
                "paper_material_id": "M001",
                "mention_list": ["TAPC"],
                "abbreviation": "TAPC",
            }
        ],
        "devices": [
            {
                "layers": [
                    {
                        "layer_role": "HTL",
                        "layer_name": "TAPC",
                        "components": [
                            {"paper_material_id": "M001", "material_mention": None}
                        ],
                    },
                    {
                        "layer_role": "anode",
                        "layer_name": "ITO",
                        "components": [],
                        "evidence_refs": ["E1"],
                    },
                ]
            }
        ],
    }

    repaired = sanitize_llm_mining_result(payload)
    layers = repaired["devices"][0]["layers"]

    assert layers[0]["layer_index"] == 1
    assert layers[0]["components"][0]["material_mention"] == "TAPC"
    assert layers[1]["layer_index"] == 2
    assert layers[1]["components"][0]["material_mention"] == "ITO"
    assert layers[1]["components"][0]["component_role"] == "electrode_material"


def test_sanitize_llm_mining_result_maps_block_id_evidence_refs() -> None:
    payload = {
        "evidence": [
            {
                "evidence_id": "E001",
                "block_id": "mineru_p2_b4",
                "source_type": "text",
            }
        ],
        "materials": [
            {
                "paper_material_id": "M001",
                "mention_list": ["TAPC"],
                "evidence_refs": ["mineru_p2_b4", "missing_ref"],
            }
        ],
    }

    repaired = sanitize_llm_mining_result(payload)

    assert repaired["materials"][0]["evidence_refs"] == ["E001"]


class _FakeMinerUClient:
    def parse_pdf(self, pdf_path: Path) -> MinerUParsedDocument:
        content_list = [
            {"type": "text", "text": "OLED device D1 was fabricated.", "page_idx": 0, "bbox": [1, 2, 3, 4]},
            {
                "type": "table",
                "table_caption": ["Table 1 Device performance."],
                "table_body": "<table><tr><td>D1</td><td>EQE 31.2%</td></tr></table>",
                "page_idx": 1,
                "bbox": [5, 6, 7, 8],
            },
        ]
        return MinerUParsedDocument(
            task_id="task-1",
            backend="hybrid-auto-engine",
            version="3.1.0",
            file_name=pdf_path.stem,
            md_content="# mock",
            content_list=content_list,
            raw_result={"results": {pdf_path.stem: {"md_content": "# mock", "content_list": content_list}}},
        )


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


def _copy_template_to_config(mining_config_path: Path) -> None:
    root = mining_config_path.parent.parent.parent
    domains_dir = root / "config" / "mining_platform" / "domains"
    domains_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_template_path(), domains_dir / "oled_device_v1.yaml")


def _template_path() -> Path:
    return Path("config/mining_platform/domains/oled_device_v1.yaml")
