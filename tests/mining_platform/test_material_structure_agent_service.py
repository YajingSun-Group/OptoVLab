from __future__ import annotations

from pathlib import Path
from threading import Lock
import time

from fastapi.testclient import TestClient
from PIL import Image
from typer.testing import CliRunner

from evolab_local.mining_platform.api.app import create_app
from evolab_local.mining_platform.candidate_ingestion_service import CandidateIngestionService
from evolab_local.mining_platform.cli import app
from evolab_local.mining_platform.core.config import load_config
from evolab_local.mining_platform.domain_template_service import load_domain_template
from evolab_local.mining_platform.external.decimer_client import (
    DecimerSegment,
    DecimerSegmentationResponse,
    DecimerSmilesResponse,
)
from evolab_local.mining_platform.external.openai_compatible_client import StaticJSONVisionClient
from evolab_local.mining_platform.external.mineru_client import MinerUParsedDocument
from evolab_local.mining_platform.library.paper_service import PaperService
from evolab_local.mining_platform.mining.mineru_parse_service import MinerUParseService
from evolab_local.mining_platform.material_structure_agent_service import (
    MaterialStructureAgentService,
    _binding_eligible_for_ocsr,
    _map_with_concurrency,
    _ocsr_structure_quality_warning,
    normalize_crop_validation_payload,
    normalize_figure_triage_payload,
    normalize_label_binding_payload,
)
from evolab_local.mining_platform.schemas.material_agent import MoleculeLabelBinding


def test_bounded_concurrency_runs_independent_decimer_work_in_parallel() -> None:
    lock = Lock()
    active = 0
    peak_active = 0

    def run_item(value: int) -> int:
        nonlocal active, peak_active
        with lock:
            active += 1
            peak_active = max(peak_active, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return value * 2

    results = _map_with_concurrency(
        [1, 2, 3, 4],
        max_concurrency=2,
        runner=run_item,
    )

    assert results == [2, 4, 6, 8]
    assert peak_active == 2


def test_ocsr_flags_disconnected_fragments_for_single_organic_molecule() -> None:
    warning = _ocsr_structure_quality_warning(
        "c1ccccc1.CCCCCCCCCCCN=C=S",
        "small_molecule_organic",
    )

    assert warning is not None
    assert "multiple disconnected components" in warning
    assert _ocsr_structure_quality_warning("c1ccccc1", "small_molecule_organic") is None
    assert _ocsr_structure_quality_warning("[Ir+3].c1ccccc1", "organometallic_complex") is None


def test_material_structure_agent_foundation_collects_all_visual_blocks(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper(config, mining_config_path, fake_pdf_factory, manifest_writer)
    run = MinerUParseService(config, client=_FakeMinerUClient()).parse_paper("10.1000/example")
    assert run is not None
    image_dir = Path(run.content_list_path).parent / "images"
    image_dir.mkdir(parents=True)
    (image_dir / "scheme.jpg").write_bytes(b"fake image")
    (image_dir / "plot.jpg").write_bytes(b"fake image")

    result = MaterialStructureAgentService(config).run_foundation("10.1000/example")

    assert result is not None
    assert result.run.status == "completed"
    assert result.run.visual_block_count == 3
    assert [block.content_type for block in result.visual_blocks] == ["image", "chart", "table"]
    assert result.visual_blocks[0].image_exists is True
    assert result.visual_blocks[1].image_exists is True
    assert result.visual_blocks[2].image_exists is False
    persisted_image = Path(result.visual_blocks[0].resolved_img_path or "")
    assert persisted_image.exists()
    assert persisted_image.is_relative_to(
        config.paths.runtime_dir / "material_agent" / "visual_blocks"
    )

    for source_image in image_dir.iterdir():
        source_image.unlink()
    refreshed = MaterialStructureAgentService(config).run_foundation("10.1000/example")

    assert refreshed is not None
    assert refreshed.run.agent_run_id != result.run.agent_run_id
    assert refreshed.run.tool_summary["image_exists_count"] == 2
    assert all(
        block.image_exists
        for block in refreshed.visual_blocks
        if block.content_type in {"image", "chart"}
    )
    assert all(
        "recovered_from_materialized_path" in block.source_json
        for block in refreshed.visual_blocks
        if block.content_type in {"image", "chart"}
    )


def test_material_structure_agent_foundation_api_and_cli(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper(config, mining_config_path, fake_pdf_factory, manifest_writer)
    run = MinerUParseService(config, client=_FakeMinerUClient()).parse_paper("10.1000/example")
    assert run is not None
    image_dir = Path(run.content_list_path).parent / "images"
    image_dir.mkdir(parents=True)
    (image_dir / "scheme.jpg").write_bytes(b"fake image")

    client = TestClient(create_app(config=config))
    response = client.post("/api/papers/10.1000%2Fexample/material-agent-runs/foundation")
    assert response.status_code == 200
    payload = response.json()
    assert payload["run"]["status"] == "completed"
    assert len(payload["visual_blocks"]) == 3

    blocks_response = client.get("/api/papers/10.1000%2Fexample/document-visual-blocks")
    assert blocks_response.status_code == 200
    visual_block_id = blocks_response.json()[0]["visual_block_id"]
    image_response = client.get(f"/api/document-visual-blocks/{visual_block_id}/image")
    assert image_response.status_code == 200

    cli_result = CliRunner().invoke(
        app,
        [
            "list-document-visual-blocks",
            "--config",
            str(mining_config_path),
            "--paper-id",
            "10.1000/example",
        ],
    )
    assert cli_result.exit_code == 0
    assert "Document Visual Blocks" in cli_result.output
    assert "Scheme 1" in cli_result.output


def test_figure_triage_agent_records_vlm_decisions(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper(config, mining_config_path, fake_pdf_factory, manifest_writer)
    run = MinerUParseService(config, client=_FakeMinerUClient()).parse_paper("10.1000/example")
    assert run is not None
    image_dir = Path(run.content_list_path).parent / "images"
    image_dir.mkdir(parents=True)
    (image_dir / "scheme.jpg").write_bytes(b"fake image")
    (image_dir / "plot.jpg").write_bytes(b"fake image")

    service = MaterialStructureAgentService(
        config,
        vision_client=StaticJSONVisionClient(
            {
                "contains_molecular_structures": True,
                "image_role": "molecule_structures",
                "has_clean_structure_depictions": True,
                "has_orbital_overlay": False,
                "has_energy_level_diagram": False,
                "has_device_stack": False,
                "should_run_decimer_segmentation": True,
                "label_candidates": ["BN-Tpl-Ph"],
                "related_paper_material_ids": [],
                "confidence": 0.88,
                "reason": "The image contains clean molecular skeletal structures.",
            }
        ),
    )

    result = service.run_figure_triage("10.1000/example", limit=1)

    assert result is not None
    assert result.run.status == "completed"
    assert len(result.results) == 1
    triage = result.results[0]
    assert triage.status == "completed"
    assert triage.image_role == "molecule_structures"
    assert triage.should_run_decimer_segmentation is True
    assert triage.label_candidates == ["BN-Tpl-Ph"]

    client = TestClient(create_app(config=config))
    response = client.get("/api/papers/10.1000%2Fexample/figure-triage-results")
    assert response.status_code == 200
    payload = response.json()
    assert payload[0]["should_run_decimer_segmentation"] is True

    cli_result = CliRunner().invoke(
        app,
        [
            "list-figure-triage",
            "--config",
            str(mining_config_path),
            "--paper-id",
            "10.1000/example",
        ],
    )
    assert cli_result.exit_code == 0
    assert "Figure Triage" in cli_result.output
    assert "Results: 1" in cli_result.output


def test_decimer_segmentation_creates_molecule_crops(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper(config, mining_config_path, fake_pdf_factory, manifest_writer)
    run = MinerUParseService(config, client=_FakeMinerUClient()).parse_paper("10.1000/example")
    assert run is not None
    image_dir = Path(run.content_list_path).parent / "images"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "scheme.jpg"
    Image.new("RGB", (120, 100), "white").save(image_path)
    Image.new("RGB", (120, 100), "white").save(image_dir / "plot.jpg")

    service = MaterialStructureAgentService(
        config,
        vision_client=StaticJSONVisionClient(
            {
                "contains_molecular_structures": True,
                "image_role": "molecule_structures",
                "has_clean_structure_depictions": True,
                "should_run_decimer_segmentation": True,
                "label_candidates": ["BN-Tpl-Ph"],
                "related_paper_material_ids": [],
                "confidence": 0.9,
                "reason": "Molecular structures are visible.",
            }
        ),
        decimer_segmentation_client=_FakeDecimerSegmentationClient(),
    )
    triage = service.run_figure_triage("10.1000/example", limit=1)
    assert triage is not None

    result = service.run_decimer_segmentation("10.1000/example", limit=1)

    assert result is not None
    assert result.errors == []
    assert len(result.crops) == 1
    crop = result.crops[0]
    assert crop.crop_id.endswith("_000")
    assert crop.bbox == [10.0, 20.0, 80.0, 90.0]
    assert crop.width == 70
    assert crop.height == 70
    assert Path(crop.crop_path).exists()

    client = TestClient(create_app(config=config))
    crops_response = client.get("/api/papers/10.1000%2Fexample/molecule-crops")
    assert crops_response.status_code == 200
    crop_id = crops_response.json()[0]["crop_id"]
    image_response = client.get(f"/api/molecule-crops/{crop_id}/image")
    assert image_response.status_code == 200

    cli_result = CliRunner().invoke(
        app,
        [
            "list-molecule-crops",
            "--config",
            str(mining_config_path),
            "--paper-id",
            "10.1000/example",
        ],
    )
    assert cli_result.exit_code == 0
    assert "Molecule Crops" in cli_result.output
    assert "Crops: 1" in cli_result.output


def test_decimer_segmentation_requires_manual_crop_when_mixed_orbital_figure_fails(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper(config, mining_config_path, fake_pdf_factory, manifest_writer)
    CandidateIngestionService(config).ingest_mining_result(
        paper_id="10.1000/example",
        template_id=str(_template_path().resolve()),
        payload=load_domain_template(_template_path()).example_output,
        source_name="mock",
    )
    run = MinerUParseService(config, client=_FakeMinerUClient()).parse_paper("10.1000/example")
    assert run is not None
    image_dir = Path(run.content_list_path).parent / "images"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (120, 100), "white").save(image_dir / "scheme.jpg")
    Image.new("RGB", (120, 100), "white").save(image_dir / "plot.jpg")

    failing_client = _FakeFailingDecimerSegmentationClient()
    service = MaterialStructureAgentService(
        config,
        vision_client=StaticJSONVisionClient(
            {
                "contains_molecular_structures": True,
                "image_role": "molecule_plus_orbital",
                "has_clean_structure_depictions": True,
                "has_orbital_overlay": True,
                "has_energy_level_diagram": False,
                "has_device_stack": False,
                "should_run_decimer_segmentation": True,
                "label_candidates": ["pACRS", "pACRSO"],
                "related_paper_material_ids": ["M001", "M002"],
                "confidence": 0.95,
                "reason": "The top row has separable 2D structures; the bottom row has HOMO/LUMO orbital panels.",
            }
        ),
        decimer_segmentation_client=failing_client,
    )
    triage = service.run_figure_triage("10.1000/example", limit=1)
    assert triage is not None

    result = service.run_decimer_segmentation("10.1000/example", limit=1)

    assert result is not None
    assert result.crops == []
    assert len(result.errors) == 1
    error = result.errors[0]
    assert error["status"] == "manual_crop_required"
    assert error["suggested_action"] == "manual_crop_or_manual_structure_input"
    assert "automatic region guessing is disabled" in error["error"]
    assert "mock DECIMER segmentation failure" in error["error"]
    assert len(failing_client.calls) == 1


def test_single_material_flat_structure_uses_full_visual_block_crop(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper(config, mining_config_path, fake_pdf_factory, manifest_writer)
    CandidateIngestionService(config).ingest_mining_result(
        paper_id="10.1000/example",
        template_id=str(_template_path().resolve()),
        payload=load_domain_template(_template_path()).example_output,
        source_name="mock",
    )
    run = MinerUParseService(config, client=_FakeMinerUClient()).parse_paper("10.1000/example")
    assert run is not None
    image_dir = Path(run.content_list_path).parent / "images"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (120, 100), "white").save(image_dir / "scheme.jpg")
    Image.new("RGB", (120, 100), "white").save(image_dir / "plot.jpg")

    service = MaterialStructureAgentService(
        config,
        vision_client=StaticJSONVisionClient(
            {
                "contains_molecular_structures": True,
                "image_role": "flat_2d_molecular_structures",
                "has_clean_structure_depictions": True,
                "has_orbital_overlay": False,
                "has_energy_level_diagram": False,
                "has_device_stack": False,
                "should_run_decimer_segmentation": True,
                "label_candidates": ["M001"],
                "related_paper_material_ids": ["M001"],
                "confidence": 0.98,
                "reason": "The visual block is already one clean 2D molecular structure.",
            }
        ),
        decimer_segmentation_client=_FakeDecimerSegmentationClient(),
    )

    assert service.run_foundation("10.1000/example") is not None
    triage = service.run_figure_triage("10.1000/example", limit=1)
    assert triage is not None
    result = service.run_decimer_segmentation("10.1000/example", limit=1)

    assert result is not None
    assert result.errors == []
    assert len(result.crops) == 1
    crop = result.crops[0]
    assert crop.bbox == [0.0, 0.0, 120.0, 100.0]
    assert crop.width == 120
    assert crop.height == 100
    assert crop.raw_segment["source"] == "full_visual_block"
    assert crop.validation_json["crop_mode"] == "full_visual_block"
    assert Path(crop.crop_path).exists()


def test_crop_validator_accepts_clean_structure_and_rejects_orbital_overlay(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper(config, mining_config_path, fake_pdf_factory, manifest_writer)
    run = MinerUParseService(config, client=_FakeMinerUClient()).parse_paper("10.1000/example")
    assert run is not None
    image_dir = Path(run.content_list_path).parent / "images"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (120, 100), "white").save(image_dir / "scheme.jpg")
    Image.new("RGB", (120, 100), "white").save(image_dir / "plot.jpg")

    segment_service = MaterialStructureAgentService(
        config,
        vision_client=StaticJSONVisionClient(
            {
                "contains_molecular_structures": True,
                "image_role": "molecule_structures",
                "has_clean_structure_depictions": True,
                "should_run_decimer_segmentation": True,
                "confidence": 0.9,
                "reason": "Molecular structures are visible.",
            }
        ),
        decimer_segmentation_client=_FakeDecimerSegmentationClient(),
    )
    assert segment_service.run_figure_triage("10.1000/example", limit=1) is not None
    assert segment_service.run_decimer_segmentation("10.1000/example", limit=1) is not None

    validation_service = MaterialStructureAgentService(
        config,
        vision_client=StaticJSONVisionClient(
            {
                "is_molecular_depiction": True,
                "is_single_molecule": True,
                "is_complete_structure": True,
                "has_benign_highlight": False,
                "is_ocsr_readable": True,
                "has_blocking_interference": False,
                "has_orbital_overlay": False,
                "has_excess_annotation": False,
                "has_multiple_structures": False,
                "has_reaction_arrow": False,
                "has_non_structural_graphics": False,
                "should_run_ocsr": True,
                "confidence": 0.93,
                "reason": "One clean complete molecular skeletal structure.",
            }
        ),
    )
    result = validation_service.run_crop_validation("10.1000/example", limit=1)

    assert result is not None
    assert len(result.validations) == 1
    assert result.validations[0].should_run_ocsr is True
    assert validation_service.list_molecule_crops("10.1000/example")[0].status == "ready_for_ocsr"

    highlighted_but_readable = normalize_crop_validation_payload(
        {
            "is_molecular_depiction": True,
            "is_single_molecule": True,
            "is_complete_structure": True,
            "has_benign_highlight": True,
            "is_ocsr_readable": True,
            "has_non_structural_graphics": True,
            "should_run_ocsr": True,
            "confidence": 0.99,
        }
    )
    assert highlighted_but_readable["should_run_ocsr"] is True

    rejected = normalize_crop_validation_payload(
        {
            "is_molecular_depiction": True,
            "is_single_molecule": True,
            "is_complete_structure": True,
            "is_ocsr_readable": False,
            "has_blocking_interference": True,
            "has_orbital_overlay": True,
            "should_run_ocsr": True,
            "confidence": 0.99,
        }
    )
    assert rejected["should_run_ocsr"] is False

    client = TestClient(create_app(config=config))
    response = client.get("/api/papers/10.1000%2Fexample/molecule-crop-validations")
    assert response.status_code == 200
    assert response.json()[0]["should_run_ocsr"] is True

    cli_result = CliRunner().invoke(
        app,
        [
            "list-molecule-crop-validations",
            "--config",
            str(mining_config_path),
            "--paper-id",
            "10.1000/example",
        ],
    )
    assert cli_result.exit_code == 0
    assert "Molecule Crop Validation" in cli_result.output
    assert "Validations: 1" in cli_result.output

    refreshed = validation_service.run_foundation("10.1000/example")
    assert refreshed is not None
    assert refreshed.run.agent_run_id != result.run.agent_run_id
    assert validation_service.list_molecule_crops("10.1000/example") == []
    assert validation_service.list_crop_validations("10.1000/example") == []
    assert {
        block.collected_by_agent_run_id
        for block in validation_service.list_visual_blocks("10.1000/example") or []
    } == {refreshed.run.agent_run_id}


def test_material_context_filters_out_materials_not_used_in_devices(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper(config, mining_config_path, fake_pdf_factory, manifest_writer)
    payload = (
        load_domain_template(_template_path()).example_output.model_copy(deep=True)
        if hasattr(load_domain_template(_template_path()).example_output, "model_copy")
        else load_domain_template(_template_path()).example_output.copy()
    )
    payload["materials"].append(
        {
            "paper_material_id": "M999",
            "mention_list": ["Unused reference emitter"],
            "full_name_in_paper": None,
            "normalized_name": "Unused reference emitter",
            "canonical_name": None,
            "abbreviation": "URE",
            "paper_specific_label": None,
            "material_class": "small_molecule_organic",
            "smiles": None,
            "inchi": None,
            "inchi_key": None,
            "structure_source": "figure",
            "structure_confidence": None,
            "evidence_refs": ["E1"],
        }
    )
    CandidateIngestionService(config).ingest_mining_result(
        paper_id="10.1000/example",
        template_id=str(_template_path().resolve()),
        payload=payload,
        source_name="mock",
    )
    run = MinerUParseService(config, client=_FakeMinerUClient()).parse_paper("10.1000/example")
    assert run is not None
    image_dir = Path(run.content_list_path).parent / "images"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (120, 100), "white").save(image_dir / "scheme.jpg")
    Image.new("RGB", (120, 100), "white").save(image_dir / "plot.jpg")

    segment_service = MaterialStructureAgentService(
        config,
        vision_client=StaticJSONVisionClient(
            {
                "contains_molecular_structures": True,
                "image_role": "molecule_structures",
                "has_clean_structure_depictions": True,
                "should_run_decimer_segmentation": True,
                "label_candidates": ["M001"],
                "related_paper_material_ids": ["M001", "M999"],
                "confidence": 0.9,
                "reason": "Visible structure label.",
            }
        ),
        decimer_segmentation_client=_FakeDecimerSegmentationClient(),
    )
    assert segment_service.run_foundation("10.1000/example") is not None
    triage = segment_service.run_figure_triage("10.1000/example", limit=1)
    assert triage is not None
    assert triage.results[0].related_paper_material_ids == ["M001"]
    assert segment_service.run_decimer_segmentation("10.1000/example", limit=1) is not None

    validator = MaterialStructureAgentService(
        config,
        vision_client=StaticJSONVisionClient(
            {
                "is_molecular_depiction": True,
                "is_single_molecule": True,
                "is_complete_structure": True,
                "is_flat_2d_structure_diagram": True,
                "has_3d_ball_and_stick_model": False,
                "has_crystal_structure_rendering": False,
                "has_photo_or_surface_rendering": False,
                "should_run_ocsr": True,
                "confidence": 0.97,
                "reason": "Clean 2D structure.",
            }
        ),
    )
    assert validator.run_crop_validation("10.1000/example", limit=1) is not None

    binding_service = MaterialStructureAgentService(
        config,
        vision_client=StaticJSONVisionClient(
            {
                "observed_label": "M999",
                "label_source": "visible_near_crop",
                "proposed_paper_material_id": "M999",
                "alternative_paper_material_ids": ["M001"],
                "decision": "matched",
                "confidence": 0.91,
                "reason": "The model tried to bind an unused material.",
            }
        ),
    )

    result = binding_service.run_label_binding("10.1000/example", limit=1)

    assert result is not None
    binding = result.bindings[0]
    assert [item["paper_material_id"] for item in binding.candidate_materials] == ["M001", "M002"]
    assert binding.model_proposed_paper_material_id is None
    assert binding.model_decision == "material_not_in_candidate_list"


def test_figure_triage_allows_molecule_plus_orbital_when_2d_structures_are_separable() -> None:
    triage = normalize_figure_triage_payload(
        {
            "contains_molecular_structures": True,
            "image_role": "molecule_plus_orbital",
            "has_clean_structure_depictions": True,
            "has_flat_2d_structure_diagrams": True,
            "has_3d_ball_and_stick_model": True,
            "has_orbital_overlay": True,
            "has_crystal_structure_rendering": False,
            "has_surface_or_photo_rendering": False,
            "has_energy_level_diagram": False,
            "has_device_stack": False,
            "should_run_decimer_segmentation": False,
            "related_paper_material_ids": ["M001", "M002"],
            "confidence": 0.95,
        },
        [{"paper_material_id": "M001"}, {"paper_material_id": "M002"}],
    )

    assert triage["image_role"] == "molecule_plus_orbital"
    assert triage["should_run_decimer_segmentation"] is True
    assert triage["related_paper_material_ids"] == ["M001", "M002"]


def test_figure_triage_allows_flat_2d_structures_with_separate_orbital_panels() -> None:
    triage = normalize_figure_triage_payload(
        {
            "contains_molecular_structures": True,
            "image_role": "flat_2d_molecular_structures",
            "has_clean_structure_depictions": True,
            "has_flat_2d_structure_diagrams": True,
            "has_orbital_overlay": True,
            "has_energy_level_diagram": False,
            "has_device_stack": False,
            "should_run_decimer_segmentation": True,
            "related_paper_material_ids": ["M001", "M002"],
            "confidence": 0.95,
        },
        [{"paper_material_id": "M001"}, {"paper_material_id": "M002"}],
    )

    assert triage["should_run_decimer_segmentation"] is True


def test_figure_triage_allows_separable_2d_structure_inside_energy_level_diagram() -> None:
    triage = normalize_figure_triage_payload(
        {
            "contains_molecular_structures": True,
            "image_role": "energy_level_diagram",
            "has_clean_structure_depictions": True,
            "has_flat_2d_structure_diagrams": True,
            "has_energy_level_diagram": True,
            "has_device_stack": False,
            "should_run_decimer_segmentation": True,
            "related_paper_material_ids": ["M001"],
            "confidence": 1.0,
        },
        [{"paper_material_id": "M001"}],
    )

    assert triage["should_run_decimer_segmentation"] is True


def test_figure_triage_rejects_pure_energy_level_diagram() -> None:
    triage = normalize_figure_triage_payload(
        {
            "contains_molecular_structures": False,
            "image_role": "energy_level_diagram",
            "has_clean_structure_depictions": False,
            "has_flat_2d_structure_diagrams": False,
            "has_energy_level_diagram": True,
            "has_device_stack": False,
            "should_run_decimer_segmentation": True,
            "related_paper_material_ids": ["M001"],
            "confidence": 1.0,
        },
        [{"paper_material_id": "M001"}],
    )

    assert triage["should_run_decimer_segmentation"] is False


def test_figure_triage_rejects_pure_orbital_without_clean_2d_structures() -> None:
    triage = normalize_figure_triage_payload(
        {
            "contains_molecular_structures": True,
            "image_role": "molecule_plus_orbital",
            "has_clean_structure_depictions": False,
            "has_flat_2d_structure_diagrams": False,
            "has_3d_ball_and_stick_model": True,
            "has_orbital_overlay": True,
            "should_run_decimer_segmentation": True,
            "related_paper_material_ids": ["M001"],
            "confidence": 0.95,
        },
        [{"paper_material_id": "M001"}],
    )

    assert triage["should_run_decimer_segmentation"] is False


def test_visual_prompt_normalizers_reject_3d_or_crystal_structure_images() -> None:
    triage = normalize_figure_triage_payload(
        {
            "contains_molecular_structures": True,
            "image_role": "crystal_structure",
            "has_clean_structure_depictions": True,
            "has_3d_ball_and_stick_model": True,
            "has_crystal_structure_rendering": True,
            "has_surface_or_photo_rendering": False,
            "should_run_decimer_segmentation": True,
            "related_paper_material_ids": ["M001"],
            "confidence": 0.95,
        },
        [{"paper_material_id": "M001"}],
    )
    assert triage["should_run_decimer_segmentation"] is False
    assert triage["has_3d_ball_and_stick_model"] is True
    assert triage["has_crystal_structure_rendering"] is True

    crop = normalize_crop_validation_payload(
        {
            "is_molecular_depiction": True,
            "is_single_molecule": True,
            "is_complete_structure": True,
            "is_flat_2d_structure_diagram": False,
            "has_3d_ball_and_stick_model": True,
            "has_crystal_structure_rendering": False,
            "is_ocsr_readable": True,
            "should_run_ocsr": True,
            "confidence": 0.99,
        }
    )
    assert crop["should_run_ocsr"] is False
    assert crop["has_3d_ball_and_stick_model"] is True


def test_label_binding_uses_single_full_visual_block_triage_match_without_vlm(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper(config, mining_config_path, fake_pdf_factory, manifest_writer)
    CandidateIngestionService(config).ingest_mining_result(
        paper_id="10.1000/example",
        template_id=str(_template_path().resolve()),
        payload=load_domain_template(_template_path()).example_output,
        source_name="mock",
    )
    run = MinerUParseService(config, client=_FakeMinerUClient()).parse_paper("10.1000/example")
    assert run is not None
    image_dir = Path(run.content_list_path).parent / "images"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (120, 100), "white").save(image_dir / "scheme.jpg")

    segment_service = MaterialStructureAgentService(
        config,
        vision_client=StaticJSONVisionClient(
            {
                "contains_molecular_structures": True,
                "image_role": "flat_2d_molecular_structures",
                "has_clean_structure_depictions": True,
                "has_orbital_overlay": False,
                "has_energy_level_diagram": False,
                "has_device_stack": False,
                "should_run_decimer_segmentation": True,
                "label_candidates": ["M001"],
                "related_paper_material_ids": ["M001"],
                "confidence": 0.98,
                "reason": "The visual block is already one clean 2D molecular structure.",
            }
        ),
        decimer_segmentation_client=_FakeDecimerSegmentationClient(),
    )
    assert segment_service.run_foundation("10.1000/example") is not None
    assert segment_service.run_figure_triage("10.1000/example", limit=1) is not None
    assert segment_service.run_decimer_segmentation("10.1000/example", limit=1) is not None

    validator = MaterialStructureAgentService(
        config,
        vision_client=StaticJSONVisionClient(
            {
                "is_molecular_depiction": True,
                "is_single_molecule": True,
                "is_complete_structure": True,
                "has_benign_highlight": True,
                "is_ocsr_readable": True,
                "has_blocking_interference": False,
                "has_orbital_overlay": False,
                "has_excess_annotation": False,
                "has_multiple_structures": False,
                "has_reaction_arrow": False,
                "has_non_structural_graphics": False,
                "should_run_ocsr": True,
                "confidence": 0.98,
                "reason": "Full visual block is a readable single molecule.",
            }
        ),
    )
    assert validator.run_crop_validation("10.1000/example", limit=1) is not None

    binding_service = MaterialStructureAgentService(
        config,
        vision_client=StaticJSONVisionClient(
            {
                "observed_label": None,
                "label_source": "unknown",
                "proposed_paper_material_id": None,
                "alternative_paper_material_ids": [],
                "decision": "no_visible_label",
                "confidence": 1.0,
                "reason": "The crop itself has no visible label.",
            }
        ),
    )

    result = binding_service.run_label_binding(
        "10.1000/example",
        model="qwen3.6-flash",
        limit=1,
    )

    assert result is not None
    binding = result.bindings[0]
    assert binding.model_decision == "matched"
    assert binding.model_proposed_paper_material_id == "M001"
    assert binding.model_observed_label == "M001"
    assert binding.model_label_source == "triage_single_full_visual_block"
    assert binding.model_confidence == 0.98
    assert binding.review_status == "pending_review"
    assert binding.raw_response["deterministic"] is True
    calls = binding_service.list_vlm_call_logs("10.1000/example")
    assert calls is not None
    assert [call.stage for call in calls].count("label_binding") == 0


def test_label_binding_preserves_visual_context_and_requires_human_review(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    config.llm.providers["qwen"].vision_enable_thinking = False
    _seed_paper(config, mining_config_path, fake_pdf_factory, manifest_writer)
    CandidateIngestionService(config).ingest_mining_result(
        paper_id="10.1000/example",
        template_id=str(_template_path().resolve()),
        payload=load_domain_template(_template_path()).example_output,
        source_name="mock",
    )
    run = MinerUParseService(config, client=_FakeMinerUClient()).parse_paper("10.1000/example")
    assert run is not None
    image_dir = Path(run.content_list_path).parent / "images"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (120, 100), "white").save(image_dir / "scheme.jpg")
    Image.new("RGB", (120, 100), "white").save(image_dir / "plot.jpg")

    segment_service = MaterialStructureAgentService(
        config,
        vision_client=StaticJSONVisionClient(
            {
                "contains_molecular_structures": True,
                "image_role": "molecule_structures",
                "has_clean_structure_depictions": True,
                "should_run_decimer_segmentation": True,
                "label_candidates": ["M001"],
                "related_paper_material_ids": ["M001"],
                "confidence": 0.9,
                "reason": "Visible structure label.",
            }
        ),
        decimer_segmentation_client=_FakeDecimerSegmentationClient(),
    )
    foundation = segment_service.run_foundation("10.1000/example")
    assert foundation is not None
    assert foundation.run.source_candidate_run_id is not None
    assert segment_service.run_figure_triage("10.1000/example", limit=1) is not None
    assert segment_service.run_decimer_segmentation("10.1000/example", limit=1) is not None
    validator = MaterialStructureAgentService(
        config,
        vision_client=StaticJSONVisionClient(
            {
                "is_molecular_depiction": True,
                "is_single_molecule": True,
                "is_complete_structure": True,
                "should_run_ocsr": True,
                "confidence": 0.97,
                "reason": "Clean structure.",
            }
        ),
    )
    assert validator.run_crop_validation("10.1000/example", limit=1) is not None
    binding_service = MaterialStructureAgentService(
        config,
        vision_client=StaticJSONVisionClient(
            {
                "observed_label": "M001",
                "label_source": "visible_near_crop",
                "proposed_paper_material_id": "M001",
                "alternative_paper_material_ids": [],
                "decision": "matched",
                "confidence": 0.91,
                "reason": "The boxed structure label matches M001.",
            }
        ),
    )

    result = binding_service.run_label_binding(
        "10.1000/example",
        model="qwen3.6-flash",
        limit=1,
    )

    assert result is not None
    assert len(result.bindings) == 1
    binding = result.bindings[0]
    assert binding.model_proposed_paper_material_id == "M001"
    assert binding.review_status == "pending_review"
    assert Path(binding.highlighted_source_figure_path).exists()
    calls = binding_service.list_vlm_call_logs("10.1000/example")
    assert calls is not None
    assert {call.stage for call in calls} == {
        "figure_triage",
        "crop_validation",
        "label_binding",
    }
    assert all(call.status == "completed" for call in calls)
    assert all(call.duration_ms is not None for call in calls)
    assert all(call.input_context["enable_thinking"] is False for call in calls)
    repeated = binding_service.run_label_binding(
        "10.1000/example",
        model="qwen3.6-flash",
        limit=1,
    )
    repeated_calls = binding_service.list_vlm_call_logs("10.1000/example")
    assert repeated is not None
    assert repeated.bindings[0].binding_id == binding.binding_id
    assert repeated_calls is not None
    assert len(repeated_calls) == len(calls)

    client = TestClient(create_app(config=config))
    list_response = client.get("/api/papers/10.1000%2Fexample/molecule-label-bindings")
    assert list_response.status_code == 200
    image_response = client.get(
        f"/api/molecule-label-bindings/{binding.binding_id}/highlighted-image"
    )
    assert image_response.status_code == 200

    stale_binding = binding.model_copy(
        update={
            "candidate_materials": [
                material
                for material in binding.candidate_materials
                if material.get("paper_material_id") == "M001"
            ]
        }
    )
    binding_service.label_bindings.upsert_proposal(stale_binding)
    current_material_correction = client.post(
        f"/api/molecule-label-bindings/{binding.binding_id}/review",
        json={
            "action": "correct",
            "actor": "tester",
            "reviewed_paper_material_id": "M002",
        },
    )
    assert current_material_correction.status_code == 200
    assert current_material_correction.json()["reviewed_paper_material_id"] == "M002"
    correction_events = client.get(
        "/api/papers/10.1000%2Fexample/molecule-label-binding-review-events"
    ).json()
    correction_event_id = next(
        event["event_id"] for event in correction_events if event["action"] == "correct"
    )
    correction_undo = client.post(
        f"/api/molecule-label-binding-review-events/{correction_event_id}/undo",
        json={"action": "undo", "actor": "tester"},
    )
    assert correction_undo.status_code == 200
    assert correction_undo.json()["review_status"] == "pending_review"

    review_response = client.post(
        f"/api/molecule-label-bindings/{binding.binding_id}/review",
        json={
            "action": "mark_material_missing",
            "actor": "tester",
            "reviewed_observed_label": "New material label",
        },
    )
    assert review_response.status_code == 200

    assert review_response.json()["review_status"] == "material_missing"
    events_response = client.get(
        "/api/papers/10.1000%2Fexample/molecule-label-binding-review-events"
    )
    assert events_response.status_code == 200
    event_id = next(
        event["event_id"]
        for event in events_response.json()
        if event["action"] == "mark_material_missing"
    )
    undo_response = client.post(
        f"/api/molecule-label-binding-review-events/{event_id}/undo",
        json={"action": "undo", "actor": "tester"},
    )
    assert undo_response.status_code == 200
    assert undo_response.json()["review_status"] == "pending_review"
    excluded_response = client.post(
        f"/api/molecule-label-bindings/{binding.binding_id}/review",
        json={"action": "mark_not_target_material", "actor": "tester"},
    )
    assert excluded_response.status_code == 200
    assert excluded_response.json()["review_status"] == "not_target_material"
    excluded_ocsr = MaterialStructureAgentService(
        config,
        decimer_smiles_client=_FakeDecimerSmilesClient(),
    ).run_decimer_ocsr("10.1000/example")
    assert excluded_ocsr is not None
    assert excluded_ocsr.eligible_binding_count == 0
    events_response = client.get(
        "/api/papers/10.1000%2Fexample/molecule-label-binding-review-events"
    )
    exclude_event_id = next(
        event["event_id"]
        for event in events_response.json()
        if event["action"] == "mark_not_target_material"
    )
    undo_excluded_response = client.post(
        f"/api/molecule-label-binding-review-events/{exclude_event_id}/undo",
        json={"action": "undo", "actor": "tester"},
    )
    assert undo_excluded_response.status_code == 200
    assert undo_excluded_response.json()["review_status"] == "pending_review"
    comparison = binding_service.run_label_binding(
        "10.1000/example",
        model="qwen3.6-plus",
        limit=1,
    )
    assert comparison is not None
    second_binding = comparison.bindings[0]
    assert second_binding.binding_id != binding.binding_id
    calls_response = client.get("/api/papers/10.1000%2Fexample/vlm-call-logs")
    assert calls_response.status_code == 200
    assert len(calls_response.json()) == 4
    assert [item["stage"] for item in calls_response.json()].count("label_binding") == 2
    confirmed_response = client.post(
        f"/api/molecule-label-bindings/{second_binding.binding_id}/review",
        json={"action": "confirm", "actor": "tester"},
    )
    assert confirmed_response.status_code == 200
    ocsr = MaterialStructureAgentService(
        config,
        decimer_smiles_client=_FakeDecimerSmilesClient(),
    ).run_decimer_ocsr("10.1000/example")
    assert ocsr is not None
    assert ocsr.eligible_binding_count == 1
    assert len(ocsr.candidates) == 1
    structure_candidate = ocsr.candidates[0]
    assert structure_candidate.provider == "decimer_ocsr"
    assert structure_candidate.canonical_smiles == "c1ccccc1"
    assert structure_candidate.status == "pending_review"
    depiction_response = client.get(
        f"/api/material-structure-candidates/{structure_candidate.structure_candidate_id}/depiction.svg"
    )
    assert depiction_response.status_code == 200
    assert "image/svg+xml" in depiction_response.headers["content-type"]
    assert depiction_response.headers["cache-control"] == "no-store, max-age=0"
    edit_response = client.post(
        f"/api/material-structure-candidates/{structure_candidate.structure_candidate_id}/edit-smiles",
        json={"actor": "tester", "smiles": "CCO", "message": "human corrected"},
    )
    assert edit_response.status_code == 200
    edited_candidate = next(
        candidate
        for candidate in edit_response.json()["structure_candidates"]
        if candidate["structure_candidate_id"] == structure_candidate.structure_candidate_id
    )
    assert edited_candidate["canonical_smiles"] == "CCO"
    edit_event_id = edit_response.json()["material_review_events"][-1]["event_id"]
    undo_edit_response = client.post(
        f"/api/material-review-events/{edit_event_id}/undo",
        json={"actor": "tester", "message": "undo corrected smiles"},
    )
    assert undo_edit_response.status_code == 200
    restored_candidate = next(
        candidate
        for candidate in undo_edit_response.json()["structure_candidates"]
        if candidate["structure_candidate_id"] == structure_candidate.structure_candidate_id
    )
    assert restored_candidate["canonical_smiles"] == "c1ccccc1"
    conflicting_response = client.post(
        f"/api/molecule-label-bindings/{binding.binding_id}/review",
        json={"action": "confirm", "actor": "tester"},
    )
    assert conflicting_response.status_code == 422

    cli_result = CliRunner().invoke(
        app,
        [
            "list-molecule-label-bindings",
            "--config",
            str(mining_config_path),
            "--paper-id",
            "10.1000/example",
        ],
    )
    assert cli_result.exit_code == 0
    assert "Molecule Label Bindings" in cli_result.output
    log_cli_result = CliRunner().invoke(
        app,
        [
            "list-vlm-call-logs",
            "--config",
            str(mining_config_path),
            "--paper-id",
            "10.1000/example",
        ],
    )
    assert log_cli_result.exit_code == 0
    assert "VLM Call Log" in log_cli_result.output
    assert "VLM calls: 4" in log_cli_result.output


def test_label_binding_groups_crops_from_one_figure_into_one_vlm_call(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper(config, mining_config_path, fake_pdf_factory, manifest_writer)
    CandidateIngestionService(config).ingest_mining_result(
        paper_id="10.1000/example",
        template_id=str(_template_path().resolve()),
        payload=load_domain_template(_template_path()).example_output,
        source_name="mock",
    )
    run = MinerUParseService(config, client=_FakeMinerUClient()).parse_paper("10.1000/example")
    assert run is not None
    image_dir = Path(run.content_list_path).parent / "images"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (140, 110), "white").save(image_dir / "scheme.jpg")
    Image.new("RGB", (140, 110), "white").save(image_dir / "plot.jpg")

    agent = MaterialStructureAgentService(
        config,
        vision_client=StaticJSONVisionClient(
            {
                "contains_molecular_structures": True,
                "image_role": "molecule_structures",
                "has_clean_structure_depictions": True,
                "should_run_decimer_segmentation": True,
                "label_candidates": ["M001"],
                "related_paper_material_ids": ["M001"],
                "confidence": 0.9,
                "reason": "Two structures are visible.",
            }
        ),
        decimer_segmentation_client=_FakeTwoCropDecimerSegmentationClient(),
    )
    assert agent.run_foundation("10.1000/example") is not None
    assert agent.run_figure_triage("10.1000/example", limit=1) is not None
    segmentation = agent.run_decimer_segmentation("10.1000/example", limit=1)
    assert segmentation is not None
    assert len(segmentation.crops) == 2

    validator = MaterialStructureAgentService(
        config,
        vision_client=StaticJSONVisionClient(
            {
                "is_molecular_depiction": True,
                "is_single_molecule": True,
                "is_complete_structure": True,
                "should_run_ocsr": True,
                "confidence": 0.97,
                "reason": "Clean structure.",
            }
        ),
    )
    assert validator.run_crop_validation("10.1000/example") is not None
    binding_payload = {
        "bindings": [
            {
                "crop_id": crop.crop_id,
                "observed_label": f"M001-{index}",
                "label_source": "visible_near_crop",
                "proposed_paper_material_id": "M001",
                "alternative_paper_material_ids": [],
                "decision": "matched",
                "confidence": 0.93,
                "reason": "Visible label and position match the material.",
            }
            for index, crop in enumerate(segmentation.crops, start=1)
        ]
    }
    binder = MaterialStructureAgentService(
        config,
        vision_client=StaticJSONVisionClient(binding_payload),
    )

    result = binder.run_label_binding("10.1000/example", model="qwen3.6-flash", max_concurrency=4)

    assert result is not None
    assert len(result.bindings) == 2
    assert all(binding.raw_response["grouped"] is True for binding in result.bindings)
    calls = binder.list_vlm_call_logs("10.1000/example")
    assert calls is not None
    assert [call.stage for call in calls].count("label_binding") == 1
    repeated = binder.run_label_binding("10.1000/example", model="qwen3.6-flash", max_concurrency=4)
    assert repeated is not None
    repeated_calls = binder.list_vlm_call_logs("10.1000/example")
    assert repeated_calls is not None
    assert [call.stage for call in repeated_calls].count("label_binding") == 1


def test_label_binding_rejects_invented_paper_material_id() -> None:
    normalized = normalize_label_binding_payload(
        {
            "observed_label": "unknown compound",
            "proposed_paper_material_id": "M999",
            "decision": "matched",
            "confidence": 0.8,
        },
        [{"paper_material_id": "M001"}],
    )

    assert normalized["model_proposed_paper_material_id"] is None
    assert normalized["model_decision"] == "material_not_in_candidate_list"
    assert normalized["review_status"] == "pending_review"


def test_label_binding_does_not_match_r_group_to_complete_material() -> None:
    normalized = normalize_label_binding_payload(
        {
            "observed_label": None,
            "label_source": "inferred_context",
            "depiction_scope": "fragment_or_r_group",
            "proposed_paper_material_id": "M003",
            "decision": "matched",
            "confidence": 1.0,
            "reason": "The extracted R-group is the substituent used by M003.",
        },
        [{"paper_material_id": "M003"}],
    )

    assert normalized["model_proposed_paper_material_id"] is None
    assert normalized["model_decision"] == "ambiguous"


def test_legacy_inferred_fragment_binding_is_not_eligible_for_ocsr() -> None:
    binding = MoleculeLabelBinding(
        binding_id="binding-1",
        paper_id="10.1000%2Ffragment",
        candidate_run_id="run-1",
        agent_run_id="agent-1",
        crop_id="crop-1",
        visual_block_id="block-1",
        provider="qwen",
        model="qwen3.6-flash",
        source_figure_path="figure.png",
        highlighted_source_figure_path="highlighted.png",
        crop_path="crop.png",
        model_observed_label=None,
        model_label_source="inferred_context",
        model_proposed_paper_material_id="M003",
        model_decision="matched",
        model_confidence=1.0,
        model_reason="The extracted R-group is the substituent defined for M003.",
        review_status="pending_review",
        created_at="2026-07-16T00:00:00+08:00",
        updated_at="2026-07-16T00:00:00+08:00",
    )

    assert (
        _binding_eligible_for_ocsr(
            binding,
            allow_unreviewed_matches=True,
            min_model_confidence=0.8,
        )
        is False
    )


class _FakeDecimerSegmentationClient:
    def segment_image(
        self,
        image_path: Path,
        *,
        expand: bool,
        return_images: bool,
        max_segments: int,
    ) -> DecimerSegmentationResponse:
        return DecimerSegmentationResponse(
            file_name=image_path.name,
            segment_count=1,
            bbox_available=True,
            bbox_source_order="xyxy",
            segments=[
                DecimerSegment(
                    index=0,
                    bbox=[10.0, 20.0, 80.0, 90.0],
                    width=70,
                    height=70,
                    raw_segment={"index": 0, "bbox": {"x0": 10, "y0": 20, "x1": 80, "y1": 90}},
                )
            ],
            raw_response={"mock": True},
        )


class _FakeTwoCropDecimerSegmentationClient:
    def segment_image(
        self,
        image_path: Path,
        *,
        expand: bool,
        return_images: bool,
        max_segments: int,
    ) -> DecimerSegmentationResponse:
        return DecimerSegmentationResponse(
            file_name=image_path.name,
            segment_count=2,
            bbox_available=True,
            bbox_source_order="xyxy",
            segments=[
                DecimerSegment(
                    index=0,
                    bbox=[10.0, 15.0, 60.0, 85.0],
                    width=50,
                    height=70,
                    raw_segment={"index": 0},
                ),
                DecimerSegment(
                    index=1,
                    bbox=[75.0, 15.0, 130.0, 85.0],
                    width=55,
                    height=70,
                    raw_segment={"index": 1},
                ),
            ],
            raw_response={"mock": True},
        )


class _FakeFailingDecimerSegmentationClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def segment_image(
        self,
        image_path: Path,
        *,
        expand: bool,
        return_images: bool,
        max_segments: int,
    ) -> DecimerSegmentationResponse:
        self.calls.append(image_path.name)
        raise RuntimeError("mock DECIMER segmentation failure")


class _FakeDecimerSmilesClient:
    def predict_smiles(self, image_path: Path) -> DecimerSmilesResponse:
        return DecimerSmilesResponse(
            file_name=image_path.name,
            smiles="c1ccccc1",
            confidence_tokens=None,
            hand_drawn=False,
            raw_response={"smiles": "c1ccccc1", "mock": True},
        )


class _FakeMinerUClient:
    def parse_pdf(self, pdf_path: Path) -> MinerUParsedDocument:
        content_list = [
            {
                "type": "text",
                "text": "The paper contains molecular structures and device performance plots.",
                "page_idx": 0,
                "bbox": [1, 2, 3, 4],
            },
            {
                "type": "image",
                "sub_type": "chemical",
                "img_path": "images/scheme.jpg",
                "image_caption": ["Scheme 1. Molecular structures of emitters."],
                "page_idx": 0,
                "bbox": [10, 20, 200, 220],
            },
            {
                "type": "chart",
                "sub_type": "line",
                "img_path": "images/plot.jpg",
                "chart_caption": ["Figure 2. EQE-luminance curves."],
                "page_idx": 1,
                "bbox": [30, 40, 250, 260],
            },
            {
                "type": "table",
                "img_path": "images/table.jpg",
                "table_caption": ["Table 1. OLED device performance."],
                "table_body": "<table><tr><td>D1</td><td>EQE 31.2%</td></tr></table>",
                "page_idx": 2,
                "bbox": [50, 60, 300, 330],
            },
        ]
        return MinerUParsedDocument(
            task_id="task-agent-foundation",
            backend="hybrid-auto-engine",
            version="3.1.0",
            file_name=pdf_path.stem,
            md_content="# mock",
            content_list=content_list,
            raw_result={
                "results": {pdf_path.stem: {"md_content": "# mock", "content_list": content_list}}
            },
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


def _template_path() -> Path:
    return Path("config/mining_platform/domains/oled_device_v1.yaml")
