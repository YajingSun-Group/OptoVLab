from __future__ import annotations

import copy
import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from evolab_local.mining_platform.api.app import create_app
from evolab_local.mining_platform.candidate_ingestion_service import CandidateIngestionService
from evolab_local.mining_platform.core.config import load_config
from evolab_local.mining_platform.domain_template_service import load_domain_template
from evolab_local.mining_platform.external.pubchem_client import PubChemCompound
from evolab_local.mining_platform.library.paper_service import PaperService
from evolab_local.mining_platform.material_public_resolver_service import (
    MaterialPublicResolverService,
)
from evolab_local.mining_platform.material_resolution_service import (
    MaterialResolutionService,
    normalize_material_alias,
)
from evolab_local.mining_platform.material_structure_review_service import (
    MaterialStructureReviewService,
)
from evolab_local.mining_platform.schemas.material_structure import (
    MaterialManualStructureAction,
    MaterialPropertyCandidate,
    MaterialReviewAction,
)
from evolab_local.mining_platform.storage.database import Database, now_iso
from evolab_local.mining_platform.storage.repositories import (
    MaterialPropertyCandidateRepository,
    PaperRepository,
)


def test_candidate_v2_api_returns_latest_bundle_and_updates_value(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _copy_template_to_config(mining_config_path)
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

    client = TestClient(create_app(config=config))
    bundle_response = client.get("/api/papers/10.1000%2Fexample/candidate-v2")

    assert bundle_response.status_code == 200
    assert bundle_response.headers["content-encoding"] == "gzip"
    bundle = bundle_response.json()
    assert bundle["run"]["status"] == "completed"
    assert bundle["entities"]
    assert bundle["values"]
    assert bundle["evidence_anchors"]
    assert bundle["template"]["template_id"] == "oled_device_v1"

    compact_response = client.get(
        "/api/papers/10.1000%2Fexample/candidate-v2?compact=true"
    )
    assert compact_response.status_code == 200
    compact_bundle = compact_response.json()
    assert compact_bundle["run"]["mining_result"] == {}
    assert compact_bundle["run"]["validation_report"] == {}
    assert compact_bundle["template"] is None
    assert all(entity["source_json"] == {} for entity in compact_bundle["entities"])
    assert compact_bundle["values"] == bundle["values"]

    eqe_value = next(
        value
        for value in bundle["values"]
        if value["template_field_path"] == "devices[].performance[].normalized_value"
        and value["value_json"] == 31.2
    )
    update_response = client.put(
        f"/api/candidate-values/{eqe_value['candidate_value_id']}",
        json={"reviewed_value_json": 30.8, "actor": "tester", "message": "corrected EQE"},
    )

    assert update_response.status_code == 200
    updated = update_response.json()
    assert updated["reviewed_value_json"] == 30.8
    assert updated["display_value"] == "30.8"
    assert updated["status"] == "modified"

    accept_response = client.post(
        f"/api/candidate-values/{eqe_value['candidate_value_id']}/confirm",
        json={"actor": "tester", "message": "accepted corrected EQE"},
    )
    assert accept_response.status_code == 200
    accepted = accept_response.json()
    assert accepted["status"] == "accepted"

    events_response = client.get(
        f"/api/candidate-values/{eqe_value['candidate_value_id']}/review-events"
    )
    assert events_response.status_code == 200
    events = events_response.json()
    assert [event["action"] for event in events] == ["modified", "accepted"]
    assert events[0]["original_value_json"] == 31.2
    assert events[0]["after_reviewed_value_json"] == 30.8
    assert events[0]["actor"] == "tester"

    undo_response = client.post(
        f"/api/candidate-value-review-events/{events[1]['event_id']}/undo",
        json={"actor": "tester", "message": "undo accepted status"},
    )
    assert undo_response.status_code == 200
    undone = undo_response.json()
    assert undone["status"] == "modified"
    assert undone["reviewed_value_json"] == 30.8

    run_events_response = client.get(
        f"/api/candidate-runs/{bundle['run']['candidate_run_id']}/value-review-events"
    )
    assert run_events_response.status_code == 200
    run_events = run_events_response.json()
    assert [event["action"] for event in run_events] == ["modified", "accepted", "undo"]
    assert run_events[-1]["before_status"] == "accepted"
    assert run_events[-1]["after_status"] == "modified"

    confirm_response = client.post(
        "/api/papers/10.1000%2Fexample/confirm-review-v2",
        json={"actor": "tester", "message": "confirm v2 in test"},
    )

    assert confirm_response.status_code == 200
    confirmed = confirm_response.json()
    assert confirmed["paper_id"] == "10.1000%2Fexample"
    assert confirmed["final_record"]["confirmed_by"] == "tester"
    assert (
        confirmed["final_record"]["final_json"]["devices"][0]["performance"][0][
            "normalized_value"
        ]
        == 30.8
    )

    final_response = client.get("/api/papers/10.1000%2Fexample/final-candidates")
    assert final_response.status_code == 200
    assert len(final_response.json()) == 1



def test_confirm_v2_filters_materials_not_used_by_devices(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    payload = _valid_result()
    materials = payload["materials"]
    assert isinstance(materials, list)
    materials.append(
        {
            "paper_material_id": "M999",
            "global_material_id": None,
            "mention_list": ["BD-02"],
            "full_name_in_paper": None,
            "normalized_name": "BD-02",
            "canonical_name": None,
            "abbreviation": "BD-02",
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
    _seed_paper_and_candidate(
        config,
        mining_config_path,
        fake_pdf_factory,
        manifest_writer,
        payload=payload,
    )

    client = TestClient(create_app(config=config))
    confirm_response = client.post(
        "/api/papers/10.1000%2Fexample/confirm-review-v2",
        json={"actor": "tester"},
    )

    assert confirm_response.status_code == 200
    final_json = confirm_response.json()["final_record"]["final_json"]
    material_ids = {
        material.get("paper_material_id")
        for material in final_json["materials"]
        if isinstance(material, dict)
    }
    assert "M001" in material_ids
    assert "M002" in material_ids
    assert "M999" not in material_ids


def test_confirm_v2_writes_accepted_public_material_structure(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper_and_candidate(config, mining_config_path, fake_pdf_factory, manifest_writer)
    resolved = MaterialPublicResolverService(
        config,
        pubchem_client=FakeMcbpPubChemClient(),
    ).resolve_paper_public("10.1000/example", paper_material_id="M002")
    assert resolved is not None
    candidate = next(
        item for item in resolved.structure_candidates if item.paper_material_id == "M002"
    )
    accepted = MaterialStructureReviewService(config).accept_structure_candidate(
        candidate.structure_candidate_id,
        MaterialReviewAction(actor="tester", message="accepted public mCBP"),
    )
    assert accepted is not None

    client = TestClient(create_app(config=config))
    confirm_response = client.post(
        "/api/papers/10.1000%2Fexample/confirm-review-v2",
        json={"actor": "tester"},
    )

    assert confirm_response.status_code == 200
    final_json = confirm_response.json()["final_record"]["final_json"]
    material = _material_by_id(final_json, "M002")
    component = _component_by_paper_material_id(final_json, "M002")
    assert material["global_material_id"]
    assert material["material_structure_status"] == "accepted"
    assert material["structure_review_status"] == "accepted"
    assert material["structure_candidate_id"] == candidate.structure_candidate_id
    assert material["structure_source"] == "pubchem"
    assert material["canonical_smiles"] == "C1=CC=C(C=C1)N2C3=CC=CC=C3C4=CC=CC=C42"
    assert material["smiles"] == material["canonical_smiles"]
    assert material["inchi_key"] == "MCBPKEY"
    assert component["global_material_id"] == material["global_material_id"]
    assert component["material_structure_status"] == "accepted"
    assert "canonical_smiles" not in component


def test_confirm_v2_writes_manual_material_structure_to_final_emitter(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper_and_candidate(config, mining_config_path, fake_pdf_factory, manifest_writer)
    saved = MaterialStructureReviewService(config).save_manual_structure(
        "10.1000/example",
        "M001",
        MaterialManualStructureAction(
            actor="tester",
            reviewed_name="BN-1",
            smiles="c1ccccc1",
            source_note="Manual structure from Scheme 1",
        ),
    )
    assert saved is not None
    candidate = next(item for item in saved.structure_candidates if item.paper_material_id == "M001")

    client = TestClient(create_app(config=config))
    confirm_response = client.post(
        "/api/papers/10.1000%2Fexample/confirm-review-v2",
        json={"actor": "tester"},
    )

    assert confirm_response.status_code == 200
    final_json = confirm_response.json()["final_record"]["final_json"]
    material = _material_by_id(final_json, "M001")
    final_emitter = final_json["devices"][0]["final_emitter"]
    assert material["material_structure_status"] == "accepted"
    assert material["structure_source"] == "manual_input"
    assert material["structure_candidate_id"] == candidate.structure_candidate_id
    assert material["canonical_smiles"] == "c1ccccc1"
    assert material["inchi_key"]
    assert final_emitter["global_material_id"] == material["global_material_id"]
    assert final_emitter["material_structure_status"] == "accepted"
    assert final_emitter["structure_candidate_id"] == candidate.structure_candidate_id
    assert "canonical_smiles" not in final_emitter


def test_confirm_v2_writes_identity_only_and_out_of_scope_without_smiles(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper_and_candidate(
        config,
        mining_config_path,
        fake_pdf_factory,
        manifest_writer,
        payload=_result_with_scope_auxiliary_materials(),
    )
    resolved = MaterialResolutionService(config).resolve_paper_materials("10.1000/example")
    assert resolved is not None

    client = TestClient(create_app(config=config))
    confirm_response = client.post(
        "/api/papers/10.1000%2Fexample/confirm-review-v2",
        json={"actor": "tester"},
    )

    assert confirm_response.status_code == 200
    final_json = confirm_response.json()["final_record"]["final_json"]
    ito = _material_by_id(final_json, "M003")
    al = _material_by_id(final_json, "M004")
    liq = _material_by_id(final_json, "M005")
    assert ito["material_structure_status"] == "out_of_scope_structure"
    assert al["material_structure_status"] == "out_of_scope_structure"
    assert liq["material_structure_status"] == "identity_only"
    for material in (ito, al, liq):
        assert material["smiles"] is None
        assert material["inchi"] is None
        assert material["inchi_key"] is None
        assert "canonical_smiles" not in material or material["canonical_smiles"] is None
        assert material["material_structure_scope"]["requires_public_resolution"] is False
    ito_component = _component_by_paper_material_id(final_json, "M003")
    liq_component = _component_by_paper_material_id(final_json, "M005")
    assert ito_component["material_structure_status"] == "out_of_scope_structure"
    assert liq_component["material_structure_status"] == "identity_only"
    assert "canonical_smiles" not in liq_component


def test_material_properties_feature_flag_hides_candidates_and_final_json(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    config.features.material_properties = False
    _seed_paper_and_candidate(config, mining_config_path, fake_pdf_factory, manifest_writer)
    bundle = CandidateIngestionService(config).get_review_bundle("10.1000/example")
    assert bundle is not None
    assert bundle.run is not None
    candidate_run_id = bundle.run.candidate_run_id

    properties = MaterialPropertyCandidateRepository(Database(config.paths.sqlite_path))
    _seed_material_property(
        properties,
        candidate_run_id,
        paper_material_id="M001",
        property_name="PLQY",
        property_category="photophysical",
        value_numeric=83.0,
        value_raw="83%",
        unit="%",
        status="accepted",
    )

    client = TestClient(create_app(config=config))
    material_response = client.get("/api/papers/10.1000%2Fexample/material-structures")
    assert material_response.status_code == 200
    material_bundle = material_response.json()
    assert material_bundle["property_candidates"] == []
    assert material_bundle["property_reviews"] == []
    assert material_bundle["property_review_events"] == []

    disabled_response = client.post(
        "/api/papers/10.1000%2Fexample/manual-material-property?paper_material_id=M001",
        json={"actor": "tester", "property_name": "HOMO"},
    )
    assert disabled_response.status_code == 404

    confirm_response = client.post(
        "/api/papers/10.1000%2Fexample/confirm-review-v2",
        json={"actor": "tester"},
    )
    assert confirm_response.status_code == 200
    final_json = confirm_response.json()["final_record"]["final_json"]
    assert "reported_properties" not in _material_by_id(final_json, "M001")


def test_confirm_v2_writes_reviewed_material_properties_to_final_json(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper_and_candidate(config, mining_config_path, fake_pdf_factory, manifest_writer)
    bundle = CandidateIngestionService(config).get_review_bundle("10.1000/example")
    assert bundle is not None
    assert bundle.run is not None
    candidate_run_id = bundle.run.candidate_run_id

    properties = MaterialPropertyCandidateRepository(Database(config.paths.sqlite_path))
    _seed_material_property(
        properties,
        candidate_run_id,
        paper_material_id="M001",
        property_name="PLQY",
        property_category="photophysical",
        value_numeric=83.0,
        value_raw="83%",
        unit="%",
        normalized_value_numeric=0.83,
        normalized_unit="fraction",
        condition={"sample_form": "doped_film", "host": "mCBP"},
        method="integrating_sphere",
        source_type="table",
        evidence_text="The PLQY of BN-1 was 83% in mCBP.",
        evidence_anchor={"page": 4, "block_id": "table-1"},
        confidence=0.91,
        status="accepted",
    )
    _seed_material_property(
        properties,
        candidate_run_id,
        paper_material_id="M001",
        property_name="HOMO",
        property_category="electronic",
        value_numeric=-5.52,
        value_raw="-5.52 eV",
        unit="eV",
        normalized_value_numeric=-5.52,
        normalized_unit="eV",
        condition={"method": "cyclic_voltammetry"},
        method="cyclic_voltammetry",
        source_type="text",
        evidence_text="The HOMO level was estimated to be -5.52 eV.",
        evidence_anchor={"page": 3, "block_id": "text-7"},
        confidence=0.84,
        status="edited_accepted",
    )
    _seed_material_property(
        properties,
        candidate_run_id,
        paper_material_id="M001",
        property_name="LUMO",
        property_category="electronic",
        value_numeric=-2.71,
        value_raw="-2.71 eV",
        unit="eV",
        status="rejected",
    )
    _seed_material_property(
        properties,
        candidate_run_id,
        paper_material_id="M002",
        property_name="Tg",
        property_category="thermal",
        value_numeric=93.0,
        value_raw="93 °C",
        unit="°C",
        normalized_value_numeric=93.0,
        normalized_unit="°C",
        method="DSC",
        source_type="manual",
        evidence_text="Manual reviewer added Tg from the paper table.",
        confidence=1.0,
        status="manual_added",
    )
    _seed_material_property(
        properties,
        candidate_run_id,
        paper_material_id="M002",
        property_name="T1",
        property_category="electronic",
        value_numeric=2.7,
        value_raw="2.7 eV",
        unit="eV",
        status="pending_review",
    )

    client = TestClient(create_app(config=config))
    confirm_response = client.post(
        "/api/papers/10.1000%2Fexample/confirm-review-v2",
        json={"actor": "tester"},
    )

    assert confirm_response.status_code == 200
    final_json = confirm_response.json()["final_record"]["final_json"]
    emitter = _material_by_id(final_json, "M001")
    host = _material_by_id(final_json, "M002")

    emitter_properties = emitter["reported_properties"]
    assert [item["property_name"] for item in emitter_properties] == ["HOMO", "PLQY"]
    assert {item["review_status"] for item in emitter_properties} == {
        "accepted",
        "edited_accepted",
    }
    plqy = next(item for item in emitter_properties if item["property_name"] == "PLQY")
    assert plqy["source"] == "paper_reported"
    assert plqy["value_numeric"] == 83.0
    assert plqy["normalized_value_numeric"] == 0.83
    assert plqy["condition"] == {"host": "mCBP", "sample_form": "doped_film"}
    assert plqy["evidence_anchor"] == {"block_id": "table-1", "page": 4}
    assert not any(item["property_name"] == "LUMO" for item in emitter_properties)

    assert [item["property_name"] for item in host["reported_properties"]] == ["Tg"]
    assert host["reported_properties"][0]["review_status"] == "manual_added"
    assert not any(item["property_name"] == "T1" for item in host["reported_properties"])


def test_candidate_v2_api_returns_empty_bundle_without_run(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _copy_template_to_config(mining_config_path)
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

    client = TestClient(create_app(config=config))
    bundle_response = client.get("/api/papers/10.1000%2Fexample/candidate-v2")

    assert bundle_response.status_code == 200
    assert bundle_response.json() == {
        "run": None,
        "entities": [],
        "values": [],
        "evidence_anchors": [],
        "template": None,
    }


def test_empty_device_result_is_valid_and_auto_confirmed_with_reason(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _copy_template_to_config(mining_config_path)
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

    result = CandidateIngestionService(config).ingest_mining_result(
        paper_id="10.1000/example",
        template_id=str(_template_path().resolve()),
        payload={"evidence": [], "materials": [], "devices": []},
        source_name="mock",
    )

    assert result is not None
    assert result.run.status == "no_device"
    assert result.value_count == 0
    paper = PaperRepository(Database(config.paths.sqlite_path)).get("10.1000%2Fexample")
    assert paper is not None
    assert paper.mining_status == "completed"
    assert paper.review_status == "confirmed"
    assert paper.review_reason == "no_device_data"


def test_devices_without_any_reported_stack_are_auto_confirmed_as_no_stack(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _copy_template_to_config(mining_config_path)
    pdf_path = fake_pdf_factory(mining_config_path)
    manifest_writer(
        mining_config_path,
        [
            {
                "doi": "10.1000/example",
                "status": "completed",
                "pdf_path": pdf_path.relative_to(config.project_root).as_posix(),
            }
        ],
    )
    PaperService(config).ingest_from_pdf_downloader(domain="oled")
    payload = {
        "evidence": [],
        "materials": [],
        "devices": [
            {
                "device_label": "Device mentioned without a reported stack",
                "architecture_text": None,
                "layers": [],
                "performance": [],
                "evidence_refs": [],
            }
        ],
    }

    result = CandidateIngestionService(config).ingest_mining_result(
        paper_id="10.1000/example",
        template_id=str(_template_path().resolve()),
        payload=payload,
        source_name="mock",
    )

    assert result is not None
    assert result.validation_report.valid is True
    assert result.run.status == "no_device"
    assert result.value_count == 0
    paper = PaperRepository(Database(config.paths.sqlite_path)).get("10.1000%2Fexample")
    assert paper is not None
    assert paper.mining_status == "completed"
    assert paper.review_status == "confirmed"
    assert paper.review_reason == "no_extractable_device_stack"


class FakeMcbpPubChemClient:
    def resolve_name(self, name: str, *, max_results: int | None = None) -> list[PubChemCompound]:
        if normalize_material_alias(name) != "mcbp":
            return []
        return [
            PubChemCompound(
                cid="999",
                query_text=name,
                iupac_name="3,3'-bis(9-carbazolyl)biphenyl",
                canonical_smiles="C1=CC=C(C=C1)N2C3=CC=CC=C3C4=CC=CC=C42",
                isomeric_smiles="C1=CC=C(C=C1)N2C3=CC=CC=C3C4=CC=CC=C42",
                inchi="InChI=1S/mcbp",
                inchi_key="MCBPKEY",
                formula="C36H24N2",
                molecular_weight=484.6,
                synonyms=["mCBP", "3,3'-di(9H-carbazol-9-yl)biphenyl"],
            )
        ]


def _seed_paper_and_candidate(
    config,
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
    *,
    payload: dict[str, object] | None = None,
) -> None:
    _copy_template_to_config(mining_config_path)
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
        payload=payload or _valid_result(),
        source_name="mock",
    )


def _result_with_scope_auxiliary_materials() -> dict[str, object]:
    payload = _valid_result()
    materials = payload["materials"]
    assert isinstance(materials, list)
    materials.extend(
        [
            {
                "paper_material_id": "M003",
                "global_material_id": None,
                "mention_list": ["ITO"],
                "full_name_in_paper": "indium tin oxide",
                "normalized_name": "ITO",
                "canonical_name": None,
                "abbreviation": "ITO",
                "paper_specific_label": None,
                "material_class": "inorganic",
                "smiles": "[bad-llm-smiles]",
                "inchi": "bad-inchi",
                "inchi_key": "bad-inchikey",
                "structure_source": None,
                "structure_confidence": None,
                "evidence_refs": ["E1"],
            },
            {
                "paper_material_id": "M004",
                "global_material_id": None,
                "mention_list": ["Al"],
                "full_name_in_paper": "aluminum",
                "normalized_name": "Al",
                "canonical_name": None,
                "abbreviation": "Al",
                "paper_specific_label": None,
                "material_class": "inorganic",
                "smiles": "[bad-llm-smiles]",
                "inchi": "bad-inchi",
                "inchi_key": "bad-inchikey",
                "structure_source": None,
                "structure_confidence": None,
                "evidence_refs": ["E1"],
            },
            {
                "paper_material_id": "M005",
                "global_material_id": None,
                "mention_list": ["Liq"],
                "full_name_in_paper": None,
                "normalized_name": "Liq",
                "canonical_name": None,
                "abbreviation": "Liq",
                "paper_specific_label": None,
                "material_class": "salt",
                "smiles": "[bad-llm-smiles]",
                "inchi": "bad-inchi",
                "inchi_key": "bad-inchikey",
                "structure_source": None,
                "structure_confidence": None,
                "evidence_refs": ["E1"],
            },
        ]
    )
    devices = payload["devices"]
    assert isinstance(devices, list)
    layers = devices[0]["layers"]
    assert isinstance(layers, list)
    layers[0]["components"][0]["paper_material_id"] = "M003"
    layers[5]["layer_name"] = "Liq"
    layers[5]["components"][0]["paper_material_id"] = "M005"
    layers[5]["components"][0]["material_mention"] = "Liq"
    layers[6]["components"][0]["paper_material_id"] = "M004"
    devices[0]["architecture_text"] = "ITO/HATCN/TAPC/mCBP:3 wt% BN-1/TPBi/Liq/Al"
    return payload


def _seed_material_property(
    repository: MaterialPropertyCandidateRepository,
    candidate_run_id: str,
    *,
    paper_material_id: str,
    property_name: str,
    property_category: str,
    value_numeric: float | None = None,
    value_text: str | None = None,
    value_raw: str | None = None,
    unit: str | None = None,
    normalized_value_numeric: float | None = None,
    normalized_unit: str | None = None,
    condition: dict[str, object] | None = None,
    method: str | None = None,
    source_type: str = "text",
    evidence_text: str | None = None,
    evidence_anchor: dict[str, object] | None = None,
    confidence: float | None = None,
    status: str = "pending_review",
) -> MaterialPropertyCandidate:
    timestamp = now_iso()
    candidate_id = f"prop-{paper_material_id}-{property_name}-{status}"
    return repository.upsert(
        MaterialPropertyCandidate(
            property_candidate_id=candidate_id,
            paper_id="10.1000%2Fexample",
            candidate_run_id=candidate_run_id,
            paper_material_id=paper_material_id,
            property_name=property_name,
            property_category=property_category,
            value_numeric=value_numeric,
            value_text=value_text,
            value_raw=value_raw,
            unit=unit,
            normalized_value_numeric=normalized_value_numeric,
            normalized_unit=normalized_unit,
            condition=condition or {},
            method=method,
            source_type=source_type,
            evidence_text=evidence_text,
            evidence_anchor=evidence_anchor or {},
            provider="test",
            model="mock",
            prompt_version="material_property_miner_v1",
            confidence=confidence,
            status=status,
            created_at=timestamp,
            updated_at=timestamp,
        )
    )


def _material_by_id(final_json: dict[str, object], paper_material_id: str) -> dict[str, object]:
    materials = final_json["materials"]
    assert isinstance(materials, list)
    return next(
        material
        for material in materials
        if isinstance(material, dict) and material.get("paper_material_id") == paper_material_id
    )


def _component_by_paper_material_id(
    final_json: dict[str, object],
    paper_material_id: str,
) -> dict[str, object]:
    devices = final_json["devices"]
    assert isinstance(devices, list)
    for device in devices:
        assert isinstance(device, dict)
        layers = device.get("layers")
        assert isinstance(layers, list)
        for layer in layers:
            assert isinstance(layer, dict)
            components = layer.get("components")
            assert isinstance(components, list)
            for component in components:
                if (
                    isinstance(component, dict)
                    and component.get("paper_material_id") == paper_material_id
                ):
                    return component
    raise AssertionError(f"Component not found: {paper_material_id}")


def _valid_result() -> dict[str, object]:
    return copy.deepcopy(load_domain_template(_template_path()).example_output)


def _copy_template_to_config(mining_config_path: Path) -> None:
    root = mining_config_path.parent.parent.parent
    domains_dir = root / "config" / "mining_platform" / "domains"
    domains_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_template_path(), domains_dir / "oled_device_v1.yaml")


def _template_path() -> Path:
    return Path("config/mining_platform/domains/oled_device_v1.yaml")
