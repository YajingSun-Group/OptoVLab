from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from evolab_local.mining_platform.api.app import create_app
from evolab_local.mining_platform.candidate_ingestion_service import CandidateIngestionService
from evolab_local.mining_platform.cli import app
from evolab_local.mining_platform.core.config import load_config
from evolab_local.mining_platform.external.anysearch_client import (
    AnySearchResult,
    parse_anysearch_response,
)
from evolab_local.mining_platform.external.openai_compatible_client import (
    LLMResponse,
    StaticJSONLLMClient,
)
from evolab_local.mining_platform.domain_template_service import load_domain_template
from evolab_local.mining_platform.external.opsin_client import OpsinCompound, parse_opsin_response
from evolab_local.mining_platform.external.pubchem_client import (
    PubChemCompound,
    parse_pubchem_property_response,
    parse_pubchem_synonyms_response,
)
from evolab_local.mining_platform.material_public_resolver_service import (
    MaterialPublicResolverService,
)
from evolab_local.mining_platform.material_identity_judge_service import (
    MaterialIdentityJudgeExecutionError,
    MaterialIdentityJudgeService,
)
from evolab_local.mining_platform.material_identity_evidence_service import (
    MaterialIdentityEvidenceService,
)
from evolab_local.mining_platform.material_auto_decision_service import (
    MaterialAutoDecisionService,
)
from evolab_local.mining_platform.library.paper_service import PaperService
from evolab_local.mining_platform.literature_verified_import_service import (
    LiteratureVerifiedImportService,
    LiteratureVerifiedStructureRecord,
)
from evolab_local.mining_platform.material_resolution_service import (
    MaterialResolutionService,
    classify_material_structure_scope,
    has_confirmed_alias_conflict,
    is_reusable_material_alias_text,
    normalize_material_alias,
)
from evolab_local.mining_platform.material_structure_review_service import (
    MaterialStructureReviewService,
)
from evolab_local.mining_platform.material_web_rescue_service import (
    MaterialWebRescueDecision,
    MaterialWebRescueService,
    MaterialWebRescueSource,
)
from evolab_local.mining_platform.material_web_rescue_report_service import (
    MaterialWebRescueReportService,
)
from evolab_local.mining_platform.material_stage3_planner_service import (
    MaterialStage3PlannerService,
)
from evolab_local.mining_platform.schemas.material_structure import (
    MaterialAlias,
    MaterialGlobal,
    MaterialUsage,
    MaterialIdentityEvidenceItem,
    MaterialIdentityEvidenceRun,
    MaterialManualStructureAction,
    MaterialStructureCandidate,
    MaterialReviewAction,
    PaperLocalMaterial,
    PaperMaterialNameReviewAction,
)
from evolab_local.mining_platform.storage.database import Database, now_iso
from evolab_local.mining_platform.storage.repositories import (
    MaterialIdentityEvidenceItemRepository,
    MaterialIdentityEvidenceRunRepository,
    MaterialResolutionTaskRepository,
    PaperMaterialLinkRepository,
)


def test_normalize_material_alias_handles_common_variants() -> None:
    assert normalize_material_alias("HAT-CN") == "hatcn"
    assert normalize_material_alias("HAT CN") == "hatcn"
    assert normalize_material_alias("ν-DABNA") == "nudabna"


@pytest.mark.parametrize(
    "alias",
    [
        "host",
        "host material",
        "BH",
        "M009",
        "compound 3",
        "final emitter",
        "C39H27N3",
        "C60",
    ],
)
def test_generic_or_paper_local_labels_are_not_reusable_global_aliases(alias: str) -> None:
    assert is_reusable_material_alias_text(alias) is False


@pytest.mark.parametrize("alias", ["mCBP", "DtBuCzB", "HAT-CN", "DOBNA-Tol"])
def test_chemical_identity_aliases_remain_reusable(alias: str) -> None:
    assert is_reusable_material_alias_text(alias) is True


def test_confirmed_alias_conflict_blocks_cross_global_registration() -> None:
    timestamp = now_iso()
    aliases = [
        MaterialAlias(
            alias_id="alias-1",
            global_material_id="correct-global",
            alias_text="TmPyPB",
            normalized_alias="tmpypb",
            review_status="confirmed",
            created_at=timestamp,
            updated_at=timestamp,
        )
    ]

    assert has_confirmed_alias_conflict(aliases, "wrong-global") is True
    assert has_confirmed_alias_conflict(aliases, "correct-global") is False


def test_material_scope_skips_polymer_and_coordination_complex_from_ocsr() -> None:
    polymer_scope = classify_material_structure_scope(
        PaperLocalMaterial(
            paper_material_id="M-POLY",
            entity_path="materials[0]",
            entity_label="PEDOT:PSS",
            mention_list=["PEDOT:PSS"],
            abbreviation="PEDOT:PSS",
            material_class="polymer_blend",
            used_in=[
                MaterialUsage(
                    layer_role="HIL",
                    component_role="injection_material",
                    material_mention="PEDOT:PSS",
                )
            ],
        )
    )
    assert polymer_scope["category"] == "identity_only"
    assert polymer_scope["requires_structure"] is False
    assert polymer_scope["rule"] == "known_identity_only_oled_auxiliary_material"

    complex_scope = classify_material_structure_scope(
        PaperLocalMaterial(
            paper_material_id="M-IR",
            entity_path="materials[1]",
            entity_label="Ir(ppy)3",
            mention_list=["Ir(ppy)3"],
            abbreviation="Ir(ppy)3",
            material_class="organometallic_complex",
            used_in=[
                MaterialUsage(
                    layer_role="EML", component_role="emitter", material_mention="Ir(ppy)3"
                )
            ],
        )
    )
    assert complex_scope["category"] == "identity_only"
    assert complex_scope["requires_structure"] is False
    assert complex_scope["rule"] == "material_class_coordination_complex_identity_only"

    class_only_complex_scope = classify_material_structure_scope(
        PaperLocalMaterial(
            paper_material_id="M-CLASS",
            entity_path="materials[3]",
            entity_label="Emitter-X",
            mention_list=["Emitter-X"],
            abbreviation="Emitter-X",
            material_class="organometallic_complex",
            used_in=[
                MaterialUsage(
                    layer_role="EML",
                    component_role="emitter",
                    material_mention="Emitter-X",
                )
            ],
        )
    )
    assert class_only_complex_scope["category"] == "identity_only"
    assert class_only_complex_scope["requires_structure"] is False
    assert class_only_complex_scope["rule"] == "material_class_coordination_complex_identity_only"

    organic_scope = classify_material_structure_scope(
        PaperLocalMaterial(
            paper_material_id="M-ORG",
            entity_path="materials[2]",
            entity_label="BN-Tpl",
            mention_list=["BN-Tpl"],
            abbreviation="BN-Tpl",
            material_class="small_molecule_organic",
            used_in=[
                MaterialUsage(
                    layer_role="EML", component_role="final_emitter", material_mention="BN-Tpl"
                )
            ],
        )
    )
    assert organic_scope["category"] == "core_structure_required"
    assert organic_scope["requires_structure"] is True

    inorganic_optical_scope = classify_material_structure_scope(
        PaperLocalMaterial(
            paper_material_id="M-TIO2",
            entity_path="materials[4]",
            entity_label="TiO2",
            mention_list=["TiO2", "titanium dioxide"],
            abbreviation="TiO2",
            full_name_in_paper="titanium dioxide",
            material_class="inorganic",
            used_in=[
                MaterialUsage(
                    layer_role="other",
                    component_role="optical_capping_material",
                    material_mention="TiO2",
                )
            ],
        )
    )
    assert inorganic_optical_scope["category"] == "out_of_scope_structure"
    assert inorganic_optical_scope["requires_structure"] is False
    assert inorganic_optical_scope["requires_public_resolution"] is False
    assert inorganic_optical_scope["rule"] == "known_non_molecular_oled_material"

    generic_inorganic_scope = classify_material_structure_scope(
        PaperLocalMaterial(
            paper_material_id="M-INORG",
            entity_path="materials[5]",
            entity_label="Inorganic optical layer X",
            mention_list=["Inorganic optical layer X"],
            material_class="inorganic",
            used_in=[
                MaterialUsage(
                    layer_role="other",
                    component_role="optical_capping_material",
                    material_mention="Inorganic optical layer X",
                )
            ],
        )
    )
    assert generic_inorganic_scope["category"] == "out_of_scope_structure"
    assert generic_inorganic_scope["requires_structure"] is False
    assert generic_inorganic_scope["rule"] == "material_class_inorganic_non_molecular"


def test_material_scope_terminates_non_material_reference_without_smiles() -> None:
    scope = classify_material_structure_scope(
        PaperLocalMaterial(
            paper_material_id="M-PLACEHOLDER",
            entity_path="materials[0]",
            entity_label="host material",
            mention_list=["host material"],
            material_class="non_material_reference",
            used_in=[
                MaterialUsage(
                    layer_role="EML",
                    component_role="host",
                    material_mention="host material",
                )
            ],
        )
    )

    assert scope["category"] == "out_of_scope_structure"
    assert scope["requires_structure"] is False
    assert scope["requires_public_resolution"] is False
    assert scope["rule"] == "material_class_non_material_reference"


@pytest.mark.parametrize("material_class", ["mixture", "proprietary"])
def test_material_scope_skips_non_single_structure_material_classes(
    material_class: str,
) -> None:
    scope = classify_material_structure_scope(
        PaperLocalMaterial(
            paper_material_id="M-NONSINGLE",
            entity_path="materials[0]",
            entity_label="Commercial formulation",
            mention_list=["Commercial formulation"],
            material_class=material_class,
            used_in=[
                MaterialUsage(
                    layer_role="EML",
                    component_role="host",
                    material_mention="Commercial formulation",
                )
            ],
        )
    )

    assert scope["category"] == "identity_only"
    assert scope["requires_structure"] is False
    assert scope["rule"] == "material_class_non_single_structure_identity_only"


@pytest.mark.parametrize(
    ("name", "full_name", "expected_category"),
    [
        ("GO", "graphene oxide", "out_of_scope_structure"),
        ("perovskite", None, "out_of_scope_structure"),
        ("Au@PS NPs", None, "out_of_scope_structure"),
        ("CFx", "fluorocarbon", "identity_only"),
    ],
)
def test_material_scope_handles_known_non_molecular_rescue_targets(
    name: str,
    full_name: str | None,
    expected_category: str,
) -> None:
    scope = classify_material_structure_scope(
        PaperLocalMaterial(
            paper_material_id="M-SCOPE",
            entity_path="materials[0]",
            entity_label=name,
            mention_list=[name, *([full_name] if full_name else [])],
            abbreviation=name,
            full_name_in_paper=full_name,
            material_class="unknown",
            used_in=[
                MaterialUsage(
                    layer_role="EML",
                    component_role="host",
                    material_mention=name,
                )
            ],
        )
    )

    assert scope["category"] == expected_category
    assert scope["requires_structure"] is False


@pytest.mark.parametrize("material_name", ["α-CN-APV", "β-CN-APV"])
def test_material_scope_does_not_treat_greek_organic_name_as_aluminium_complex(
    material_name: str,
) -> None:
    scope = classify_material_structure_scope(
        PaperLocalMaterial(
            paper_material_id="M-GREEK",
            entity_path="materials[0]",
            entity_label=material_name,
            mention_list=[material_name],
            abbreviation=material_name,
            material_class="small_molecule_organic",
            used_in=[
                MaterialUsage(
                    layer_role="EML",
                    component_role="final_emitter",
                    material_mention=material_name,
                )
            ],
        )
    )

    assert scope["category"] == "core_structure_required"
    assert scope["requires_structure"] is True
    assert scope["rule"] == "default_core_resolution"


def test_material_scope_uses_polymer_class_before_metal_prefix_heuristic() -> None:
    scope = classify_material_structure_scope(
        PaperLocalMaterial(
            paper_material_id="M-PDBT",
            entity_path="materials[0]",
            entity_label="pDBT3710",
            mention_list=["pDBT3710"],
            abbreviation="pDBT3710",
            material_class="polymer",
            used_in=[
                MaterialUsage(
                    layer_role="EML",
                    component_role="emitter",
                    material_mention="pDBT3710",
                )
            ],
        )
    )

    assert scope["category"] == "identity_only"
    assert scope["requires_structure"] is False
    assert scope["rule"] == "material_class_polymer_identity_only"


def test_material_scope_treats_paper_described_oligomer_as_identity_only() -> None:
    scope = classify_material_structure_scope(
        PaperLocalMaterial(
            paper_material_id="M-TPAF3",
            entity_path="materials[0]",
            entity_label="(TPAF)3",
            mention_list=["(TPAF)3", "oligomer", "host"],
            abbreviation="(TPAF)3",
            material_class="small_molecule_organic",
            used_in=[
                MaterialUsage(
                    layer_role="EML",
                    component_role="host",
                    material_mention="(TPAF)3",
                )
            ],
        )
    )

    assert scope["category"] == "identity_only"
    assert scope["requires_structure"] is False
    assert scope["rule"] == "polymer_or_composite_identity_only"


def test_material_scope_treats_dendrimer_depiction_as_identity_only() -> None:
    scope = classify_material_structure_scope(
        PaperLocalMaterial(
            paper_material_id="M-G0",
            entity_path="materials[0]",
            entity_label="G0",
            mention_list=["G0", "zeroth generation dendrimer"],
            abbreviation="G0",
            material_class="small_molecule_organic",
            used_in=[
                MaterialUsage(
                    layer_role="EML",
                    component_role="host",
                    material_mention="G0",
                )
            ],
        )
    )

    assert scope["category"] == "identity_only"
    assert scope["requires_structure"] is False
    assert scope["rule"] == "dendrimer_identity_only"


def test_material_scope_treats_px2cz_as_known_polymer_when_llm_class_is_wrong() -> None:
    scope = classify_material_structure_scope(
        PaperLocalMaterial(
            paper_material_id="M-PX2CZ",
            entity_path="materials[0]",
            entity_label="PX2Cz",
            mention_list=["PX2Cz"],
            abbreviation="PX2Cz",
            material_class="small_molecule_organic",
            used_in=[
                MaterialUsage(
                    layer_role="HTL",
                    component_role="transport_material",
                    material_mention="PX2Cz",
                )
            ],
        )
    )

    assert scope["category"] == "identity_only"
    assert scope["requires_structure"] is False
    assert scope["rule"] == "known_identity_only_oled_auxiliary_material"


@pytest.mark.parametrize("material_name", ["PtON1-tBu", "Pt2L1", "Cu-12F", "Alq3"])
def test_material_scope_keeps_strong_formula_style_complex_aliases(
    material_name: str,
) -> None:
    scope = classify_material_structure_scope(
        PaperLocalMaterial(
            paper_material_id="M-COMPLEX",
            entity_path="materials[0]",
            entity_label=material_name,
            mention_list=[material_name],
            abbreviation=material_name,
            material_class="unknown",
            used_in=[
                MaterialUsage(
                    layer_role="EML",
                    component_role="emitter",
                    material_mention=material_name,
                )
            ],
        )
    )

    assert scope["category"] == "identity_only"
    assert scope["requires_structure"] is False
    assert scope["rule"] == "coordination_complex_identity_only"


def test_material_scope_recognizes_spelled_out_aluminum_complex() -> None:
    complex_scope = classify_material_structure_scope(
        PaperLocalMaterial(
            paper_material_id="M-BALQ",
            entity_path="materials[0]",
            entity_label="Balq",
            mention_list=[
                "Balq",
                "bis(2-methyl-8-quinolinolato-N1,O8)-(1,1-biphenyl-4-olato)aluminum",
            ],
            abbreviation="Balq",
            full_name_in_paper=(
                "bis(2-methyl-8-quinolinolato-N1,O8)-(1,1-biphenyl-4-olato)aluminum"
            ),
            material_class="small_molecule_organic",
            used_in=[
                MaterialUsage(
                    layer_role="HBL",
                    component_role="blocking_material",
                    material_mention="Balq",
                )
            ],
        )
    )
    assert complex_scope["category"] == "identity_only"
    assert complex_scope["requires_structure"] is False
    assert complex_scope["rule"] == "coordination_complex_identity_only"

    elemental_scope = classify_material_structure_scope(
        PaperLocalMaterial(
            paper_material_id="M-AL",
            entity_path="materials[1]",
            entity_label="Al",
            mention_list=["Al", "aluminum"],
            abbreviation="Al",
            full_name_in_paper="aluminum",
            material_class="inorganic",
            used_in=[
                MaterialUsage(
                    layer_role="cathode",
                    component_role="electrode_material",
                    material_mention="Al",
                )
            ],
        )
    )
    assert elemental_scope["category"] == "out_of_scope_structure"
    assert elemental_scope["requires_structure"] is False


def test_material_scope_skips_hole_blocking_layer_placeholder() -> None:
    scope = classify_material_structure_scope(
        PaperLocalMaterial(
            paper_material_id="M-HBL",
            entity_path="materials[0]",
            entity_label="HBL",
            mention_list=["HBL"],
            abbreviation="HBL",
            paper_specific_label="hole blocking layer",
            material_class="unknown",
            used_in=[
                MaterialUsage(
                    layer_role="HBL",
                    component_role="blocking_material",
                    material_mention="HBL",
                )
            ],
        )
    )

    assert scope["category"] == "out_of_scope_structure"
    assert scope["requires_structure"] is False
    assert scope["rule"] == "known_non_molecular_oled_material"


def test_material_resolution_links_confirmed_local_alias_and_tasks_unresolved(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper_and_candidate(config, mining_config_path, fake_pdf_factory, manifest_writer)
    service = MaterialResolutionService(config)
    global_material = service.create_global_material(
        canonical_name="mCBP",
        aliases=["3,3'-di(9H-carbazol-9-yl)biphenyl"],
        material_class="small_molecule_organic",
        review_status="confirmed",
        confidence=0.99,
    )

    bundle = service.resolve_paper_materials("10.1000/example")

    assert bundle is not None
    assert bundle.candidate_run_id
    assert len(bundle.materials) == 2
    links = {link.paper_material_id: link for link in bundle.links}
    assert links["M002"].global_material_id == global_material.global_material_id
    assert links["M002"].match_status == "matched_local"
    assert links["M002"].match_method == "normalized_alias"
    assert links["M001"].match_status == "unresolved"
    tasks = {task.paper_material_id: task for task in bundle.tasks}
    assert tasks["M001"].status == "pending"
    assert tasks["M001"].priority == "high"
    assert tasks["M001"].assigned_strategy == "manual_structure_required"
    assert "BN-1" in tasks["M001"].material_mentions


def test_material_resolution_does_not_auto_link_structure_candidate_synonym(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper_and_candidate(config, mining_config_path, fake_pdf_factory, manifest_writer)
    service = MaterialResolutionService(config)
    unrelated = service.create_global_material(
        canonical_name="unrelated material",
        canonical_smiles="C",
        inchi_key="UNRELATEDKEY",
        review_status="confirmed",
        confidence=0.99,
    )
    timestamp = now_iso()
    service.aliases.add(
        MaterialAlias(
            alias_id="noisy-pubchem-synonym",
            global_material_id=unrelated.global_material_id,
            alias_text="mCBP",
            normalized_alias="mcbp",
            alias_type="structure_candidate",
            source="pubchem",
            confidence=0.99,
            review_status="confirmed",
            created_at=timestamp,
            updated_at=timestamp,
        )
    )

    bundle = service.resolve_paper_materials("10.1000/example")

    assert bundle is not None
    link = next(item for item in bundle.links if item.paper_material_id == "M002")
    assert link.global_material_id is None
    assert link.match_status == "unresolved"


def test_material_resolution_reuses_confirmed_paper_local_structure_alias(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper_and_candidate(config, mining_config_path, fake_pdf_factory, manifest_writer)
    service = MaterialResolutionService(config)
    confirmed = service.create_global_material(
        canonical_name="reviewed mCBP",
        canonical_smiles="C1=CC=CC=C1",
        inchi_key="REVIEWEDKEY",
        review_status="confirmed",
        confidence=1.0,
    )
    timestamp = now_iso()
    service.aliases.add(
        MaterialAlias(
            alias_id="reviewed-paper-local-alias",
            global_material_id=confirmed.global_material_id,
            alias_text="mCBP",
            normalized_alias="mcbp",
            alias_type="structure_candidate",
            source_paper_id="10.1000/source-paper",
            source="pubchem",
            confidence=1.0,
            review_status="confirmed",
            created_at=timestamp,
            updated_at=timestamp,
        )
    )

    bundle = service.resolve_paper_materials("10.1000/example")

    assert bundle is not None
    link = next(item for item in bundle.links if item.paper_material_id == "M002")
    assert link.global_material_id == confirmed.global_material_id
    assert link.match_status == "matched_local"


def test_material_resolution_marks_conflicting_trusted_aliases_ambiguous(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    payload = _valid_result()
    material = payload["materials"][1]
    assert isinstance(material, dict)
    material["full_name_in_paper"] = "explicit paper full name"
    _seed_paper_and_candidate(
        config,
        mining_config_path,
        fake_pdf_factory,
        manifest_writer,
        payload=payload,
    )
    service = MaterialResolutionService(config)
    first = service.create_global_material(
        canonical_name="first structure",
        aliases=["explicit paper full name"],
        canonical_smiles="C",
        inchi_key="FIRSTKEY",
        review_status="confirmed",
        confidence=0.99,
    )
    second = service.create_global_material(
        canonical_name="mCBP",
        canonical_smiles="N",
        inchi_key="SECONDKEY",
        review_status="confirmed",
        confidence=0.99,
    )

    bundle = service.resolve_paper_materials("10.1000/example")

    assert bundle is not None
    link = next(item for item in bundle.links if item.paper_material_id == "M002")
    assert link.match_status == "ambiguous"
    assert set(link.evidence["candidate_global_material_ids"]) == {
        first.global_material_id,
        second.global_material_id,
    }
    task = next(item for item in bundle.tasks if item.paper_material_id == "M002")
    assert task.status == "needs_review"
    assert task.assigned_strategy == "ambiguous_local_alias"
    assert task.current_stage == "unresolved"
    assert task.next_action == "resolve"


def test_material_resolution_reopens_stale_terminal_scope_after_classifier_fix(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    payload = _valid_result()
    materials = payload["materials"]
    assert isinstance(materials, list)
    material = materials[0]
    assert isinstance(material, dict)
    material.update(
        {
            "mention_list": ["α-CN-APV"],
            "normalized_name": "α-CN-APV",
            "abbreviation": "α-CN-APV",
            "material_class": "small_molecule_organic",
        }
    )
    _seed_paper_and_candidate(
        config,
        mining_config_path,
        fake_pdf_factory,
        manifest_writer,
        payload=payload,
    )
    service = MaterialResolutionService(config)
    initial = service.resolve_paper_materials("10.1000/example")
    assert initial and initial.candidate_run_id
    initial_link = next(link for link in initial.links if link.paper_material_id == "M001")
    initial_task = next(task for task in initial.tasks if task.paper_material_id == "M001")
    timestamp = now_iso()
    PaperMaterialLinkRepository(Database(config.paths.sqlite_path)).upsert(
        initial_link.model_copy(
            update={
                "match_method": "structure_scope_rule",
                "match_confidence": 0.86,
                "match_status": "identity_only",
                "evidence": {
                    "structure_scope": {
                        "category": "identity_only",
                        "rule": "coordination_complex_identity_only",
                    }
                },
                "confirmed_at": timestamp,
            }
        )
    )
    MaterialResolutionTaskRepository(Database(config.paths.sqlite_path)).upsert(
        initial_task.model_copy(
            update={
                "status": "completed",
                "assigned_strategy": "identity_only",
                "current_stage": "completed",
                "next_action": "none",
                "completed_at": timestamp,
            }
        )
    )

    repaired = service.resolve_paper_materials("10.1000/example")

    assert repaired is not None
    repaired_link = next(link for link in repaired.links if link.paper_material_id == "M001")
    repaired_task = next(task for task in repaired.tasks if task.paper_material_id == "M001")
    assert repaired_link.match_status == "unresolved"
    assert repaired_link.match_method == "none"
    assert repaired_task.status == "pending"
    assert repaired_task.assigned_strategy == "unresolved"
    assert repaired_task.current_stage == "unresolved"
    assert repaired_task.next_action == "resolve"
    assert repaired_task.completed_at is None


def test_material_resolution_prefers_confirmed_structured_alias_over_seed_placeholder(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper_and_candidate(config, mining_config_path, fake_pdf_factory, manifest_writer)
    service = MaterialResolutionService(config)
    seed_placeholder = service.create_global_material(
        canonical_name="mCBP",
        aliases=[],
        material_class="small_molecule_organic",
        source="common_oled_seed",
        review_status="candidate",
        confidence=0.8,
    )
    confirmed_structured = service.create_global_material(
        canonical_name="mCBP",
        aliases=[],
        material_class="small_molecule_organic",
        representation_type="small_molecule",
        canonical_smiles="C1=CC=C(C=C1)N2C3=CC=CC=C3C4=CC=CC=C42",
        inchi_key="MCBPKEY",
        source="pubchem",
        review_status="confirmed",
        confidence=0.99,
    )

    bundle = service.resolve_paper_materials("10.1000/example")

    assert bundle is not None
    links = {link.paper_material_id: link for link in bundle.links}
    assert links["M002"].global_material_id == confirmed_structured.global_material_id
    assert links["M002"].global_material_id != seed_placeholder.global_material_id
    assert links["M002"].match_status == "matched_local"
    assert links["M002"].match_method == "normalized_alias_preferred_confirmed"
    tasks = {task.paper_material_id: task for task in bundle.tasks}
    assert tasks.get("M002") is None or tasks["M002"].status != "needs_review"


def test_material_name_review_updates_name_without_changing_id(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    payload = _valid_result()
    payload["materials"][0]["abbreviation"] = "Ir(ppy)"
    payload["materials"][0]["mention_list"] = ["Ir(ppy)"]
    _seed_paper_and_candidate(
        config,
        mining_config_path,
        fake_pdf_factory,
        manifest_writer,
        payload=payload,
    )

    service = MaterialResolutionService(config)
    bundle = service.review_paper_material_name(
        "10.1000/example",
        "M001",
        PaperMaterialNameReviewAction(
            reviewed_name="Ir(ppy)3",
            reviewed_abbreviation="Ir(ppy)3",
            reviewed_normalized_name="Ir(ppy)",
            actor="tester",
        ),
    )

    assert bundle is not None
    material = next(item for item in bundle.materials if item.paper_material_id == "M001")
    assert material.paper_material_id == "M001"
    assert material.abbreviation == "Ir(ppy)3"
    assert material.normalized_name == "Ir(ppy)3"
    assert material.mention_list[0] == "Ir(ppy)3"
    assert bundle.material_name_reviews[0].reviewed_name == "Ir(ppy)3"
    assert bundle.material_name_reviews[0].reviewed_normalized_name == "Ir(ppy)3"
    assert bundle.material_review_events[-1].action == "correct_material_name"

    assert bundle.candidate_run_id is not None
    material_values = [
        value
        for value in service.candidates.list_values_by_run(bundle.candidate_run_id)
        if value.concrete_path.startswith("materials[0].")
    ]
    by_path = {value.template_field_path: value for value in material_values}
    assert by_path["materials[].paper_material_id"].value_json == "M001"
    assert by_path["materials[].paper_material_id"].reviewed_value_json is None
    assert by_path["materials[].abbreviation"].reviewed_value_json == "Ir(ppy)3"
    assert by_path["materials[].normalized_name"].reviewed_value_json == "Ir(ppy)3"
    assert by_path["materials[].mention_list"].reviewed_value_json[0] == "Ir(ppy)3"


def test_material_structure_bundle_hides_unused_materials_and_historical_candidates(
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

    service = MaterialResolutionService(config)
    initial_bundle = service.get_material_structure_bundle("10.1000/example")
    assert initial_bundle is not None
    assert initial_bundle.candidate_run_id is not None
    service.resolve_paper_materials("10.1000/example")
    raw_tasks = service.tasks.list_by_run(initial_bundle.candidate_run_id)
    assert all(task.paper_material_id != "M999" for task in raw_tasks)
    timestamp = now_iso()
    service.structure_candidates.upsert(
        MaterialStructureCandidate(
            structure_candidate_id="used_candidate",
            paper_id=initial_bundle.paper_id,
            candidate_run_id=initial_bundle.candidate_run_id,
            paper_material_id="M001",
            provider="decimer_ocsr",
            resolver_name="test",
            query_text="BN-1",
            source_identifier="used-crop",
            canonical_name="BN-1",
            raw_smiles="C1=CC=CC=C1",
            canonical_smiles="c1ccccc1",
            confidence=0.9,
            created_at=timestamp,
            updated_at=timestamp,
        )
    )
    service.structure_candidates.upsert(
        MaterialStructureCandidate(
            structure_candidate_id="unused_candidate",
            paper_id=initial_bundle.paper_id,
            candidate_run_id=initial_bundle.candidate_run_id,
            paper_material_id="M999",
            provider="decimer_ocsr",
            resolver_name="test",
            query_text="BD-02",
            source_identifier="unused-crop",
            canonical_name="BD-02",
            raw_smiles="C1=CC=CC=C1",
            canonical_smiles="c1ccccc1",
            confidence=0.9,
            created_at=timestamp,
            updated_at=timestamp,
        )
    )

    bundle = service.get_material_structure_bundle("10.1000/example")

    assert bundle is not None
    material_ids = {material.paper_material_id for material in bundle.materials}
    candidate_material_ids = {
        candidate.paper_material_id for candidate in bundle.structure_candidates
    }
    assert "M001" in material_ids
    assert "M999" not in material_ids
    assert "M001" in candidate_material_ids
    assert "M999" not in candidate_material_ids


def test_material_name_agent_suggests_ir_ppy3(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    payload = _valid_result()
    payload["materials"][0]["abbreviation"] = "Ir(ppy)"
    payload["materials"][0]["mention_list"] = ["Ir(ppy)"]
    _seed_paper_and_candidate(
        config,
        mining_config_path,
        fake_pdf_factory,
        manifest_writer,
        payload=payload,
    )

    service = MaterialResolutionService(
        config,
        pubchem_client=FakeIrppyPubChemClient(),
        anysearch_client=FakeIrppyAnySearchClient(),
    )
    bundle = service.validate_material_names("10.1000/example", "M001")

    assert bundle is not None
    suggestions = [
        item for item in bundle.material_name_suggestions if item.paper_material_id == "M001"
    ]
    assert len(suggestions) == 1
    suggestion = suggestions[0]
    assert suggestion.agent_name == "material_name_agent_v2"
    assert suggestion.original_name == "Ir(ppy)"
    assert suggestion.suggested_name == "Ir(ppy)3"
    assert suggestion.suggested_normalized_name == "Ir(ppy)3"
    assert suggestion.confidence and suggestion.confidence >= 0.9
    assert suggestion.evidence["agent_version"] == "material_name_agent_v2"
    reference_evidence = suggestion.evidence["reference_evidence"]
    assert any(
        item["source_type"] == "pubchem" and item["status"] == "found"
        for item in reference_evidence
    )
    assert any(
        item["source_type"] == "web_search" and item["status"] == "found"
        for item in reference_evidence
    )


def test_material_resolution_api_and_cli(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper_and_candidate(config, mining_config_path, fake_pdf_factory, manifest_writer)
    MaterialResolutionService(config).create_global_material(
        canonical_name="mCBP",
        aliases=[],
        material_class="small_molecule_organic",
        review_status="confirmed",
    )

    client = TestClient(create_app(config=config))
    resolve_response = client.post("/api/papers/10.1000%2Fexample/resolve-materials")
    assert resolve_response.status_code == 200
    resolved = resolve_response.json()
    assert resolved["paper_id"] == "10.1000%2Fexample"
    assert len(resolved["materials"]) == 2
    assert any(link["match_status"] == "matched_local" for link in resolved["links"])
    assert any(task["paper_material_id"] == "M001" for task in resolved["tasks"])

    get_response = client.get("/api/papers/10.1000%2Fexample/material-structures")
    assert get_response.status_code == 200
    assert get_response.json()["candidate_run_id"] == resolved["candidate_run_id"]

    cli_result = CliRunner().invoke(
        app,
        [
            "resolve-paper-materials",
            "--config",
            str(mining_config_path),
            "--paper-id",
            "10.1000/example",
        ],
    )
    assert cli_result.exit_code == 0
    assert "Material Resolution" in cli_result.output
    assert "M001" in cli_result.output


def test_pubchem_response_parsers() -> None:
    property_payload = {
        "PropertyTable": {
            "Properties": [
                {
                    "CID": 123,
                    "IUPACName": "test iupac",
                    "CanonicalSMILES": "C1=CC=CC=C1",
                    "IsomericSMILES": "C1=CC=CC=C1",
                    "InChI": "InChI=1S/test",
                    "InChIKey": "TESTKEY",
                    "MolecularFormula": "C6H6",
                    "MolecularWeight": 78.11,
                }
            ]
        }
    }
    compounds = parse_pubchem_property_response(property_payload, query_text="benzene")

    assert len(compounds) == 1
    assert compounds[0].cid == "123"
    assert compounds[0].canonical_smiles == "C1=CC=CC=C1"
    assert compounds[0].molecular_weight == 78.11

    current_pubchem_payload = {
        "PropertyTable": {
            "Properties": [
                {
                    "CID": 456,
                    "ConnectivitySMILES": "C1=CC=CC=C1",
                    "SMILES": "C1=CC=CC=C1",
                    "InChIKey": "CURRENTKEY",
                }
            ]
        }
    }
    current_compounds = parse_pubchem_property_response(
        current_pubchem_payload,
        query_text="current-response",
    )
    assert current_compounds[0].canonical_smiles == "C1=CC=CC=C1"
    assert current_compounds[0].isomeric_smiles == "C1=CC=CC=C1"

    synonyms = parse_pubchem_synonyms_response(
        {
            "InformationList": {
                "Information": [{"Synonym": ["Benzene", "benzene", "Cyclohexatriene"]}]
            }
        }
    )
    assert synonyms == ["Benzene", "Cyclohexatriene"]

    search_results = parse_anysearch_response(
        {
            "data": {
                "results": [
                    {
                        "title": "BPBPA | CAS 164724-35-0",
                        "url": "https://example.test/bpbpa",
                        "content": "OLED hole transport material",
                        "score": 91.2,
                    }
                ]
            }
        }
    )
    assert search_results[0].cas_numbers() == ["164724-35-0"]

    opsin = parse_opsin_response(
        {
            "status": "SUCCESS",
            "smiles": "c1ccccc1",
            "stdinchi": "InChI=1S/C6H6",
            "stdinchikey": "UHOVQNZJYSORNB-UHFFFAOYSA-N",
        },
        query_text="benzene",
    )
    assert opsin is not None
    assert opsin.smiles == "c1ccccc1"
    assert opsin.inchi_key == "UHOVQNZJYSORNB-UHFFFAOYSA-N"


def test_material_scope_classifies_identity_only_and_out_of_scope_auxiliaries(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    payload = _result_with_scope_auxiliary_materials()
    _seed_paper_and_candidate(
        config,
        mining_config_path,
        fake_pdf_factory,
        manifest_writer,
        payload=payload,
    )

    bundle = MaterialResolutionService(config).resolve_paper_materials("10.1000/example")

    assert bundle is not None
    links = {link.paper_material_id: link for link in bundle.links}
    tasks = {task.paper_material_id: task for task in bundle.tasks}
    assert links["M003"].match_status == "out_of_scope_structure"
    assert links["M003"].evidence["structure_scope"]["rule"] == "known_non_molecular_oled_material"
    assert tasks["M003"].status == "completed"
    assert tasks["M003"].assigned_strategy == "out_of_scope_structure"
    assert tasks["M003"].material_context["structure_scope"]["requires_public_resolution"] is False
    assert links["M004"].match_status == "out_of_scope_structure"
    assert tasks["M004"].assigned_strategy == "out_of_scope_structure"
    assert links["M005"].match_status == "identity_only"
    assert tasks["M005"].status == "completed"
    assert tasks["M005"].assigned_strategy == "identity_only"

    public_client = FakePublicResolverClient()
    resolved_public = MaterialPublicResolverService(
        config,
        pubchem_client=public_client,
    ).resolve_paper_public("10.1000/example", max_queries_per_material=2)

    assert resolved_public is not None
    assert "mCBP" in public_client.queries
    assert "ITO" not in public_client.queries
    assert "Al" not in public_client.queries
    assert "Liq" not in public_client.queries


def test_public_resolver_creates_pubchem_structure_candidate(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper_and_candidate(config, mining_config_path, fake_pdf_factory, manifest_writer)
    client = FakePublicResolverClient()
    service = MaterialPublicResolverService(config, pubchem_client=client)

    bundle = service.resolve_paper_public("10.1000/example", max_queries_per_material=2)

    assert bundle is not None
    candidates = {
        candidate.paper_material_id: candidate for candidate in bundle.structure_candidates
    }
    assert "M002" in candidates
    assert candidates["M002"].provider == "pubchem"
    assert candidates["M002"].source_identifier == "999"
    assert candidates["M002"].canonical_smiles == "C1=CC=C(C=C1)N2C3=CC=CC=C3C4=CC=CC=C42"
    assert candidates["M002"].status == "pending_review"
    tasks = {task.paper_material_id: task for task in bundle.tasks}
    assert tasks["M002"].status == "needs_review"
    assert tasks["M002"].assigned_strategy == "public_database_review"
    assert "BN-1" not in client.queries


def test_public_resolver_ignores_direct_pubchem_candidate_without_name_match(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper_and_candidate(config, mining_config_path, fake_pdf_factory, manifest_writer)
    service = MaterialPublicResolverService(
        config,
        pubchem_client=FakeUnrelatedPublicResolverClient(),
    )

    bundle = service.resolve_paper_public(
        "10.1000/example",
        paper_material_id="M002",
        max_queries_per_material=1,
    )

    assert bundle is not None
    candidates = [item for item in bundle.structure_candidates if item.paper_material_id == "M002"]
    assert candidates == []
    task = next(item for item in bundle.tasks if item.paper_material_id == "M002")
    assert task.assigned_strategy == "public_database_not_found"
    assert task.next_action == "run_visual_ocsr"
    assert "no lexical identity match" in task.material_context["resolver_warnings"][0]


def test_material_structure_candidate_accept_and_undo(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper_and_candidate(config, mining_config_path, fake_pdf_factory, manifest_writer)
    resolver = MaterialPublicResolverService(
        config,
        pubchem_client=FakePublicResolverClient(),
    )
    bundle = resolver.resolve_paper_public("10.1000/example", max_queries_per_material=2)
    assert bundle is not None
    candidate = next(
        item for item in bundle.structure_candidates if item.paper_material_id == "M002"
    )

    review_service = MaterialStructureReviewService(config)
    timestamp = now_iso()
    existing_global = review_service.global_materials.upsert(
        MaterialGlobal(
            global_material_id="global-old-mcbp",
            canonical_name="mCBP old structure",
            material_class="small_molecule_organic",
            representation_type="small_molecule",
            raw_smiles="C",
            canonical_smiles="C",
            isomeric_smiles="C",
            inchi=None,
            inchi_key=candidate.inchi_key,
            formula="CH4",
            molecular_weight=16.04,
            source="legacy",
            source_detail={"note": "old local library record"},
            confidence=0.1,
            review_status="confirmed",
            created_at=timestamp,
            updated_at=timestamp,
            confirmed_at=timestamp,
        )
    )
    accepted = review_service.accept_structure_candidate(
        candidate.structure_candidate_id,
        MaterialReviewAction(actor="tester", message="verified mCBP"),
    )

    assert accepted is not None
    accepted_candidate = next(
        item
        for item in accepted.structure_candidates
        if item.structure_candidate_id == candidate.structure_candidate_id
    )
    assert accepted_candidate.status == "accepted"
    links = {link.paper_material_id: link for link in accepted.links}
    assert links["M002"].match_status == "matched_candidate"
    assert links["M002"].global_material_id == existing_global.global_material_id
    accepted_global = next(
        item
        for item in accepted.global_materials
        if item.global_material_id == existing_global.global_material_id
    )
    assert accepted_global.review_status == "confirmed"
    assert accepted_global.canonical_name == "mCBP old structure"
    assert accepted_global.inchi_key == "MCBPKEY"
    assert accepted_global.canonical_smiles == candidate.canonical_smiles
    assert accepted_global.formula == candidate.formula
    assert accepted_global.source == candidate.provider
    assert (
        accepted_global.source_detail["last_accepted_structure_candidate"]["structure_candidate_id"]
        == candidate.structure_candidate_id
    )
    assert accepted.material_review_events[-1].action == "accept"
    paper_local_aliases = [
        alias
        for alias in review_service.aliases.find_by_normalized("mcbp")
        if alias.global_material_id == links["M002"].global_material_id
        and alias.source_paper_id == accepted.paper_id
    ]
    assert any(alias.alias_type.startswith("paper_local_") for alias in paper_local_aliases)

    refreshed = resolver.resolve_paper_public(
        "10.1000/example",
        paper_material_id="M002",
        max_queries_per_material=2,
    )
    assert refreshed is not None
    refreshed_candidate = next(
        item
        for item in refreshed.structure_candidates
        if item.structure_candidate_id == candidate.structure_candidate_id
    )
    refreshed_task = next(item for item in refreshed.tasks if item.paper_material_id == "M002")
    assert refreshed_candidate.status == "accepted"
    assert refreshed_task.status == "completed"

    local_resolved = MaterialResolutionService(config).resolve_paper_materials("10.1000/example")
    assert local_resolved is not None
    local_links = {link.paper_material_id: link for link in local_resolved.links}
    assert local_links["M002"].match_status == "matched_candidate"
    assert local_links["M002"].global_material_id == links["M002"].global_material_id

    MaterialResolutionService(config).links.delete_by_paper_material(
        accepted.candidate_run_id or "",
        "M002",
    )
    restored = MaterialResolutionService(config).resolve_paper_materials("10.1000/example")
    assert restored is not None
    restored_links = {link.paper_material_id: link for link in restored.links}
    assert restored_links["M002"].match_status == "matched_candidate"
    assert restored_links["M002"].global_material_id == links["M002"].global_material_id
    restored_tasks = {task.paper_material_id: task for task in restored.tasks}
    assert restored_tasks["M002"].status == "completed"

    undone = review_service.undo_material_review_event(
        accepted.material_review_events[-1].event_id,
        MaterialReviewAction(actor="tester", message="undo test"),
    )

    assert undone is not None
    undone_candidate = next(
        item
        for item in undone.structure_candidates
        if item.structure_candidate_id == candidate.structure_candidate_id
    )
    assert undone_candidate.status == "pending_review"
    undone_links = {link.paper_material_id: link for link in undone.links}
    assert undone_links["M002"].match_status == "unresolved"
    assert undone.material_review_events[-1].action == "undo"


def test_verified_existing_global_link_and_undo(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper_and_candidate(config, mining_config_path, fake_pdf_factory, manifest_writer)
    resolver = MaterialResolutionService(config)
    initial = resolver.resolve_paper_materials("10.1000/example")
    assert initial is not None
    initial_link = next(item for item in initial.links if item.paper_material_id == "M002")
    initial_task = next(item for item in initial.tasks if item.paper_material_id == "M002")
    assert initial_link.match_status == "unresolved"

    timestamp = now_iso()
    review_service = MaterialStructureReviewService(config)
    global_material = review_service.global_materials.upsert(
        MaterialGlobal(
            global_material_id="verified-global-mcbp",
            canonical_name="1,3-bis(9H-carbazol-9-yl)benzene",
            material_class="small_molecule_organic",
            representation_type="small_molecule",
            raw_smiles="c1ccc2c(c1)c1ccccc1n2-c1cccc(-n2c3ccccc3c3ccccc32)c1",
            canonical_smiles=(
                "c1ccc2c(c1)c1ccccc1n2-c1cccc(-n2c3ccccc3c3ccccc32)c1"
            ),
            isomeric_smiles=None,
            inchi=None,
            inchi_key="MZYDBGLUVPLRKR-UHFFFAOYSA-N",
            formula="C30H20N2",
            molecular_weight=408.5,
            source="pubchem",
            confidence=1.0,
            review_status="confirmed",
            created_at=timestamp,
            updated_at=timestamp,
            confirmed_at=timestamp,
        )
    )

    linked = review_service.link_existing_global_material(
        "10.1000/example",
        "M002",
        MaterialReviewAction(
            actor="chemical_agent",
            message="Exact paper full-name and PubChem InChIKey match.",
            global_material_id=global_material.global_material_id,
            evidence={
                "confidence": 0.99,
                "source_url": "https://pubchem.ncbi.nlm.nih.gov/compound/22020377",
                "matched_inchi_key": "MZYDBGLUVPLRKR-UHFFFAOYSA-N",
            },
        ),
    )

    assert linked is not None
    link = next(item for item in linked.links if item.paper_material_id == "M002")
    assert link.global_material_id == global_material.global_material_id
    assert link.match_method == "chemical_agent_verified_global"
    assert link.match_status == "matched_candidate"
    assert link.evidence["verification"]["matched_inchi_key"] == global_material.inchi_key
    task = next(item for item in linked.tasks if item.paper_material_id == "M002")
    assert task.status == "completed"
    assert task.assigned_strategy == "chemical_agent_verified_global"
    event = linked.material_review_events[-1]
    assert event.action == "link_existing_global"
    assert event.global_material_id == global_material.global_material_id

    conflicting_global = review_service.global_materials.upsert(
        global_material.model_copy(
            update={
                "global_material_id": "conflicting-global-mcbp",
                "canonical_name": "Different confirmed material",
                "raw_smiles": "C",
                "canonical_smiles": "C",
                "inchi_key": "VNWKTOKETHGBQD-UHFFFAOYSA-N",
                "formula": "CH4",
                "molecular_weight": 16.04,
            }
        )
    )
    with pytest.raises(ValueError, match="already linked to a different confirmed structure"):
        review_service.link_existing_global_material(
            "10.1000/example",
            "M002",
            MaterialReviewAction(
                actor="chemical_agent",
                global_material_id=conflicting_global.global_material_id,
            ),
        )

    undone = review_service.undo_material_review_event(
        event.event_id,
        MaterialReviewAction(actor="chemical_agent", message="Undo verified link."),
    )

    assert undone is not None
    restored_link = next(item for item in undone.links if item.paper_material_id == "M002")
    assert restored_link.match_status == "unresolved"
    restored_task = next(item for item in undone.tasks if item.paper_material_id == "M002")
    restored_payload = restored_task.model_dump(mode="json", exclude={"updated_at"})
    initial_payload = initial_task.model_dump(mode="json", exclude={"updated_at"})
    assert restored_payload == initial_payload
    assert undone.material_review_events[-1].action == "undo"


def test_web_search_cas_fallback_creates_pubchem_candidate_for_review(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper_and_candidate(config, mining_config_path, fake_pdf_factory, manifest_writer)
    service = MaterialPublicResolverService(
        config,
        pubchem_client=FakeIdentifierPubChemClient(),
        anysearch_client=FakeAnySearchClient(),
    )

    bundle = service.resolve_paper_public(
        "10.1000/example",
        paper_material_id="M002",
        max_queries_per_material=1,
    )

    assert bundle is not None
    candidate = next(
        item for item in bundle.structure_candidates if item.paper_material_id == "M002"
    )
    assert candidate.provider == "pubchem"
    assert candidate.resolver_name == "anysearch_to_pubchem"
    assert candidate.status == "pending_review"
    assert candidate.evidence["discovered_identifier"] == "164724-35-0"
    task = next(item for item in bundle.tasks if item.paper_material_id == "M002")
    assert task.assigned_strategy == "web_search_to_pubchem_review"
    assert task.material_context["web_search_results"][0]["cas_numbers"] == ["164724-35-0"]


def test_opsin_systematic_name_fallback_creates_review_candidate(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    payload = _valid_result()
    payload["materials"][1]["full_name_in_paper"] = "benzene"
    payload["materials"][1]["normalized_name"] = None
    _seed_paper_and_candidate(
        config,
        mining_config_path,
        fake_pdf_factory,
        manifest_writer,
        payload=payload,
    )
    service = MaterialPublicResolverService(
        config,
        pubchem_client=FakeEmptyPublicResolverClient(),
        opsin_client=FakeOpsinClient(),
    )

    bundle = service.resolve_paper_public(
        "10.1000/example",
        paper_material_id="M002",
        max_queries_per_material=2,
    )

    assert bundle is not None
    candidate = next(
        item for item in bundle.structure_candidates if item.paper_material_id == "M002"
    )
    assert candidate.provider == "opsin"
    assert candidate.resolver_name == "opsin_systematic_name"
    assert candidate.canonical_smiles == "c1ccccc1"
    assert candidate.inchi_key == "UHOVQNZJYSORNB-UHFFFAOYSA-N"
    assert candidate.status == "pending_review"
    task = next(item for item in bundle.tasks if item.paper_material_id == "M002")
    assert task.assigned_strategy == "opsin_name_review"


def test_explicit_full_name_opsin_candidate_is_not_blocked_by_abbreviation_hit(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    payload = _valid_result()
    payload["materials"][1]["full_name_in_paper"] = "benzene"
    payload["materials"][1]["normalized_name"] = None
    _seed_paper_and_candidate(
        config,
        mining_config_path,
        fake_pdf_factory,
        manifest_writer,
        payload=payload,
    )
    service = MaterialPublicResolverService(
        config,
        pubchem_client=FakePublicResolverClient(),
        opsin_client=FakeOpsinClient(),
    )

    bundle = service.resolve_paper_public(
        "10.1000/example",
        paper_material_id="M002",
        max_queries_per_material=2,
    )

    assert bundle is not None
    candidates = [item for item in bundle.structure_candidates if item.paper_material_id == "M002"]
    assert {candidate.provider for candidate in candidates} == {"pubchem", "opsin"}
    pubchem_candidate = next(item for item in candidates if item.provider == "pubchem")
    opsin_candidate = next(item for item in candidates if item.provider == "opsin")
    assert pubchem_candidate.query_type == "abbreviation"
    assert pubchem_candidate.confidence == 0.72
    assert opsin_candidate.query_type == "full_name"
    assert opsin_candidate.inchi_key == "UHOVQNZJYSORNB-UHFFFAOYSA-N"
    task = next(item for item in bundle.tasks if item.paper_material_id == "M002")
    assert task.assigned_strategy == "opsin_name_review"
    assert (
        opsin_candidate.structure_candidate_id in task.material_context["structure_candidate_ids"]
    )


def test_abbreviation_pubchem_hit_does_not_block_web_identity_support(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    payload = _valid_result()
    payload["materials"][1]["normalized_name"] = None
    _seed_paper_and_candidate(
        config,
        mining_config_path,
        fake_pdf_factory,
        manifest_writer,
        payload=payload,
    )
    service = MaterialPublicResolverService(
        config,
        pubchem_client=FakeAmbiguousAbbreviationPubChemClient(),
        opsin_client=FakeOpsinClient(),
        anysearch_client=FakeAnySearchClient(),
    )

    bundle = service.resolve_paper_public(
        "10.1000/example",
        paper_material_id="M002",
        max_queries_per_material=1,
    )

    assert bundle is not None
    candidates = [item for item in bundle.structure_candidates if item.paper_material_id == "M002"]
    assert {candidate.resolver_name for candidate in candidates} == {
        "pubchem_name",
        "anysearch_to_pubchem",
    }
    task = next(item for item in bundle.tasks if item.paper_material_id == "M002")
    assert task.assigned_strategy == "web_search_to_pubchem_review"
    assert task.material_context["web_search_results"]


def test_web_search_tries_targeted_cas_smiles_query_variant(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper_and_candidate(config, mining_config_path, fake_pdf_factory, manifest_writer)
    anysearch = FakeTargetedAnySearchClient()
    service = MaterialPublicResolverService(
        config,
        pubchem_client=FakeIdentifierPubChemClient(),
        opsin_client=FakeOpsinClient(),
        anysearch_client=anysearch,
    )

    bundle = service.resolve_paper_public(
        "10.1000/example",
        paper_material_id="M002",
        max_queries_per_material=1,
    )

    assert bundle is not None
    assert any("CAS SMILES" in query for query in anysearch.queries)
    candidate = next(
        item for item in bundle.structure_candidates if item.paper_material_id == "M002"
    )
    assert candidate.resolver_name == "anysearch_to_pubchem"


def test_manual_structure_input_creates_accepted_candidate_and_link(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper_and_candidate(config, mining_config_path, fake_pdf_factory, manifest_writer)
    initial = MaterialResolutionService(config).resolve_paper_materials("10.1000/example")
    assert initial is not None
    initial_task = next(item for item in initial.tasks if item.paper_material_id == "M001")
    assert initial_task.assigned_strategy == "manual_structure_required"

    bundle = MaterialStructureReviewService(config).save_manual_structure(
        "10.1000/example",
        "M001",
        MaterialManualStructureAction(
            actor="tester",
            reviewed_name="BN-1",
            smiles="c1ccccc1",
            source_note="Manually entered from Scheme 1",
        ),
    )

    assert bundle is not None
    candidate = next(
        item for item in bundle.structure_candidates if item.paper_material_id == "M001"
    )
    assert candidate.provider == "manual_input"
    assert candidate.resolver_name == "human_manual_structure"
    assert candidate.status == "accepted"
    assert candidate.canonical_smiles == "c1ccccc1"
    links = {link.paper_material_id: link for link in bundle.links}
    assert links["M001"].match_status == "matched_candidate"
    assert links["M001"].match_method == "manual_input_candidate"
    task = next(item for item in bundle.tasks if item.paper_material_id == "M001")
    assert task.status == "completed"
    assert task.assigned_strategy == "manual_structure_input"
    assert task.current_stage == "completed"
    assert task.next_action == "none"
    assert bundle.material_review_events[-1].action == "accept"
    assert "Manual structure input" in (bundle.material_review_events[-1].message or "")
    aliases = MaterialStructureReviewService(config).aliases.find_by_normalized("bn1")
    assert any(alias.source == "manual_input" for alias in aliases)


def test_literature_verified_import_is_auditable_and_idempotent(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper_and_candidate(config, mining_config_path, fake_pdf_factory, manifest_writer)
    record = LiteratureVerifiedStructureRecord(
        paper_id="10.1000%2Fexample",
        doi="10.1000/example",
        doi_url="https://doi.org/10.1000/example",
        paper_material_id="M001",
        primary_material_name="BN-1",
        resolved_full_name="Literature verified BN-1",
        material_class="small_molecule_organic",
        smiles="c1ccccc1",
        evidence_level="A",
        evidence_status="Cross-checked against the paper and an independent source.",
        source_references=[
            "https://doi.org/10.1000/example",
            "https://example.test/structure",
        ],
        reported_formula="C6H6",
        reported_molecular_weight=78.114,
        rdkit_validation="PASS",
        workbook_path="/tmp/verified.xlsx",
        workbook_sha256="abc123",
        main_sheet_row=2,
        evidence_sheet_row=2,
    )
    service = LiteratureVerifiedImportService(config)

    dry_run = service.run([record])
    applied = service.run([record], apply=True, actor="test_verified_import")
    repeated = service.run([record])

    assert dry_run.counts == {"import": 1}
    assert applied.counts == {"imported": 1}
    assert applied.decisions[0].imported_candidate_id
    assert applied.decisions[0].global_material_id
    assert repeated.counts == {"skip_already_resolved_same": 1}
    bundle = MaterialResolutionService(config).get_material_structure_bundle("10.1000/example")
    assert bundle is not None
    candidate = next(
        item
        for item in bundle.structure_candidates
        if item.structure_candidate_id == applied.decisions[0].imported_candidate_id
    )
    assert candidate.provider == "manual_input"
    assert candidate.status == "accepted"
    assert '"import_type": "literature_verified_excel"' in candidate.evidence["source_note"]
    task = next(item for item in bundle.tasks if item.paper_material_id == "M001")
    assert task.status == "completed"
    assert task.current_stage == "completed"
    assert task.next_action == "none"
    assert bundle.material_review_events[-1].actor == "test_verified_import"


def test_literature_verified_import_accepts_user_supplied_disconnected_structure(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper_and_candidate(config, mining_config_path, fake_pdf_factory, manifest_writer)
    record = LiteratureVerifiedStructureRecord(
        paper_id="10.1000%2Fexample",
        doi="10.1000/example",
        paper_material_id="M001",
        primary_material_name="Complex",
        material_class="coordination_complex",
        smiles="[Al+3].[O-]c1ccccc1.[O-]c1ccccc1.[O-]c1ccccc1",
        evidence_level="main_sheet",
        evidence_status="SMILES supplied in the first workbook sheet by the user.",
        workbook_path="/tmp/verified.xlsx",
        workbook_sha256="abc123",
        main_sheet_row=2,
        evidence_sheet_row=0,
    )

    report = LiteratureVerifiedImportService(config).run([record], apply=True)

    assert report.counts == {"imported": 1}
    bundle = MaterialResolutionService(config).get_material_structure_bundle("10.1000/example")
    assert bundle is not None
    candidate = next(
        item
        for item in bundle.structure_candidates
        if item.structure_candidate_id == report.decisions[0].imported_candidate_id
    )
    assert candidate.status == "accepted"
    assert candidate.representation_type == "multi_component"


def test_public_resolver_does_not_requeue_rejected_candidate(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper_and_candidate(config, mining_config_path, fake_pdf_factory, manifest_writer)
    resolver = MaterialPublicResolverService(config, pubchem_client=FakePublicResolverClient())
    bundle = resolver.resolve_paper_public("10.1000/example", max_queries_per_material=2)
    assert bundle is not None
    candidate = next(
        item for item in bundle.structure_candidates if item.paper_material_id == "M002"
    )

    rejected = MaterialStructureReviewService(config).reject_structure_candidate(
        candidate.structure_candidate_id,
        MaterialReviewAction(actor="tester", message="wrong public structure"),
    )
    assert rejected is not None
    refreshed = resolver.resolve_paper_public(
        "10.1000/example",
        paper_material_id="M002",
        max_queries_per_material=2,
    )
    assert refreshed is not None
    refreshed_candidate = next(
        item
        for item in refreshed.structure_candidates
        if item.structure_candidate_id == candidate.structure_candidate_id
    )
    refreshed_task = next(item for item in refreshed.tasks if item.paper_material_id == "M002")
    assert refreshed_candidate.status == "rejected"
    assert refreshed_task.status == "pending"
    assert refreshed_task.assigned_strategy == "public_candidate_rejected_continue_resolution"
    assert refreshed_task.current_stage == "visual_ocsr_pending"
    assert refreshed_task.next_action == "run_visual_ocsr"


def test_material_structure_candidate_review_api(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper_and_candidate(config, mining_config_path, fake_pdf_factory, manifest_writer)
    bundle = MaterialPublicResolverService(
        config,
        pubchem_client=FakePublicResolverClient(),
    ).resolve_paper_public("10.1000/example", max_queries_per_material=2)
    assert bundle is not None
    candidate = next(
        item for item in bundle.structure_candidates if item.paper_material_id == "M002"
    )

    client = TestClient(create_app(config=config))
    accept_response = client.post(
        f"/api/material-structure-candidates/{candidate.structure_candidate_id}/accept",
        json={"actor": "tester", "message": "accepted by API"},
    )
    assert accept_response.status_code == 200
    accepted = accept_response.json()
    assert any(item["status"] == "accepted" for item in accepted["structure_candidates"])
    event_id = accepted["material_review_events"][-1]["event_id"]

    undo_response = client.post(
        f"/api/material-review-events/{event_id}/undo",
        json={"actor": "tester", "message": "undo by API"},
    )
    assert undo_response.status_code == 200
    undone = undo_response.json()
    assert undone["material_review_events"][-1]["action"] == "undo"


def test_manual_structure_input_api(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper_and_candidate(config, mining_config_path, fake_pdf_factory, manifest_writer)
    client = TestClient(create_app(config=config))

    response = client.post(
        "/api/papers/10.1000%2Fexample/manual-material-structure?paper_material_id=M001",
        json={
            "actor": "tester",
            "reviewed_name": "BN-1",
            "smiles": "c1ccccc1",
            "source_note": "Manual API entry",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    candidate = next(
        item for item in payload["structure_candidates"] if item["paper_material_id"] == "M001"
    )
    assert candidate["provider"] == "manual_input"
    assert candidate["status"] == "accepted"
    link = next(item for item in payload["links"] if item["paper_material_id"] == "M001")
    assert link["match_status"] == "matched_candidate"


def test_material_auto_decision_accepts_high_confidence_public_candidate(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper_and_candidate(config, mining_config_path, fake_pdf_factory, manifest_writer)
    bundle = MaterialPublicResolverService(
        config,
        pubchem_client=FakePublicResolverClient(),
    ).resolve_paper_public("10.1000/example", max_queries_per_material=2)
    assert bundle is not None
    candidate = next(
        item for item in bundle.structure_candidates if item.paper_material_id == "M002"
    )

    MaterialIdentityJudgeService(
        config,
        llm_client=StaticJSONLLMClient(
            {
                "verdict": "exact_match",
                "confidence": 0.94,
                "supporting_evidence": ["The candidate matches the paper material alias."],
                "conflicts": [],
                "recommended_action": "ready_for_human_accept",
            }
        ),
    ).judge_paper_candidates("10.1000/example", paper_material_id="M002")

    result = MaterialAutoDecisionService(config).apply_paper_auto_decisions("10.1000/example")

    assert result is not None
    assert result.accepted_count == 1
    assert result.rejected_count == 0
    accepted = MaterialStructureReviewService(config).structure_candidates.get(
        candidate.structure_candidate_id
    )
    assert accepted is not None
    assert accepted.status == "accepted"
    bundle_after = MaterialResolutionService(config).get_material_structure_bundle(
        "10.1000/example"
    )
    assert bundle_after is not None
    event = bundle_after.material_review_events[-1]
    assert event.action == "accept"
    assert event.actor == "automation_policy"
    assert "Auto accepted" in (event.message or "")


def test_material_web_rescue_accepts_verified_candidate_through_audit_path(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper_and_candidate(config, mining_config_path, fake_pdf_factory, manifest_writer)
    service = MaterialWebRescueService(config)
    bundle = service.material_resolution.get_material_structure_bundle("10.1000/example")
    assert bundle is not None
    assert bundle.candidate_run_id is not None

    result = service.apply_decision(
        MaterialWebRescueDecision(
            paper_id="10.1000/example",
            candidate_run_id=bundle.candidate_run_id,
            paper_material_id="M001",
            expected_mentions=["BN-1"],
            action="accept",
            reviewed_name="BN-1",
            canonical_name="Verified BN-1 test material",
            smiles="c1ccccc1",
            identity_verdict="exact_match",
            confidence=0.98,
            structure_method="database_record",
            sources=[
                MaterialWebRescueSource(
                    title="Paper defining BN-1",
                    url="https://doi.org/10.1000/example",
                    source_type="article",
                    roles=["paper_identity"],
                    evidence="The paper explicitly labels the device material BN-1.",
                ),
                MaterialWebRescueSource(
                    title="Independent structured record",
                    url="https://pubchem.ncbi.nlm.nih.gov/compound/241",
                    source_type="pubchem",
                    roles=["independent_identity", "structure"],
                    identifier="CID 241",
                    evidence="The structured record links the verified name to this SMILES.",
                ),
            ],
            notes="Synthetic test of the audited web-rescue path.",
        ),
        actor="test_web_rescue",
        research_run_id="test-run",
    )

    assert result.status == "accepted"
    assert result.structure_candidate_id is not None
    candidate = service.structure_candidates.get(result.structure_candidate_id)
    assert candidate is not None
    assert candidate.provider == "web_rescue_agent"
    assert candidate.status == "accepted"
    judgment = service.identity_judgments.latest_by_candidate(result.structure_candidate_id)
    assert judgment is not None
    assert judgment.verdict == "exact_match"
    refreshed = service.material_resolution.get_material_structure_bundle("10.1000/example")
    assert refreshed is not None
    link = next(item for item in refreshed.links if item.paper_material_id == "M001")
    assert link.match_status == "matched_candidate"
    assert refreshed.material_review_events[-1].actor == "test_web_rescue"


def test_material_web_rescue_does_not_auto_accept_uncorroborated_decimer_output() -> None:
    decision = MaterialWebRescueDecision(
        paper_id="10.1000/example",
        candidate_run_id="run-1",
        paper_material_id="M001",
        expected_mentions=["Molecule A"],
        action="accept",
        reviewed_name="Molecule A",
        smiles="c1ccccc1",
        identity_verdict="exact_match",
        confidence=0.99,
        structure_method="decimer_ocsr",
        sources=[
            MaterialWebRescueSource(
                title="Article",
                url="https://doi.org/10.1000/example",
                source_type="article",
                roles=["paper_identity"],
                evidence="The article binds Molecule A to a displayed structure.",
            ),
            MaterialWebRescueSource(
                title="Article structure image",
                url="https://publisher.example/figure-1.png",
                source_type="supporting_information",
                roles=["image", "structure"],
                evidence="DECIMER produced the proposed SMILES from the labeled image.",
            ),
        ],
    )

    errors = MaterialWebRescueService._acceptance_errors(decision)

    assert "DECIMER output requires an independent structured-source match" in errors


def test_material_web_rescue_accepts_jglobal_as_strong_structure_source() -> None:
    decision = MaterialWebRescueDecision(
        paper_id="10.1000%2Fexample",
        candidate_run_id="run-1",
        paper_material_id="M001",
        expected_mentions=["Example"],
        action="accept",
        reviewed_name="Example",
        smiles="c1ccccc1",
        identity_verdict="exact_match",
        confidence=0.99,
        structure_method="database_record",
        sources=[
            MaterialWebRescueSource(
                title="Target article",
                url="https://doi.org/10.1000/example",
                source_type="article",
                roles=["paper_identity"],
                evidence="The article binds the material label to its full name.",
            ),
            MaterialWebRescueSource(
                title="J-GLOBAL substance record",
                url="https://jglobal.jst.go.jp/en/detail?JGLOBAL_ID=example",
                source_type="jglobal",
                roles=["independent_identity", "structure"],
                evidence="The decided-structure record supplies the structure and identifier.",
            ),
        ],
    )

    assert MaterialWebRescueService._acceptance_errors(decision) == []


def test_material_web_rescue_cannot_reaccept_identical_rejected_candidate(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper_and_candidate(config, mining_config_path, fake_pdf_factory, manifest_writer)
    service = MaterialWebRescueService(config)
    bundle = service.material_resolution.get_material_structure_bundle("10.1000/example")
    assert bundle is not None
    assert bundle.candidate_run_id is not None
    decision = MaterialWebRescueDecision(
        paper_id="10.1000/example",
        candidate_run_id=bundle.candidate_run_id,
        paper_material_id="M001",
        expected_mentions=["BN-1"],
        action="accept",
        reviewed_name="BN-1",
        smiles="c1ccccc1",
        identity_verdict="exact_match",
        confidence=0.99,
        structure_method="database_record",
        sources=[
            MaterialWebRescueSource(
                title="Target article",
                url="https://doi.org/10.1000/example",
                source_type="article",
                roles=["paper_identity"],
                evidence="The article identifies BN-1.",
            ),
            MaterialWebRescueSource(
                title="Structured record",
                url="https://pubchem.ncbi.nlm.nih.gov/compound/241",
                source_type="pubchem",
                roles=["independent_identity", "structure"],
                evidence="The structured record supplies the verified graph.",
            ),
        ],
    )
    accepted = service.apply_decision(
        decision,
        actor="test_web_rescue",
        research_run_id="test-reject-guard",
    )
    assert accepted.structure_candidate_id is not None
    accepted_bundle = service.material_resolution.get_material_structure_bundle("10.1000/example")
    assert accepted_bundle is not None
    accept_event = next(
        event
        for event in reversed(accepted_bundle.material_review_events)
        if event.structure_candidate_id == accepted.structure_candidate_id
        and event.action == "accept"
    )
    service.review_service.undo_material_review_event(
        accept_event.event_id,
        MaterialReviewAction(actor="test_web_rescue", message="Undo semantic mismatch."),
    )
    service.review_service.reject_structure_candidate(
        accepted.structure_candidate_id,
        MaterialReviewAction(actor="test_web_rescue", message="Semantic mismatch."),
    )

    with pytest.raises(ValueError, match="previously rejected"):
        service.apply_decision(
            decision,
            actor="test_web_rescue",
            research_run_id="test-reject-guard-retry",
            dry_run=True,
        )


def test_material_web_rescue_report_verifies_accepted_audit_chain(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
    tmp_path: Path,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper_and_candidate(config, mining_config_path, fake_pdf_factory, manifest_writer)
    rescue = MaterialWebRescueService(config)
    bundle = rescue.material_resolution.get_material_structure_bundle("10.1000/example")
    assert bundle is not None
    assert bundle.candidate_run_id is not None
    rescue.apply_decision(
        MaterialWebRescueDecision(
            paper_id="10.1000/example",
            candidate_run_id=bundle.candidate_run_id,
            paper_material_id="M001",
            expected_mentions=["BN-1"],
            action="accept",
            reviewed_name="BN-1",
            canonical_name="Verified BN-1 test material",
            smiles="c1ccccc1",
            identity_verdict="exact_match",
            confidence=0.98,
            structure_method="database_record",
            sources=[
                MaterialWebRescueSource(
                    title="Paper defining BN-1",
                    url="https://doi.org/10.1000/example",
                    source_type="article",
                    roles=["paper_identity"],
                    evidence="The paper explicitly labels the device material BN-1.",
                ),
                MaterialWebRescueSource(
                    title="Independent structured record",
                    url="https://pubchem.ncbi.nlm.nih.gov/compound/241",
                    source_type="pubchem",
                    roles=["independent_identity", "structure"],
                    evidence="The structured record links the verified name to this SMILES.",
                ),
            ],
        ),
        actor="custom_high_intelligence_rescue_agent",
        research_run_id="test-report-run",
    )
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "run_id": "test-report-run",
                "items": [
                    {
                        "target_id": "target-1",
                        "paper_id": "10.1000%2Fexample",
                        "doi": "10.1000/example",
                        "title": "Example",
                        "journal": "Test",
                        "candidate_run_id": bundle.candidate_run_id,
                        "paper_material_id": "M001",
                        "material_mentions": ["BN-1"],
                        "material_context": {},
                        "identity_judgments": [],
                        "visual_counts": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = MaterialWebRescueReportService(config).generate(
        inventory,
        tmp_path / "report",
    )

    assert report["summary"] == {"accepted_web_rescue": 1}
    assert report["audit_error_count"] == 0
    assert (tmp_path / "report" / "material-web-rescue-report.json").exists()
    assert (tmp_path / "report" / "material-web-rescue-report.md").exists()


def test_material_auto_decision_defers_bundle_refresh_and_completion(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
    monkeypatch,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper_and_candidate(config, mining_config_path, fake_pdf_factory, manifest_writer)
    bundle = MaterialPublicResolverService(
        config,
        pubchem_client=FakePublicResolverClient(),
    ).resolve_paper_public("10.1000/example", max_queries_per_material=2)
    assert bundle is not None
    MaterialIdentityJudgeService(
        config,
        llm_client=StaticJSONLLMClient(
            {
                "verdict": "exact_match",
                "confidence": 0.95,
                "supporting_evidence": ["The candidate matches the paper material alias."],
                "conflicts": [],
                "recommended_action": "ready_for_human_accept",
            }
        ),
    ).judge_paper_candidates("10.1000/example", paper_material_id="M002")
    service = MaterialAutoDecisionService(config)
    accept_calls: list[dict[str, object]] = []
    completion_calls: list[str] = []

    def record_accept(_candidate_id, _action, **kwargs):
        accept_calls.append(kwargs)
        return None

    monkeypatch.setattr(service.review_service, "accept_structure_candidate", record_accept)
    monkeypatch.setattr(
        service.review_service.material_completion,
        "confirm_paper_if_materials_complete",
        lambda paper_id: completion_calls.append(paper_id) or True,
    )

    result = service.apply_paper_auto_decisions("10.1000/example")

    assert result is not None
    assert result.accepted_count == 1
    assert len(accept_calls) == 1
    assert accept_calls[0]["defer_completion"] is True
    assert accept_calls[0]["return_bundle"] is False
    assert isinstance(accept_calls[0]["paper_material"], PaperLocalMaterial)
    assert completion_calls == ["10.1000%2Fexample"]


def test_material_auto_decision_rejects_high_confidence_conflict(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper_and_candidate(config, mining_config_path, fake_pdf_factory, manifest_writer)
    bundle = MaterialPublicResolverService(
        config,
        pubchem_client=FakePublicResolverClient(),
    ).resolve_paper_public("10.1000/example", max_queries_per_material=2)
    assert bundle is not None
    candidate = next(
        item for item in bundle.structure_candidates if item.paper_material_id == "M002"
    )

    MaterialIdentityJudgeService(
        config,
        llm_client=StaticJSONLLMClient(
            {
                "verdict": "conflict",
                "confidence": 0.97,
                "supporting_evidence": [],
                "conflicts": ["Candidate source names a different material."],
                "recommended_action": "reject_candidate",
            }
        ),
    ).judge_paper_candidates("10.1000/example", paper_material_id="M002")

    result = MaterialAutoDecisionService(config).apply_paper_auto_decisions("10.1000/example")

    assert result is not None
    assert result.accepted_count == 0
    assert result.rejected_count == 1
    rejected = MaterialStructureReviewService(config).structure_candidates.get(
        candidate.structure_candidate_id
    )
    assert rejected is not None
    assert rejected.status == "rejected"
    updated_bundle = MaterialResolutionService(config).get_material_structure_bundle(
        "10.1000/example"
    )
    assert updated_bundle is not None
    task = next(item for item in updated_bundle.tasks if item.paper_material_id == "M002")
    assert task.status == "pending"
    assert task.current_stage == "visual_ocsr_pending"
    assert task.next_action == "run_visual_ocsr"
    bundle_after = MaterialResolutionService(config).get_material_structure_bundle(
        "10.1000/example"
    )
    assert bundle_after is not None
    event = bundle_after.material_review_events[-1]
    assert event.action == "reject"
    assert event.actor == "automation_policy"
    assert "Auto rejected" in (event.message or "")


def test_material_auto_decision_rejects_legacy_direct_pubchem_query_mismatch(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper_and_candidate(config, mining_config_path, fake_pdf_factory, manifest_writer)
    bundle = MaterialResolutionService(config).resolve_paper_materials("10.1000/example")
    assert bundle is not None
    assert bundle.candidate_run_id is not None
    timestamp = now_iso()
    candidate = MaterialStructureCandidate(
        structure_candidate_id="legacy-unrelated-pubchem",
        paper_id=bundle.paper_id,
        candidate_run_id=bundle.candidate_run_id,
        paper_material_id="M002",
        provider="pubchem",
        resolver_name="pubchem_name",
        query_text="mCBP",
        query_type="abbreviation",
        source_identifier="117770458",
        canonical_name="unrelated triazine",
        canonical_smiles="ClC1=NC(-c2ccccc2)=NC(-c2ccccc2)=N1",
        inchi_key="JKHCVYDYGWHIFJ-UHFFFAOYSA-N",
        confidence=0.62,
        status="pending_review",
        created_at=timestamp,
        updated_at=timestamp,
    )
    review_service = MaterialStructureReviewService(config)
    review_service.structure_candidates.upsert(candidate)

    result = MaterialAutoDecisionService(config).apply_paper_auto_decisions("10.1000/example")

    assert result is not None
    decision = next(
        item
        for item in result.decisions
        if item.structure_candidate_id == candidate.structure_candidate_id
    )
    assert decision.action == "auto_reject"
    assert decision.reason == "direct_pubchem_query_mismatch"
    assert decision.confidence == 0.62
    assert decision.applied is True
    rejected = review_service.structure_candidates.get(candidate.structure_candidate_id)
    assert rejected is not None
    assert rejected.status == "rejected"
    updated_bundle = MaterialResolutionService(config).get_material_structure_bundle(
        "10.1000/example"
    )
    assert updated_bundle is not None
    task = next(item for item in updated_bundle.tasks if item.paper_material_id == "M002")
    assert task.current_stage == "visual_ocsr_pending"
    assert task.next_action == "run_visual_ocsr"


def test_rejecting_competing_bad_candidate_preserves_confirmed_material_task(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper_and_candidate(config, mining_config_path, fake_pdf_factory, manifest_writer)
    bundle = MaterialPublicResolverService(
        config,
        pubchem_client=FakePublicResolverClient(),
    ).resolve_paper_public("10.1000/example", max_queries_per_material=2)
    assert bundle is not None
    accepted_candidate = next(
        item for item in bundle.structure_candidates if item.paper_material_id == "M002"
    )
    review_service = MaterialStructureReviewService(config)
    accepted_bundle = review_service.accept_structure_candidate(
        accepted_candidate.structure_candidate_id,
        MaterialReviewAction(actor="tester", message="verified structure"),
    )
    assert accepted_bundle is not None
    completed_task = next(
        item for item in accepted_bundle.tasks if item.paper_material_id == "M002"
    )
    assert completed_task.status == "completed"
    timestamp = now_iso()
    bad_candidate = MaterialStructureCandidate(
        structure_candidate_id="competing-bad-pubchem",
        paper_id=accepted_bundle.paper_id,
        candidate_run_id=accepted_bundle.candidate_run_id or "",
        paper_material_id="M002",
        provider="pubchem",
        resolver_name="pubchem_name",
        query_text="mCBP",
        query_type="abbreviation",
        source_identifier="117770458",
        canonical_name="unrelated triazine",
        canonical_smiles="ClC1=NC(-c2ccccc2)=NC(-c2ccccc2)=N1",
        inchi_key="JKHCVYDYGWHIFJ-UHFFFAOYSA-N",
        confidence=0.62,
        status="pending_review",
        created_at=timestamp,
        updated_at=timestamp,
    )
    review_service.structure_candidates.upsert(bad_candidate)

    result = MaterialAutoDecisionService(config).apply_paper_auto_decisions("10.1000/example")

    assert result is not None
    decision = next(
        item
        for item in result.decisions
        if item.structure_candidate_id == bad_candidate.structure_candidate_id
    )
    assert decision.action == "auto_reject"
    assert decision.applied is True
    refreshed = MaterialResolutionService(config).get_material_structure_bundle("10.1000/example")
    assert refreshed is not None
    task = next(item for item in refreshed.tasks if item.paper_material_id == "M002")
    link = next(item for item in refreshed.links if item.paper_material_id == "M002")
    assert task.status == "completed"
    assert task.next_action == completed_task.next_action
    assert link.match_status == "matched_candidate"
    assert link.global_material_id is not None


def test_material_auto_decision_rejects_low_numeric_confidence_with_deterministic_conflict(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper_and_candidate(config, mining_config_path, fake_pdf_factory, manifest_writer)
    bundle = MaterialPublicResolverService(
        config,
        pubchem_client=FakeIdentifierPubChemClient(),
        anysearch_client=FakeAnySearchClient(),
    ).resolve_paper_public("10.1000/example", paper_material_id="M002", max_queries_per_material=1)
    assert bundle is not None
    candidate = next(
        item for item in bundle.structure_candidates if item.paper_material_id == "M002"
    )
    judged = MaterialIdentityJudgeService(
        config,
        llm_client=StaticJSONLLMClient(
            {
                "verdict": "conflict",
                "confidence": 0.1,
                "supporting_evidence": [],
                "conflicts": ["The discovered CAS belongs to a differently named material."],
                "recommended_action": "reject_candidate",
            }
        ),
        opsin_client=FakeOpsinClient(),
    ).judge_paper_candidates("10.1000/example", paper_material_id="M002")
    assert judged is not None
    judgment = next(
        item
        for item in judged.identity_judgments
        if item.structure_candidate_id == candidate.structure_candidate_id
    )
    assert judgment.confidence == 0.1
    assert judgment.deterministic_checks["identifier_source_title_matches_paper_alias"] is False

    result = MaterialAutoDecisionService(config).apply_paper_auto_decisions("10.1000/example")

    assert result is not None
    decision = next(
        item
        for item in result.decisions
        if item.structure_candidate_id == candidate.structure_candidate_id
    )
    assert decision.action == "auto_reject"
    assert decision.reason == "deterministic_identity_conflict"
    assert decision.applied is True
    rejected = MaterialStructureReviewService(config).structure_candidates.get(
        candidate.structure_candidate_id
    )
    assert rejected is not None
    assert rejected.status == "rejected"

    updated_bundle = MaterialResolutionService(config).get_material_structure_bundle(
        "10.1000/example"
    )
    assert updated_bundle is not None
    task = next(item for item in updated_bundle.tasks if item.paper_material_id == "M002")
    assert task.status == "pending"
    assert task.current_stage == "visual_ocsr_pending"
    assert task.next_action == "run_visual_ocsr"


def test_material_auto_decision_does_not_accept_decimer_ocsr_by_default(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper_and_candidate(config, mining_config_path, fake_pdf_factory, manifest_writer)
    bundle = MaterialResolutionService(config).get_material_structure_bundle("10.1000/example")
    assert bundle is not None
    assert bundle.candidate_run_id is not None
    timestamp = now_iso()
    candidate = MaterialStructureReviewService(config).structure_candidates.upsert(
        MaterialStructureCandidate(
            structure_candidate_id="decimer-candidate",
            paper_id=bundle.paper_id,
            candidate_run_id=bundle.candidate_run_id,
            paper_material_id="M001",
            provider="decimer_ocsr",
            resolver_name="decimer_image_to_smiles",
            query_text="BN-1",
            source_identifier="crop-1",
            canonical_name="BN-1",
            raw_smiles="C1=CC=CC=C1",
            canonical_smiles="c1ccccc1",
            inchi_key="UHOVQNZJYSORNB-UHFFFAOYSA-N",
            confidence=0.99,
            created_at=timestamp,
            updated_at=timestamp,
        )
    )
    MaterialIdentityJudgeService(
        config,
        llm_client=StaticJSONLLMClient(
            {
                "verdict": "exact_match",
                "confidence": 0.99,
                "supporting_evidence": ["The figure label matches the material."],
                "conflicts": [],
                "recommended_action": "ready_for_human_accept",
            }
        ),
    ).judge_paper_candidates("10.1000/example", paper_material_id="M001")

    result = MaterialAutoDecisionService(config).apply_paper_auto_decisions("10.1000/example")

    assert result is not None
    assert result.accepted_count == 0
    decision = next(
        item
        for item in result.decisions
        if item.structure_candidate_id == candidate.structure_candidate_id
    )
    assert decision.action == "skip"
    assert decision.reason == "decimer_ocsr_auto_accept_disabled"
    refreshed = MaterialStructureReviewService(config).structure_candidates.get(
        candidate.structure_candidate_id
    )
    assert refreshed is not None
    assert refreshed.status == "pending_review"


def test_material_identity_judge_api_persists_verdict_and_blocks_acceptance(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper_and_candidate(config, mining_config_path, fake_pdf_factory, manifest_writer)
    bundle = MaterialPublicResolverService(
        config,
        pubchem_client=FakePublicResolverClient(),
    ).resolve_paper_public("10.1000/example", max_queries_per_material=2)
    assert bundle is not None
    candidate = next(
        item for item in bundle.structure_candidates if item.paper_material_id == "M002"
    )
    fake_judge = StaticJSONLLMClient(
        {
            "verdict": "conflict",
            "confidence": 0.96,
            "supporting_evidence": [],
            "conflicts": ["The supplier title names a different compound."],
            "recommended_action": "reject_candidate",
        }
    )
    application = create_app(config=config)
    application.state.material_identity_judge_service = MaterialIdentityJudgeService(
        config,
        llm_client=fake_judge,
    )
    client = TestClient(application)

    judge_response = client.post(
        "/api/papers/10.1000%2Fexample/judge-material-identities?paper_material_id=M002"
    )

    assert judge_response.status_code == 200
    judgments = judge_response.json()["identity_judgments"]
    assert judgments[0]["structure_candidate_id"] == candidate.structure_candidate_id
    assert judgments[0]["verdict"] == "conflict"
    accept_response = client.post(
        f"/api/material-structure-candidates/{candidate.structure_candidate_id}/accept",
        json={"actor": "tester"},
    )
    assert accept_response.status_code == 422
    assert (
        "blocked by the latest Material Identity Judge result" in accept_response.json()["detail"]
    )


def test_material_identity_judge_forces_conflict_on_opsin_inchi_key_mismatch(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    payload = _valid_result()
    payload["materials"][1]["full_name_in_paper"] = "benzene"
    _seed_paper_and_candidate(
        config,
        mining_config_path,
        fake_pdf_factory,
        manifest_writer,
        payload=payload,
    )
    bundle = MaterialPublicResolverService(
        config,
        pubchem_client=FakePublicResolverClient(),
    ).resolve_paper_public("10.1000/example", paper_material_id="M002", max_queries_per_material=2)
    assert bundle is not None

    judged = MaterialIdentityJudgeService(
        config,
        llm_client=StaticJSONLLMClient(
            {
                "verdict": "exact_match",
                "confidence": 0.99,
                "supporting_evidence": ["Model guessed a match."],
                "conflicts": [],
                "recommended_action": "ready_for_human_accept",
            }
        ),
        opsin_client=FakeOpsinClient(),
    ).judge_paper_candidates("10.1000/example", paper_material_id="M002")

    assert judged is not None
    judgment = judged.identity_judgments[0]
    assert judgment.verdict == "conflict"
    assert judgment.recommended_action == "reject_candidate"
    assert judgment.deterministic_checks["opsin_inchi_key_comparison"] == "conflict"
    with pytest.raises(ValueError, match="blocked by the latest Material Identity Judge"):
        MaterialStructureReviewService(config).accept_structure_candidate(
            judgment.structure_candidate_id,
            MaterialReviewAction(actor="tester"),
        )


def test_material_identity_judge_allows_confirmed_evidence_to_override_lossy_paper_name_opsin_conflict(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    payload = _valid_result()
    payload["materials"][1]["full_name_in_paper"] = "lossy paper-local full name"
    _seed_paper_and_candidate(
        config,
        mining_config_path,
        fake_pdf_factory,
        manifest_writer,
        payload=payload,
    )
    bundle = MaterialPublicResolverService(
        config,
        pubchem_client=FakeConfirmedEvidencePublicResolverClient(),
    ).resolve_paper_public("10.1000/example", paper_material_id="M002", max_queries_per_material=2)
    assert bundle is not None
    candidate = next(
        item for item in bundle.structure_candidates if item.paper_material_id == "M002"
    )
    _add_confirmed_identity_evidence(
        config,
        paper_id=bundle.paper_id,
        candidate_run_id=bundle.candidate_run_id or "",
        paper_material_id="M002",
        full_name="confirmed evidence full name",
    )

    judged = MaterialIdentityJudgeService(
        config,
        llm_client=StaticJSONLLMClient(
            {
                "verdict": "conflict",
                "confidence": 0.3,
                "supporting_evidence": ["LLM followed the conservative OPSIN conflict rule."],
                "conflicts": ["Paper full-name OPSIN result differs."],
                "recommended_action": "reject_candidate",
            }
        ),
        opsin_client=FakeConfirmedEvidenceOpsinClient(),
    ).judge_paper_candidates("10.1000/example", paper_material_id="M002")

    assert judged is not None
    judgment = next(
        item
        for item in judged.identity_judgments
        if item.structure_candidate_id == candidate.structure_candidate_id
    )
    assert judgment.verdict == "likely_match"
    assert judgment.recommended_action == "ready_for_human_accept"
    assert judgment.deterministic_checks["opsin_inchi_key_comparison"] == "conflict"
    assert judgment.deterministic_checks["confirmed_evidence_inchi_key_comparison"] == "same"
    assert "Human-confirmed identity evidence" in judgment.supporting_evidence[0]

    accepted = MaterialStructureReviewService(config).accept_structure_candidate(
        judgment.structure_candidate_id,
        MaterialReviewAction(actor="tester"),
    )
    assert accepted is not None
    accepted_candidate = next(
        item
        for item in accepted.structure_candidates
        if item.structure_candidate_id == candidate.structure_candidate_id
    )
    assert accepted_candidate.status == "accepted"


def test_material_identity_judge_downgrades_web_cas_candidate_with_mismatched_source_name(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper_and_candidate(config, mining_config_path, fake_pdf_factory, manifest_writer)
    bundle = MaterialPublicResolverService(
        config,
        pubchem_client=FakeIdentifierPubChemClient(),
        anysearch_client=FakeAnySearchClient(),
    ).resolve_paper_public("10.1000/example", paper_material_id="M002", max_queries_per_material=1)
    assert bundle is not None
    candidate = next(
        item for item in bundle.structure_candidates if item.paper_material_id == "M002"
    )
    with pytest.raises(ValueError, match="require Material Identity Judge review"):
        MaterialStructureReviewService(config).accept_structure_candidate(
            candidate.structure_candidate_id,
            MaterialReviewAction(actor="tester"),
        )

    judged = MaterialIdentityJudgeService(
        config,
        llm_client=StaticJSONLLMClient(
            {
                "verdict": "likely_match",
                "confidence": 0.91,
                "supporting_evidence": ["CAS lookup produced a structure."],
                "conflicts": [],
                "recommended_action": "ready_for_human_accept",
            }
        ),
        opsin_client=FakeOpsinClient(),
    ).judge_paper_candidates("10.1000/example", paper_material_id="M002")

    assert judged is not None
    judgment = judged.identity_judgments[0]
    assert candidate.resolver_name == "anysearch_to_pubchem"
    assert judgment.verdict == "ambiguous"
    assert judgment.recommended_action == "search_more_evidence"
    assert "does not match any paper alias" in judgment.conflicts[0]
    assert judgment.deterministic_checks["identifier_source_title_matches_paper_alias"] is False


def test_identity_evidence_enrichment_generates_candidate_and_requires_confirmed_source(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper_and_candidate(config, mining_config_path, fake_pdf_factory, manifest_writer)
    initial = MaterialPublicResolverService(
        config,
        pubchem_client=FakeIdentifierPubChemClient(),
        anysearch_client=FakeAnySearchClient(),
    ).resolve_paper_public("10.1000/example", paper_material_id="M002", max_queries_per_material=1)
    assert initial is not None
    old_candidate = next(
        item for item in initial.structure_candidates if item.paper_material_id == "M002"
    )
    judge_service = MaterialIdentityJudgeService(
        config,
        llm_client=StaticJSONLLMClient(
            {
                "verdict": "likely_match",
                "confidence": 0.91,
                "supporting_evidence": ["Evidence states the material identity."],
                "conflicts": [],
                "recommended_action": "ready_for_human_accept",
            }
        ),
        opsin_client=FakeOpsinClient(),
    )
    evidence_service = MaterialIdentityEvidenceService(
        config,
        anysearch_client=FakeIdentityEvidenceSearchClient(),
        llm_client=StaticJSONLLMClient(
            {
                "evidence_items": [
                    {
                        "source_index": 0,
                        "source_tier": "B",
                        "explicitly_linked": True,
                        "alias": "mCBP",
                        "full_name": "benzene",
                        "cas_number": None,
                        "pubchem_cid": None,
                        "excerpt": "mCBP (benzene) was employed as the host.",
                        "confidence": 0.94,
                        "query_text": '"mCBP" OLED full chemical name',
                    }
                ]
            }
        ),
        opsin_client=FakeOpsinClient(),
        pubchem_client=FakeEmptyPublicResolverClient(),
        judge_service=judge_service,
    )
    application = create_app(config=config)
    application.state.material_identity_evidence_service = evidence_service
    client = TestClient(application)

    enriched_response = client.post(
        "/api/papers/10.1000%2Fexample/enrich-material-identity?paper_material_id=M002"
    )

    assert enriched_response.status_code == 200
    enriched = enriched_response.json()
    assert enriched["identity_evidence_runs"][0]["status"] == "completed"
    item = enriched["identity_evidence_items"][0]
    assert item["source_tier"] == "B"
    assert item["full_name"] == "benzene"
    assert item["review_status"] == "pending_review"
    new_candidate = next(
        candidate
        for candidate in enriched["structure_candidates"]
        if candidate["resolver_name"] == "identity_evidence_opsin"
    )
    blocked_before_confirm = client.post(
        f"/api/material-structure-candidates/{new_candidate['structure_candidate_id']}/accept",
        json={"actor": "tester"},
    )
    assert blocked_before_confirm.status_code == 422
    assert "confirmed evidence source" in blocked_before_confirm.json()["detail"]

    confirmed_response = client.post(
        f"/api/material-identity-evidence-items/{item['evidence_item_id']}/review",
        json={"decision": "confirm", "actor": "tester"},
    )

    assert confirmed_response.status_code == 200
    confirmed = confirmed_response.json()
    latest_by_candidate: dict[str, dict[str, object]] = {}
    for judgment in confirmed["identity_judgments"]:
        latest_by_candidate.setdefault(judgment["structure_candidate_id"], judgment)
    assert latest_by_candidate[old_candidate.structure_candidate_id]["verdict"] == "conflict"
    accepted_response = client.post(
        f"/api/material-structure-candidates/{new_candidate['structure_candidate_id']}/accept",
        json={"actor": "tester"},
    )
    assert accepted_response.status_code == 200


class FakeIrppyPubChemClient:
    def resolve_name(self, name: str, *, max_results: int | None = None) -> list[PubChemCompound]:
        if normalize_material_alias(name) != "irppy3":
            return []
        return [
            PubChemCompound(
                cid="160214",
                query_text=name,
                iupac_name="tris(2-phenylpyridine)iridium(III)",
                canonical_smiles="C1=CC=CC=C1",
                inchi_key="IRPPY3KEY",
                formula="C33H24IrN3",
                synonyms=["Ir(ppy)3", "Tris(2-phenylpyridine)iridium"],
            )
        ]


def test_material_stage3_planner_persists_material_level_next_action(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper_and_candidate(config, mining_config_path, fake_pdf_factory, manifest_writer)

    plan = MaterialStage3PlannerService(config).plan_papers(["10.1000/example"])

    assert plan.paper_count == 1
    assert plan.material_count >= 1
    assert plan.public_pending_count >= 1
    public_item = next(item for item in plan.items if item.route == "public_resolution")
    task = MaterialResolutionTaskRepository(
        Database(config.paths.sqlite_path)
    ).get_by_paper_material(
        public_item.candidate_run_id,
        public_item.paper_material_id,
    )
    assert task is not None
    assert task.current_stage == "planned"
    assert task.next_action == "resolve_public"


def test_material_stage3_planner_reuses_resolved_bundle_for_triage(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
    monkeypatch,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper_and_candidate(config, mining_config_path, fake_pdf_factory, manifest_writer)
    planner = MaterialStage3PlannerService(config)

    def fail_redundant_reload(_paper_id: str):
        raise AssertionError("triage_paper should not reload an already resolved bundle")

    monkeypatch.setattr(planner.triage, "triage_paper", fail_redundant_reload)

    plan = planner.plan_papers(
        ["10.1000/example"],
        refresh_local=True,
        max_workers=2,
    )

    assert plan.paper_count == 1
    assert plan.material_count >= 1


def test_material_stage3_planner_clears_stale_action_for_accepted_structure(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper_and_candidate(config, mining_config_path, fake_pdf_factory, manifest_writer)
    resolver = MaterialPublicResolverService(
        config,
        pubchem_client=FakePublicResolverClient(),
    )
    bundle = resolver.resolve_paper_public("10.1000/example", max_queries_per_material=2)
    assert bundle is not None
    candidate = next(
        item for item in bundle.structure_candidates if item.paper_material_id == "M002"
    )
    MaterialStructureReviewService(config).accept_structure_candidate(
        candidate.structure_candidate_id,
        MaterialReviewAction(actor="tester"),
    )

    plan = MaterialStage3PlannerService(config).plan_papers(
        ["10.1000/example"],
        refresh_local=False,
    )

    accepted_item = next(item for item in plan.items if item.paper_material_id == "M002")
    task = MaterialResolutionTaskRepository(
        Database(config.paths.sqlite_path)
    ).get_by_paper_material(
        accepted_item.candidate_run_id,
        accepted_item.paper_material_id,
    )
    assert accepted_item.route == "local_or_accepted"
    assert task is not None
    assert task.status == "completed"
    assert task.current_stage == "completed"
    assert task.next_action == "none"
    assert task.assigned_strategy == "accepted_or_local_structure"


def test_single_material_public_resolution_reuses_completed_attempt(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper_and_candidate(config, mining_config_path, fake_pdf_factory, manifest_writer)
    fake_public = FakePublicResolverClient()
    service = MaterialPublicResolverService(
        config,
        pubchem_client=fake_public,
        opsin_client=FakeOpsinClient(),
    )
    service.resolve_paper_public("10.1000/example", paper_material_id="M002")
    first_query_count = len(fake_public.queries)

    service.resolve_material_public("10.1000/example", "M002")

    assert first_query_count > 0
    assert len(fake_public.queries) == first_query_count


def test_identity_judge_groups_candidates_and_reuses_fingerprint(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper_and_candidate(config, mining_config_path, fake_pdf_factory, manifest_writer)
    bundle = MaterialPublicResolverService(
        config,
        pubchem_client=FakePublicResolverClient(),
        opsin_client=FakeOpsinClient(),
    ).resolve_paper_public("10.1000/example", paper_material_id="M002")
    assert bundle is not None and bundle.candidate_run_id
    timestamp = now_iso()
    MaterialStructureReviewService(config).structure_candidates.upsert(
        MaterialStructureCandidate(
            structure_candidate_id="second-public-candidate",
            paper_id=bundle.paper_id,
            candidate_run_id=bundle.candidate_run_id,
            paper_material_id="M002",
            provider="pubchem",
            resolver_name="pubchem_name",
            query_text="mCBP",
            source_identifier="second-cid",
            canonical_name="second candidate",
            canonical_smiles="c1ccccc1",
            inchi_key="SECONDKEY",
            created_at=timestamp,
            updated_at=timestamp,
        )
    )

    class CountingGroupedClient:
        def __init__(self) -> None:
            self.calls = 0

        def generate_json(self, messages, *, model=None, temperature=None, max_tokens=None):
            del temperature, max_tokens
            self.calls += 1
            context = json.loads(messages[-1]["content"].split("\n", 1)[1])
            judgments = [
                {
                    "structure_candidate_id": candidate["structure_candidate_id"],
                    "verdict": "ambiguous",
                    "confidence": 0.7,
                    "supporting_evidence": [],
                    "conflicts": ["Two candidates require comparison."],
                    "recommended_action": "manual_review",
                }
                for candidate in context["candidates"]
            ]
            return LLMResponse(
                content=json.dumps({"judgments": judgments}),
                parsed_json={"judgments": judgments},
                raw_response={"mock": True, "model": model},
            )

    client = CountingGroupedClient()
    judge = MaterialIdentityJudgeService(
        config,
        llm_client=client,
        opsin_client=FakeOpsinClient(),
    )
    first = judge.judge_material_candidates("10.1000/example", "M002")
    second = judge.judge_material_candidates("10.1000/example", "M002")

    assert len(first) == 2
    assert len(second) == 2
    assert client.calls == 1
    assert {item.input_context["group_fingerprint"] for item in first}


def test_identity_judge_operational_failure_is_persisted_and_raised(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper_and_candidate(config, mining_config_path, fake_pdf_factory, manifest_writer)
    bundle = MaterialPublicResolverService(
        config,
        pubchem_client=FakePublicResolverClient(),
        opsin_client=FakeOpsinClient(),
    ).resolve_paper_public("10.1000/example", paper_material_id="M002")
    assert bundle is not None
    candidate = next(
        item
        for item in bundle.structure_candidates
        if item.paper_material_id == "M002" and item.status not in {"accepted", "rejected"}
    )

    class FailingClient:
        def generate_json(self, messages, *, model=None, temperature=None, max_tokens=None):
            del messages, model, temperature, max_tokens
            raise RuntimeError("provider temporarily unavailable")

    judge = MaterialIdentityJudgeService(
        config,
        llm_client=FailingClient(),
        opsin_client=FakeOpsinClient(),
    )

    with pytest.raises(MaterialIdentityJudgeExecutionError, match="temporarily unavailable"):
        judge.judge_material_candidates("10.1000/example", "M002")

    failed = judge.judgments.latest_by_candidate(candidate.structure_candidate_id)
    assert failed is not None
    assert failed.status == "failed"
    assert failed.error_message == "provider temporarily unavailable"


class FakeIrppyAnySearchClient:
    def search(self, query: str, *, max_results: int | None = None) -> list[AnySearchResult]:
        return [
            AnySearchResult(
                title="Ir(ppy)3 OLED emitter reference",
                url="https://example.test/irppy3",
                content="Ir(ppy)3 is commonly used as an OLED phosphorescent sensitizer.",
                score=91.0,
            )
        ]


class FakePublicResolverClient:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def resolve_name(self, name: str, *, max_results: int | None = None) -> list[PubChemCompound]:
        self.queries.append(name)
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


class FakeUnrelatedPublicResolverClient:
    def resolve_name(self, name: str, *, max_results: int | None = None) -> list[PubChemCompound]:
        return [
            PubChemCompound(
                cid="117770458",
                query_text=name,
                iupac_name="2-chloro-4-(biphenyl-4-yl)-6-phenyl-1,3,5-triazine",
                canonical_smiles="ClC1=NC(-c2ccccc2)=NC(-c2ccccc2)=N1",
                inchi_key="JKHCVYDYGWHIFJ-UHFFFAOYSA-N",
                formula="C21H14ClN3",
                synonyms=["unrelated triazine"],
            )
        ]


class FakeConfirmedEvidencePublicResolverClient:
    def resolve_name(self, name: str, *, max_results: int | None = None) -> list[PubChemCompound]:
        if normalize_material_alias(name) != "mcbp":
            return []
        return [
            PubChemCompound(
                cid="confirmed-999",
                query_text=name,
                iupac_name="confirmed evidence full name",
                canonical_smiles="c1ccccc1",
                isomeric_smiles="c1ccccc1",
                inchi="InChI=1S/C6H6",
                inchi_key="UHOVQNZJYSORNB-UHFFFAOYSA-N",
                formula="C6H6",
                molecular_weight=78.11,
                synonyms=["mCBP", "confirmed evidence full name"],
            )
        ]


class FakeIdentifierPubChemClient:
    def resolve_name(self, name: str, *, max_results: int | None = None) -> list[PubChemCompound]:
        if name != "164724-35-0":
            return []
        return [
            PubChemCompound(
                cid="16153173",
                query_text=name,
                canonical_smiles="C1=CC=CC=C1",
                inchi_key="BPBPAKEY",
                formula="C60H44N2",
                synonyms=["BPBPA"],
            )
        ]


class FakeAmbiguousAbbreviationPubChemClient:
    def resolve_name(self, name: str, *, max_results: int | None = None) -> list[PubChemCompound]:
        normalized = normalize_material_alias(name)
        if normalized == "mcbp":
            return [
                PubChemCompound(
                    cid="wrong-abbreviation-hit",
                    query_text=name,
                    iupac_name="unrelated material sharing the abbreviation",
                    canonical_smiles="C",
                    inchi_key="WRONGABBREVIATIONKEY",
                    formula="CH4",
                    synonyms=["mCBP"],
                )
            ]
        if name == "164724-35-0":
            return [
                PubChemCompound(
                    cid="16153173",
                    query_text=name,
                    canonical_smiles="c1ccccc1",
                    inchi_key="WEBIDENTITYKEY",
                    formula="C6H6",
                    synonyms=["mCBP"],
                )
            ]
        return []


class FakeEmptyPublicResolverClient:
    def resolve_name(self, name: str, *, max_results: int | None = None) -> list[PubChemCompound]:
        return []


class FakeOpsinClient:
    def resolve_name(self, name: str) -> OpsinCompound | None:
        if name != "benzene":
            return None
        return OpsinCompound(
            query_text=name,
            smiles="c1ccccc1",
            inchi="InChI=1S/C6H6",
            inchi_key="UHOVQNZJYSORNB-UHFFFAOYSA-N",
            raw_result={"status": "SUCCESS", "smiles": "c1ccccc1"},
        )


class FakeConfirmedEvidenceOpsinClient:
    def resolve_name(self, name: str) -> OpsinCompound | None:
        if name == "lossy paper-local full name":
            return OpsinCompound(
                query_text=name,
                smiles="C",
                inchi="InChI=1S/CH4/h1H4",
                inchi_key="VNWKTOKETHGBQD-UHFFFAOYSA-N",
                raw_result={"status": "SUCCESS", "smiles": "C"},
            )
        if name == "confirmed evidence full name":
            return OpsinCompound(
                query_text=name,
                smiles="c1ccccc1",
                inchi="InChI=1S/C6H6",
                inchi_key="UHOVQNZJYSORNB-UHFFFAOYSA-N",
                raw_result={"status": "SUCCESS", "smiles": "c1ccccc1"},
            )
        return None


class FakeAnySearchClient:
    def search(self, query: str, *, max_results: int | None = None) -> list[AnySearchResult]:
        return [
            AnySearchResult(
                title="BPBPA | CAS 164724-35-0",
                url="https://example.test/bpbpa",
                content="OLED material CAS 164724-35-0",
            )
        ]


class FakeTargetedAnySearchClient:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, *, max_results: int | None = None) -> list[AnySearchResult]:
        self.queries.append(query)
        if "CAS SMILES" not in query:
            return []
        return FakeAnySearchClient().search(query, max_results=max_results)


class FakeIdentityEvidenceSearchClient:
    def search(self, query: str, *, max_results: int | None = None) -> list[AnySearchResult]:
        return [
            AnySearchResult(
                title="Authoritative OLED article: mCBP is benzene",
                url="https://example.test/mcbp-evidence",
                description="The paper reports mCBP (benzene) as the host material.",
                content="mCBP (benzene) was employed as the host.",
            )
        ]


def _add_confirmed_identity_evidence(
    config,
    *,
    paper_id: str,
    candidate_run_id: str,
    paper_material_id: str,
    full_name: str,
) -> None:
    database = Database(config.paths.sqlite_path)
    timestamp = now_iso()
    run = MaterialIdentityEvidenceRun(
        evidence_run_id="confirmed-evidence-run",
        paper_id=paper_id,
        candidate_run_id=candidate_run_id,
        paper_material_id=paper_material_id,
        provider="test",
        model="test",
        query_plan=["confirmed evidence"],
        status="completed",
        recommended_next_action="manual_review",
        created_at=timestamp,
        updated_at=timestamp,
        completed_at=timestamp,
    )
    MaterialIdentityEvidenceRunRepository(database).upsert(run)
    MaterialIdentityEvidenceItemRepository(database).add(
        MaterialIdentityEvidenceItem(
            evidence_item_id="confirmed-evidence-item",
            evidence_run_id=run.evidence_run_id,
            paper_id=paper_id,
            candidate_run_id=candidate_run_id,
            paper_material_id=paper_material_id,
            source_type="web_search",
            source_tier="B",
            source_title="Confirmed identity source",
            source_url="https://example.test/confirmed",
            query_text="mCBP",
            excerpt="mCBP is confirmed evidence full name.",
            alias="mCBP",
            full_name=full_name,
            cas_number="123-45-6",
            explicitly_linked=True,
            confidence=0.95,
            review_status="confirmed",
            reviewed_by="tester",
            review_note="Confirmed in test",
            reviewed_at=timestamp,
            created_at=timestamp,
            updated_at=timestamp,
        )
    )


def _seed_paper_and_candidate(
    config,
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
    *,
    payload: dict[str, object] | None = None,
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
                "smiles": None,
                "inchi": None,
                "inchi_key": None,
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
                "smiles": None,
                "inchi": None,
                "inchi_key": None,
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
                "smiles": None,
                "inchi": None,
                "inchi_key": None,
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


def _valid_result() -> dict[str, object]:
    return copy.deepcopy(load_domain_template(_template_path()).example_output)


def _template_path() -> Path:
    return Path("config/mining_platform/domains/oled_device_v1.yaml")
