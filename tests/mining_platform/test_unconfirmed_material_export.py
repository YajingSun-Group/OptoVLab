from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "export_unconfirmed_materials.py"
_SPEC = importlib.util.spec_from_file_location("export_unconfirmed_materials", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_material_status = _MODULE._material_status


def _candidate(*, status: str = "pending_review") -> SimpleNamespace:
    return SimpleNamespace(
        structure_candidate_id="candidate-1",
        status=status,
        provider="pubchem",
        canonical_smiles="c1ccccc1",
        isomeric_smiles=None,
        raw_smiles=None,
    )


def _judgment() -> SimpleNamespace:
    return SimpleNamespace(
        verdict="exact_match",
        recommended_action="ready_for_human_accept",
    )


def _status_for(
    candidate: SimpleNamespace,
    *,
    link: SimpleNamespace | None = None,
    linked_global: SimpleNamespace | None = None,
) -> dict[str, str]:
    return _material_status(
        material=SimpleNamespace(),
        link=link,
        linked_global=linked_global,
        task=None,
        candidates=[candidate],
        judgments_by_candidate={candidate.structure_candidate_id: _judgment()},
        evidence_runs=[],
        evidence_items=[],
        binding_stats={"confirmed": 0, "pending": 0},
        visual=_MODULE._empty_visual_stats(),
        name_suggestion_count=0,
    )


def test_export_marks_accepted_candidate_ready_before_judgment_status() -> None:
    status = _status_for(_candidate(status="accepted"))

    assert status["label"] == "Ready"


def test_export_marks_structured_global_link_ready_before_judgment_status() -> None:
    status = _status_for(
        _candidate(),
        link=SimpleNamespace(global_material_id="global-1"),
        linked_global=SimpleNamespace(
            canonical_smiles="c1ccccc1",
            isomeric_smiles=None,
            raw_smiles=None,
        ),
    )

    assert status["label"] == "Ready"


def test_export_keeps_unaccepted_judged_candidate_ready_for_review() -> None:
    status = _status_for(_candidate())

    assert status["label"] == "Candidate ready"


def test_export_prefers_manual_input_after_completed_failed_resolution() -> None:
    candidate = _candidate(status="rejected")
    status = _material_status(
        material=SimpleNamespace(),
        link=None,
        linked_global=None,
        task=SimpleNamespace(
            next_action="manual_structure_input",
            assigned_strategy="manual_structure_required",
            status="needs_review",
            priority="normal",
            material_context={},
        ),
        candidates=[candidate],
        judgments_by_candidate={},
        evidence_runs=[SimpleNamespace(recommended_next_action="run_figure_pipeline")],
        evidence_items=[],
        binding_stats={"confirmed": 0, "pending": 0},
        visual=_MODULE._empty_visual_stats(),
        name_suggestion_count=0,
    )

    assert status["label"] == "Manual input required"
