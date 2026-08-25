from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response

from evolab_local.mining_platform.chemical_figure_collector_service import (
    ChemicalFigureCollectorService,
)
from evolab_local.mining_platform.material_public_resolver_service import (
    MaterialPublicResolverService,
)
from evolab_local.mining_platform.material_identity_judge_service import (
    MaterialIdentityJudgeService,
)
from evolab_local.mining_platform.material_identity_evidence_service import (
    MaterialIdentityEvidenceService,
)
from evolab_local.mining_platform.material_auto_decision_service import (
    MaterialAutoDecisionService,
)
from evolab_local.mining_platform.material_property_review_service import (
    MaterialPropertyReviewService,
)
from evolab_local.mining_platform.material_resolution_service import MaterialResolutionService
from evolab_local.mining_platform.material_structure_review_service import (
    MaterialStructureReviewService,
)
from evolab_local.mining_platform.material_structure_agent_service import (
    MaterialStructureAgentService,
)
from evolab_local.mining_platform.schemas.material_agent import (
    DecimerSegmentationBatchResult,
    DocumentVisualBlock,
    FigureTriageBatchResult,
    FigureTriageResult,
    MaterialAgentFoundationResult,
    MaterialAgentRun,
    MoleculeCrop,
    MoleculeCropValidation,
    MoleculeCropValidationBatchResult,
    MoleculeLabelBinding,
    MoleculeLabelBindingBatchResult,
    MoleculeLabelBindingReviewAction,
    MoleculeLabelBindingReviewEvent,
    VLMCallLog,
)
from evolab_local.mining_platform.schemas.material_structure import (
    ChemicalFigureBlock,
    DecimerOCSRBatchResult,
    MaterialAutoDecisionResult,
    MaterialManualStructureAction,
    MaterialPropertyManualAddAction,
    MaterialPropertyReviewAction,
    MaterialReviewAction,
    MaterialIdentityEvidenceReviewAction,
    PaperMaterialNameReviewAction,
    MaterialResolutionTask,
    MaterialStructureCandidate,
    MaterialStructureEditAction,
    PaperMaterialStructureBundle,
)

router = APIRouter(tags=["material-structures"])


def _service(request: Request) -> MaterialResolutionService:
    return request.app.state.material_resolution_service


def _public_service(request: Request) -> MaterialPublicResolverService:
    return request.app.state.material_public_resolver_service


def _identity_service(request: Request) -> MaterialIdentityJudgeService:
    return request.app.state.material_identity_judge_service


def _identity_evidence_service(request: Request) -> MaterialIdentityEvidenceService:
    return request.app.state.material_identity_evidence_service


def _auto_decision_service(request: Request) -> MaterialAutoDecisionService:
    return request.app.state.material_auto_decision_service


def _property_review_service(request: Request) -> MaterialPropertyReviewService:
    return request.app.state.material_property_review_service


def _ensure_material_properties_enabled(request: Request) -> None:
    if not request.app.state.config.features.material_properties:
        raise HTTPException(
            status_code=404,
            detail="Material property extraction and review is disabled in the current runtime version.",
        )


def _review_service(request: Request) -> MaterialStructureReviewService:
    return request.app.state.material_structure_review_service


def _figure_service(request: Request) -> ChemicalFigureCollectorService:
    return request.app.state.chemical_figure_collector_service


def _agent_service(request: Request) -> MaterialStructureAgentService:
    return request.app.state.material_structure_agent_service


@router.get(
    "/api/papers/{paper_id:path}/material-structures",
    response_model=PaperMaterialStructureBundle,
)
def get_paper_material_structures(
    request: Request,
    paper_id: str,
) -> PaperMaterialStructureBundle:
    bundle = _service(request).get_material_structure_bundle(paper_id)
    if bundle is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return bundle


@router.post(
    "/api/papers/{paper_id:path}/resolve-materials",
    response_model=PaperMaterialStructureBundle,
)
def resolve_paper_materials(
    request: Request,
    paper_id: str,
) -> PaperMaterialStructureBundle:
    bundle = _service(request).resolve_paper_materials(paper_id)
    if bundle is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return bundle


@router.post(
    "/api/papers/{paper_id:path}/review-material-name",
    response_model=PaperMaterialStructureBundle,
)
def review_paper_material_name(
    request: Request,
    paper_id: str,
    paper_material_id: str,
    action: PaperMaterialNameReviewAction,
) -> PaperMaterialStructureBundle:
    try:
        bundle = _service(request).review_paper_material_name(
            paper_id,
            paper_material_id=paper_material_id,
            action=action,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if bundle is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return bundle


@router.post(
    "/api/papers/{paper_id:path}/validate-material-names",
    response_model=PaperMaterialStructureBundle,
)
def validate_paper_material_names(
    request: Request,
    paper_id: str,
    paper_material_id: str | None = None,
) -> PaperMaterialStructureBundle:
    bundle = _service(request).validate_material_names(
        paper_id,
        paper_material_id=paper_material_id,
    )
    if bundle is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return bundle


@router.post(
    "/api/papers/{paper_id:path}/resolve-public-materials",
    response_model=PaperMaterialStructureBundle,
)
def resolve_paper_materials_public(
    request: Request,
    paper_id: str,
    paper_material_id: str | None = None,
) -> PaperMaterialStructureBundle:
    bundle = _public_service(request).resolve_paper_public(
        paper_id,
        paper_material_id=paper_material_id,
    )
    if bundle is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return bundle


@router.post(
    "/api/papers/{paper_id:path}/judge-material-identities",
    response_model=PaperMaterialStructureBundle,
)
def judge_paper_material_identities(
    request: Request,
    paper_id: str,
    paper_material_id: str | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> PaperMaterialStructureBundle:
    try:
        bundle = _identity_service(request).judge_paper_candidates(
            paper_id,
            paper_material_id=paper_material_id,
            provider=provider,
            model=model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if bundle is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return bundle


@router.post(
    "/api/papers/{paper_id:path}/auto-decide-materials",
    response_model=MaterialAutoDecisionResult,
)
def auto_decide_paper_materials(
    request: Request,
    paper_id: str,
    dry_run: bool = False,
    accept_min_confidence: float | None = None,
    reject_min_confidence: float | None = None,
) -> MaterialAutoDecisionResult:
    bundle = _auto_decision_service(request).apply_paper_auto_decisions(
        paper_id,
        dry_run=dry_run,
        accept_min_confidence=accept_min_confidence,
        reject_min_confidence=reject_min_confidence,
    )
    if bundle is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return bundle


@router.post(
    "/api/papers/{paper_id:path}/enrich-material-identity",
    response_model=PaperMaterialStructureBundle,
)
def enrich_paper_material_identity(
    request: Request,
    paper_id: str,
    paper_material_id: str,
    provider: str | None = None,
    model: str | None = None,
) -> PaperMaterialStructureBundle:
    try:
        bundle = _identity_evidence_service(request).enrich_material_identity(
            paper_id,
            paper_material_id=paper_material_id,
            provider=provider,
            model=model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if bundle is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return bundle


@router.post(
    "/api/material-identity-evidence-items/{evidence_item_id}/review",
    response_model=PaperMaterialStructureBundle,
)
def review_material_identity_evidence_item(
    request: Request,
    evidence_item_id: str,
    action: MaterialIdentityEvidenceReviewAction,
) -> PaperMaterialStructureBundle:
    try:
        bundle = _identity_evidence_service(request).review_evidence_item(
            evidence_item_id,
            action,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if bundle is None:
        raise HTTPException(status_code=404, detail="Evidence item not found")
    return bundle


@router.get(
    "/api/material-resolution-tasks",
    response_model=list[MaterialResolutionTask],
)
def list_material_resolution_tasks(request: Request) -> list[MaterialResolutionTask]:
    return _service(request).list_resolution_tasks()


@router.post(
    "/api/material-property-candidates/{property_candidate_id}/accept",
    response_model=PaperMaterialStructureBundle,
)
def accept_material_property_candidate(
    request: Request,
    property_candidate_id: str,
    action: MaterialPropertyReviewAction,
) -> PaperMaterialStructureBundle:
    _ensure_material_properties_enabled(request)
    bundle = _property_review_service(request).accept_candidate(property_candidate_id, action)
    if bundle is None:
        raise HTTPException(status_code=404, detail="Material property candidate not found")
    return bundle


@router.put(
    "/api/material-property-candidates/{property_candidate_id}",
    response_model=PaperMaterialStructureBundle,
)
def edit_material_property_candidate(
    request: Request,
    property_candidate_id: str,
    action: MaterialPropertyReviewAction,
) -> PaperMaterialStructureBundle:
    _ensure_material_properties_enabled(request)
    try:
        bundle = _property_review_service(request).edit_candidate(property_candidate_id, action)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if bundle is None:
        raise HTTPException(status_code=404, detail="Material property candidate not found")
    return bundle


@router.post(
    "/api/material-property-candidates/{property_candidate_id}/reject",
    response_model=PaperMaterialStructureBundle,
)
def reject_material_property_candidate(
    request: Request,
    property_candidate_id: str,
    action: MaterialPropertyReviewAction,
) -> PaperMaterialStructureBundle:
    _ensure_material_properties_enabled(request)
    bundle = _property_review_service(request).reject_candidate(property_candidate_id, action)
    if bundle is None:
        raise HTTPException(status_code=404, detail="Material property candidate not found")
    return bundle


@router.post(
    "/api/papers/{paper_id:path}/manual-material-property",
    response_model=PaperMaterialStructureBundle,
)
def add_manual_material_property(
    request: Request,
    paper_id: str,
    paper_material_id: str,
    action: MaterialPropertyManualAddAction,
) -> PaperMaterialStructureBundle:
    _ensure_material_properties_enabled(request)
    try:
        bundle = _property_review_service(request).add_manual_property(
            paper_id,
            paper_material_id=paper_material_id,
            action=action,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if bundle is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return bundle


@router.get(
    "/api/papers/{paper_id:path}/material-structure-candidates",
    response_model=list[MaterialStructureCandidate],
)
def list_paper_material_structure_candidates(
    request: Request,
    paper_id: str,
) -> list[MaterialStructureCandidate]:
    return _public_service(request).list_structure_candidates(paper_id)


@router.get(
    "/api/papers/{paper_id:path}/material-agent-runs",
    response_model=list[MaterialAgentRun],
)
def list_material_agent_runs(
    request: Request,
    paper_id: str,
) -> list[MaterialAgentRun]:
    runs = _agent_service(request).list_runs(paper_id)
    if runs is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return runs


@router.post(
    "/api/papers/{paper_id:path}/material-agent-runs/foundation",
    response_model=MaterialAgentFoundationResult,
)
def run_material_agent_foundation(
    request: Request,
    paper_id: str,
) -> MaterialAgentFoundationResult:
    result = _agent_service(request).run_foundation(paper_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return result


@router.get(
    "/api/papers/{paper_id:path}/document-visual-blocks",
    response_model=list[DocumentVisualBlock],
)
def list_document_visual_blocks(
    request: Request,
    paper_id: str,
) -> list[DocumentVisualBlock]:
    blocks = _agent_service(request).list_visual_blocks(paper_id)
    if blocks is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return blocks


@router.get(
    "/api/papers/{paper_id:path}/figure-triage-results",
    response_model=list[FigureTriageResult],
)
def list_figure_triage_results(
    request: Request,
    paper_id: str,
) -> list[FigureTriageResult]:
    results = _agent_service(request).list_figure_triage_results(paper_id)
    if results is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return results


@router.post(
    "/api/papers/{paper_id:path}/figure-triage",
    response_model=FigureTriageBatchResult,
)
def run_figure_triage(
    request: Request,
    paper_id: str,
    provider: str = "qwen",
    model: str | None = None,
    limit: int | None = None,
) -> FigureTriageBatchResult:
    try:
        result = _agent_service(request).run_figure_triage(
            paper_id,
            provider=provider,
            model=model,
            limit=limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return result


@router.get(
    "/api/papers/{paper_id:path}/molecule-crops",
    response_model=list[MoleculeCrop],
)
def list_molecule_crops(
    request: Request,
    paper_id: str,
) -> list[MoleculeCrop]:
    crops = _agent_service(request).list_molecule_crops(paper_id)
    if crops is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return crops


@router.post(
    "/api/papers/{paper_id:path}/decimer-segmentation",
    response_model=DecimerSegmentationBatchResult,
)
def run_decimer_segmentation(
    request: Request,
    paper_id: str,
    limit: int | None = None,
    max_segments: int | None = None,
) -> DecimerSegmentationBatchResult:
    result = _agent_service(request).run_decimer_segmentation(
        paper_id,
        limit=limit,
        max_segments=max_segments,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return result


@router.post(
    "/api/papers/{paper_id:path}/decimer-ocsr",
    response_model=DecimerOCSRBatchResult,
)
def run_decimer_ocsr(
    request: Request,
    paper_id: str,
    limit: int | None = None,
) -> DecimerOCSRBatchResult:
    try:
        result = _agent_service(request).run_decimer_ocsr(paper_id, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return result


@router.get(
    "/api/papers/{paper_id:path}/molecule-crop-validations",
    response_model=list[MoleculeCropValidation],
)
def list_molecule_crop_validations(
    request: Request,
    paper_id: str,
) -> list[MoleculeCropValidation]:
    validations = _agent_service(request).list_crop_validations(paper_id)
    if validations is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return validations


@router.post(
    "/api/papers/{paper_id:path}/validate-molecule-crops",
    response_model=MoleculeCropValidationBatchResult,
)
def validate_molecule_crops(
    request: Request,
    paper_id: str,
    provider: str = "qwen",
    model: str | None = None,
    limit: int | None = None,
) -> MoleculeCropValidationBatchResult:
    result = _agent_service(request).run_crop_validation(
        paper_id,
        provider=provider,
        model=model,
        limit=limit,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return result


@router.get(
    "/api/papers/{paper_id:path}/molecule-label-bindings",
    response_model=list[MoleculeLabelBinding],
)
def list_molecule_label_bindings(
    request: Request,
    paper_id: str,
) -> list[MoleculeLabelBinding]:
    bindings = _agent_service(request).list_label_bindings(paper_id)
    if bindings is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return bindings


@router.post(
    "/api/papers/{paper_id:path}/bind-molecule-labels",
    response_model=MoleculeLabelBindingBatchResult,
)
def bind_molecule_labels(
    request: Request,
    paper_id: str,
    provider: str = "qwen",
    model: str | None = None,
    limit: int | None = None,
) -> MoleculeLabelBindingBatchResult:
    try:
        result = _agent_service(request).run_label_binding(
            paper_id,
            provider=provider,
            model=model,
            limit=limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return result


@router.get(
    "/api/papers/{paper_id:path}/molecule-label-binding-review-events",
    response_model=list[MoleculeLabelBindingReviewEvent],
)
def list_molecule_label_binding_review_events(
    request: Request,
    paper_id: str,
) -> list[MoleculeLabelBindingReviewEvent]:
    events = _agent_service(request).list_label_binding_review_events(paper_id)
    if events is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return events


@router.get(
    "/api/papers/{paper_id:path}/vlm-call-logs",
    response_model=list[VLMCallLog],
)
def list_vlm_call_logs(
    request: Request,
    paper_id: str,
    limit: int = 100,
) -> list[VLMCallLog]:
    calls = _agent_service(request).list_vlm_call_logs(paper_id, limit=limit)
    if calls is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return calls


@router.post(
    "/api/molecule-label-bindings/{binding_id}/review",
    response_model=MoleculeLabelBinding,
)
def review_molecule_label_binding(
    request: Request,
    binding_id: str,
    action: MoleculeLabelBindingReviewAction,
) -> MoleculeLabelBinding:
    try:
        binding = _agent_service(request).review_label_binding(binding_id, action)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if binding is None:
        raise HTTPException(status_code=404, detail="Molecule label binding not found")
    return binding


@router.post(
    "/api/molecule-label-binding-review-events/{event_id}/undo",
    response_model=MoleculeLabelBinding,
)
def undo_molecule_label_binding_review_event(
    request: Request,
    event_id: str,
    action: MoleculeLabelBindingReviewAction,
) -> MoleculeLabelBinding:
    binding = _agent_service(request).undo_label_binding_review_event(event_id, action)
    if binding is None:
        raise HTTPException(status_code=404, detail="Molecule label binding event not found")
    return binding


@router.get("/api/document-visual-blocks/{visual_block_id}/image")
def get_document_visual_block_image(
    request: Request,
    visual_block_id: str,
) -> FileResponse:
    image_path = _agent_service(request).get_visual_block_image_path(visual_block_id)
    if not image_path:
        raise HTTPException(status_code=404, detail="Document visual block image not found")
    return FileResponse(image_path, headers={"Cache-Control": "no-store, max-age=0"})


@router.get("/api/molecule-crops/{crop_id}/image")
def get_molecule_crop_image(
    request: Request,
    crop_id: str,
) -> FileResponse:
    image_path = _agent_service(request).get_molecule_crop_image_path(crop_id)
    if not image_path:
        raise HTTPException(status_code=404, detail="Molecule crop image not found")
    return FileResponse(image_path, headers={"Cache-Control": "no-store, max-age=0"})


@router.get("/api/molecule-label-bindings/{binding_id}/highlighted-image")
def get_molecule_label_binding_highlighted_image(
    request: Request,
    binding_id: str,
) -> FileResponse:
    image_path = _agent_service(request).get_label_binding_highlighted_image_path(binding_id)
    if not image_path:
        raise HTTPException(status_code=404, detail="Highlighted binding image not found")
    return FileResponse(image_path, headers={"Cache-Control": "no-store, max-age=0"})


@router.get(
    "/api/papers/{paper_id:path}/chemical-figures",
    response_model=list[ChemicalFigureBlock],
)
def list_paper_chemical_figures(
    request: Request,
    paper_id: str,
) -> list[ChemicalFigureBlock]:
    blocks = _figure_service(request).list_for_paper(paper_id)
    if blocks is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return blocks


@router.post(
    "/api/papers/{paper_id:path}/collect-chemical-figures",
    response_model=list[ChemicalFigureBlock],
)
def collect_paper_chemical_figures(
    request: Request,
    paper_id: str,
) -> list[ChemicalFigureBlock]:
    blocks = _figure_service(request).collect_for_paper(paper_id)
    if blocks is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return blocks


@router.get("/api/chemical-figure-blocks/{figure_block_id}/image")
def get_chemical_figure_image(
    request: Request,
    figure_block_id: str,
) -> FileResponse:
    image_path = _figure_service(request).get_image_path(figure_block_id)
    if not image_path:
        raise HTTPException(status_code=404, detail="Chemical figure image not found")
    return FileResponse(image_path)


@router.post(
    "/api/papers/{paper_id:path}/manual-material-structure",
    response_model=PaperMaterialStructureBundle,
)
def save_manual_material_structure(
    request: Request,
    paper_id: str,
    paper_material_id: str,
    action: MaterialManualStructureAction,
) -> PaperMaterialStructureBundle:
    try:
        bundle = _review_service(request).save_manual_structure(
            paper_id,
            paper_material_id,
            action,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if bundle is None:
        raise HTTPException(status_code=404, detail="Paper or material not found")
    return bundle


@router.post(
    "/api/material-structure-candidates/{structure_candidate_id}/accept",
    response_model=PaperMaterialStructureBundle,
)
def accept_material_structure_candidate(
    request: Request,
    structure_candidate_id: str,
    action: MaterialReviewAction = MaterialReviewAction(),
) -> PaperMaterialStructureBundle:
    try:
        bundle = _review_service(request).accept_structure_candidate(
            structure_candidate_id,
            action,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if bundle is None:
        raise HTTPException(status_code=404, detail="Material structure candidate not found")
    return bundle


@router.post(
    "/api/material-structure-candidates/{structure_candidate_id}/edit-smiles",
    response_model=PaperMaterialStructureBundle,
)
def edit_material_structure_candidate_smiles(
    request: Request,
    structure_candidate_id: str,
    action: MaterialStructureEditAction,
) -> PaperMaterialStructureBundle:
    try:
        bundle = _review_service(request).correct_structure_candidate(
            structure_candidate_id,
            action,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if bundle is None:
        raise HTTPException(status_code=404, detail="Material structure candidate not found")
    return bundle


@router.get("/api/material-structure-candidates/{structure_candidate_id}/depiction.svg")
def get_material_structure_candidate_depiction(
    request: Request,
    structure_candidate_id: str,
) -> Response:
    try:
        svg = _review_service(request).get_structure_candidate_depiction_svg(structure_candidate_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if svg is None:
        raise HTTPException(status_code=404, detail="Material structure candidate not found")
    return Response(
        content=svg,
        media_type="image/svg+xml",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@router.get("/api/materials-global/{global_material_id}/depiction.svg")
def get_global_material_depiction(
    request: Request,
    global_material_id: str,
) -> Response:
    try:
        svg = _review_service(request).get_global_material_depiction_svg(global_material_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if svg is None:
        raise HTTPException(status_code=404, detail="Global material not found")
    return Response(
        content=svg,
        media_type="image/svg+xml",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@router.post(
    "/api/material-structure-candidates/{structure_candidate_id}/reject",
    response_model=PaperMaterialStructureBundle,
)
def reject_material_structure_candidate(
    request: Request,
    structure_candidate_id: str,
    action: MaterialReviewAction = MaterialReviewAction(),
) -> PaperMaterialStructureBundle:
    bundle = _review_service(request).reject_structure_candidate(
        structure_candidate_id,
        action,
    )
    if bundle is None:
        raise HTTPException(status_code=404, detail="Material structure candidate not found")
    return bundle


@router.post(
    "/api/material-review-events/{event_id}/undo",
    response_model=PaperMaterialStructureBundle,
)
def undo_material_review_event(
    request: Request,
    event_id: str,
    action: MaterialReviewAction = MaterialReviewAction(),
) -> PaperMaterialStructureBundle:
    bundle = _review_service(request).undo_material_review_event(event_id, action)
    if bundle is None:
        raise HTTPException(status_code=404, detail="Material review event not found")
    return bundle
