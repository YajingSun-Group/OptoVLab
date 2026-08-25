from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from evolab_local.mining_platform.candidate_ingestion_service import CandidateIngestionService
from evolab_local.mining_platform.candidate_review_service import CandidateReviewService
from evolab_local.mining_platform.schemas.candidate import CandidateFieldUpdate, CandidateFieldValue
from evolab_local.mining_platform.schemas.candidate_ingestion import (
    CandidateFinalConfirmResult,
    CandidateFinalRecord,
    CandidateReviewV2Bundle,
    CandidateValue,
    CandidateValueReviewEvent,
    CandidateValueUpdate,
)
from evolab_local.mining_platform.schemas.evidence import EvidenceAnchor
from evolab_local.mining_platform.schemas.final_device import ConfirmPaperResult, OledDeviceFinal
from evolab_local.mining_platform.schemas.review import ReviewAction

router = APIRouter(tags=["candidate-review"])


def _service(request: Request) -> CandidateReviewService:
    return request.app.state.candidate_review_service


def _v2_service(request: Request) -> CandidateIngestionService:
    return request.app.state.candidate_ingestion_service


@router.get(
    "/api/papers/{paper_id:path}/candidate-v2",
    response_model=CandidateReviewV2Bundle,
)
def get_candidate_review_v2(
    request: Request,
    paper_id: str,
    compact: bool = Query(default=False),
) -> CandidateReviewV2Bundle:
    bundle = _v2_service(request).get_review_bundle(paper_id, compact=compact)
    if bundle is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return bundle


@router.put("/api/candidate-values/{candidate_value_id}", response_model=CandidateValue)
def update_candidate_value(
    request: Request,
    candidate_value_id: str,
    payload: CandidateValueUpdate,
) -> CandidateValue:
    value = _v2_service(request).update_candidate_value(candidate_value_id, payload)
    if not value:
        raise HTTPException(status_code=404, detail="Candidate value not found")
    return value


@router.post("/api/candidate-values/{candidate_value_id}/confirm", response_model=CandidateValue)
def confirm_candidate_value(
    request: Request,
    candidate_value_id: str,
    action: ReviewAction = ReviewAction(),
) -> CandidateValue:
    value = _v2_service(request).set_candidate_value_status(
        candidate_value_id,
        "accepted",
        actor=action.actor,
        message=action.message,
    )
    if not value:
        raise HTTPException(status_code=404, detail="Candidate value not found")
    return value


@router.post("/api/candidate-values/{candidate_value_id}/reject", response_model=CandidateValue)
def reject_candidate_value(
    request: Request,
    candidate_value_id: str,
    action: ReviewAction = ReviewAction(),
) -> CandidateValue:
    value = _v2_service(request).set_candidate_value_status(
        candidate_value_id,
        "rejected",
        actor=action.actor,
        message=action.message,
    )
    if not value:
        raise HTTPException(status_code=404, detail="Candidate value not found")
    return value


@router.get(
    "/api/candidate-values/{candidate_value_id}/review-events",
    response_model=list[CandidateValueReviewEvent],
)
def list_candidate_value_review_events(
    request: Request,
    candidate_value_id: str,
) -> list[CandidateValueReviewEvent]:
    return _v2_service(request).list_value_review_events_by_value(candidate_value_id)


@router.get(
    "/api/candidate-runs/{candidate_run_id}/value-review-events",
    response_model=list[CandidateValueReviewEvent],
)
def list_candidate_run_value_review_events(
    request: Request,
    candidate_run_id: str,
) -> list[CandidateValueReviewEvent]:
    return _v2_service(request).list_value_review_events_by_run(candidate_run_id)


@router.post(
    "/api/candidate-value-review-events/{event_id}/undo",
    response_model=CandidateValue,
)
def undo_candidate_value_review_event(
    request: Request,
    event_id: str,
    action: ReviewAction = ReviewAction(),
) -> CandidateValue:
    value = _v2_service(request).undo_value_review_event(
        event_id,
        actor=action.actor,
        message=action.message,
    )
    if not value:
        raise HTTPException(status_code=404, detail="Candidate value review event not found")
    return value


@router.post(
    "/api/papers/{paper_id:path}/confirm-review-v2",
    response_model=CandidateFinalConfirmResult,
)
def confirm_paper_review_v2(
    request: Request,
    paper_id: str,
    action: ReviewAction = ReviewAction(),
) -> CandidateFinalConfirmResult:
    result = _v2_service(request).confirm_review_v2(paper_id, actor=action.actor)
    if not result:
        raise HTTPException(status_code=404, detail="Candidate v2 run not found")
    return result


@router.get(
    "/api/papers/{paper_id:path}/final-candidates",
    response_model=list[CandidateFinalRecord],
)
def list_final_candidate_records(request: Request, paper_id: str) -> list[CandidateFinalRecord]:
    records = _v2_service(request).list_final_records(paper_id)
    if records is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return records


@router.get(
    "/api/papers/{paper_id:path}/candidate-fields",
    response_model=list[CandidateFieldValue],
)
def list_candidate_fields(request: Request, paper_id: str) -> list[CandidateFieldValue]:
    return _service(request).list_candidate_fields(paper_id)


@router.put("/api/candidate-fields/{candidate_field_id}", response_model=CandidateFieldValue)
def update_candidate_field(
    request: Request,
    candidate_field_id: str,
    payload: CandidateFieldUpdate,
) -> CandidateFieldValue:
    field = _service(request).update_candidate_field(candidate_field_id, payload)
    if not field:
        raise HTTPException(status_code=404, detail="Candidate field not found")
    return field


@router.post(
    "/api/candidate-fields/{candidate_field_id}/confirm",
    response_model=CandidateFieldValue,
)
def confirm_candidate_field(
    request: Request,
    candidate_field_id: str,
    action: ReviewAction = ReviewAction(),
) -> CandidateFieldValue:
    field = _service(request).set_candidate_field_status(candidate_field_id, "confirmed", action)
    if not field:
        raise HTTPException(status_code=404, detail="Candidate field not found")
    return field


@router.post(
    "/api/candidate-fields/{candidate_field_id}/reject",
    response_model=CandidateFieldValue,
)
def reject_candidate_field(
    request: Request,
    candidate_field_id: str,
    action: ReviewAction = ReviewAction(),
) -> CandidateFieldValue:
    field = _service(request).set_candidate_field_status(candidate_field_id, "rejected", action)
    if not field:
        raise HTTPException(status_code=404, detail="Candidate field not found")
    return field


@router.get(
    "/api/papers/{paper_id:path}/evidence-anchors",
    response_model=list[EvidenceAnchor],
)
def list_evidence_anchors(request: Request, paper_id: str) -> list[EvidenceAnchor]:
    return _service(request).list_evidence_anchors(paper_id)


@router.get("/api/evidence-anchors/{evidence_anchor_id}", response_model=EvidenceAnchor)
def get_evidence_anchor(request: Request, evidence_anchor_id: str) -> EvidenceAnchor:
    anchor = _service(request).get_evidence_anchor(evidence_anchor_id)
    if not anchor:
        raise HTTPException(status_code=404, detail="Evidence anchor not found")
    return anchor


@router.post("/api/papers/{paper_id:path}/confirm-review", response_model=ConfirmPaperResult)
def confirm_paper_review(
    request: Request,
    paper_id: str,
    action: ReviewAction = ReviewAction(),
) -> ConfirmPaperResult:
    result = _service(request).confirm_paper(paper_id, action)
    if not result:
        raise HTTPException(status_code=404, detail="Paper not found")
    return result


@router.get("/api/papers/{paper_id:path}/final-devices", response_model=list[OledDeviceFinal])
def list_final_devices(request: Request, paper_id: str) -> list[OledDeviceFinal]:
    return _service(request).list_final_devices(paper_id)
