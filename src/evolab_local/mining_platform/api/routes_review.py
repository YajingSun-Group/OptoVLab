from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from evolab_local.mining_platform.review_service import ReviewService
from evolab_local.mining_platform.schemas.device_record import (
    DeviceRecordCreate,
    DeviceRecordReviewed,
    DeviceRecordUpdate,
)
from evolab_local.mining_platform.schemas.review import ReviewAction, ReviewEvent

router = APIRouter(tags=["review"])


def _service(request: Request) -> ReviewService:
    return request.app.state.review_service


@router.get("/api/papers/{paper_id:path}/device-records", response_model=list[DeviceRecordReviewed])
def list_device_records(request: Request, paper_id: str) -> list[DeviceRecordReviewed]:
    return _service(request).list_device_records(paper_id)


@router.post("/api/papers/{paper_id:path}/device-records", response_model=DeviceRecordReviewed)
def create_device_record(
    request: Request,
    paper_id: str,
    payload: DeviceRecordCreate,
) -> DeviceRecordReviewed:
    record = _service(request).create_device_record(paper_id, payload)
    if not record:
        raise HTTPException(status_code=404, detail="Paper not found")
    return record


@router.get("/api/papers/{paper_id:path}/review-events", response_model=list[ReviewEvent])
def list_review_events(request: Request, paper_id: str) -> list[ReviewEvent]:
    return _service(request).list_review_events(paper_id)


@router.get("/api/device-records/{record_id}", response_model=DeviceRecordReviewed)
def get_device_record(request: Request, record_id: str) -> DeviceRecordReviewed:
    record = _service(request).get_device_record(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Device record not found")
    return record


@router.put("/api/device-records/{record_id}", response_model=DeviceRecordReviewed)
def update_device_record(
    request: Request,
    record_id: str,
    payload: DeviceRecordUpdate,
) -> DeviceRecordReviewed:
    record = _service(request).update_device_record(record_id, payload)
    if not record:
        raise HTTPException(status_code=404, detail="Device record not found")
    return record


@router.post("/api/device-records/{record_id}/confirm", response_model=DeviceRecordReviewed)
def confirm_device_record(
    request: Request,
    record_id: str,
    action: ReviewAction = ReviewAction(),
) -> DeviceRecordReviewed:
    record = _service(request).set_device_record_status(record_id, "confirmed", action)
    if not record:
        raise HTTPException(status_code=404, detail="Device record not found")
    return record


@router.post("/api/device-records/{record_id}/reject", response_model=DeviceRecordReviewed)
def reject_device_record(
    request: Request,
    record_id: str,
    action: ReviewAction = ReviewAction(),
) -> DeviceRecordReviewed:
    record = _service(request).set_device_record_status(record_id, "rejected", action)
    if not record:
        raise HTTPException(status_code=404, detail="Device record not found")
    return record
