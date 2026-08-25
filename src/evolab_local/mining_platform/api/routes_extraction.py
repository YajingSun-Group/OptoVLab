from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from evolab_local.mining_platform.extraction_service import ExtractionService
from evolab_local.mining_platform.schemas.device_record import DeviceRecordReviewed
from evolab_local.mining_platform.schemas.extraction import (
    DeviceRecordRaw,
    ExtractionResult,
    ExtractionRun,
)
from evolab_local.mining_platform.schemas.review import ReviewAction

router = APIRouter(tags=["extraction"])


def _service(request: Request) -> ExtractionService:
    return request.app.state.extraction_service


@router.post("/api/papers/{paper_id:path}/extract-oled", response_model=ExtractionResult)
def extract_oled(request: Request, paper_id: str) -> ExtractionResult:
    result = _service(request).extract_oled(paper_id)
    if not result:
        raise HTTPException(status_code=404, detail="Paper not found")
    return result


@router.get("/api/papers/{paper_id:path}/extraction-runs", response_model=list[ExtractionRun])
def list_extraction_runs(request: Request, paper_id: str) -> list[ExtractionRun]:
    return _service(request).list_runs(paper_id)


@router.get(
    "/api/papers/{paper_id:path}/raw-device-records",
    response_model=list[DeviceRecordRaw],
)
def list_raw_device_records(request: Request, paper_id: str) -> list[DeviceRecordRaw]:
    return _service(request).list_raw_device_records(paper_id)


@router.get("/api/raw-device-records/{raw_record_id}", response_model=DeviceRecordRaw)
def get_raw_device_record(request: Request, raw_record_id: str) -> DeviceRecordRaw:
    record = _service(request).get_raw_device_record(raw_record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Raw device record not found")
    return record


@router.post(
    "/api/raw-device-records/{raw_record_id}/accept",
    response_model=DeviceRecordReviewed,
)
def accept_raw_device_record(
    request: Request,
    raw_record_id: str,
    action: ReviewAction = ReviewAction(),
) -> DeviceRecordReviewed:
    record = _service(request).accept_raw_device_record(raw_record_id, action)
    if not record:
        raise HTTPException(status_code=404, detail="Raw device record not found")
    return record


@router.post("/api/raw-device-records/{raw_record_id}/reject", response_model=DeviceRecordRaw)
def reject_raw_device_record(
    request: Request,
    raw_record_id: str,
    action: ReviewAction = ReviewAction(),
) -> DeviceRecordRaw:
    record = _service(request).reject_raw_device_record(raw_record_id, action)
    if not record:
        raise HTTPException(status_code=404, detail="Raw device record not found")
    return record
