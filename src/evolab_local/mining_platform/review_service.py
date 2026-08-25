from __future__ import annotations

import json

from evolab_local.mining_platform.core.config import MiningPlatformConfig
from evolab_local.mining_platform.library.paper_service import PaperService
from evolab_local.mining_platform.schemas.device_record import (
    DeviceRecordCreate,
    DeviceRecordReviewed,
    DeviceRecordUpdate,
)
from evolab_local.mining_platform.schemas.review import ReviewAction, ReviewEvent
from evolab_local.mining_platform.storage.database import Database
from evolab_local.mining_platform.storage.repositories import (
    DeviceRecordRepository,
    ReviewEventRepository,
)


def _record_json(record: DeviceRecordReviewed | None) -> str | None:
    if not record:
        return None
    return json.dumps(record.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)


class ReviewService:
    def __init__(self, config: MiningPlatformConfig) -> None:
        self.config = config
        self.database = Database(config.paths.sqlite_path)
        self.paper_service = PaperService(config)
        self.device_records = DeviceRecordRepository(self.database)
        self.review_events = ReviewEventRepository(self.database)

    def init_runtime(self) -> None:
        self.paper_service.init_runtime()

    def list_device_records(self, paper_id: str) -> list[DeviceRecordReviewed]:
        self.init_runtime()
        normalized_paper_id = self.paper_service.normalize_paper_id(paper_id)
        return self.device_records.list_by_paper(normalized_paper_id)

    def create_device_record(
        self, paper_id: str, payload: DeviceRecordCreate
    ) -> DeviceRecordReviewed | None:
        self.init_runtime()
        normalized_paper_id = self.paper_service.normalize_paper_id(paper_id)
        if not self.paper_service.get_paper(normalized_paper_id):
            return None
        record = self.device_records.create(normalized_paper_id, payload)
        self.review_events.add(
            paper_id=normalized_paper_id,
            record_id=record.record_id,
            event_type="created",
            actor=payload.actor,
            message=payload.message,
            after_json=_record_json(record),
        )
        return record

    def get_device_record(self, record_id: str) -> DeviceRecordReviewed | None:
        self.init_runtime()
        return self.device_records.get(record_id)

    def update_device_record(
        self, record_id: str, payload: DeviceRecordUpdate
    ) -> DeviceRecordReviewed | None:
        self.init_runtime()
        before = self.device_records.get(record_id)
        if not before:
            return None
        updated = self.device_records.update(record_id, payload)
        if not updated:
            return None
        self.review_events.add(
            paper_id=updated.paper_id,
            record_id=updated.record_id,
            event_type="updated",
            actor=payload.actor,
            message=payload.message,
            before_json=_record_json(before),
            after_json=_record_json(updated),
        )
        return updated

    def set_device_record_status(
        self, record_id: str, status: str, action: ReviewAction
    ) -> DeviceRecordReviewed | None:
        self.init_runtime()
        before = self.device_records.get(record_id)
        if not before:
            return None
        updated = self.device_records.set_review_status(record_id, status)
        if not updated:
            return None
        self.review_events.add(
            paper_id=updated.paper_id,
            record_id=updated.record_id,
            event_type=status,
            actor=action.actor,
            message=action.message,
            before_json=_record_json(before),
            after_json=_record_json(updated),
        )
        return updated

    def list_review_events(self, paper_id: str) -> list[ReviewEvent]:
        self.init_runtime()
        normalized_paper_id = self.paper_service.normalize_paper_id(paper_id)
        return self.review_events.list_by_paper(normalized_paper_id)
