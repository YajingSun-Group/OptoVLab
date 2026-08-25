from __future__ import annotations

import json
from collections import defaultdict
from uuid import uuid4

from evolab_local.mining_platform.core.config import MiningPlatformConfig
from evolab_local.mining_platform.library.paper_service import PaperService
from evolab_local.mining_platform.schemas.candidate import (
    CandidateFieldUpdate,
    CandidateFieldValue,
    CandidateSeedResult,
)
from evolab_local.mining_platform.schemas.evidence import EvidenceAnchor
from evolab_local.mining_platform.schemas.extraction import DeviceRecordRaw, FieldEvidence
from evolab_local.mining_platform.schemas.final_device import ConfirmPaperResult, OledDeviceFinal
from evolab_local.mining_platform.schemas.review import ReviewAction
from evolab_local.mining_platform.storage.database import Database, now_iso
from evolab_local.mining_platform.storage.repositories import (
    CandidateFieldValueRepository,
    DeviceRecordRawRepository,
    EvidenceAnchorRepository,
    OledDeviceFinalRepository,
    PaperRepository,
    ReviewEventRepository,
)


CANDIDATE_FIELD_LABELS: dict[str, str] = {
    "device_label": "Device label",
    "architecture": "Architecture",
    "notes": "Notes",
    "substrate": "Substrate",
    "anode": "Anode",
    "hil": "HIL",
    "htl": "HTL",
    "ebl": "EBL",
    "eml_host": "EML host",
    "eml_dopant": "EML dopant",
    "eml_emitter": "EML emitter",
    "hbl": "HBL",
    "etl": "ETL",
    "eil": "EIL",
    "cathode": "Cathode",
    "layer_thicknesses": "Layer thicknesses",
    "eqe_max": "EQE max",
    "ce_max": "Current efficiency max",
    "pe_max": "Power efficiency max",
    "luminance_max": "Luminance max",
    "turn_on_voltage": "Turn-on voltage",
    "cie_x": "CIE x",
    "cie_y": "CIE y",
    "el_peak": "EL peak",
    "fwhm": "FWHM",
    "lifetime": "Lifetime",
}

CANDIDATE_FIELD_NAMES = tuple(CANDIDATE_FIELD_LABELS)


def _model_json(value: object | None) -> str | None:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return json.dumps(value.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


class CandidateReviewService:
    def __init__(self, config: MiningPlatformConfig) -> None:
        self.config = config
        self.database = Database(config.paths.sqlite_path)
        self.paper_service = PaperService(config)
        self.papers = PaperRepository(self.database)
        self.raw_records = DeviceRecordRawRepository(self.database)
        self.anchors = EvidenceAnchorRepository(self.database)
        self.candidate_fields = CandidateFieldValueRepository(self.database)
        self.final_devices = OledDeviceFinalRepository(self.database)
        self.review_events = ReviewEventRepository(self.database)

    def init_runtime(self) -> None:
        self.paper_service.init_runtime()

    def list_candidate_fields(self, paper_id: str) -> list[CandidateFieldValue]:
        self.init_runtime()
        normalized_paper_id = self.paper_service.normalize_paper_id(paper_id)
        return self.candidate_fields.list_by_paper(normalized_paper_id)

    def update_candidate_field(
        self, candidate_field_id: str, payload: CandidateFieldUpdate
    ) -> CandidateFieldValue | None:
        self.init_runtime()
        before = self.candidate_fields.get(candidate_field_id)
        if not before:
            return None
        updated = self.candidate_fields.update(
            candidate_field_id,
            payload.reviewed_value,
            payload.field_status,
        )
        if not updated:
            return None
        event_type = (
            f"candidate_field_{updated.field_status}"
            if payload.field_status in {"confirmed", "rejected"}
            else "candidate_field_updated"
        )
        self.review_events.add(
            paper_id=updated.paper_id,
            record_id=updated.record_id,
            event_type=event_type,
            actor=payload.actor,
            message=payload.message,
            before_json=_model_json(before),
            after_json=_model_json(updated),
        )
        return updated

    def set_candidate_field_status(
        self,
        candidate_field_id: str,
        status: str,
        action: ReviewAction,
    ) -> CandidateFieldValue | None:
        self.init_runtime()
        before = self.candidate_fields.get(candidate_field_id)
        if not before:
            return None
        updated = self.candidate_fields.set_status(candidate_field_id, status)
        if not updated:
            return None
        self.review_events.add(
            paper_id=updated.paper_id,
            record_id=updated.record_id,
            event_type=f"candidate_field_{status}",
            actor=action.actor,
            message=action.message,
            before_json=_model_json(before),
            after_json=_model_json(updated),
        )
        return updated

    def list_evidence_anchors(self, paper_id: str) -> list[EvidenceAnchor]:
        self.init_runtime()
        normalized_paper_id = self.paper_service.normalize_paper_id(paper_id)
        return self.anchors.list_by_paper(normalized_paper_id)

    def get_evidence_anchor(self, evidence_anchor_id: str) -> EvidenceAnchor | None:
        self.init_runtime()
        return self.anchors.get(evidence_anchor_id)

    def list_final_devices(self, paper_id: str) -> list[OledDeviceFinal]:
        self.init_runtime()
        normalized_paper_id = self.paper_service.normalize_paper_id(paper_id)
        return self.final_devices.list_by_paper(normalized_paper_id)

    def seed_candidate_fields_from_raw(self, paper_id: str) -> CandidateSeedResult | None:
        self.init_runtime()
        normalized_paper_id = self.paper_service.normalize_paper_id(paper_id)
        if not self.paper_service.get_paper(normalized_paper_id):
            return None

        raw_records = self.raw_records.list_by_paper(normalized_paper_id)
        timestamp = now_iso()
        anchors: list[EvidenceAnchor] = []
        fields: list[CandidateFieldValue] = []

        self.candidate_fields.delete_by_paper(normalized_paper_id)
        self.anchors.delete_by_paper(normalized_paper_id)

        for raw in raw_records:
            extractor_name = self._raw_payload_str(raw, "extractor")
            extractor_version = self._raw_payload_str(raw, "version")
            for field_name in CANDIDATE_FIELD_NAMES:
                value = getattr(raw, field_name)
                if value is None or str(value).strip() == "":
                    continue
                evidence = raw.field_evidence.get(field_name)
                anchor = self._anchor_from_raw(raw, field_name, evidence, timestamp)
                anchors.append(anchor)
                fields.append(
                    CandidateFieldValue(
                        candidate_field_id=uuid4().hex,
                        paper_id=normalized_paper_id,
                        record_id=raw.raw_record_id,
                        field_name=field_name,
                        field_label=CANDIDATE_FIELD_LABELS[field_name],
                        mined_value=str(value),
                        reviewed_value=str(value),
                        confidence=self._confidence_score(raw),
                        confidence_json=raw.confidence,
                        evidence_anchor_id=anchor.evidence_anchor_id,
                        extractor_name=extractor_name,
                        extractor_version=extractor_version,
                        field_status="pending",
                        created_at=timestamp,
                        updated_at=timestamp,
                    )
                )

        self.anchors.add_many(anchors)
        self.candidate_fields.add_many(fields)
        if fields:
            self.papers.set_review_status(normalized_paper_id, "needs_review")
        return CandidateSeedResult(
            paper_id=normalized_paper_id,
            raw_record_count=len(raw_records),
            field_count=len(fields),
            evidence_count=len(anchors),
        )

    def confirm_paper(
        self, paper_id: str, action: ReviewAction = ReviewAction()
    ) -> ConfirmPaperResult | None:
        self.init_runtime()
        normalized_paper_id = self.paper_service.normalize_paper_id(paper_id)
        if not self.paper_service.get_paper(normalized_paper_id):
            return None

        fields = self.candidate_fields.list_by_paper(normalized_paper_id)
        grouped: dict[str, list[CandidateFieldValue]] = defaultdict(list)
        for field in fields:
            if field.field_status != "rejected":
                grouped[field.record_id].append(field)

        timestamp = now_iso()
        final_records: list[OledDeviceFinal] = []
        for record_id, record_fields in grouped.items():
            values: dict[str, object] = {}
            source_candidate_ids: list[str] = []
            evidence_text: str | None = None
            evidence_page: int | None = None
            for field in record_fields:
                value = (
                    field.reviewed_value if field.reviewed_value is not None else field.mined_value
                )
                if value is None or str(value).strip() == "":
                    continue
                values[field.field_name] = value
                source_candidate_ids.append(field.candidate_field_id)
                if field.evidence_anchor_id and evidence_text is None:
                    anchor = self.anchors.get(field.evidence_anchor_id)
                    if anchor:
                        evidence_text = anchor.source_text
                        evidence_page = anchor.page_id
            if not values:
                continue
            values.setdefault("evidence_text", evidence_text)
            values.setdefault("evidence_page", evidence_page)
            final_records.append(
                OledDeviceFinal(
                    final_device_id=uuid4().hex,
                    paper_id=normalized_paper_id,
                    source_candidate_ids=source_candidate_ids,
                    confirmed_by=action.actor,
                    confirmed_at=timestamp,
                    created_at=timestamp,
                    updated_at=timestamp,
                    **values,
                )
            )

        stored = self.final_devices.replace_for_paper(normalized_paper_id, final_records)
        if stored:
            self.papers.set_review_status(normalized_paper_id, "confirmed")
        self.review_events.add(
            paper_id=normalized_paper_id,
            event_type="paper_confirmed",
            actor=action.actor,
            message=action.message,
            after_json=json.dumps(
                [record.model_dump(mode="json") for record in stored],
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
        return ConfirmPaperResult(
            paper_id=normalized_paper_id,
            final_devices=stored,
            final_count=len(stored),
        )

    def _anchor_from_raw(
        self,
        raw: DeviceRecordRaw,
        field_name: str,
        evidence: FieldEvidence | None,
        timestamp: str,
    ) -> EvidenceAnchor:
        block_id = evidence.block_ids[0] if evidence and evidence.block_ids else None
        page_id = evidence.page_id if evidence else raw.evidence_page
        source_text = evidence.source_text if evidence else raw.evidence_text
        return EvidenceAnchor(
            evidence_anchor_id=uuid4().hex,
            paper_id=raw.paper_id,
            page_id=page_id,
            block_id=block_id,
            bbox=[],
            source_text=source_text,
            source_type=f"field:{field_name}",
            created_at=timestamp,
        )

    @staticmethod
    def _confidence_score(raw: DeviceRecordRaw) -> float | None:
        score = raw.confidence.get("score")
        if isinstance(score, (int, float)):
            return float(score)
        return None

    @staticmethod
    def _raw_payload_str(raw: DeviceRecordRaw, key: str) -> str | None:
        value = raw.raw_payload.get(key)
        return value if isinstance(value, str) else None
