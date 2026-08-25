from __future__ import annotations

from evolab_local.mining_platform.core.config import MiningPlatformConfig
from evolab_local.mining_platform.extractors.oled_rules import OledRuleBasedExtractor
from evolab_local.mining_platform.library.paper_service import PaperService
from evolab_local.mining_platform.parse_service import ParseService
from evolab_local.mining_platform.review_service import ReviewService
from evolab_local.mining_platform.schemas.device_record import (
    DeviceRecordCreate,
    DeviceRecordReviewed,
)
from evolab_local.mining_platform.schemas.extraction import (
    DeviceRecordRaw,
    ExtractionResult,
    ExtractionRun,
)
from evolab_local.mining_platform.schemas.review import ReviewAction
from evolab_local.mining_platform.storage.database import Database
from evolab_local.mining_platform.storage.repositories import (
    DeviceRecordRawRepository,
    DeviceRecordRepository,
    DocumentBlockRepository,
    ExtractionRunRepository,
    PaperRepository,
    ReviewEventRepository,
)


class ExtractionService:
    def __init__(self, config: MiningPlatformConfig) -> None:
        self.config = config
        self.database = Database(config.paths.sqlite_path)
        self.paper_service = PaperService(config)
        self.parse_service = ParseService(config)
        self.review_service = ReviewService(config)
        self.papers = PaperRepository(self.database)
        self.blocks = DocumentBlockRepository(self.database)
        self.runs = ExtractionRunRepository(self.database)
        self.raw_records = DeviceRecordRawRepository(self.database)
        self.reviewed_records = DeviceRecordRepository(self.database)
        self.review_events = ReviewEventRepository(self.database)
        self.oled_extractor = OledRuleBasedExtractor()

    def init_runtime(self) -> None:
        self.paper_service.init_runtime()

    def extract_oled(self, paper_id: str) -> ExtractionResult | None:
        self.init_runtime()
        normalized_paper_id = self.paper_service.normalize_paper_id(paper_id)
        if not self.paper_service.get_paper(normalized_paper_id):
            return None

        blocks = self.blocks.list_by_paper(normalized_paper_id)
        if not blocks:
            self.parse_service.parse_paper(normalized_paper_id)
            blocks = self.blocks.list_by_paper(normalized_paper_id)

        self.papers.set_mining_status(normalized_paper_id, "extracting")
        run = self.runs.create(
            paper_id=normalized_paper_id,
            extractor_name=self.oled_extractor.name,
            extractor_version=self.oled_extractor.version,
            input_block_count=len(blocks),
        )
        try:
            candidates = self.oled_extractor.extract(normalized_paper_id, blocks)
            raw_records = self.raw_records.add_many(run.run_id, normalized_paper_id, candidates)
            completed_run = self.runs.complete(run.run_id, len(raw_records))
            self.papers.set_mining_status(normalized_paper_id, "mined")
            if not completed_run:
                return None
            return ExtractionResult(run=completed_run, raw_records=raw_records)
        except Exception as exc:
            failed_run = self.runs.fail(run.run_id, str(exc))
            self.papers.set_mining_status(normalized_paper_id, "failed")
            if not failed_run:
                raise
            raise

    def list_runs(self, paper_id: str) -> list[ExtractionRun]:
        self.init_runtime()
        normalized_paper_id = self.paper_service.normalize_paper_id(paper_id)
        return self.runs.list_by_paper(normalized_paper_id)

    def list_raw_device_records(self, paper_id: str) -> list[DeviceRecordRaw]:
        self.init_runtime()
        normalized_paper_id = self.paper_service.normalize_paper_id(paper_id)
        return self.raw_records.list_by_paper(normalized_paper_id)

    def get_raw_device_record(self, raw_record_id: str) -> DeviceRecordRaw | None:
        self.init_runtime()
        return self.raw_records.get(raw_record_id)

    def accept_raw_device_record(
        self,
        raw_record_id: str,
        action: ReviewAction,
    ) -> DeviceRecordReviewed | None:
        self.init_runtime()
        raw = self.raw_records.get(raw_record_id)
        if not raw:
            return None
        if raw.reviewed_record_id:
            return self.reviewed_records.get(raw.reviewed_record_id)

        payload = DeviceRecordCreate(
            **self._reviewed_payload_from_raw(raw),
            actor=action.actor,
            message=action.message or "Accepted raw extraction candidate",
        )
        reviewed = self.review_service.create_device_record(raw.paper_id, payload)
        if not reviewed:
            return None
        self.raw_records.set_review_status(raw_record_id, "accepted", reviewed.record_id)
        return reviewed

    def reject_raw_device_record(
        self,
        raw_record_id: str,
        action: ReviewAction,
    ) -> DeviceRecordRaw | None:
        self.init_runtime()
        raw = self.raw_records.get(raw_record_id)
        if not raw:
            return None
        updated = self.raw_records.set_review_status(raw_record_id, "rejected")
        if updated:
            self.review_events.add(
                paper_id=updated.paper_id,
                event_type="raw_rejected",
                actor=action.actor,
                message=action.message or "Rejected raw extraction candidate",
                after_json=updated.model_dump_json(),
            )
        return updated

    def _reviewed_payload_from_raw(self, raw: DeviceRecordRaw) -> dict[str, object]:
        return {
            "device_label": raw.device_label,
            "architecture": raw.architecture,
            "notes": raw.notes,
            "substrate": raw.substrate,
            "anode": raw.anode,
            "hil": raw.hil,
            "htl": raw.htl,
            "ebl": raw.ebl,
            "eml_host": raw.eml_host,
            "eml_dopant": raw.eml_dopant,
            "eml_emitter": raw.eml_emitter,
            "hbl": raw.hbl,
            "etl": raw.etl,
            "eil": raw.eil,
            "cathode": raw.cathode,
            "layer_thicknesses": raw.layer_thicknesses,
            "eqe_max": raw.eqe_max,
            "ce_max": raw.ce_max,
            "pe_max": raw.pe_max,
            "luminance_max": raw.luminance_max,
            "turn_on_voltage": raw.turn_on_voltage,
            "cie_x": raw.cie_x,
            "cie_y": raw.cie_y,
            "el_peak": raw.el_peak,
            "fwhm": raw.fwhm,
            "lifetime": raw.lifetime,
            "evidence_text": raw.evidence_text,
            "evidence_page": raw.evidence_page,
        }
