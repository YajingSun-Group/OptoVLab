from __future__ import annotations

import json
import sqlite3
from uuid import uuid4

from evolab_local.mining_platform.schemas.batch import BatchJob
from evolab_local.mining_platform.schemas.candidate_ingestion import (
    CandidateEntity,
    CandidateFinalRecord,
    CandidateIngestionRun,
    CandidateValue,
    CandidateValueReviewEvent,
)
from evolab_local.mining_platform.schemas.device_record import (
    DeviceRecordCreate,
    DeviceRecordReviewed,
    DeviceRecordUpdate,
)
from evolab_local.mining_platform.schemas.candidate import CandidateFieldValue
from evolab_local.mining_platform.schemas.document import DocumentBlock
from evolab_local.mining_platform.schemas.evidence import EvidenceAnchor
from evolab_local.mining_platform.schemas.external_runs import LLMMiningRun, MinerUParseRun
from evolab_local.mining_platform.schemas.extraction import (
    DeviceRecordRaw,
    ExtractionRun,
    RawDeviceCandidate,
)
from evolab_local.mining_platform.schemas.final_device import OledDeviceFinal
from evolab_local.mining_platform.schemas.material_agent import (
    DocumentVisualBlock,
    FigureTriageResult,
    MaterialAgentRun,
    MoleculeCrop,
    MoleculeCropValidation,
    MoleculeLabelBinding,
    MoleculeLabelBindingReviewEvent,
    VLMCallLog,
)
from evolab_local.mining_platform.schemas.material_structure import (
    ChemicalFigureBlock,
    MaterialAlias,
    MaterialGlobal,
    MaterialIdentityEvidenceItem,
    MaterialIdentityEvidenceRun,
    MaterialIdentityJudgment,
    MaterialPropertyCandidate,
    MaterialPropertyReview,
    MaterialPropertyReviewEvent,
    MaterialReviewEvent,
    PaperMaterialNameReview,
    PaperMaterialNameSuggestion,
    MaterialResolutionTask,
    MaterialStructureCandidate,
    PaperMaterialLink,
)
from evolab_local.mining_platform.schemas.paper import Paper
from evolab_local.mining_platform.schemas.review import ReviewEvent
from evolab_local.mining_platform.storage.database import Database, now_iso


DEVICE_RECORD_FIELD_COLUMNS = (
    "device_label",
    "architecture",
    "notes",
    "substrate",
    "anode",
    "hil",
    "htl",
    "ebl",
    "eml_host",
    "eml_dopant",
    "eml_emitter",
    "hbl",
    "etl",
    "eil",
    "cathode",
    "layer_thicknesses",
    "eqe_max",
    "ce_max",
    "pe_max",
    "luminance_max",
    "turn_on_voltage",
    "cie_x",
    "cie_y",
    "el_peak",
    "fwhm",
    "lifetime",
    "evidence_text",
    "evidence_page",
)

FINAL_DEVICE_COLUMNS = (
    "final_device_id",
    "paper_id",
    *DEVICE_RECORD_FIELD_COLUMNS,
    "source_candidate_ids_json",
    "confirmed_by",
    "confirmed_at",
    "created_at",
    "updated_at",
)


class BatchJobRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def upsert(self, job: BatchJob) -> BatchJob:
        with self.database.connect() as conn:
            conn.execute(
                """
                INSERT INTO batch_jobs (
                  job_id, paper_id, doi, source_pdf_path, inbox_pdf_path,
                  pdf_sha256, pdf_size_bytes, status, current_stage,
                  last_completed_stage, retry_count, max_retries, error_message,
                  stage_timings_json, stage_errors_json, options_json,
                  created_at, updated_at, started_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(paper_id) DO UPDATE SET
                  doi=excluded.doi,
                  source_pdf_path=COALESCE(excluded.source_pdf_path, batch_jobs.source_pdf_path),
                  inbox_pdf_path=excluded.inbox_pdf_path,
                  pdf_sha256=excluded.pdf_sha256,
                  pdf_size_bytes=excluded.pdf_size_bytes,
                  status=CASE
                    WHEN batch_jobs.pdf_sha256 = excluded.pdf_sha256 THEN batch_jobs.status
                    ELSE excluded.status
                  END,
                  current_stage=CASE
                    WHEN batch_jobs.pdf_sha256 = excluded.pdf_sha256 THEN batch_jobs.current_stage
                    ELSE excluded.current_stage
                  END,
                  last_completed_stage=CASE
                    WHEN batch_jobs.pdf_sha256 = excluded.pdf_sha256 THEN batch_jobs.last_completed_stage
                    ELSE excluded.last_completed_stage
                  END,
                  retry_count=CASE
                    WHEN batch_jobs.pdf_sha256 = excluded.pdf_sha256 THEN batch_jobs.retry_count
                    ELSE excluded.retry_count
                  END,
                  max_retries=excluded.max_retries,
                  error_message=CASE
                    WHEN batch_jobs.pdf_sha256 = excluded.pdf_sha256 THEN batch_jobs.error_message
                    ELSE excluded.error_message
                  END,
                  stage_timings_json=CASE
                    WHEN batch_jobs.pdf_sha256 = excluded.pdf_sha256 THEN batch_jobs.stage_timings_json
                    ELSE excluded.stage_timings_json
                  END,
                  stage_errors_json=CASE
                    WHEN batch_jobs.pdf_sha256 = excluded.pdf_sha256 THEN batch_jobs.stage_errors_json
                    ELSE excluded.stage_errors_json
                  END,
                  options_json=excluded.options_json,
                  updated_at=excluded.updated_at,
                  started_at=CASE
                    WHEN batch_jobs.pdf_sha256 = excluded.pdf_sha256 THEN batch_jobs.started_at
                    ELSE excluded.started_at
                  END,
                  completed_at=CASE
                    WHEN batch_jobs.pdf_sha256 = excluded.pdf_sha256 THEN batch_jobs.completed_at
                    ELSE excluded.completed_at
                  END
                """,
                self._job_values(job),
            )
        return self.get_by_paper(job.paper_id) or job

    def update(self, job: BatchJob) -> BatchJob:
        with self.database.connect() as conn:
            conn.execute(
                """
                UPDATE batch_jobs
                SET doi = ?, source_pdf_path = ?, inbox_pdf_path = ?, pdf_sha256 = ?,
                    pdf_size_bytes = ?, status = ?, current_stage = ?,
                    last_completed_stage = ?, retry_count = ?, max_retries = ?,
                    error_message = ?, stage_timings_json = ?, stage_errors_json = ?,
                    options_json = ?, updated_at = ?, started_at = ?, completed_at = ?
                WHERE job_id = ?
                """,
                (
                    job.doi,
                    job.source_pdf_path,
                    job.inbox_pdf_path,
                    job.pdf_sha256,
                    job.pdf_size_bytes,
                    job.status,
                    job.current_stage,
                    job.last_completed_stage,
                    job.retry_count,
                    job.max_retries,
                    job.error_message,
                    json.dumps(job.stage_timings, ensure_ascii=False, sort_keys=True),
                    json.dumps(job.stage_errors, ensure_ascii=False, sort_keys=True),
                    json.dumps(job.options, ensure_ascii=False, sort_keys=True),
                    job.updated_at,
                    job.started_at,
                    job.completed_at,
                    job.job_id,
                ),
            )
        return job

    def update_many(self, jobs: list[BatchJob]) -> list[BatchJob]:
        if not jobs:
            return []
        with self.database.connect() as conn:
            conn.executemany(
                """
                UPDATE batch_jobs
                SET doi = ?, source_pdf_path = ?, inbox_pdf_path = ?, pdf_sha256 = ?,
                    pdf_size_bytes = ?, status = ?, current_stage = ?,
                    last_completed_stage = ?, retry_count = ?, max_retries = ?,
                    error_message = ?, stage_timings_json = ?, stage_errors_json = ?,
                    options_json = ?, updated_at = ?, started_at = ?, completed_at = ?
                WHERE job_id = ?
                """,
                [
                    (
                        job.doi,
                        job.source_pdf_path,
                        job.inbox_pdf_path,
                        job.pdf_sha256,
                        job.pdf_size_bytes,
                        job.status,
                        job.current_stage,
                        job.last_completed_stage,
                        job.retry_count,
                        job.max_retries,
                        job.error_message,
                        json.dumps(job.stage_timings, ensure_ascii=False, sort_keys=True),
                        json.dumps(job.stage_errors, ensure_ascii=False, sort_keys=True),
                        json.dumps(job.options, ensure_ascii=False, sort_keys=True),
                        job.updated_at,
                        job.started_at,
                        job.completed_at,
                        job.job_id,
                    )
                    for job in jobs
                ],
            )
        return jobs

    def get(self, job_id: str) -> BatchJob | None:
        with self.database.connect() as conn:
            row = conn.execute("SELECT * FROM batch_jobs WHERE job_id = ?", (job_id,)).fetchone()
        return self._row_to_job(row) if row else None

    def get_by_paper(self, paper_id: str) -> BatchJob | None:
        with self.database.connect() as conn:
            row = conn.execute(
                "SELECT * FROM batch_jobs WHERE paper_id = ?",
                (paper_id,),
            ).fetchone()
        return self._row_to_job(row) if row else None

    def list(self, status: str | None = None, limit: int | None = None) -> list[BatchJob]:
        sql = "SELECT * FROM batch_jobs"
        params: list[object] = []
        if status:
            sql += " WHERE status = ?"
            params.append(status)
        sql += " ORDER BY updated_at DESC, paper_id ASC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        with self.database.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_job(row) for row in rows]

    def list_for_review_batches(self) -> list[BatchJob]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM batch_jobs
                ORDER BY created_at ASC, paper_id ASC
                """
            ).fetchall()
        return [self._row_to_job(row) for row in rows]

    def next_runnable(
        self,
        *,
        limit: int,
        include_failed_retries: bool = False,
    ) -> list[BatchJob]:
        statuses = ["registered"]
        if include_failed_retries:
            statuses.append("failed")
        placeholders = ", ".join("?" for _ in statuses)
        with self.database.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT batch_jobs.*
                FROM batch_jobs
                JOIN papers ON papers.paper_id = batch_jobs.paper_id
                WHERE batch_jobs.status IN ({placeholders})
                  AND batch_jobs.retry_count < batch_jobs.max_retries
                  AND papers.review_status NOT IN ('confirmed', 'excluded')
                ORDER BY batch_jobs.created_at ASC, batch_jobs.paper_id ASC
                LIMIT ?
                """,
                [*statuses, limit],
            ).fetchall()
        return [self._row_to_job(row) for row in rows]

    @staticmethod
    def _job_values(job: BatchJob) -> tuple[object, ...]:
        return (
            job.job_id,
            job.paper_id,
            job.doi,
            job.source_pdf_path,
            job.inbox_pdf_path,
            job.pdf_sha256,
            job.pdf_size_bytes,
            job.status,
            job.current_stage,
            job.last_completed_stage,
            job.retry_count,
            job.max_retries,
            job.error_message,
            json.dumps(job.stage_timings, ensure_ascii=False, sort_keys=True),
            json.dumps(job.stage_errors, ensure_ascii=False, sort_keys=True),
            json.dumps(job.options, ensure_ascii=False, sort_keys=True),
            job.created_at,
            job.updated_at,
            job.started_at,
            job.completed_at,
        )

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> BatchJob:
        payload = dict(row)
        payload["stage_timings"] = json.loads(payload.pop("stage_timings_json") or "{}")
        payload["stage_errors"] = json.loads(payload.pop("stage_errors_json") or "{}")
        payload["options"] = json.loads(payload.pop("options_json") or "{}")
        return BatchJob.model_validate(payload)


class PaperRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def upsert(self, paper: Paper) -> Paper:
        timestamp = now_iso()
        created_at = paper.created_at or timestamp
        updated = paper.model_copy(update={"created_at": created_at, "updated_at": timestamp})
        with self.database.connect() as conn:
            conn.execute(
                """
                INSERT INTO papers (
                  paper_id, doi, title, journal, publisher, year, pdf_path, pdf_sha256,
                  pdf_size_bytes, source, download_status, parse_status, mining_status,
                  review_status, review_reason, domain, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(paper_id) DO UPDATE SET
                  doi=excluded.doi,
                  title=excluded.title,
                  journal=excluded.journal,
                  publisher=excluded.publisher,
                  year=excluded.year,
                  pdf_path=excluded.pdf_path,
                  pdf_sha256=excluded.pdf_sha256,
                  pdf_size_bytes=excluded.pdf_size_bytes,
                  source=excluded.source,
                  download_status=excluded.download_status,
                  parse_status=CASE
                    WHEN papers.pdf_sha256 = excluded.pdf_sha256 THEN papers.parse_status
                    ELSE excluded.parse_status
                  END,
                  mining_status=CASE
                    WHEN papers.pdf_sha256 = excluded.pdf_sha256 THEN papers.mining_status
                    ELSE excluded.mining_status
                  END,
                  review_status=CASE
                    WHEN papers.pdf_sha256 = excluded.pdf_sha256 THEN papers.review_status
                    ELSE excluded.review_status
                  END,
                  review_reason=CASE
                    WHEN papers.pdf_sha256 = excluded.pdf_sha256 THEN papers.review_reason
                    ELSE excluded.review_reason
                  END,
                  domain=excluded.domain,
                  updated_at=excluded.updated_at
                """,
                (
                    updated.paper_id,
                    updated.doi,
                    updated.title,
                    updated.journal,
                    updated.publisher,
                    updated.year,
                    updated.pdf_path,
                    updated.pdf_sha256,
                    updated.pdf_size_bytes,
                    updated.source,
                    updated.download_status,
                    updated.parse_status,
                    updated.mining_status,
                    updated.review_status,
                    updated.review_reason,
                    updated.domain,
                    updated.created_at,
                    updated.updated_at,
                ),
            )
        return updated

    def list(self) -> list[Paper]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM papers
                ORDER BY
                  CASE
                    WHEN review_status IN ("needs_review", "in_progress") THEN 0
                    WHEN mining_status = "completed" THEN 1
                    ELSE 2
                  END ASC,
                  updated_at DESC,
                  doi ASC
                """
            ).fetchall()
        return [self._row_to_paper(row) for row in rows]

    def get(self, paper_id: str) -> Paper | None:
        with self.database.connect() as conn:
            row = conn.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
        return self._row_to_paper(row) if row else None

    def count(self) -> int:
        with self.database.connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM papers").fetchone()
        return int(row["count"])

    def set_parse_status(self, paper_id: str, status: str) -> Paper | None:
        timestamp = now_iso()
        with self.database.connect() as conn:
            conn.execute(
                """
                UPDATE papers
                SET parse_status = ?, updated_at = ?
                WHERE paper_id = ?
                """,
                (status, timestamp, paper_id),
            )
        return self.get(paper_id)

    def set_mining_status(self, paper_id: str, status: str) -> Paper | None:
        timestamp = now_iso()
        with self.database.connect() as conn:
            conn.execute(
                """
                UPDATE papers
                SET mining_status = ?, updated_at = ?
                WHERE paper_id = ?
                """,
                (status, timestamp, paper_id),
            )
        return self.get(paper_id)

    def set_review_status(
        self,
        paper_id: str,
        status: str,
        *,
        reason: str | None = None,
    ) -> Paper | None:
        timestamp = now_iso()
        with self.database.connect() as conn:
            conn.execute(
                """
                UPDATE papers
                SET review_status = ?, review_reason = ?, updated_at = ?
                WHERE paper_id = ?
                """,
                (status, reason, timestamp, paper_id),
            )
        return self.get(paper_id)

    def set_review_status_many(
        self,
        paper_ids: list[str],
        status: str,
        *,
        reason: str | None = None,
    ) -> None:
        if not paper_ids:
            return
        timestamp = now_iso()
        with self.database.connect() as conn:
            conn.executemany(
                """
                UPDATE papers
                SET review_status = ?, review_reason = ?, updated_at = ?
                WHERE paper_id = ?
                """,
                [(status, reason, timestamp, paper_id) for paper_id in paper_ids],
            )

    @staticmethod
    def _row_to_paper(row: sqlite3.Row) -> Paper:
        return Paper.model_validate(dict(row))


class DocumentBlockRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def replace_for_paper(self, paper_id: str, blocks: list[DocumentBlock]) -> list[DocumentBlock]:
        timestamp = now_iso()
        rows = [
            block.model_copy(update={"paper_id": paper_id, "created_at": timestamp})
            for block in blocks
        ]
        with self.database.connect() as conn:
            conn.execute("DELETE FROM document_blocks WHERE paper_id = ?", (paper_id,))
            conn.executemany(
                """
                INSERT INTO document_blocks (
                  paper_id, block_id, page_id, block_index, block_type, text,
                  bbox_json, source, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        block.paper_id,
                        block.block_id,
                        block.page_id,
                        block.block_index,
                        block.block_type,
                        block.text,
                        json.dumps(block.bbox, ensure_ascii=False),
                        block.source,
                        block.created_at,
                    )
                    for block in rows
                ],
            )
        return rows

    def list_by_paper(self, paper_id: str) -> list[DocumentBlock]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM document_blocks
                WHERE paper_id = ?
                ORDER BY page_id ASC, block_index ASC
                """,
                (paper_id,),
            ).fetchall()
        return [self._row_to_block(row) for row in rows]

    def get(self, paper_id: str, block_id: str) -> DocumentBlock | None:
        with self.database.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM document_blocks
                WHERE paper_id = ? AND block_id = ?
                """,
                (paper_id, block_id),
            ).fetchone()
        return self._row_to_block(row) if row else None

    @staticmethod
    def _row_to_block(row: sqlite3.Row) -> DocumentBlock:
        payload = dict(row)
        payload["bbox"] = json.loads(payload.pop("bbox_json"))
        return DocumentBlock.model_validate(payload)


class ExtractionRunRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def create(
        self,
        paper_id: str,
        extractor_name: str,
        extractor_version: str,
        input_block_count: int,
    ) -> ExtractionRun:
        run = ExtractionRun(
            run_id=uuid4().hex,
            paper_id=paper_id,
            extractor_name=extractor_name,
            extractor_version=extractor_version,
            status="running",
            input_block_count=input_block_count,
            raw_record_count=0,
            created_at=now_iso(),
        )
        with self.database.connect() as conn:
            conn.execute(
                """
                INSERT INTO extraction_runs (
                  run_id, paper_id, extractor_name, extractor_version, status,
                  input_block_count, raw_record_count, error_message, created_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.paper_id,
                    run.extractor_name,
                    run.extractor_version,
                    run.status,
                    run.input_block_count,
                    run.raw_record_count,
                    run.error_message,
                    run.created_at,
                    run.completed_at,
                ),
            )
        return run

    def complete(self, run_id: str, raw_record_count: int) -> ExtractionRun | None:
        timestamp = now_iso()
        with self.database.connect() as conn:
            conn.execute(
                """
                UPDATE extraction_runs
                SET status = 'completed', raw_record_count = ?, completed_at = ?
                WHERE run_id = ?
                """,
                (raw_record_count, timestamp, run_id),
            )
        return self.get(run_id)

    def fail(self, run_id: str, error_message: str) -> ExtractionRun | None:
        timestamp = now_iso()
        with self.database.connect() as conn:
            conn.execute(
                """
                UPDATE extraction_runs
                SET status = 'failed', error_message = ?, completed_at = ?
                WHERE run_id = ?
                """,
                (error_message, timestamp, run_id),
            )
        return self.get(run_id)

    def get(self, run_id: str) -> ExtractionRun | None:
        with self.database.connect() as conn:
            row = conn.execute(
                "SELECT * FROM extraction_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return self._row_to_run(row) if row else None

    def list_by_paper(self, paper_id: str) -> list[ExtractionRun]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM extraction_runs
                WHERE paper_id = ?
                ORDER BY created_at DESC
                """,
                (paper_id,),
            ).fetchall()
        return [self._row_to_run(row) for row in rows]

    @staticmethod
    def _row_to_run(row: sqlite3.Row) -> ExtractionRun:
        return ExtractionRun.model_validate(dict(row))


class DeviceRecordRawRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def add_many(
        self,
        run_id: str,
        paper_id: str,
        candidates: list[RawDeviceCandidate],
    ) -> list[DeviceRecordRaw]:
        timestamp = now_iso()
        records = [
            DeviceRecordRaw(
                raw_record_id=uuid4().hex,
                run_id=run_id,
                paper_id=paper_id,
                **candidate.model_dump(),
                created_at=timestamp,
                updated_at=timestamp,
            )
            for candidate in candidates
        ]
        columns = (
            "raw_record_id",
            "run_id",
            "paper_id",
            *DEVICE_RECORD_FIELD_COLUMNS,
            "evidence_block_ids_json",
            "field_evidence_json",
            "confidence_json",
            "raw_payload_json",
            "review_status",
            "reviewed_record_id",
            "created_at",
            "updated_at",
        )
        with self.database.connect() as conn:
            conn.executemany(
                f"INSERT INTO device_records_raw ({', '.join(columns)}) "
                f"VALUES ({', '.join('?' for _ in columns)})",
                [self._record_values(record, columns) for record in records],
            )
        return records

    def list_by_paper(self, paper_id: str) -> list[DeviceRecordRaw]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM device_records_raw
                WHERE paper_id = ?
                ORDER BY created_at DESC
                """,
                (paper_id,),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def get(self, raw_record_id: str) -> DeviceRecordRaw | None:
        with self.database.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM device_records_raw
                WHERE raw_record_id = ?
                """,
                (raw_record_id,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def set_review_status(
        self,
        raw_record_id: str,
        status: str,
        reviewed_record_id: str | None = None,
    ) -> DeviceRecordRaw | None:
        current = self.get(raw_record_id)
        if not current:
            return None
        timestamp = now_iso()
        next_reviewed_record_id = reviewed_record_id or current.reviewed_record_id
        with self.database.connect() as conn:
            conn.execute(
                """
                UPDATE device_records_raw
                SET review_status = ?, reviewed_record_id = ?, updated_at = ?
                WHERE raw_record_id = ?
                """,
                (status, next_reviewed_record_id, timestamp, raw_record_id),
            )
        return self.get(raw_record_id)

    @staticmethod
    def _record_values(record: DeviceRecordRaw, columns: tuple[str, ...]) -> tuple[object, ...]:
        values: list[object] = []
        for column in columns:
            if column == "evidence_block_ids_json":
                values.append(json.dumps(record.evidence_block_ids, ensure_ascii=False))
            elif column == "field_evidence_json":
                values.append(
                    json.dumps(
                        {
                            key: value.model_dump(mode="json")
                            for key, value in record.field_evidence.items()
                        },
                        ensure_ascii=False,
                    )
                )
            elif column == "confidence_json":
                values.append(json.dumps(record.confidence, ensure_ascii=False))
            elif column == "raw_payload_json":
                values.append(json.dumps(record.raw_payload, ensure_ascii=False))
            else:
                values.append(getattr(record, column))
        return tuple(values)

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> DeviceRecordRaw:
        payload = dict(row)
        payload["evidence_block_ids"] = json.loads(payload.pop("evidence_block_ids_json"))
        payload["field_evidence"] = json.loads(payload.pop("field_evidence_json"))
        payload["confidence"] = json.loads(payload.pop("confidence_json"))
        payload["raw_payload"] = json.loads(payload.pop("raw_payload_json"))
        return DeviceRecordRaw.model_validate(payload)


class DeviceRecordRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def create(self, paper_id: str, payload: DeviceRecordCreate) -> DeviceRecordReviewed:
        timestamp = now_iso()
        record = DeviceRecordReviewed(
            record_id=uuid4().hex,
            paper_id=paper_id,
            **payload.model_dump(exclude={"actor", "message"}),
            created_at=timestamp,
            updated_at=timestamp,
        )
        columns = (
            "record_id",
            "paper_id",
            *DEVICE_RECORD_FIELD_COLUMNS,
            "review_status",
            "created_at",
            "updated_at",
            "confirmed_at",
        )
        values = [getattr(record, column) for column in columns]
        placeholders = ", ".join("?" for _ in columns)
        with self.database.connect() as conn:
            conn.execute(
                f"INSERT INTO device_records_reviewed ({', '.join(columns)}) VALUES ({placeholders})",
                values,
            )
        return record

    def update(self, record_id: str, payload: DeviceRecordUpdate) -> DeviceRecordReviewed | None:
        current = self.get(record_id)
        if not current:
            return None
        timestamp = now_iso()
        update_data = payload.model_dump(exclude={"actor", "message"}, exclude_unset=True)
        updated = current.model_copy(update={**update_data, "updated_at": timestamp})
        assignments = [f"{column}=?" for column in (*DEVICE_RECORD_FIELD_COLUMNS, "updated_at")]
        values = [getattr(updated, column) for column in DEVICE_RECORD_FIELD_COLUMNS]
        values.append(updated.updated_at)
        values.append(record_id)
        with self.database.connect() as conn:
            conn.execute(
                f"UPDATE device_records_reviewed SET {', '.join(assignments)} WHERE record_id=?",
                values,
            )
        return updated

    def set_review_status(self, record_id: str, status: str) -> DeviceRecordReviewed | None:
        current = self.get(record_id)
        if not current:
            return None
        timestamp = now_iso()
        confirmed_at = timestamp if status == "confirmed" else current.confirmed_at
        with self.database.connect() as conn:
            conn.execute(
                """
                UPDATE device_records_reviewed
                SET review_status=?, updated_at=?, confirmed_at=?
                WHERE record_id=?
                """,
                (status, timestamp, confirmed_at, record_id),
            )
        return current.model_copy(
            update={"review_status": status, "updated_at": timestamp, "confirmed_at": confirmed_at}
        )

    def list_by_paper(self, paper_id: str) -> list[DeviceRecordReviewed]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM device_records_reviewed
                WHERE paper_id = ?
                ORDER BY updated_at DESC, created_at DESC
                """,
                (paper_id,),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def get(self, record_id: str) -> DeviceRecordReviewed | None:
        with self.database.connect() as conn:
            row = conn.execute(
                "SELECT * FROM device_records_reviewed WHERE record_id = ?",
                (record_id,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> DeviceRecordReviewed:
        return DeviceRecordReviewed.model_validate(dict(row))


class ReviewEventRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def add(
        self,
        paper_id: str,
        event_type: str,
        actor: str,
        record_id: str | None = None,
        message: str | None = None,
        before_json: str | None = None,
        after_json: str | None = None,
    ) -> ReviewEvent:
        event = ReviewEvent(
            event_id=uuid4().hex,
            paper_id=paper_id,
            record_id=record_id,
            event_type=event_type,
            actor=actor,
            message=message,
            before_json=before_json,
            after_json=after_json,
            created_at=now_iso(),
        )
        with self.database.connect() as conn:
            conn.execute(
                """
                INSERT INTO review_events (
                  event_id, paper_id, record_id, event_type, actor, message,
                  before_json, after_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.paper_id,
                    event.record_id,
                    event.event_type,
                    event.actor,
                    event.message,
                    event.before_json,
                    event.after_json,
                    event.created_at,
                ),
            )
        return event

    def list_by_paper(self, paper_id: str) -> list[ReviewEvent]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM review_events
                WHERE paper_id = ?
                ORDER BY rowid DESC
                """,
                (paper_id,),
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> ReviewEvent:
        return ReviewEvent.model_validate(dict(row))


class EvidenceAnchorRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def add_many(self, anchors: list[EvidenceAnchor]) -> list[EvidenceAnchor]:
        with self.database.connect() as conn:
            conn.executemany(
                """
                INSERT INTO evidence_anchors (
                  evidence_anchor_id, paper_id, page_id, block_id, bbox_json,
                  source_text, source_type, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        anchor.evidence_anchor_id,
                        anchor.paper_id,
                        anchor.page_id,
                        anchor.block_id,
                        json.dumps(anchor.bbox, ensure_ascii=False),
                        anchor.source_text,
                        anchor.source_type,
                        anchor.created_at,
                    )
                    for anchor in anchors
                ],
            )
        return anchors

    def list_by_paper(self, paper_id: str) -> list[EvidenceAnchor]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                  evidence_anchors.*,
                  document_blocks.bbox_json AS block_bbox_json
                FROM evidence_anchors
                LEFT JOIN document_blocks
                  ON document_blocks.paper_id = evidence_anchors.paper_id
                 AND document_blocks.block_id = evidence_anchors.block_id
                WHERE evidence_anchors.paper_id = ?
                ORDER BY evidence_anchors.page_id ASC, evidence_anchors.created_at ASC
                """,
                (paper_id,),
            ).fetchall()
        return [self._row_to_anchor(row) for row in rows]

    def get(self, evidence_anchor_id: str) -> EvidenceAnchor | None:
        with self.database.connect() as conn:
            row = conn.execute(
                """
                SELECT
                  evidence_anchors.*,
                  document_blocks.bbox_json AS block_bbox_json
                FROM evidence_anchors
                LEFT JOIN document_blocks
                  ON document_blocks.paper_id = evidence_anchors.paper_id
                 AND document_blocks.block_id = evidence_anchors.block_id
                WHERE evidence_anchors.evidence_anchor_id = ?
                """,
                (evidence_anchor_id,),
            ).fetchone()
        return self._row_to_anchor(row) if row else None

    def delete_by_paper(self, paper_id: str) -> None:
        with self.database.connect() as conn:
            conn.execute("DELETE FROM evidence_anchors WHERE paper_id = ?", (paper_id,))

    @staticmethod
    def _row_to_anchor(row: sqlite3.Row) -> EvidenceAnchor:
        payload = dict(row)
        bbox = json.loads(payload.pop("bbox_json"))
        block_bbox_json = payload.pop("block_bbox_json", None)
        if not bbox and block_bbox_json:
            bbox = json.loads(block_bbox_json)
        payload["bbox"] = bbox
        return EvidenceAnchor.model_validate(payload)


class CandidateFieldValueRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def add_many(self, fields: list[CandidateFieldValue]) -> list[CandidateFieldValue]:
        with self.database.connect() as conn:
            conn.executemany(
                """
                INSERT INTO candidate_field_values (
                  candidate_field_id, paper_id, record_scope, record_id, field_name,
                  field_label, mined_value, reviewed_value, confidence, confidence_json,
                  evidence_anchor_id, extractor_name, extractor_version, field_status,
                  created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [self._field_values(field) for field in fields],
            )
        return fields

    def list_by_paper(self, paper_id: str) -> list[CandidateFieldValue]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM candidate_field_values
                WHERE paper_id = ?
                ORDER BY record_id ASC, created_at ASC, field_name ASC
                """,
                (paper_id,),
            ).fetchall()
        return [self._row_to_field(row) for row in rows]

    def get(self, candidate_field_id: str) -> CandidateFieldValue | None:
        with self.database.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM candidate_field_values
                WHERE candidate_field_id = ?
                """,
                (candidate_field_id,),
            ).fetchone()
        return self._row_to_field(row) if row else None

    def update(
        self,
        candidate_field_id: str,
        reviewed_value: str | None,
        field_status: str | None = None,
    ) -> CandidateFieldValue | None:
        current = self.get(candidate_field_id)
        if not current:
            return None
        timestamp = now_iso()
        next_status = field_status or (
            "edited" if reviewed_value != current.mined_value else current.field_status
        )
        with self.database.connect() as conn:
            conn.execute(
                """
                UPDATE candidate_field_values
                SET reviewed_value = ?, field_status = ?, updated_at = ?
                WHERE candidate_field_id = ?
                """,
                (reviewed_value, next_status, timestamp, candidate_field_id),
            )
        return self.get(candidate_field_id)

    def set_status(self, candidate_field_id: str, field_status: str) -> CandidateFieldValue | None:
        current = self.get(candidate_field_id)
        if not current:
            return None
        timestamp = now_iso()
        with self.database.connect() as conn:
            conn.execute(
                """
                UPDATE candidate_field_values
                SET field_status = ?, updated_at = ?
                WHERE candidate_field_id = ?
                """,
                (field_status, timestamp, candidate_field_id),
            )
        return self.get(candidate_field_id)

    def delete_by_paper(self, paper_id: str) -> None:
        with self.database.connect() as conn:
            conn.execute("DELETE FROM candidate_field_values WHERE paper_id = ?", (paper_id,))

    @staticmethod
    def _field_values(field: CandidateFieldValue) -> tuple[object, ...]:
        return (
            field.candidate_field_id,
            field.paper_id,
            field.record_scope,
            field.record_id,
            field.field_name,
            field.field_label,
            field.mined_value,
            field.reviewed_value,
            field.confidence,
            json.dumps(field.confidence_json, ensure_ascii=False),
            field.evidence_anchor_id,
            field.extractor_name,
            field.extractor_version,
            field.field_status,
            field.created_at,
            field.updated_at,
        )

    @staticmethod
    def _row_to_field(row: sqlite3.Row) -> CandidateFieldValue:
        payload = dict(row)
        payload["confidence_json"] = json.loads(payload["confidence_json"])
        return CandidateFieldValue.model_validate(payload)


class OledDeviceFinalRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def replace_for_paper(
        self, paper_id: str, records: list[OledDeviceFinal]
    ) -> list[OledDeviceFinal]:
        placeholders = ", ".join("?" for _ in FINAL_DEVICE_COLUMNS)
        with self.database.connect() as conn:
            conn.execute("DELETE FROM oled_devices_final WHERE paper_id = ?", (paper_id,))
            conn.executemany(
                f"INSERT INTO oled_devices_final ({', '.join(FINAL_DEVICE_COLUMNS)}) "
                f"VALUES ({placeholders})",
                [self._record_values(record) for record in records],
            )
        return records

    def list_by_paper(self, paper_id: str) -> list[OledDeviceFinal]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM oled_devices_final
                WHERE paper_id = ?
                ORDER BY created_at ASC
                """,
                (paper_id,),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    @staticmethod
    def _record_values(record: OledDeviceFinal) -> tuple[object, ...]:
        values: list[object] = []
        for column in FINAL_DEVICE_COLUMNS:
            if column == "source_candidate_ids_json":
                values.append(json.dumps(record.source_candidate_ids, ensure_ascii=False))
            else:
                values.append(getattr(record, column))
        return tuple(values)

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> OledDeviceFinal:
        payload = dict(row)
        payload["source_candidate_ids"] = json.loads(payload.pop("source_candidate_ids_json"))
        return OledDeviceFinal.model_validate(payload)


class CandidateIngestionRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def create_run(
        self,
        *,
        paper_id: str,
        template_id: str,
        template_version: str,
        source_name: str,
        source_version: str | None,
        status: str,
        validation_report: dict[str, object],
        mining_result: dict[str, object],
        error_message: str | None = None,
    ) -> CandidateIngestionRun:
        timestamp = now_iso()
        run = CandidateIngestionRun(
            candidate_run_id=uuid4().hex,
            paper_id=paper_id,
            template_id=template_id,
            template_version=template_version,
            source_name=source_name,
            source_version=source_version,
            status=status,
            validation_report=validation_report,
            mining_result=mining_result,
            error_message=error_message,
            created_at=timestamp,
            completed_at=timestamp,
        )
        with self.database.connect() as conn:
            conn.execute(
                """
                INSERT INTO candidate_ingestion_runs (
                  candidate_run_id, paper_id, template_id, template_version,
                  source_name, source_version, status, validation_report_json,
                  mining_result_json, error_message, created_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.candidate_run_id,
                    run.paper_id,
                    run.template_id,
                    run.template_version,
                    run.source_name,
                    run.source_version,
                    run.status,
                    json.dumps(run.validation_report, ensure_ascii=False, sort_keys=True),
                    json.dumps(run.mining_result, ensure_ascii=False, sort_keys=True),
                    run.error_message,
                    run.created_at,
                    run.completed_at,
                ),
            )
        return run

    def add_entities(self, entities: list[CandidateEntity]) -> list[CandidateEntity]:
        with self.database.connect() as conn:
            conn.executemany(
                """
                INSERT INTO candidate_entities (
                  candidate_entity_id, candidate_run_id, paper_id, template_id,
                  entity_type, entity_path, entity_label, parent_entity_id,
                  sort_order, source_json, review_status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        entity.candidate_entity_id,
                        entity.candidate_run_id,
                        entity.paper_id,
                        entity.template_id,
                        entity.entity_type,
                        entity.entity_path,
                        entity.entity_label,
                        entity.parent_entity_id,
                        entity.sort_order,
                        json.dumps(entity.source_json, ensure_ascii=False, sort_keys=True),
                        entity.review_status,
                        entity.created_at,
                        entity.updated_at,
                    )
                    for entity in entities
                ],
            )
        return entities

    def add_values(self, values: list[CandidateValue]) -> list[CandidateValue]:
        with self.database.connect() as conn:
            conn.executemany(
                """
                INSERT INTO candidate_values (
                  candidate_value_id, candidate_run_id, candidate_entity_id,
                  paper_id, template_id, template_field_path, concrete_path,
                  field_label, data_type, value_json, reviewed_value_json,
                  display_value, evidence_anchor_ids_json, status, created_at,
                  updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        value.candidate_value_id,
                        value.candidate_run_id,
                        value.candidate_entity_id,
                        value.paper_id,
                        value.template_id,
                        value.template_field_path,
                        value.concrete_path,
                        value.field_label,
                        value.data_type,
                        json.dumps(value.value_json, ensure_ascii=False, sort_keys=True),
                        (
                            json.dumps(
                                value.reviewed_value_json,
                                ensure_ascii=False,
                                sort_keys=True,
                            )
                            if value.reviewed_value_json is not None
                            else None
                        ),
                        value.display_value,
                        json.dumps(value.evidence_anchor_ids, ensure_ascii=False),
                        value.status,
                        value.created_at,
                        value.updated_at,
                    )
                    for value in values
                ],
            )
        return values

    def get_run(self, candidate_run_id: str) -> CandidateIngestionRun | None:
        with self.database.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM candidate_ingestion_runs
                WHERE candidate_run_id = ?
                """,
                (candidate_run_id,),
            ).fetchone()
        return self._row_to_run(row) if row else None

    def list_runs_by_paper(
        self,
        paper_id: str,
        *,
        include_payload: bool = True,
    ) -> list[CandidateIngestionRun]:
        payload_columns = (
            "validation_report_json, mining_result_json"
            if include_payload
            else "'{}' AS validation_report_json, '{}' AS mining_result_json"
        )
        with self.database.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                  candidate_run_id, paper_id, template_id, template_version,
                  source_name, source_version, status, {payload_columns},
                  error_message, created_at, completed_at
                FROM candidate_ingestion_runs
                WHERE paper_id = ?
                ORDER BY created_at DESC
                """,  # noqa: S608 - payload_columns is selected from fixed SQL literals above.
                (paper_id,),
            ).fetchall()
        return [self._row_to_run(row) for row in rows]

    def list_entities_by_run(self, candidate_run_id: str) -> list[CandidateEntity]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM candidate_entities
                WHERE candidate_run_id = ?
                ORDER BY entity_path ASC, sort_order ASC
                """,
                (candidate_run_id,),
            ).fetchall()
        return [self._row_to_entity(row) for row in rows]

    def list_values_by_run(self, candidate_run_id: str) -> list[CandidateValue]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM candidate_values
                WHERE candidate_run_id = ?
                ORDER BY concrete_path ASC
                """,
                (candidate_run_id,),
            ).fetchall()
        return [self._row_to_value(row) for row in rows]

    def get_value(self, candidate_value_id: str) -> CandidateValue | None:
        with self.database.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM candidate_values
                WHERE candidate_value_id = ?
                """,
                (candidate_value_id,),
            ).fetchone()
        return self._row_to_value(row) if row else None

    def update_value(
        self,
        candidate_value_id: str,
        reviewed_value_json: object | None,
        status: str | None = None,
    ) -> CandidateValue | None:
        current = self.get_value(candidate_value_id)
        if not current:
            return None
        timestamp = now_iso()
        normalized_status = _normalize_candidate_value_status(status)
        next_status = normalized_status or (
            "modified" if reviewed_value_json != current.value_json else current.status
        )
        display_value = (
            _candidate_display_value(current.value_json)
            if reviewed_value_json is None
            else _candidate_display_value(reviewed_value_json)
        )
        with self.database.connect() as conn:
            conn.execute(
                """
                UPDATE candidate_values
                SET reviewed_value_json = ?, display_value = ?, status = ?, updated_at = ?
                WHERE candidate_value_id = ?
                """,
                (
                    (
                        json.dumps(reviewed_value_json, ensure_ascii=False, sort_keys=True)
                        if reviewed_value_json is not None
                        else None
                    ),
                    display_value,
                    next_status,
                    timestamp,
                    candidate_value_id,
                ),
            )
        return self.get_value(candidate_value_id)

    @staticmethod
    def _row_to_run(row: sqlite3.Row) -> CandidateIngestionRun:
        payload = dict(row)
        payload["validation_report"] = json.loads(payload.pop("validation_report_json"))
        payload["mining_result"] = json.loads(payload.pop("mining_result_json"))
        return CandidateIngestionRun.model_validate(payload)

    @staticmethod
    def _row_to_entity(row: sqlite3.Row) -> CandidateEntity:
        payload = dict(row)
        payload["source_json"] = json.loads(payload.pop("source_json"))
        return CandidateEntity.model_validate(payload)

    @staticmethod
    def _row_to_value(row: sqlite3.Row) -> CandidateValue:
        payload = dict(row)
        payload["value_json"] = json.loads(payload["value_json"])
        if payload["reviewed_value_json"] is not None:
            payload["reviewed_value_json"] = json.loads(payload["reviewed_value_json"])
        payload["evidence_anchor_ids"] = json.loads(payload.pop("evidence_anchor_ids_json"))
        return CandidateValue.model_validate(payload)


class CandidateValueReviewEventRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def add(
        self,
        *,
        before: CandidateValue,
        after: CandidateValue,
        action: str,
        actor: str,
        message: str | None = None,
    ) -> CandidateValueReviewEvent:
        event = CandidateValueReviewEvent(
            event_id=uuid4().hex,
            candidate_value_id=after.candidate_value_id,
            candidate_run_id=after.candidate_run_id,
            candidate_entity_id=after.candidate_entity_id,
            paper_id=after.paper_id,
            template_id=after.template_id,
            template_field_path=after.template_field_path,
            concrete_path=after.concrete_path,
            action=action,
            actor=actor,
            message=message,
            original_value_json=before.value_json,
            before_reviewed_value_json=before.reviewed_value_json,
            after_reviewed_value_json=after.reviewed_value_json,
            before_status=before.status,
            after_status=after.status,
            created_at=now_iso(),
        )
        with self.database.connect() as conn:
            conn.execute(
                """
                INSERT INTO candidate_value_review_events (
                  event_id, candidate_value_id, candidate_run_id, candidate_entity_id,
                  paper_id, template_id, template_field_path, concrete_path, action,
                  actor, message, original_value_json, before_reviewed_value_json,
                  after_reviewed_value_json, before_status, after_status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.candidate_value_id,
                    event.candidate_run_id,
                    event.candidate_entity_id,
                    event.paper_id,
                    event.template_id,
                    event.template_field_path,
                    event.concrete_path,
                    event.action,
                    event.actor,
                    event.message,
                    json.dumps(event.original_value_json, ensure_ascii=False, sort_keys=True),
                    (
                        json.dumps(
                            event.before_reviewed_value_json,
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        if event.before_reviewed_value_json is not None
                        else None
                    ),
                    (
                        json.dumps(
                            event.after_reviewed_value_json,
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        if event.after_reviewed_value_json is not None
                        else None
                    ),
                    event.before_status,
                    event.after_status,
                    event.created_at,
                ),
            )
        return event

    def list_by_run(self, candidate_run_id: str) -> list[CandidateValueReviewEvent]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM candidate_value_review_events
                WHERE candidate_run_id = ?
                ORDER BY created_at ASC, rowid ASC
                """,
                (candidate_run_id,),
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def list_by_value(self, candidate_value_id: str) -> list[CandidateValueReviewEvent]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM candidate_value_review_events
                WHERE candidate_value_id = ?
                ORDER BY created_at ASC, rowid ASC
                """,
                (candidate_value_id,),
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def get(self, event_id: str) -> CandidateValueReviewEvent | None:
        with self.database.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM candidate_value_review_events
                WHERE event_id = ?
                """,
                (event_id,),
            ).fetchone()
        return self._row_to_event(row) if row else None

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> CandidateValueReviewEvent:
        payload = dict(row)
        payload["original_value_json"] = json.loads(payload["original_value_json"])
        if payload["before_reviewed_value_json"] is not None:
            payload["before_reviewed_value_json"] = json.loads(
                payload["before_reviewed_value_json"]
            )
        if payload["after_reviewed_value_json"] is not None:
            payload["after_reviewed_value_json"] = json.loads(payload["after_reviewed_value_json"])
        return CandidateValueReviewEvent.model_validate(payload)


class CandidateFinalRecordRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def replace_for_paper_template(
        self,
        paper_id: str,
        template_id: str,
        record: CandidateFinalRecord,
    ) -> CandidateFinalRecord:
        with self.database.connect() as conn:
            conn.execute(
                """
                DELETE FROM candidate_final_records
                WHERE paper_id = ? AND template_id = ?
                """,
                (paper_id, template_id),
            )
            conn.execute(
                """
                INSERT INTO candidate_final_records (
                  final_record_id, paper_id, candidate_run_id, template_id,
                  template_version, final_json, source_candidate_value_ids_json,
                  confirmed_by, status, created_at, updated_at, confirmed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.final_record_id,
                    record.paper_id,
                    record.candidate_run_id,
                    record.template_id,
                    record.template_version,
                    json.dumps(record.final_json, ensure_ascii=False, sort_keys=True),
                    json.dumps(record.source_candidate_value_ids, ensure_ascii=False),
                    record.confirmed_by,
                    record.status,
                    record.created_at,
                    record.updated_at,
                    record.confirmed_at,
                ),
            )
        return record

    def list_by_paper(self, paper_id: str) -> list[CandidateFinalRecord]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM candidate_final_records
                WHERE paper_id = ?
                ORDER BY confirmed_at DESC
                """,
                (paper_id,),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def get_by_run(self, candidate_run_id: str) -> CandidateFinalRecord | None:
        with self.database.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM candidate_final_records
                WHERE candidate_run_id = ?
                ORDER BY confirmed_at DESC
                LIMIT 1
                """,
                (candidate_run_id,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> CandidateFinalRecord:
        payload = dict(row)
        payload["final_json"] = json.loads(payload["final_json"])
        payload["source_candidate_value_ids"] = json.loads(
            payload.pop("source_candidate_value_ids_json")
        )
        return CandidateFinalRecord.model_validate(payload)


class MaterialGlobalRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def upsert(self, material: MaterialGlobal) -> MaterialGlobal:
        timestamp = now_iso()
        created_at = material.created_at or timestamp
        updated = material.model_copy(update={"created_at": created_at, "updated_at": timestamp})
        with self.database.connect() as conn:
            conn.execute(
                """
                INSERT INTO materials_global (
                  global_material_id, canonical_name, material_class,
                  representation_type, raw_smiles, canonical_smiles,
                  isomeric_smiles, inchi, inchi_key, formula, molecular_weight,
                  source, source_detail_json, confidence, review_status,
                  created_at, updated_at, confirmed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(global_material_id) DO UPDATE SET
                  canonical_name=excluded.canonical_name,
                  material_class=excluded.material_class,
                  representation_type=excluded.representation_type,
                  raw_smiles=excluded.raw_smiles,
                  canonical_smiles=excluded.canonical_smiles,
                  isomeric_smiles=excluded.isomeric_smiles,
                  inchi=excluded.inchi,
                  inchi_key=excluded.inchi_key,
                  formula=excluded.formula,
                  molecular_weight=excluded.molecular_weight,
                  source=excluded.source,
                  source_detail_json=excluded.source_detail_json,
                  confidence=excluded.confidence,
                  review_status=excluded.review_status,
                  updated_at=excluded.updated_at,
                  confirmed_at=excluded.confirmed_at
                """,
                self._material_values(updated),
            )
        return updated

    def get(self, global_material_id: str) -> MaterialGlobal | None:
        with self.database.connect() as conn:
            row = conn.execute(
                "SELECT * FROM materials_global WHERE global_material_id = ?",
                (global_material_id,),
            ).fetchone()
        return self._row_to_material(row) if row else None

    def get_many(self, global_material_ids: list[str]) -> list[MaterialGlobal]:
        if not global_material_ids:
            return []
        placeholders = ", ".join("?" for _ in global_material_ids)
        with self.database.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM materials_global
                WHERE global_material_id IN ({placeholders})
                ORDER BY canonical_name ASC, global_material_id ASC
                """,
                global_material_ids,
            ).fetchall()
        return [self._row_to_material(row) for row in rows]

    def get_by_inchi_key(self, inchi_key: str) -> MaterialGlobal | None:
        with self.database.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM materials_global
                WHERE inchi_key = ?
                ORDER BY
                  CASE WHEN review_status = 'confirmed' THEN 0 ELSE 1 END,
                  updated_at DESC
                LIMIT 1
                """,
                (inchi_key,),
            ).fetchone()
        return self._row_to_material(row) if row else None

    def list(self) -> list[MaterialGlobal]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM materials_global
                ORDER BY canonical_name ASC, global_material_id ASC
                """
            ).fetchall()
        return [self._row_to_material(row) for row in rows]

    @staticmethod
    def _material_values(material: MaterialGlobal) -> tuple[object, ...]:
        return (
            material.global_material_id,
            material.canonical_name,
            material.material_class,
            material.representation_type,
            material.raw_smiles,
            material.canonical_smiles,
            material.isomeric_smiles,
            material.inchi,
            material.inchi_key,
            material.formula,
            material.molecular_weight,
            material.source,
            json.dumps(material.source_detail, ensure_ascii=False, sort_keys=True),
            material.confidence,
            material.review_status,
            material.created_at,
            material.updated_at,
            material.confirmed_at,
        )

    @staticmethod
    def _row_to_material(row: sqlite3.Row) -> MaterialGlobal:
        payload = dict(row)
        payload["source_detail"] = json.loads(payload.pop("source_detail_json") or "{}")
        return MaterialGlobal.model_validate(payload)


class MaterialAliasRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def add(self, alias: MaterialAlias) -> MaterialAlias:
        with self.database.connect() as conn:
            conn.execute(
                """
                INSERT INTO material_aliases (
                  alias_id, global_material_id, alias_text, normalized_alias,
                  alias_type, source_paper_id, source, confidence,
                  review_status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._alias_values(alias),
            )
        return alias

    def add_if_missing(self, alias: MaterialAlias) -> MaterialAlias:
        with self.database.connect() as conn:
            existing = conn.execute(
                """
                SELECT *
                FROM material_aliases
                WHERE global_material_id = ? AND normalized_alias = ?
                LIMIT 1
                """,
                (alias.global_material_id, alias.normalized_alias),
            ).fetchone()
            if existing:
                return self._row_to_alias(existing)
            conn.execute(
                """
                INSERT INTO material_aliases (
                  alias_id, global_material_id, alias_text, normalized_alias,
                  alias_type, source_paper_id, source, confidence,
                  review_status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._alias_values(alias),
            )
        return alias

    def set_review_status_by_global_material(
        self,
        global_material_id: str,
        review_status: str,
    ) -> None:
        timestamp = now_iso()
        with self.database.connect() as conn:
            conn.execute(
                """
                UPDATE material_aliases
                SET review_status = ?, updated_at = ?
                WHERE global_material_id = ?
                """,
                (review_status, timestamp, global_material_id),
            )

    def find_by_normalized(self, normalized_alias: str) -> list[MaterialAlias]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM material_aliases
                WHERE normalized_alias = ?
                ORDER BY confidence DESC, alias_text ASC
                """,
                (normalized_alias,),
            ).fetchall()
        return [self._row_to_alias(row) for row in rows]

    def list_by_global_material(self, global_material_id: str) -> list[MaterialAlias]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM material_aliases
                WHERE global_material_id = ?
                ORDER BY alias_text ASC
                """,
                (global_material_id,),
            ).fetchall()
        return [self._row_to_alias(row) for row in rows]

    def list(self) -> list[MaterialAlias]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM material_aliases
                ORDER BY normalized_alias ASC, alias_text ASC
                """
            ).fetchall()
        return [self._row_to_alias(row) for row in rows]

    @staticmethod
    def _alias_values(alias: MaterialAlias) -> tuple[object, ...]:
        return (
            alias.alias_id,
            alias.global_material_id,
            alias.alias_text,
            alias.normalized_alias,
            alias.alias_type,
            alias.source_paper_id,
            alias.source,
            alias.confidence,
            alias.review_status,
            alias.created_at,
            alias.updated_at,
        )

    @staticmethod
    def _row_to_alias(row: sqlite3.Row) -> MaterialAlias:
        return MaterialAlias.model_validate(dict(row))


class PaperMaterialLinkRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def upsert(self, link: PaperMaterialLink) -> PaperMaterialLink:
        current = self.get_by_paper_material(link.candidate_run_id, link.paper_material_id)
        created_at = current.created_at if current else link.created_at
        updated = link.model_copy(update={"created_at": created_at, "updated_at": now_iso()})
        with self.database.connect() as conn:
            conn.execute(
                """
                INSERT INTO paper_material_links (
                  paper_material_link_id, paper_id, candidate_run_id,
                  paper_material_id, global_material_id, match_method,
                  match_confidence, match_status, evidence_json, created_at,
                  updated_at, confirmed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(candidate_run_id, paper_material_id) DO UPDATE SET
                  global_material_id=excluded.global_material_id,
                  match_method=excluded.match_method,
                  match_confidence=excluded.match_confidence,
                  match_status=excluded.match_status,
                  evidence_json=excluded.evidence_json,
                  updated_at=excluded.updated_at,
                  confirmed_at=excluded.confirmed_at
                """,
                self._link_values(updated),
            )
        return updated

    def get_by_paper_material(
        self,
        candidate_run_id: str,
        paper_material_id: str,
    ) -> PaperMaterialLink | None:
        with self.database.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM paper_material_links
                WHERE candidate_run_id = ? AND paper_material_id = ?
                """,
                (candidate_run_id, paper_material_id),
            ).fetchone()
        return self._row_to_link(row) if row else None

    def list_by_run(self, candidate_run_id: str) -> list[PaperMaterialLink]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM paper_material_links
                WHERE candidate_run_id = ?
                ORDER BY paper_material_id ASC
                """,
                (candidate_run_id,),
            ).fetchall()
        return [self._row_to_link(row) for row in rows]

    def delete_by_paper_material(self, candidate_run_id: str, paper_material_id: str) -> None:
        with self.database.connect() as conn:
            conn.execute(
                """
                DELETE FROM paper_material_links
                WHERE candidate_run_id = ? AND paper_material_id = ?
                """,
                (candidate_run_id, paper_material_id),
            )

    @staticmethod
    def _link_values(link: PaperMaterialLink) -> tuple[object, ...]:
        return (
            link.paper_material_link_id,
            link.paper_id,
            link.candidate_run_id,
            link.paper_material_id,
            link.global_material_id,
            link.match_method,
            link.match_confidence,
            link.match_status,
            json.dumps(link.evidence, ensure_ascii=False, sort_keys=True),
            link.created_at,
            link.updated_at,
            link.confirmed_at,
        )

    @staticmethod
    def _row_to_link(row: sqlite3.Row) -> PaperMaterialLink:
        payload = dict(row)
        payload["evidence"] = json.loads(payload.pop("evidence_json") or "{}")
        return PaperMaterialLink.model_validate(payload)


class PaperMaterialNameReviewRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def upsert(self, review: PaperMaterialNameReview) -> PaperMaterialNameReview:
        current = self.get_by_paper_material(review.candidate_run_id, review.paper_material_id)
        created_at = current.created_at if current else review.created_at
        updated = review.model_copy(update={"created_at": created_at, "updated_at": now_iso()})
        with self.database.connect() as conn:
            conn.execute(
                """
                INSERT INTO paper_material_name_reviews (
                  review_id, paper_id, candidate_run_id, paper_material_id,
                  reviewed_name, reviewed_full_name_in_paper, reviewed_abbreviation,
                  reviewed_normalized_name, reviewed_canonical_name, review_status,
                  actor, message, source, source_detail_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(candidate_run_id, paper_material_id) DO UPDATE SET
                  reviewed_name=excluded.reviewed_name,
                  reviewed_full_name_in_paper=excluded.reviewed_full_name_in_paper,
                  reviewed_abbreviation=excluded.reviewed_abbreviation,
                  reviewed_normalized_name=excluded.reviewed_normalized_name,
                  reviewed_canonical_name=excluded.reviewed_canonical_name,
                  review_status=excluded.review_status,
                  actor=excluded.actor,
                  message=excluded.message,
                  source=excluded.source,
                  source_detail_json=excluded.source_detail_json,
                  updated_at=excluded.updated_at
                """,
                self._review_values(updated),
            )
        return updated

    def get_by_paper_material(
        self,
        candidate_run_id: str,
        paper_material_id: str,
    ) -> PaperMaterialNameReview | None:
        with self.database.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM paper_material_name_reviews
                WHERE candidate_run_id = ? AND paper_material_id = ?
                """,
                (candidate_run_id, paper_material_id),
            ).fetchone()
        return self._row_to_review(row) if row else None

    def list_by_run(self, candidate_run_id: str) -> list[PaperMaterialNameReview]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM paper_material_name_reviews
                WHERE candidate_run_id = ?
                ORDER BY paper_material_id ASC
                """,
                (candidate_run_id,),
            ).fetchall()
        return [self._row_to_review(row) for row in rows]

    @staticmethod
    def _review_values(review: PaperMaterialNameReview) -> tuple[object, ...]:
        return (
            review.review_id,
            review.paper_id,
            review.candidate_run_id,
            review.paper_material_id,
            review.reviewed_name,
            review.reviewed_full_name_in_paper,
            review.reviewed_abbreviation,
            review.reviewed_normalized_name,
            review.reviewed_canonical_name,
            review.review_status,
            review.actor,
            review.message,
            review.source,
            json.dumps(review.source_detail, ensure_ascii=False, sort_keys=True),
            review.created_at,
            review.updated_at,
        )

    @staticmethod
    def _row_to_review(row: sqlite3.Row) -> PaperMaterialNameReview:
        payload = dict(row)
        payload["source_detail"] = json.loads(payload.pop("source_detail_json") or "{}")
        return PaperMaterialNameReview.model_validate(payload)


class PaperMaterialNameSuggestionRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def upsert(self, suggestion: PaperMaterialNameSuggestion) -> PaperMaterialNameSuggestion:
        current = self.get_by_unique(
            suggestion.candidate_run_id,
            suggestion.paper_material_id,
            suggestion.agent_name,
            suggestion.suggested_name,
        )
        created_at = current.created_at if current else suggestion.created_at
        updated = suggestion.model_copy(update={"created_at": created_at, "updated_at": now_iso()})
        with self.database.connect() as conn:
            conn.execute(
                """
                INSERT INTO paper_material_name_suggestions (
                  suggestion_id, paper_id, candidate_run_id, paper_material_id,
                  agent_name, original_name, suggested_name,
                  suggested_full_name_in_paper, suggested_abbreviation,
                  suggested_normalized_name, suggested_canonical_name, confidence,
                  reason, evidence_json, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(candidate_run_id, paper_material_id, agent_name, suggested_name)
                DO UPDATE SET
                  original_name=excluded.original_name,
                  suggested_full_name_in_paper=excluded.suggested_full_name_in_paper,
                  suggested_abbreviation=excluded.suggested_abbreviation,
                  suggested_normalized_name=excluded.suggested_normalized_name,
                  suggested_canonical_name=excluded.suggested_canonical_name,
                  confidence=excluded.confidence,
                  reason=excluded.reason,
                  evidence_json=excluded.evidence_json,
                  status=excluded.status,
                  updated_at=excluded.updated_at
                """,
                self._suggestion_values(updated),
            )
        return updated

    def get_by_unique(
        self,
        candidate_run_id: str,
        paper_material_id: str,
        agent_name: str,
        suggested_name: str,
    ) -> PaperMaterialNameSuggestion | None:
        with self.database.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM paper_material_name_suggestions
                WHERE candidate_run_id = ?
                  AND paper_material_id = ?
                  AND agent_name = ?
                  AND suggested_name = ?
                """,
                (candidate_run_id, paper_material_id, agent_name, suggested_name),
            ).fetchone()
        return self._row_to_suggestion(row) if row else None

    def list_by_run(self, candidate_run_id: str) -> list[PaperMaterialNameSuggestion]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM paper_material_name_suggestions
                WHERE candidate_run_id = ?
                ORDER BY paper_material_id ASC, confidence DESC, created_at DESC
                """,
                (candidate_run_id,),
            ).fetchall()
        return [self._row_to_suggestion(row) for row in rows]

    @staticmethod
    def _suggestion_values(suggestion: PaperMaterialNameSuggestion) -> tuple[object, ...]:
        return (
            suggestion.suggestion_id,
            suggestion.paper_id,
            suggestion.candidate_run_id,
            suggestion.paper_material_id,
            suggestion.agent_name,
            suggestion.original_name,
            suggestion.suggested_name,
            suggestion.suggested_full_name_in_paper,
            suggestion.suggested_abbreviation,
            suggestion.suggested_normalized_name,
            suggestion.suggested_canonical_name,
            suggestion.confidence,
            suggestion.reason,
            json.dumps(suggestion.evidence, ensure_ascii=False, sort_keys=True),
            suggestion.status,
            suggestion.created_at,
            suggestion.updated_at,
        )

    @staticmethod
    def _row_to_suggestion(row: sqlite3.Row) -> PaperMaterialNameSuggestion:
        payload = dict(row)
        payload["evidence"] = json.loads(payload.pop("evidence_json") or "{}")
        return PaperMaterialNameSuggestion.model_validate(payload)


class MaterialResolutionTaskRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def upsert(self, task: MaterialResolutionTask) -> MaterialResolutionTask:
        current = self.get_by_paper_material(task.candidate_run_id, task.paper_material_id)
        created_at = current.created_at if current else task.created_at
        if current:
            preserve_orchestration = (
                task.current_stage == "unresolved" and current.current_stage != "unresolved"
            )
            task = task.model_copy(
                update={
                    "status": current.status if preserve_orchestration else task.status,
                    "assigned_strategy": (
                        current.assigned_strategy
                        if preserve_orchestration
                        else task.assigned_strategy
                    ),
                    "current_stage": (
                        current.current_stage
                        if task.current_stage == "unresolved"
                        else task.current_stage
                    ),
                    "next_action": (
                        current.next_action if task.next_action == "resolve" else task.next_action
                    ),
                    "retry_count": max(current.retry_count, task.retry_count),
                    "stage_timings": {
                        **current.stage_timings,
                        **task.stage_timings,
                    },
                    "stage_errors": {
                        **current.stage_errors,
                        **task.stage_errors,
                    },
                    "error_message": (
                        current.error_message if preserve_orchestration else task.error_message
                    ),
                    "completed_at": (
                        current.completed_at if preserve_orchestration else task.completed_at
                    ),
                }
            )
        updated = task.model_copy(update={"created_at": created_at, "updated_at": now_iso()})
        with self.database.connect() as conn:
            conn.execute(
                """
                INSERT INTO material_resolution_tasks (
                  task_id, paper_id, candidate_run_id, paper_material_id,
                  material_mentions_json, material_context_json, priority,
                  status, assigned_strategy, current_stage, next_action,
                  retry_count, stage_timings_json, stage_errors_json,
                  error_message, created_at, updated_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(candidate_run_id, paper_material_id) DO UPDATE SET
                  material_mentions_json=excluded.material_mentions_json,
                  material_context_json=excluded.material_context_json,
                  priority=excluded.priority,
                  status=excluded.status,
                  assigned_strategy=excluded.assigned_strategy,
                  current_stage=excluded.current_stage,
                  next_action=excluded.next_action,
                  retry_count=excluded.retry_count,
                  stage_timings_json=excluded.stage_timings_json,
                  stage_errors_json=excluded.stage_errors_json,
                  error_message=excluded.error_message,
                  updated_at=excluded.updated_at,
                  completed_at=excluded.completed_at
                """,
                self._task_values(updated),
            )
        return updated

    def get_by_paper_material(
        self,
        candidate_run_id: str,
        paper_material_id: str,
    ) -> MaterialResolutionTask | None:
        with self.database.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM material_resolution_tasks
                WHERE candidate_run_id = ? AND paper_material_id = ?
                """,
                (candidate_run_id, paper_material_id),
            ).fetchone()
        return self._row_to_task(row) if row else None

    def list_by_run(self, candidate_run_id: str) -> list[MaterialResolutionTask]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM material_resolution_tasks
                WHERE candidate_run_id = ?
                ORDER BY paper_material_id ASC
                """,
                (candidate_run_id,),
            ).fetchall()
        return [self._row_to_task(row) for row in rows]

    def list(self) -> list[MaterialResolutionTask]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM material_resolution_tasks
                ORDER BY updated_at DESC, paper_id ASC, paper_material_id ASC
                """
            ).fetchall()
        return [self._row_to_task(row) for row in rows]

    def delete_by_paper_material(self, candidate_run_id: str, paper_material_id: str) -> None:
        with self.database.connect() as conn:
            conn.execute(
                """
                DELETE FROM material_resolution_tasks
                WHERE candidate_run_id = ? AND paper_material_id = ?
                """,
                (candidate_run_id, paper_material_id),
            )

    @staticmethod
    def _task_values(task: MaterialResolutionTask) -> tuple[object, ...]:
        return (
            task.task_id,
            task.paper_id,
            task.candidate_run_id,
            task.paper_material_id,
            json.dumps(task.material_mentions, ensure_ascii=False),
            json.dumps(task.material_context, ensure_ascii=False, sort_keys=True),
            task.priority,
            task.status,
            task.assigned_strategy,
            task.current_stage,
            task.next_action,
            task.retry_count,
            json.dumps(task.stage_timings, ensure_ascii=False, sort_keys=True),
            json.dumps(task.stage_errors, ensure_ascii=False, sort_keys=True),
            task.error_message,
            task.created_at,
            task.updated_at,
            task.completed_at,
        )

    @staticmethod
    def _row_to_task(row: sqlite3.Row) -> MaterialResolutionTask:
        payload = dict(row)
        payload["material_mentions"] = json.loads(payload.pop("material_mentions_json") or "[]")
        payload["material_context"] = json.loads(payload.pop("material_context_json") or "{}")
        payload["stage_timings"] = json.loads(payload.pop("stage_timings_json") or "{}")
        payload["stage_errors"] = json.loads(payload.pop("stage_errors_json") or "{}")
        return MaterialResolutionTask.model_validate(payload)


class MaterialStructureCandidateRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def upsert(self, candidate: MaterialStructureCandidate) -> MaterialStructureCandidate:
        current = self.get_by_source(
            candidate.candidate_run_id,
            candidate.paper_material_id,
            candidate.provider,
            candidate.source_identifier,
        )
        created_at = current.created_at if current else candidate.created_at
        updated = candidate.model_copy(update={"created_at": created_at, "updated_at": now_iso()})
        with self.database.connect() as conn:
            conn.execute(
                """
                INSERT INTO material_structure_candidates (
                  structure_candidate_id, paper_id, candidate_run_id,
                  paper_material_id, provider, resolver_name, query_text,
                  query_type, source_identifier, source_url, canonical_name,
                  material_class, representation_type, raw_smiles,
                  canonical_smiles, isomeric_smiles, inchi, inchi_key,
                  formula, molecular_weight, synonyms_json, evidence_json,
                  confidence, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(candidate_run_id, paper_material_id, provider, source_identifier)
                DO UPDATE SET
                  resolver_name=excluded.resolver_name,
                  query_text=excluded.query_text,
                  query_type=excluded.query_type,
                  source_url=excluded.source_url,
                  canonical_name=excluded.canonical_name,
                  material_class=excluded.material_class,
                  representation_type=excluded.representation_type,
                  raw_smiles=excluded.raw_smiles,
                  canonical_smiles=excluded.canonical_smiles,
                  isomeric_smiles=excluded.isomeric_smiles,
                  inchi=excluded.inchi,
                  inchi_key=excluded.inchi_key,
                  formula=excluded.formula,
                  molecular_weight=excluded.molecular_weight,
                  synonyms_json=excluded.synonyms_json,
                  evidence_json=excluded.evidence_json,
                  confidence=excluded.confidence,
                  status=excluded.status,
                  updated_at=excluded.updated_at
                """,
                self._candidate_values(updated),
            )
        return updated

    def get_by_source(
        self,
        candidate_run_id: str,
        paper_material_id: str,
        provider: str,
        source_identifier: str | None,
    ) -> MaterialStructureCandidate | None:
        if source_identifier is None:
            return None
        with self.database.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM material_structure_candidates
                WHERE candidate_run_id = ?
                  AND paper_material_id = ?
                  AND provider = ?
                  AND source_identifier = ?
                """,
                (candidate_run_id, paper_material_id, provider, source_identifier),
            ).fetchone()
        return self._row_to_candidate(row) if row else None

    def get(self, structure_candidate_id: str) -> MaterialStructureCandidate | None:
        with self.database.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM material_structure_candidates
                WHERE structure_candidate_id = ?
                """,
                (structure_candidate_id,),
            ).fetchone()
        return self._row_to_candidate(row) if row else None

    def set_status(
        self,
        structure_candidate_id: str,
        status: str,
    ) -> MaterialStructureCandidate | None:
        timestamp = now_iso()
        with self.database.connect() as conn:
            conn.execute(
                """
                UPDATE material_structure_candidates
                SET status = ?, updated_at = ?
                WHERE structure_candidate_id = ?
                """,
                (status, timestamp, structure_candidate_id),
            )
        return self.get(structure_candidate_id)

    def update(self, candidate: MaterialStructureCandidate) -> MaterialStructureCandidate:
        with self.database.connect() as conn:
            conn.execute(
                """
                UPDATE material_structure_candidates
                SET resolver_name = ?, query_text = ?, query_type = ?,
                    source_url = ?, canonical_name = ?, material_class = ?,
                    representation_type = ?, raw_smiles = ?, canonical_smiles = ?,
                    isomeric_smiles = ?, inchi = ?, inchi_key = ?, formula = ?,
                    molecular_weight = ?, synonyms_json = ?, evidence_json = ?,
                    confidence = ?, status = ?, updated_at = ?
                WHERE structure_candidate_id = ?
                """,
                (
                    candidate.resolver_name,
                    candidate.query_text,
                    candidate.query_type,
                    candidate.source_url,
                    candidate.canonical_name,
                    candidate.material_class,
                    candidate.representation_type,
                    candidate.raw_smiles,
                    candidate.canonical_smiles,
                    candidate.isomeric_smiles,
                    candidate.inchi,
                    candidate.inchi_key,
                    candidate.formula,
                    candidate.molecular_weight,
                    json.dumps(candidate.synonyms, ensure_ascii=False),
                    json.dumps(candidate.evidence, ensure_ascii=False, sort_keys=True),
                    candidate.confidence,
                    candidate.status,
                    candidate.updated_at,
                    candidate.structure_candidate_id,
                ),
            )
        return self.get(candidate.structure_candidate_id) or candidate

    def list_by_run(self, candidate_run_id: str) -> list[MaterialStructureCandidate]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM material_structure_candidates
                WHERE candidate_run_id = ?
                ORDER BY paper_material_id ASC, confidence DESC, provider ASC
                """,
                (candidate_run_id,),
            ).fetchall()
        return [self._row_to_candidate(row) for row in rows]

    def list_by_paper_material(
        self,
        candidate_run_id: str,
        paper_material_id: str,
    ) -> list[MaterialStructureCandidate]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM material_structure_candidates
                WHERE candidate_run_id = ? AND paper_material_id = ?
                ORDER BY confidence DESC, provider ASC, source_identifier ASC
                """,
                (candidate_run_id, paper_material_id),
            ).fetchall()
        return [self._row_to_candidate(row) for row in rows]

    @staticmethod
    def _candidate_values(candidate: MaterialStructureCandidate) -> tuple[object, ...]:
        return (
            candidate.structure_candidate_id,
            candidate.paper_id,
            candidate.candidate_run_id,
            candidate.paper_material_id,
            candidate.provider,
            candidate.resolver_name,
            candidate.query_text,
            candidate.query_type,
            candidate.source_identifier,
            candidate.source_url,
            candidate.canonical_name,
            candidate.material_class,
            candidate.representation_type,
            candidate.raw_smiles,
            candidate.canonical_smiles,
            candidate.isomeric_smiles,
            candidate.inchi,
            candidate.inchi_key,
            candidate.formula,
            candidate.molecular_weight,
            json.dumps(candidate.synonyms, ensure_ascii=False),
            json.dumps(candidate.evidence, ensure_ascii=False, sort_keys=True),
            candidate.confidence,
            candidate.status,
            candidate.created_at,
            candidate.updated_at,
        )

    @staticmethod
    def _row_to_candidate(row: sqlite3.Row) -> MaterialStructureCandidate:
        payload = dict(row)
        payload["synonyms"] = json.loads(payload.pop("synonyms_json") or "[]")
        payload["evidence"] = json.loads(payload.pop("evidence_json") or "{}")
        return MaterialStructureCandidate.model_validate(payload)


class MaterialIdentityJudgmentRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def add(self, judgment: MaterialIdentityJudgment) -> MaterialIdentityJudgment:
        with self.database.connect() as conn:
            conn.execute(
                """
                INSERT INTO material_identity_judgments (
                  judgment_id, paper_id, candidate_run_id, paper_material_id,
                  structure_candidate_id, provider, model, prompt_version,
                  verdict, confidence, supporting_evidence_json, conflicts_json,
                  recommended_action, deterministic_checks_json,
                  input_context_json, raw_response_json, status,
                  error_message, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._values(judgment),
            )
        return judgment

    def list_by_run(self, candidate_run_id: str) -> list[MaterialIdentityJudgment]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM material_identity_judgments
                WHERE candidate_run_id = ?
                ORDER BY created_at DESC, rowid DESC
                """,
                (candidate_run_id,),
            ).fetchall()
        return [self._row_to_judgment(row) for row in rows]

    def latest_by_candidate(self, structure_candidate_id: str) -> MaterialIdentityJudgment | None:
        with self.database.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM material_identity_judgments
                WHERE structure_candidate_id = ?
                ORDER BY created_at DESC, rowid DESC
                LIMIT 1
                """,
                (structure_candidate_id,),
            ).fetchone()
        return self._row_to_judgment(row) if row else None

    @staticmethod
    def _values(judgment: MaterialIdentityJudgment) -> tuple[object, ...]:
        return (
            judgment.judgment_id,
            judgment.paper_id,
            judgment.candidate_run_id,
            judgment.paper_material_id,
            judgment.structure_candidate_id,
            judgment.provider,
            judgment.model,
            judgment.prompt_version,
            judgment.verdict,
            judgment.confidence,
            json.dumps(judgment.supporting_evidence, ensure_ascii=False),
            json.dumps(judgment.conflicts, ensure_ascii=False),
            judgment.recommended_action,
            json.dumps(judgment.deterministic_checks, ensure_ascii=False, sort_keys=True),
            json.dumps(judgment.input_context, ensure_ascii=False, sort_keys=True),
            json.dumps(judgment.raw_response, ensure_ascii=False, sort_keys=True),
            judgment.status,
            judgment.error_message,
            judgment.created_at,
            judgment.updated_at,
        )

    @staticmethod
    def _row_to_judgment(row: sqlite3.Row) -> MaterialIdentityJudgment:
        payload = dict(row)
        payload["supporting_evidence"] = json.loads(payload.pop("supporting_evidence_json") or "[]")
        payload["conflicts"] = json.loads(payload.pop("conflicts_json") or "[]")
        payload["deterministic_checks"] = json.loads(
            payload.pop("deterministic_checks_json") or "{}"
        )
        payload["input_context"] = json.loads(payload.pop("input_context_json") or "{}")
        payload["raw_response"] = json.loads(payload.pop("raw_response_json") or "{}")
        return MaterialIdentityJudgment.model_validate(payload)


class MaterialIdentityEvidenceRunRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def upsert(self, run: MaterialIdentityEvidenceRun) -> MaterialIdentityEvidenceRun:
        with self.database.connect() as conn:
            conn.execute(
                """
                INSERT INTO material_identity_evidence_runs (
                  evidence_run_id, paper_id, candidate_run_id, paper_material_id,
                  trigger_judgment_id, provider, model, prompt_version, strategy,
                  query_plan_json, status, generated_candidate_ids_json,
                  recommended_next_action, raw_response_json, error_message,
                  created_at, updated_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(evidence_run_id) DO UPDATE SET
                  status=excluded.status,
                  generated_candidate_ids_json=excluded.generated_candidate_ids_json,
                  recommended_next_action=excluded.recommended_next_action,
                  raw_response_json=excluded.raw_response_json,
                  error_message=excluded.error_message,
                  updated_at=excluded.updated_at,
                  completed_at=excluded.completed_at
                """,
                self._values(run),
            )
        return run

    def list_by_run(self, candidate_run_id: str) -> list[MaterialIdentityEvidenceRun]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM material_identity_evidence_runs
                WHERE candidate_run_id = ?
                ORDER BY created_at DESC, rowid DESC
                """,
                (candidate_run_id,),
            ).fetchall()
        return [self._row_to_run(row) for row in rows]

    @staticmethod
    def _values(run: MaterialIdentityEvidenceRun) -> tuple[object, ...]:
        return (
            run.evidence_run_id,
            run.paper_id,
            run.candidate_run_id,
            run.paper_material_id,
            run.trigger_judgment_id,
            run.provider,
            run.model,
            run.prompt_version,
            run.strategy,
            json.dumps(run.query_plan, ensure_ascii=False),
            run.status,
            json.dumps(run.generated_candidate_ids, ensure_ascii=False),
            run.recommended_next_action,
            json.dumps(run.raw_response, ensure_ascii=False, sort_keys=True),
            run.error_message,
            run.created_at,
            run.updated_at,
            run.completed_at,
        )

    @staticmethod
    def _row_to_run(row: sqlite3.Row) -> MaterialIdentityEvidenceRun:
        payload = dict(row)
        payload["query_plan"] = json.loads(payload.pop("query_plan_json") or "[]")
        payload["generated_candidate_ids"] = json.loads(
            payload.pop("generated_candidate_ids_json") or "[]"
        )
        payload["raw_response"] = json.loads(payload.pop("raw_response_json") or "{}")
        return MaterialIdentityEvidenceRun.model_validate(payload)


class MaterialIdentityEvidenceItemRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def add(self, item: MaterialIdentityEvidenceItem) -> MaterialIdentityEvidenceItem:
        with self.database.connect() as conn:
            conn.execute(
                """
                INSERT INTO material_identity_evidence_items (
                  evidence_item_id, evidence_run_id, paper_id, candidate_run_id,
                  paper_material_id, source_type, source_tier, source_title,
                  source_url, query_text, excerpt, alias, full_name, cas_number,
                  pubchem_cid, explicitly_linked, confidence, extraction_json,
                  raw_source_json, review_status, reviewed_by, review_note,
                  reviewed_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._values(item),
            )
        return item

    def get(self, evidence_item_id: str) -> MaterialIdentityEvidenceItem | None:
        with self.database.connect() as conn:
            row = conn.execute(
                "SELECT * FROM material_identity_evidence_items WHERE evidence_item_id = ?",
                (evidence_item_id,),
            ).fetchone()
        return self._row_to_item(row) if row else None

    def update_review(
        self,
        evidence_item_id: str,
        *,
        review_status: str,
        reviewed_by: str,
        review_note: str | None,
    ) -> MaterialIdentityEvidenceItem | None:
        timestamp = now_iso()
        with self.database.connect() as conn:
            conn.execute(
                """
                UPDATE material_identity_evidence_items
                SET review_status = ?, reviewed_by = ?, review_note = ?,
                    reviewed_at = ?, updated_at = ?
                WHERE evidence_item_id = ?
                """,
                (
                    review_status,
                    reviewed_by,
                    review_note,
                    timestamp,
                    timestamp,
                    evidence_item_id,
                ),
            )
        return self.get(evidence_item_id)

    def list_by_run(self, candidate_run_id: str) -> list[MaterialIdentityEvidenceItem]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM material_identity_evidence_items
                WHERE candidate_run_id = ?
                ORDER BY created_at DESC, rowid DESC
                """,
                (candidate_run_id,),
            ).fetchall()
        return [self._row_to_item(row) for row in rows]

    def list_by_material(
        self,
        candidate_run_id: str,
        paper_material_id: str,
    ) -> list[MaterialIdentityEvidenceItem]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM material_identity_evidence_items
                WHERE candidate_run_id = ? AND paper_material_id = ?
                ORDER BY created_at DESC, rowid DESC
                """,
                (candidate_run_id, paper_material_id),
            ).fetchall()
        return [self._row_to_item(row) for row in rows]

    @staticmethod
    def _values(item: MaterialIdentityEvidenceItem) -> tuple[object, ...]:
        return (
            item.evidence_item_id,
            item.evidence_run_id,
            item.paper_id,
            item.candidate_run_id,
            item.paper_material_id,
            item.source_type,
            item.source_tier,
            item.source_title,
            item.source_url,
            item.query_text,
            item.excerpt,
            item.alias,
            item.full_name,
            item.cas_number,
            item.pubchem_cid,
            int(item.explicitly_linked),
            item.confidence,
            json.dumps(item.extraction, ensure_ascii=False, sort_keys=True),
            json.dumps(item.raw_source, ensure_ascii=False, sort_keys=True),
            item.review_status,
            item.reviewed_by,
            item.review_note,
            item.reviewed_at,
            item.created_at,
            item.updated_at,
        )

    @staticmethod
    def _row_to_item(row: sqlite3.Row) -> MaterialIdentityEvidenceItem:
        payload = dict(row)
        payload["explicitly_linked"] = bool(payload["explicitly_linked"])
        payload["extraction"] = json.loads(payload.pop("extraction_json") or "{}")
        payload["raw_source"] = json.loads(payload.pop("raw_source_json") or "{}")
        return MaterialIdentityEvidenceItem.model_validate(payload)


class MaterialReviewEventRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def add(self, event: MaterialReviewEvent) -> MaterialReviewEvent:
        with self.database.connect() as conn:
            conn.execute(
                """
                INSERT INTO material_review_events (
                  event_id, paper_id, candidate_run_id, paper_material_id,
                  structure_candidate_id, global_material_id, action, actor,
                  message, before_candidate_status, after_candidate_status,
                  before_link_json, after_link_json, before_task_json,
                  after_task_json, before_candidate_json, after_candidate_json,
                  created_global_material_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._event_values(event),
            )
        return event

    def list_by_run(self, candidate_run_id: str) -> list[MaterialReviewEvent]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM material_review_events
                WHERE candidate_run_id = ?
                ORDER BY created_at ASC, rowid ASC
                """,
                (candidate_run_id,),
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def get(self, event_id: str) -> MaterialReviewEvent | None:
        with self.database.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM material_review_events
                WHERE event_id = ?
                """,
                (event_id,),
            ).fetchone()
        return self._row_to_event(row) if row else None

    @staticmethod
    def _event_values(event: MaterialReviewEvent) -> tuple[object, ...]:
        return (
            event.event_id,
            event.paper_id,
            event.candidate_run_id,
            event.paper_material_id,
            event.structure_candidate_id,
            event.global_material_id,
            event.action,
            event.actor,
            event.message,
            event.before_candidate_status,
            event.after_candidate_status,
            json.dumps(event.before_link, ensure_ascii=False, sort_keys=True)
            if event.before_link is not None
            else None,
            json.dumps(event.after_link, ensure_ascii=False, sort_keys=True)
            if event.after_link is not None
            else None,
            json.dumps(event.before_task, ensure_ascii=False, sort_keys=True)
            if event.before_task is not None
            else None,
            json.dumps(event.after_task, ensure_ascii=False, sort_keys=True)
            if event.after_task is not None
            else None,
            json.dumps(event.before_candidate, ensure_ascii=False, sort_keys=True)
            if event.before_candidate is not None
            else None,
            json.dumps(event.after_candidate, ensure_ascii=False, sort_keys=True)
            if event.after_candidate is not None
            else None,
            event.created_global_material_id,
            event.created_at,
        )

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> MaterialReviewEvent:
        payload = dict(row)
        for key in (
            "before_link",
            "after_link",
            "before_task",
            "after_task",
            "before_candidate",
            "after_candidate",
        ):
            raw = payload.pop(f"{key}_json")
            payload[key] = json.loads(raw) if raw else None
        return MaterialReviewEvent.model_validate(payload)


class MaterialPropertyCandidateRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def upsert(self, candidate: MaterialPropertyCandidate) -> MaterialPropertyCandidate:
        current = self.get(candidate.property_candidate_id)
        created_at = current.created_at if current else candidate.created_at
        updated = candidate.model_copy(update={"created_at": created_at, "updated_at": now_iso()})
        with self.database.connect() as conn:
            conn.execute(
                """
                INSERT INTO material_property_candidates (
                  property_candidate_id, paper_id, candidate_run_id, paper_material_id,
                  global_material_id, property_name, property_category, value_numeric,
                  value_text, value_raw, unit, normalized_value_numeric, normalized_unit,
                  condition_json, method, source_type, evidence_text, llm_evidence_text,
                  source_block_text, evidence_anchor_json, provider, model, prompt_version,
                  confidence, status, error_message, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(property_candidate_id) DO UPDATE SET
                  global_material_id=excluded.global_material_id,
                  property_name=excluded.property_name,
                  property_category=excluded.property_category,
                  value_numeric=excluded.value_numeric,
                  value_text=excluded.value_text,
                  value_raw=excluded.value_raw,
                  unit=excluded.unit,
                  normalized_value_numeric=excluded.normalized_value_numeric,
                  normalized_unit=excluded.normalized_unit,
                  condition_json=excluded.condition_json,
                  method=excluded.method,
                  source_type=excluded.source_type,
                  evidence_text=excluded.evidence_text,
                  llm_evidence_text=excluded.llm_evidence_text,
                  source_block_text=excluded.source_block_text,
                  evidence_anchor_json=excluded.evidence_anchor_json,
                  provider=excluded.provider,
                  model=excluded.model,
                  prompt_version=excluded.prompt_version,
                  confidence=excluded.confidence,
                  status=excluded.status,
                  error_message=excluded.error_message,
                  updated_at=excluded.updated_at
                """,
                self._candidate_values(updated),
            )
        return updated

    def get(self, property_candidate_id: str) -> MaterialPropertyCandidate | None:
        with self.database.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM material_property_candidates
                WHERE property_candidate_id = ?
                """,
                (property_candidate_id,),
            ).fetchone()
        return self._row_to_candidate(row) if row else None

    def list_by_run(self, candidate_run_id: str) -> list[MaterialPropertyCandidate]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM material_property_candidates
                WHERE candidate_run_id = ?
                ORDER BY paper_material_id ASC, property_name ASC, confidence DESC, rowid ASC
                """,
                (candidate_run_id,),
            ).fetchall()
        return [self._row_to_candidate(row) for row in rows]

    def list_by_material(
        self,
        candidate_run_id: str,
        paper_material_id: str,
    ) -> list[MaterialPropertyCandidate]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM material_property_candidates
                WHERE candidate_run_id = ? AND paper_material_id = ?
                ORDER BY property_name ASC, confidence DESC, rowid ASC
                """,
                (candidate_run_id, paper_material_id),
            ).fetchall()
        return [self._row_to_candidate(row) for row in rows]

    def update_status(
        self,
        property_candidate_id: str,
        status: str,
        *,
        error_message: str | None = None,
    ) -> MaterialPropertyCandidate | None:
        timestamp = now_iso()
        with self.database.connect() as conn:
            conn.execute(
                """
                UPDATE material_property_candidates
                SET status = ?, error_message = ?, updated_at = ?
                WHERE property_candidate_id = ?
                """,
                (status, error_message, timestamp, property_candidate_id),
            )
        return self.get(property_candidate_id)

    @staticmethod
    def _candidate_values(candidate: MaterialPropertyCandidate) -> tuple[object, ...]:
        return (
            candidate.property_candidate_id,
            candidate.paper_id,
            candidate.candidate_run_id,
            candidate.paper_material_id,
            candidate.global_material_id,
            candidate.property_name,
            candidate.property_category,
            candidate.value_numeric,
            candidate.value_text,
            candidate.value_raw,
            candidate.unit,
            candidate.normalized_value_numeric,
            candidate.normalized_unit,
            json.dumps(candidate.condition, ensure_ascii=False, sort_keys=True),
            candidate.method,
            candidate.source_type,
            candidate.evidence_text,
            candidate.llm_evidence_text,
            candidate.source_block_text,
            json.dumps(candidate.evidence_anchor, ensure_ascii=False, sort_keys=True),
            candidate.provider,
            candidate.model,
            candidate.prompt_version,
            candidate.confidence,
            candidate.status,
            candidate.error_message,
            candidate.created_at,
            candidate.updated_at,
        )

    @staticmethod
    def _row_to_candidate(row: sqlite3.Row) -> MaterialPropertyCandidate:
        payload = dict(row)
        payload["condition"] = json.loads(payload.pop("condition_json") or "{}")
        payload["evidence_anchor"] = json.loads(payload.pop("evidence_anchor_json") or "{}")
        return MaterialPropertyCandidate.model_validate(payload)


class MaterialPropertyReviewRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def add(self, review: MaterialPropertyReview) -> MaterialPropertyReview:
        with self.database.connect() as conn:
            conn.execute(
                """
                INSERT INTO material_property_reviews (
                  review_id, property_candidate_id, paper_id, candidate_run_id,
                  paper_material_id, decision, reviewed_property_name,
                  reviewed_value_numeric, reviewed_value_text, reviewed_unit,
                  reviewed_condition_json, reviewed_evidence_anchor_json,
                  actor, message, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._review_values(review),
            )
        return review

    def list_by_run(self, candidate_run_id: str) -> list[MaterialPropertyReview]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM material_property_reviews
                WHERE candidate_run_id = ?
                ORDER BY created_at DESC, rowid DESC
                """,
                (candidate_run_id,),
            ).fetchall()
        return [self._row_to_review(row) for row in rows]

    def list_by_candidate(self, property_candidate_id: str) -> list[MaterialPropertyReview]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM material_property_reviews
                WHERE property_candidate_id = ?
                ORDER BY created_at DESC, rowid DESC
                """,
                (property_candidate_id,),
            ).fetchall()
        return [self._row_to_review(row) for row in rows]

    @staticmethod
    def _review_values(review: MaterialPropertyReview) -> tuple[object, ...]:
        return (
            review.review_id,
            review.property_candidate_id,
            review.paper_id,
            review.candidate_run_id,
            review.paper_material_id,
            review.decision,
            review.reviewed_property_name,
            review.reviewed_value_numeric,
            review.reviewed_value_text,
            review.reviewed_unit,
            json.dumps(review.reviewed_condition, ensure_ascii=False, sort_keys=True),
            json.dumps(review.reviewed_evidence_anchor, ensure_ascii=False, sort_keys=True),
            review.actor,
            review.message,
            review.created_at,
        )

    @staticmethod
    def _row_to_review(row: sqlite3.Row) -> MaterialPropertyReview:
        payload = dict(row)
        payload["reviewed_condition"] = json.loads(payload.pop("reviewed_condition_json") or "{}")
        payload["reviewed_evidence_anchor"] = json.loads(
            payload.pop("reviewed_evidence_anchor_json") or "{}"
        )
        return MaterialPropertyReview.model_validate(payload)


class MaterialPropertyReviewEventRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def add(self, event: MaterialPropertyReviewEvent) -> MaterialPropertyReviewEvent:
        with self.database.connect() as conn:
            conn.execute(
                """
                INSERT INTO material_property_review_events (
                  event_id, paper_id, candidate_run_id, paper_material_id,
                  property_candidate_id, event_type, before_json, after_json,
                  actor, message, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._event_values(event),
            )
        return event

    def get(self, event_id: str) -> MaterialPropertyReviewEvent | None:
        with self.database.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM material_property_review_events
                WHERE event_id = ?
                """,
                (event_id,),
            ).fetchone()
        return self._row_to_event(row) if row else None

    def list_by_run(self, candidate_run_id: str) -> list[MaterialPropertyReviewEvent]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM material_property_review_events
                WHERE candidate_run_id = ?
                ORDER BY created_at ASC, rowid ASC
                """,
                (candidate_run_id,),
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    @staticmethod
    def _event_values(event: MaterialPropertyReviewEvent) -> tuple[object, ...]:
        return (
            event.event_id,
            event.paper_id,
            event.candidate_run_id,
            event.paper_material_id,
            event.property_candidate_id,
            event.event_type,
            json.dumps(event.before, ensure_ascii=False, sort_keys=True)
            if event.before is not None
            else None,
            json.dumps(event.after, ensure_ascii=False, sort_keys=True)
            if event.after is not None
            else None,
            event.actor,
            event.message,
            event.created_at,
        )

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> MaterialPropertyReviewEvent:
        payload = dict(row)
        for key in ("before", "after"):
            raw = payload.pop(f"{key}_json")
            payload[key] = json.loads(raw) if raw else None
        return MaterialPropertyReviewEvent.model_validate(payload)


class MaterialAgentRunRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def create(self, run: MaterialAgentRun) -> MaterialAgentRun:
        with self.database.connect() as conn:
            conn.execute(
                """
                INSERT INTO material_agent_runs (
                  agent_run_id, paper_id, status, strategy,
                  source_candidate_run_id, mineru_run_id, material_count,
                  visual_block_count, tool_summary_json, error_message,
                  created_at, updated_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._run_values(run),
            )
        return run

    def update(self, run: MaterialAgentRun) -> MaterialAgentRun:
        with self.database.connect() as conn:
            conn.execute(
                """
                UPDATE material_agent_runs
                SET status = ?, strategy = ?, source_candidate_run_id = ?,
                    mineru_run_id = ?, material_count = ?, visual_block_count = ?,
                    tool_summary_json = ?, error_message = ?, updated_at = ?,
                    completed_at = ?
                WHERE agent_run_id = ?
                """,
                (
                    run.status,
                    run.strategy,
                    run.source_candidate_run_id,
                    run.mineru_run_id,
                    run.material_count,
                    run.visual_block_count,
                    json.dumps(run.tool_summary, ensure_ascii=False, sort_keys=True),
                    run.error_message,
                    run.updated_at,
                    run.completed_at,
                    run.agent_run_id,
                ),
            )
        return run

    def get(self, agent_run_id: str) -> MaterialAgentRun | None:
        with self.database.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM material_agent_runs
                WHERE agent_run_id = ?
                """,
                (agent_run_id,),
            ).fetchone()
        return self._row_to_run(row) if row else None

    def list_by_paper(self, paper_id: str) -> list[MaterialAgentRun]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM material_agent_runs
                WHERE paper_id = ?
                ORDER BY created_at DESC, rowid DESC
                """,
                (paper_id,),
            ).fetchall()
        return [self._row_to_run(row) for row in rows]

    @staticmethod
    def _run_values(run: MaterialAgentRun) -> tuple[object, ...]:
        return (
            run.agent_run_id,
            run.paper_id,
            run.status,
            run.strategy,
            run.source_candidate_run_id,
            run.mineru_run_id,
            run.material_count,
            run.visual_block_count,
            json.dumps(run.tool_summary, ensure_ascii=False, sort_keys=True),
            run.error_message,
            run.created_at,
            run.updated_at,
            run.completed_at,
        )

    @staticmethod
    def _row_to_run(row: sqlite3.Row) -> MaterialAgentRun:
        payload = dict(row)
        payload["tool_summary"] = json.loads(payload.pop("tool_summary_json") or "{}")
        return MaterialAgentRun.model_validate(payload)


class DocumentVisualBlockRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def replace_for_mineru_run(
        self,
        mineru_run_id: str,
        blocks: list[DocumentVisualBlock],
    ) -> list[DocumentVisualBlock]:
        timestamp = now_iso()
        rows = [
            block.model_copy(
                update={"created_at": block.created_at or timestamp, "updated_at": timestamp}
            )
            for block in blocks
        ]
        with self.database.connect() as conn:
            conn.execute(
                "DELETE FROM document_visual_blocks WHERE mineru_run_id = ?",
                (mineru_run_id,),
            )
            conn.executemany(
                """
                INSERT INTO document_visual_blocks (
                  visual_block_id, paper_id, mineru_run_id, collected_by_agent_run_id,
                  content_index, content_type, sub_type, page_idx, page_id,
                  bbox_json, img_path, resolved_img_path, image_exists,
                  caption, nearby_text, source_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [self._block_values(block) for block in rows],
            )
        return rows

    def get(self, visual_block_id: str) -> DocumentVisualBlock | None:
        with self.database.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM document_visual_blocks
                WHERE visual_block_id = ?
                """,
                (visual_block_id,),
            ).fetchone()
        return self._row_to_block(row) if row else None

    def list_by_paper(self, paper_id: str) -> list[DocumentVisualBlock]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM document_visual_blocks
                WHERE paper_id = ?
                ORDER BY page_id ASC, content_index ASC
                """,
                (paper_id,),
            ).fetchall()
        return [self._row_to_block(row) for row in rows]

    def list_by_mineru_run(self, mineru_run_id: str) -> list[DocumentVisualBlock]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM document_visual_blocks
                WHERE mineru_run_id = ?
                ORDER BY content_index ASC
                """,
                (mineru_run_id,),
            ).fetchall()
        return [self._row_to_block(row) for row in rows]

    @staticmethod
    def _block_values(block: DocumentVisualBlock) -> tuple[object, ...]:
        return (
            block.visual_block_id,
            block.paper_id,
            block.mineru_run_id,
            block.collected_by_agent_run_id,
            block.content_index,
            block.content_type,
            block.sub_type,
            block.page_idx,
            block.page_id,
            json.dumps(block.bbox, ensure_ascii=False),
            block.img_path,
            block.resolved_img_path,
            1 if block.image_exists else 0,
            block.caption,
            block.nearby_text,
            json.dumps(block.source_json, ensure_ascii=False, sort_keys=True),
            block.created_at,
            block.updated_at,
        )

    @staticmethod
    def _row_to_block(row: sqlite3.Row) -> DocumentVisualBlock:
        payload = dict(row)
        payload["bbox"] = json.loads(payload.pop("bbox_json") or "[]")
        payload["source_json"] = json.loads(payload.pop("source_json") or "{}")
        payload["image_exists"] = bool(payload["image_exists"])
        return DocumentVisualBlock.model_validate(payload)


class FigureTriageResultRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def upsert(self, result: FigureTriageResult) -> FigureTriageResult:
        current = self.get_by_run_block(result.agent_run_id, result.visual_block_id)
        created_at = current.created_at if current else result.created_at
        updated = result.model_copy(update={"created_at": created_at, "updated_at": now_iso()})
        with self.database.connect() as conn:
            conn.execute(
                """
                INSERT INTO figure_triage_results (
                  triage_result_id, agent_run_id, visual_block_id, paper_id,
                  provider, model, contains_molecular_structures, image_role,
                  has_clean_structure_depictions, has_orbital_overlay,
                  has_energy_level_diagram, has_device_stack,
                  should_run_decimer_segmentation, label_candidates_json,
                  related_paper_material_ids_json, confidence, reason,
                  raw_response_json, status, error_message, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(agent_run_id, visual_block_id) DO UPDATE SET
                  provider=excluded.provider,
                  model=excluded.model,
                  contains_molecular_structures=excluded.contains_molecular_structures,
                  image_role=excluded.image_role,
                  has_clean_structure_depictions=excluded.has_clean_structure_depictions,
                  has_orbital_overlay=excluded.has_orbital_overlay,
                  has_energy_level_diagram=excluded.has_energy_level_diagram,
                  has_device_stack=excluded.has_device_stack,
                  should_run_decimer_segmentation=excluded.should_run_decimer_segmentation,
                  label_candidates_json=excluded.label_candidates_json,
                  related_paper_material_ids_json=excluded.related_paper_material_ids_json,
                  confidence=excluded.confidence,
                  reason=excluded.reason,
                  raw_response_json=excluded.raw_response_json,
                  status=excluded.status,
                  error_message=excluded.error_message,
                  updated_at=excluded.updated_at
                """,
                self._result_values(updated),
            )
        return updated

    def get_by_run_block(
        self,
        agent_run_id: str,
        visual_block_id: str,
    ) -> FigureTriageResult | None:
        with self.database.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM figure_triage_results
                WHERE agent_run_id = ? AND visual_block_id = ?
                """,
                (agent_run_id, visual_block_id),
            ).fetchone()
        return self._row_to_result(row) if row else None

    def list_by_run(self, agent_run_id: str) -> list[FigureTriageResult]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM figure_triage_results
                WHERE agent_run_id = ?
                ORDER BY visual_block_id ASC
                """,
                (agent_run_id,),
            ).fetchall()
        return [self._row_to_result(row) for row in rows]

    def list_by_paper(self, paper_id: str) -> list[FigureTriageResult]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM figure_triage_results
                WHERE paper_id = ?
                ORDER BY created_at DESC, visual_block_id ASC
                """,
                (paper_id,),
            ).fetchall()
        return [self._row_to_result(row) for row in rows]

    @staticmethod
    def _result_values(result: FigureTriageResult) -> tuple[object, ...]:
        return (
            result.triage_result_id,
            result.agent_run_id,
            result.visual_block_id,
            result.paper_id,
            result.provider,
            result.model,
            1 if result.contains_molecular_structures else 0,
            result.image_role,
            1 if result.has_clean_structure_depictions else 0,
            1 if result.has_orbital_overlay else 0,
            1 if result.has_energy_level_diagram else 0,
            1 if result.has_device_stack else 0,
            1 if result.should_run_decimer_segmentation else 0,
            json.dumps(result.label_candidates, ensure_ascii=False),
            json.dumps(result.related_paper_material_ids, ensure_ascii=False),
            result.confidence,
            result.reason,
            json.dumps(result.raw_response, ensure_ascii=False, sort_keys=True),
            result.status,
            result.error_message,
            result.created_at,
            result.updated_at,
        )

    @staticmethod
    def _row_to_result(row: sqlite3.Row) -> FigureTriageResult:
        payload = dict(row)
        payload["contains_molecular_structures"] = bool(payload["contains_molecular_structures"])
        payload["has_clean_structure_depictions"] = bool(payload["has_clean_structure_depictions"])
        payload["has_orbital_overlay"] = bool(payload["has_orbital_overlay"])
        payload["has_energy_level_diagram"] = bool(payload["has_energy_level_diagram"])
        payload["has_device_stack"] = bool(payload["has_device_stack"])
        payload["should_run_decimer_segmentation"] = bool(
            payload["should_run_decimer_segmentation"]
        )
        payload["label_candidates"] = json.loads(payload.pop("label_candidates_json") or "[]")
        payload["related_paper_material_ids"] = json.loads(
            payload.pop("related_paper_material_ids_json") or "[]"
        )
        payload["raw_response"] = json.loads(payload.pop("raw_response_json") or "{}")
        return FigureTriageResult.model_validate(payload)


class MoleculeCropRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def replace_for_triage_result(
        self,
        triage_result_id: str,
        crops: list[MoleculeCrop],
    ) -> list[MoleculeCrop]:
        timestamp = now_iso()
        rows = [crop.model_copy(update={"updated_at": timestamp}) for crop in crops]
        with self.database.connect() as conn:
            conn.execute(
                "DELETE FROM molecule_crops WHERE triage_result_id = ?",
                (triage_result_id,),
            )
            conn.executemany(
                """
                INSERT INTO molecule_crops (
                  crop_id, paper_id, agent_run_id, triage_result_id, visual_block_id,
                  segment_index, bbox_json, source_image_path, crop_path, width, height,
                  segmentation_confidence, validation_json, raw_segment_json, status,
                  created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [self._crop_values(crop) for crop in rows],
            )
        return rows

    def list_by_paper(self, paper_id: str) -> list[MoleculeCrop]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM molecule_crops
                WHERE paper_id = ?
                ORDER BY created_at DESC, visual_block_id ASC, segment_index ASC
                """,
                (paper_id,),
            ).fetchall()
        return [self._row_to_crop(row) for row in rows]

    def list_by_run(self, agent_run_id: str) -> list[MoleculeCrop]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM molecule_crops
                WHERE agent_run_id = ?
                ORDER BY visual_block_id ASC, segment_index ASC
                """,
                (agent_run_id,),
            ).fetchall()
        return [self._row_to_crop(row) for row in rows]

    def get(self, crop_id: str) -> MoleculeCrop | None:
        with self.database.connect() as conn:
            row = conn.execute(
                "SELECT * FROM molecule_crops WHERE crop_id = ?",
                (crop_id,),
            ).fetchone()
        return self._row_to_crop(row) if row else None

    def apply_validation(self, validation: MoleculeCropValidation) -> MoleculeCrop | None:
        crop = self.get(validation.crop_id)
        if not crop:
            return None
        summary = {
            **crop.validation_json,
            "latest_crop_validation_id": validation.validation_id,
            "latest_crop_validation_model": validation.model,
            "should_run_ocsr": validation.should_run_ocsr,
            "validation_confidence": validation.confidence,
            "validation_reason": validation.reason,
        }
        status = (
            "ready_for_ocsr"
            if validation.status == "completed" and validation.should_run_ocsr
            else "rejected_by_validator"
            if validation.status == "completed"
            else "validation_failed"
        )
        with self.database.connect() as conn:
            conn.execute(
                """
                UPDATE molecule_crops
                SET validation_json = ?, status = ?, updated_at = ?
                WHERE crop_id = ?
                """,
                (
                    json.dumps(summary, ensure_ascii=False, sort_keys=True),
                    status,
                    now_iso(),
                    validation.crop_id,
                ),
            )
        return self.get(validation.crop_id)

    @staticmethod
    def _crop_values(crop: MoleculeCrop) -> tuple[object, ...]:
        return (
            crop.crop_id,
            crop.paper_id,
            crop.agent_run_id,
            crop.triage_result_id,
            crop.visual_block_id,
            crop.segment_index,
            json.dumps(crop.bbox, ensure_ascii=False),
            crop.source_image_path,
            crop.crop_path,
            crop.width,
            crop.height,
            crop.segmentation_confidence,
            json.dumps(crop.validation_json, ensure_ascii=False, sort_keys=True),
            json.dumps(crop.raw_segment, ensure_ascii=False, sort_keys=True),
            crop.status,
            crop.created_at,
            crop.updated_at,
        )

    @staticmethod
    def _row_to_crop(row: sqlite3.Row) -> MoleculeCrop:
        payload = dict(row)
        payload["bbox"] = json.loads(payload.pop("bbox_json") or "[]")
        payload["validation_json"] = json.loads(payload.pop("validation_json") or "{}")
        payload["raw_segment"] = json.loads(payload.pop("raw_segment_json") or "{}")
        return MoleculeCrop.model_validate(payload)


class MoleculeCropValidationRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def upsert(self, validation: MoleculeCropValidation) -> MoleculeCropValidation:
        current = self.get_by_crop_provider_model(
            validation.crop_id,
            validation.provider,
            validation.model,
        )
        created_at = current.created_at if current else validation.created_at
        updated = validation.model_copy(update={"created_at": created_at, "updated_at": now_iso()})
        with self.database.connect() as conn:
            conn.execute(
                """
                INSERT INTO molecule_crop_validations (
                  validation_id, crop_id, paper_id, agent_run_id, visual_block_id,
                  provider, model, is_molecular_depiction, is_single_molecule,
                  is_complete_structure, has_benign_highlight, is_ocsr_readable,
                  has_blocking_interference, has_orbital_overlay, has_excess_annotation,
                  has_multiple_structures, has_reaction_arrow, has_non_structural_graphics,
                  should_run_ocsr, confidence, reason, raw_response_json, status,
                  error_message, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(crop_id, provider, model) DO UPDATE SET
                  validation_id=excluded.validation_id,
                  is_molecular_depiction=excluded.is_molecular_depiction,
                  is_single_molecule=excluded.is_single_molecule,
                  is_complete_structure=excluded.is_complete_structure,
                  has_benign_highlight=excluded.has_benign_highlight,
                  is_ocsr_readable=excluded.is_ocsr_readable,
                  has_blocking_interference=excluded.has_blocking_interference,
                  has_orbital_overlay=excluded.has_orbital_overlay,
                  has_excess_annotation=excluded.has_excess_annotation,
                  has_multiple_structures=excluded.has_multiple_structures,
                  has_reaction_arrow=excluded.has_reaction_arrow,
                  has_non_structural_graphics=excluded.has_non_structural_graphics,
                  should_run_ocsr=excluded.should_run_ocsr,
                  confidence=excluded.confidence,
                  reason=excluded.reason,
                  raw_response_json=excluded.raw_response_json,
                  status=excluded.status,
                  error_message=excluded.error_message,
                  updated_at=excluded.updated_at
                """,
                self._validation_values(updated),
            )
        return updated

    def get_by_crop_provider_model(
        self,
        crop_id: str,
        provider: str,
        model: str,
    ) -> MoleculeCropValidation | None:
        with self.database.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM molecule_crop_validations
                WHERE crop_id = ? AND provider = ? AND model = ?
                """,
                (crop_id, provider, model),
            ).fetchone()
        return self._row_to_validation(row) if row else None

    def list_by_paper(self, paper_id: str) -> list[MoleculeCropValidation]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM molecule_crop_validations
                WHERE paper_id = ?
                ORDER BY updated_at DESC, crop_id ASC
                """,
                (paper_id,),
            ).fetchall()
        return [self._row_to_validation(row) for row in rows]

    @staticmethod
    def _validation_values(validation: MoleculeCropValidation) -> tuple[object, ...]:
        return (
            validation.validation_id,
            validation.crop_id,
            validation.paper_id,
            validation.agent_run_id,
            validation.visual_block_id,
            validation.provider,
            validation.model,
            1 if validation.is_molecular_depiction else 0,
            1 if validation.is_single_molecule else 0,
            1 if validation.is_complete_structure else 0,
            1 if validation.has_benign_highlight else 0,
            1 if validation.is_ocsr_readable else 0,
            1 if validation.has_blocking_interference else 0,
            1 if validation.has_orbital_overlay else 0,
            1 if validation.has_excess_annotation else 0,
            1 if validation.has_multiple_structures else 0,
            1 if validation.has_reaction_arrow else 0,
            1 if validation.has_non_structural_graphics else 0,
            1 if validation.should_run_ocsr else 0,
            validation.confidence,
            validation.reason,
            json.dumps(validation.raw_response, ensure_ascii=False, sort_keys=True),
            validation.status,
            validation.error_message,
            validation.created_at,
            validation.updated_at,
        )

    @staticmethod
    def _row_to_validation(row: sqlite3.Row) -> MoleculeCropValidation:
        payload = dict(row)
        for key in (
            "is_molecular_depiction",
            "is_single_molecule",
            "is_complete_structure",
            "has_benign_highlight",
            "is_ocsr_readable",
            "has_blocking_interference",
            "has_orbital_overlay",
            "has_excess_annotation",
            "has_multiple_structures",
            "has_reaction_arrow",
            "has_non_structural_graphics",
            "should_run_ocsr",
        ):
            payload[key] = bool(payload[key])
        payload["raw_response"] = json.loads(payload.pop("raw_response_json") or "{}")
        return MoleculeCropValidation.model_validate(payload)


class MoleculeLabelBindingRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def upsert_proposal(self, binding: MoleculeLabelBinding) -> MoleculeLabelBinding:
        current = self.get_by_crop_provider_model(binding.crop_id, binding.provider, binding.model)
        if current:
            binding = binding.model_copy(
                update={
                    "binding_id": current.binding_id,
                    "created_at": current.created_at,
                    "reviewed_paper_material_id": current.reviewed_paper_material_id,
                    "reviewed_observed_label": current.reviewed_observed_label,
                    "review_status": current.review_status,
                    "reviewed_by": current.reviewed_by,
                    "reviewed_at": current.reviewed_at,
                    "review_note": current.review_note,
                    "updated_at": now_iso(),
                }
            )
        with self.database.connect() as conn:
            conn.execute(
                """
                INSERT INTO molecule_label_bindings (
                  binding_id, paper_id, candidate_run_id, agent_run_id, crop_id,
                  visual_block_id, provider, model, source_figure_path,
                  highlighted_source_figure_path, crop_path, caption_text, nearby_text,
                  triage_label_candidates_json, candidate_materials_json,
                  model_observed_label, model_label_source, model_proposed_paper_material_id,
                  model_alternative_paper_material_ids_json, model_decision, model_confidence,
                  model_reason, raw_response_json, status, error_message,
                  reviewed_paper_material_id, reviewed_observed_label, review_status,
                  reviewed_by, reviewed_at, review_note, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(crop_id, provider, model) DO UPDATE SET
                  source_figure_path=excluded.source_figure_path,
                  highlighted_source_figure_path=excluded.highlighted_source_figure_path,
                  crop_path=excluded.crop_path,
                  caption_text=excluded.caption_text,
                  nearby_text=excluded.nearby_text,
                  triage_label_candidates_json=excluded.triage_label_candidates_json,
                  candidate_materials_json=excluded.candidate_materials_json,
                  model_observed_label=excluded.model_observed_label,
                  model_label_source=excluded.model_label_source,
                  model_proposed_paper_material_id=excluded.model_proposed_paper_material_id,
                  model_alternative_paper_material_ids_json=excluded.model_alternative_paper_material_ids_json,
                  model_decision=excluded.model_decision,
                  model_confidence=excluded.model_confidence,
                  model_reason=excluded.model_reason,
                  raw_response_json=excluded.raw_response_json,
                  status=excluded.status,
                  error_message=excluded.error_message,
                  updated_at=excluded.updated_at
                """,
                self._binding_values(binding),
            )
        return binding

    def get(self, binding_id: str) -> MoleculeLabelBinding | None:
        with self.database.connect() as conn:
            row = conn.execute(
                "SELECT * FROM molecule_label_bindings WHERE binding_id = ?",
                (binding_id,),
            ).fetchone()
        return self._row_to_binding(row) if row else None

    def get_by_crop_provider_model(
        self, crop_id: str, provider: str, model: str
    ) -> MoleculeLabelBinding | None:
        with self.database.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM molecule_label_bindings
                WHERE crop_id = ? AND provider = ? AND model = ?
                """,
                (crop_id, provider, model),
            ).fetchone()
        return self._row_to_binding(row) if row else None

    def list_by_paper(self, paper_id: str) -> list[MoleculeLabelBinding]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM molecule_label_bindings
                WHERE paper_id = ?
                ORDER BY updated_at DESC, crop_id ASC
                """,
                (paper_id,),
            ).fetchall()
        return [self._row_to_binding(row) for row in rows]

    def list_by_run(self, agent_run_id: str) -> list[MoleculeLabelBinding]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM molecule_label_bindings
                WHERE agent_run_id = ?
                ORDER BY crop_id ASC
                """,
                (agent_run_id,),
            ).fetchall()
        return [self._row_to_binding(row) for row in rows]

    def list_by_crop(self, crop_id: str) -> list[MoleculeLabelBinding]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM molecule_label_bindings
                WHERE crop_id = ?
                ORDER BY updated_at DESC
                """,
                (crop_id,),
            ).fetchall()
        return [self._row_to_binding(row) for row in rows]

    def update_review(self, binding: MoleculeLabelBinding) -> MoleculeLabelBinding:
        with self.database.connect() as conn:
            conn.execute(
                """
                UPDATE molecule_label_bindings
                SET reviewed_paper_material_id = ?, reviewed_observed_label = ?,
                    review_status = ?, reviewed_by = ?, reviewed_at = ?,
                    review_note = ?, updated_at = ?
                WHERE binding_id = ?
                """,
                (
                    binding.reviewed_paper_material_id,
                    binding.reviewed_observed_label,
                    binding.review_status,
                    binding.reviewed_by,
                    binding.reviewed_at,
                    binding.review_note,
                    binding.updated_at,
                    binding.binding_id,
                ),
            )
        return binding

    @staticmethod
    def _binding_values(binding: MoleculeLabelBinding) -> tuple[object, ...]:
        return (
            binding.binding_id,
            binding.paper_id,
            binding.candidate_run_id,
            binding.agent_run_id,
            binding.crop_id,
            binding.visual_block_id,
            binding.provider,
            binding.model,
            binding.source_figure_path,
            binding.highlighted_source_figure_path,
            binding.crop_path,
            binding.caption_text,
            binding.nearby_text,
            json.dumps(binding.triage_label_candidates, ensure_ascii=False),
            json.dumps(binding.candidate_materials, ensure_ascii=False, sort_keys=True),
            binding.model_observed_label,
            binding.model_label_source,
            binding.model_proposed_paper_material_id,
            json.dumps(binding.model_alternative_paper_material_ids, ensure_ascii=False),
            binding.model_decision,
            binding.model_confidence,
            binding.model_reason,
            json.dumps(binding.raw_response, ensure_ascii=False, sort_keys=True),
            binding.status,
            binding.error_message,
            binding.reviewed_paper_material_id,
            binding.reviewed_observed_label,
            binding.review_status,
            binding.reviewed_by,
            binding.reviewed_at,
            binding.review_note,
            binding.created_at,
            binding.updated_at,
        )

    @staticmethod
    def _row_to_binding(row: sqlite3.Row) -> MoleculeLabelBinding:
        payload = dict(row)
        payload["triage_label_candidates"] = json.loads(
            payload.pop("triage_label_candidates_json") or "[]"
        )
        payload["candidate_materials"] = json.loads(payload.pop("candidate_materials_json") or "[]")
        payload["model_alternative_paper_material_ids"] = json.loads(
            payload.pop("model_alternative_paper_material_ids_json") or "[]"
        )
        payload["raw_response"] = json.loads(payload.pop("raw_response_json") or "{}")
        return MoleculeLabelBinding.model_validate(payload)


class MoleculeLabelBindingReviewEventRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def add(self, event: MoleculeLabelBindingReviewEvent) -> MoleculeLabelBindingReviewEvent:
        with self.database.connect() as conn:
            conn.execute(
                """
                INSERT INTO molecule_label_binding_review_events (
                  event_id, binding_id, paper_id, crop_id, action, actor, message,
                  before_reviewed_paper_material_id, after_reviewed_paper_material_id,
                  before_observed_label, after_observed_label, before_review_status,
                  after_review_status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.binding_id,
                    event.paper_id,
                    event.crop_id,
                    event.action,
                    event.actor,
                    event.message,
                    event.before_reviewed_paper_material_id,
                    event.after_reviewed_paper_material_id,
                    event.before_observed_label,
                    event.after_observed_label,
                    event.before_review_status,
                    event.after_review_status,
                    event.created_at,
                ),
            )
        return event

    def get(self, event_id: str) -> MoleculeLabelBindingReviewEvent | None:
        with self.database.connect() as conn:
            row = conn.execute(
                "SELECT * FROM molecule_label_binding_review_events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        return MoleculeLabelBindingReviewEvent.model_validate(dict(row)) if row else None

    def list_by_paper(self, paper_id: str) -> list[MoleculeLabelBindingReviewEvent]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM molecule_label_binding_review_events
                WHERE paper_id = ?
                ORDER BY created_at DESC
                """,
                (paper_id,),
            ).fetchall()
        return [MoleculeLabelBindingReviewEvent.model_validate(dict(row)) for row in rows]


class VLMCallLogRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def create(self, call: VLMCallLog) -> VLMCallLog:
        with self.database.connect() as conn:
            conn.execute(
                """
                INSERT INTO vlm_call_logs (
                  vlm_call_id, paper_id, agent_run_id, stage, input_entity_type,
                  input_entity_id, provider, model, prompt_version,
                  input_image_paths_json, input_context_json, parsed_response_json,
                  usage_json, status, error_message, started_at, finished_at, duration_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._values(call),
            )
        return call

    def update(self, call: VLMCallLog) -> VLMCallLog:
        with self.database.connect() as conn:
            conn.execute(
                """
                UPDATE vlm_call_logs
                SET parsed_response_json = ?, usage_json = ?, status = ?,
                    error_message = ?, finished_at = ?, duration_ms = ?
                WHERE vlm_call_id = ?
                """,
                (
                    json.dumps(call.parsed_response, ensure_ascii=False, sort_keys=True),
                    json.dumps(call.usage, ensure_ascii=False, sort_keys=True),
                    call.status,
                    call.error_message,
                    call.finished_at,
                    call.duration_ms,
                    call.vlm_call_id,
                ),
            )
        return call

    def list_by_run(self, agent_run_id: str) -> list[VLMCallLog]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM vlm_call_logs
                WHERE agent_run_id = ?
                ORDER BY started_at DESC, vlm_call_id DESC
                """,
                (agent_run_id,),
            ).fetchall()
        return [self._row_to_call(row) for row in rows]

    @staticmethod
    def _values(call: VLMCallLog) -> tuple[object, ...]:
        return (
            call.vlm_call_id,
            call.paper_id,
            call.agent_run_id,
            call.stage,
            call.input_entity_type,
            call.input_entity_id,
            call.provider,
            call.model,
            call.prompt_version,
            json.dumps(call.input_image_paths, ensure_ascii=False),
            json.dumps(call.input_context, ensure_ascii=False, sort_keys=True),
            json.dumps(call.parsed_response, ensure_ascii=False, sort_keys=True),
            json.dumps(call.usage, ensure_ascii=False, sort_keys=True),
            call.status,
            call.error_message,
            call.started_at,
            call.finished_at,
            call.duration_ms,
        )

    @staticmethod
    def _row_to_call(row: sqlite3.Row) -> VLMCallLog:
        payload = dict(row)
        payload["input_image_paths"] = json.loads(payload.pop("input_image_paths_json") or "[]")
        payload["input_context"] = json.loads(payload.pop("input_context_json") or "{}")
        payload["parsed_response"] = json.loads(payload.pop("parsed_response_json") or "{}")
        payload["usage"] = json.loads(payload.pop("usage_json") or "{}")
        return VLMCallLog.model_validate(payload)


class ChemicalFigureBlockRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def replace_for_run(
        self,
        mineru_run_id: str,
        blocks: list[ChemicalFigureBlock],
    ) -> list[ChemicalFigureBlock]:
        timestamp = now_iso()
        rows = [
            block.model_copy(
                update={"created_at": block.created_at or timestamp, "updated_at": timestamp}
            )
            for block in blocks
        ]
        with self.database.connect() as conn:
            conn.execute(
                "DELETE FROM chemical_figure_blocks WHERE mineru_run_id = ?",
                (mineru_run_id,),
            )
            conn.executemany(
                """
                INSERT INTO chemical_figure_blocks (
                  figure_block_id, paper_id, mineru_run_id, content_index,
                  content_type, sub_type, page_idx, page_id, bbox_json,
                  img_path, resolved_img_path, image_exists, caption,
                  nearby_text, heuristic_tags_json, confidence, status,
                  source_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [self._block_values(block) for block in rows],
            )
        return rows

    def get(self, figure_block_id: str) -> ChemicalFigureBlock | None:
        with self.database.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM chemical_figure_blocks
                WHERE figure_block_id = ?
                """,
                (figure_block_id,),
            ).fetchone()
        return self._row_to_block(row) if row else None

    def list_by_paper(self, paper_id: str) -> list[ChemicalFigureBlock]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM chemical_figure_blocks
                WHERE paper_id = ?
                ORDER BY page_id ASC, content_index ASC
                """,
                (paper_id,),
            ).fetchall()
        return [self._row_to_block(row) for row in rows]

    def list_by_run(self, mineru_run_id: str) -> list[ChemicalFigureBlock]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM chemical_figure_blocks
                WHERE mineru_run_id = ?
                ORDER BY content_index ASC
                """,
                (mineru_run_id,),
            ).fetchall()
        return [self._row_to_block(row) for row in rows]

    @staticmethod
    def _block_values(block: ChemicalFigureBlock) -> tuple[object, ...]:
        return (
            block.figure_block_id,
            block.paper_id,
            block.mineru_run_id,
            block.content_index,
            block.content_type,
            block.sub_type,
            block.page_idx,
            block.page_id,
            json.dumps(block.bbox, ensure_ascii=False),
            block.img_path,
            block.resolved_img_path,
            1 if block.image_exists else 0,
            block.caption,
            block.nearby_text,
            json.dumps(block.heuristic_tags, ensure_ascii=False),
            block.confidence,
            block.status,
            json.dumps(block.source_json, ensure_ascii=False, sort_keys=True),
            block.created_at,
            block.updated_at,
        )

    @staticmethod
    def _row_to_block(row: sqlite3.Row) -> ChemicalFigureBlock:
        payload = dict(row)
        payload["bbox"] = json.loads(payload.pop("bbox_json") or "[]")
        payload["heuristic_tags"] = json.loads(payload.pop("heuristic_tags_json") or "[]")
        payload["source_json"] = json.loads(payload.pop("source_json") or "{}")
        payload["image_exists"] = bool(payload["image_exists"])
        return ChemicalFigureBlock.model_validate(payload)


class MinerUParseRunRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def create(self, run: MinerUParseRun) -> MinerUParseRun:
        with self.database.connect() as conn:
            conn.execute(
                """
                INSERT INTO mineru_parse_runs (
                  mineru_run_id, paper_id, task_id, status, service_base_url,
                  parser_version, content_item_count, result_path, content_list_path,
                  markdown_path, error_message, created_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.mineru_run_id,
                    run.paper_id,
                    run.task_id,
                    run.status,
                    run.service_base_url,
                    run.parser_version,
                    run.content_item_count,
                    run.result_path,
                    run.content_list_path,
                    run.markdown_path,
                    run.error_message,
                    run.created_at,
                    run.completed_at,
                ),
            )
        return run

    def update(self, run: MinerUParseRun) -> MinerUParseRun:
        with self.database.connect() as conn:
            conn.execute(
                """
                UPDATE mineru_parse_runs
                SET task_id = ?, status = ?, parser_version = ?, content_item_count = ?,
                    result_path = ?, content_list_path = ?, markdown_path = ?,
                    error_message = ?, completed_at = ?
                WHERE mineru_run_id = ?
                """,
                (
                    run.task_id,
                    run.status,
                    run.parser_version,
                    run.content_item_count,
                    run.result_path,
                    run.content_list_path,
                    run.markdown_path,
                    run.error_message,
                    run.completed_at,
                    run.mineru_run_id,
                ),
            )
        return run

    def get(self, mineru_run_id: str) -> MinerUParseRun | None:
        with self.database.connect() as conn:
            row = conn.execute(
                "SELECT * FROM mineru_parse_runs WHERE mineru_run_id = ?",
                (mineru_run_id,),
            ).fetchone()
        return self._row_to_run(row) if row else None

    def latest_completed_by_paper(self, paper_id: str) -> MinerUParseRun | None:
        with self.database.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM mineru_parse_runs
                WHERE paper_id = ? AND status = 'completed'
                ORDER BY completed_at DESC, created_at DESC
                LIMIT 1
                """,
                (paper_id,),
            ).fetchone()
        return self._row_to_run(row) if row else None

    @staticmethod
    def _row_to_run(row: sqlite3.Row) -> MinerUParseRun:
        return MinerUParseRun.model_validate(dict(row))


class LLMMiningRunRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def create(self, run: LLMMiningRun) -> LLMMiningRun:
        with self.database.connect() as conn:
            conn.execute(
                """
                INSERT INTO llm_mining_runs (
                  llm_run_id, paper_id, template_id, provider, model, status,
                  source_parser, input_item_count, prompt_path, raw_response_path,
                  mining_result_path, validation_report_path, candidate_run_id,
                  error_message, token_usage_json, created_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._run_values(run),
            )
        return run

    def update(self, run: LLMMiningRun) -> LLMMiningRun:
        with self.database.connect() as conn:
            conn.execute(
                """
                UPDATE llm_mining_runs
                SET status = ?, input_item_count = ?, prompt_path = ?, raw_response_path = ?,
                    mining_result_path = ?, validation_report_path = ?, candidate_run_id = ?,
                    error_message = ?, token_usage_json = ?, completed_at = ?
                WHERE llm_run_id = ?
                """,
                (
                    run.status,
                    run.input_item_count,
                    run.prompt_path,
                    run.raw_response_path,
                    run.mining_result_path,
                    run.validation_report_path,
                    run.candidate_run_id,
                    run.error_message,
                    json.dumps(run.token_usage, ensure_ascii=False, sort_keys=True),
                    run.completed_at,
                    run.llm_run_id,
                ),
            )
        return run

    def get(self, llm_run_id: str) -> LLMMiningRun | None:
        with self.database.connect() as conn:
            row = conn.execute(
                "SELECT * FROM llm_mining_runs WHERE llm_run_id = ?",
                (llm_run_id,),
            ).fetchone()
        return self._row_to_run(row) if row else None

    def list_by_paper(self, paper_id: str) -> list[LLMMiningRun]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM llm_mining_runs
                WHERE paper_id = ?
                ORDER BY created_at DESC
                """,
                (paper_id,),
            ).fetchall()
        return [self._row_to_run(row) for row in rows]

    @staticmethod
    def _run_values(run: LLMMiningRun) -> tuple[object, ...]:
        return (
            run.llm_run_id,
            run.paper_id,
            run.template_id,
            run.provider,
            run.model,
            run.status,
            run.source_parser,
            run.input_item_count,
            run.prompt_path,
            run.raw_response_path,
            run.mining_result_path,
            run.validation_report_path,
            run.candidate_run_id,
            run.error_message,
            json.dumps(run.token_usage, ensure_ascii=False, sort_keys=True),
            run.created_at,
            run.completed_at,
        )

    @staticmethod
    def _row_to_run(row: sqlite3.Row) -> LLMMiningRun:
        payload = dict(row)
        payload["token_usage"] = json.loads(payload.pop("token_usage_json") or "{}")
        return LLMMiningRun.model_validate(payload)


def _candidate_display_value(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{value:g}"
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _normalize_candidate_value_status(status: str | None) -> str | None:
    if status is None:
        return None
    aliases = {
        "edited": "modified",
        "confirmed": "accepted",
    }
    normalized = aliases.get(status, status)
    allowed = {"pending", "modified", "accepted", "rejected"}
    return normalized if normalized in allowed else status
