from __future__ import annotations

import base64
import io
import json
import sqlite3
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from threading import Condition, Event, Lock, Thread
from typing import Any, Callable, Protocol
from uuid import uuid4

import httpx
from PIL import Image

from evolab_local.mining_platform.core.config import LLMProviderConfig, VisionBatchConfig
from evolab_local.mining_platform.external.openai_compatible_client import (
    LLMResponse,
    OpenAICompatibleVisionClient,
    extract_json_object,
)


_TERMINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled"}
_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class QwenBatchAPIProtocol(Protocol):
    def upload_file(self, path: Path) -> str: ...

    def create_batch(
        self,
        *,
        input_file_id: str,
        completion_window: str,
        name: str,
        description: str,
        endpoint: str = "/v1/chat/completions",
    ) -> dict[str, Any]: ...

    def retrieve_batch(self, batch_id: str) -> dict[str, Any]: ...

    def download_file(self, file_id: str, destination: Path) -> None: ...

    def close(self) -> None: ...


class DashScopeBatchAPI:
    """Small HTTP client for DashScope's OpenAI-compatible Batch File API."""

    def __init__(self, config: LLMProviderConfig) -> None:
        if not config.api_key:
            raise ValueError("Qwen API key is not configured.")
        self.config = config
        self.base_url = config.base_url.rstrip("/")
        self.client = httpx.Client(
            headers={"Authorization": f"Bearer {config.api_key}"},
            timeout=config.timeout_seconds,
            trust_env=False,
        )

    def upload_file(self, path: Path) -> str:
        response = self._request_with_retry(
            "POST",
            "/files",
            file_path=path,
            data={"purpose": "batch"},
        )
        payload = self._json_object(response)
        file_id = payload.get("id")
        if not isinstance(file_id, str) or not file_id:
            raise ValueError("Qwen Batch file upload did not return a file ID.")
        return file_id

    def create_batch(
        self,
        *,
        input_file_id: str,
        completion_window: str,
        name: str,
        description: str,
        endpoint: str = "/v1/chat/completions",
    ) -> dict[str, Any]:
        response = self._request_with_retry(
            "POST",
            "/batches",
            json_body={
                "input_file_id": input_file_id,
                "endpoint": endpoint,
                "completion_window": completion_window,
                "metadata": {
                    "ds_name": name[:100],
                    "ds_description": description[:200],
                },
            },
        )
        return self._json_object(response)

    def retrieve_batch(self, batch_id: str) -> dict[str, Any]:
        response = self._request_with_retry("GET", f"/batches/{batch_id}")
        return self._json_object(response)

    def download_file(self, file_id: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        attempts = max(1, self.config.request_max_attempts)
        for attempt in range(attempts):
            try:
                with self.client.stream("GET", f"{self.base_url}/files/{file_id}/content") as response:
                    response.raise_for_status()
                    with destination.open("wb") as output:
                        for chunk in response.iter_bytes():
                            output.write(chunk)
                return
            except (httpx.RequestError, httpx.HTTPStatusError):
                if attempt + 1 >= attempts:
                    raise
                time.sleep(max(0.0, self.config.retry_backoff_seconds) * (2**attempt))

    def close(self) -> None:
        self.client.close()

    def _request_with_retry(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        file_path: Path | None = None,
        data: dict[str, str] | None = None,
    ) -> httpx.Response:
        attempts = max(1, self.config.request_max_attempts)
        for attempt in range(attempts):
            try:
                if file_path is None:
                    response = self.client.request(
                        method,
                        f"{self.base_url}{path}",
                        json=json_body,
                    )
                else:
                    with file_path.open("rb") as file_handle:
                        response = self.client.request(
                            method,
                            f"{self.base_url}{path}",
                            data=data,
                            files={
                                "file": (
                                    file_path.name,
                                    file_handle,
                                    "application/jsonl",
                                )
                            },
                        )
                response.raise_for_status()
                return response
            except (httpx.RequestError, httpx.HTTPStatusError) as exc:
                retryable = isinstance(exc, httpx.RequestError) or (
                    exc.response.status_code in _RETRYABLE_STATUS_CODES
                )
                if attempt + 1 >= attempts or not retryable:
                    raise
                time.sleep(max(0.0, self.config.retry_backoff_seconds) * (2**attempt))
        raise RuntimeError("Qwen Batch request retry loop ended without a response.")

    @staticmethod
    def _json_object(response: httpx.Response) -> dict[str, Any]:
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Qwen Batch API response must be a JSON object.")
        return payload


@dataclass
class _QueuedVisionRequest:
    request_hash: str
    custom_id: str
    body: dict[str, Any]
    line: bytes
    event: Event = field(default_factory=Event)
    result: LLMResponse | None = None
    error: BaseException | None = None


class QwenBatchVisionClient:
    """Expose Batch File inference through the existing synchronous VisionClient API.

    Concurrent callers are coalesced into durable JSONL jobs. Results are cached by a
    content hash, so a restarted group run can recover an already-paid provider result.
    """

    def __init__(
        self,
        provider_config: LLMProviderConfig,
        batch_config: VisionBatchConfig,
        *,
        runtime_dir: Path,
        batch_api: QwenBatchAPIProtocol | None = None,
        realtime_client: OpenAICompatibleVisionClient | None = None,
        progress: Callable[[str], None] | None = None,
        recover_incomplete: bool = True,
    ) -> None:
        self.provider_config = provider_config
        self.batch_config = batch_config
        self.runtime_dir = runtime_dir
        self.jobs_dir = runtime_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = runtime_dir / "batch_cache.sqlite"
        self.api = batch_api or DashScopeBatchAPI(provider_config)
        self.realtime_client = realtime_client or OpenAICompatibleVisionClient(provider_config)
        self.progress = progress or (lambda message: print(message, flush=True))
        self._database_lock = Lock()
        self._condition = Condition()
        self._queue: deque[_QueuedVisionRequest] = deque()
        self._inflight: dict[str, _QueuedVisionRequest] = {}
        self._closed = False
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, batch_config.max_active_jobs),
            thread_name_prefix="qwen-batch-job",
        )
        self._init_cache()
        if recover_incomplete:
            self.recover_incomplete_jobs()
        self._coordinator = Thread(
            target=self._coordinate,
            name="qwen-batch-coordinator",
            daemon=True,
        )
        self._coordinator.start()

    def generate_json(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        selected_model = model or self.provider_config.vision_model or self.provider_config.default_model
        original_messages = messages
        body = self._build_body(
            messages,
            model=selected_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        body = self._fit_body_to_line_limit(body)
        request_hash = _request_hash(body)
        cached = self._cached_response(request_hash)
        if cached is not None:
            return cached
        custom_id = f"vlm-{request_hash[:32]}"
        line = _request_line(custom_id, body)
        if len(line) > self.batch_config.max_line_bytes:
            if not self.batch_config.realtime_fallback:
                raise ValueError(
                    f"Qwen Batch request is {len(line)} bytes; limit is "
                    f"{self.batch_config.max_line_bytes}."
                )
            self.progress(
                f"[qwen-batch] oversized request {custom_id} ({len(line)} bytes); "
                "using realtime fallback"
            )
            response = self.realtime_client.generate_json(
                original_messages,
                model=selected_model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            self._store_standalone_response(request_hash, custom_id, response)
            return response

        with self._condition:
            if self._closed:
                raise RuntimeError("Qwen Batch vision client is closed.")
            cached = self._cached_response(request_hash)
            if cached is not None:
                return cached
            request = self._inflight.get(request_hash)
            if request is None:
                request = _QueuedVisionRequest(
                    request_hash=request_hash,
                    custom_id=custom_id,
                    body=body,
                    line=line,
                )
                self._inflight[request_hash] = request
                self._queue.append(request)
                self._condition.notify_all()
        request.event.wait()
        if request.error is not None:
            raise request.error
        if request.result is None:
            raise RuntimeError(f"Qwen Batch request {custom_id} completed without a result.")
        return request.result

    def recover_incomplete_jobs(self) -> int:
        jobs = self._recoverable_jobs()
        if not jobs:
            return 0
        self.progress(f"[qwen-batch] recovering {len(jobs)} unfinished provider job(s)")
        worker_count = min(max(1, self.batch_config.max_active_jobs), len(jobs))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            list(executor.map(self._recover_job, jobs))
        return len(jobs)

    def statistics(self) -> dict[str, Any]:
        with self._database_lock, self._connect() as conn:
            job_rows = conn.execute(
                "SELECT status, COUNT(*) AS count FROM batch_jobs GROUP BY status"
            ).fetchall()
            request_rows = conn.execute(
                "SELECT status, COUNT(*) AS count FROM batch_requests GROUP BY status"
            ).fetchall()
            completed_rows = conn.execute(
                """
                SELECT local_job_id, response_json
                FROM batch_requests
                WHERE status = 'completed' AND response_json IS NOT NULL
                """
            ).fetchall()
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        realtime_fallback_count = 0
        for row in completed_rows:
            if row["local_job_id"] == "realtime-fallback":
                realtime_fallback_count += 1
            try:
                payload = json.loads(row["response_json"])
            except json.JSONDecodeError:
                continue
            response_usage = payload.get("usage") if isinstance(payload, dict) else None
            if not isinstance(response_usage, dict):
                continue
            for key in usage:
                value = response_usage.get(key)
                if isinstance(value, int | float):
                    usage[key] += int(value)
        return {
            "jobs_by_status": {str(row["status"]): int(row["count"]) for row in job_rows},
            "requests_by_status": {
                str(row["status"]): int(row["count"]) for row in request_rows
            },
            "realtime_fallback_count": realtime_fallback_count,
            "usage": usage,
        }

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.notify_all()
        self._coordinator.join()
        self._executor.shutdown(wait=True)
        self.api.close()

    def __enter__(self) -> QwenBatchVisionClient:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _coordinate(self) -> None:
        while True:
            with self._condition:
                while not self._queue and not self._closed:
                    self._condition.wait()
                if not self._queue and self._closed:
                    return
                if not self._closed and self.batch_config.flush_seconds > 0:
                    deadline = time.monotonic() + self.batch_config.flush_seconds
                    while not self._closed:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        self._condition.wait(timeout=remaining)
                requests = list(self._queue)
                self._queue.clear()
            for chunk in self._partition_requests(requests):
                self._executor.submit(self._process_chunk, chunk)

    def _partition_requests(
        self,
        requests: list[_QueuedVisionRequest],
    ) -> list[list[_QueuedVisionRequest]]:
        chunks: list[list[_QueuedVisionRequest]] = []
        current: list[_QueuedVisionRequest] = []
        current_bytes = 0
        for request in requests:
            exceeds_count = len(current) >= self.batch_config.max_requests_per_job
            exceeds_bytes = current and (
                current_bytes + len(request.line) > self.batch_config.max_file_bytes
            )
            if exceeds_count or exceeds_bytes:
                chunks.append(current)
                current = []
                current_bytes = 0
            current.append(request)
            current_bytes += len(request.line)
        if current:
            chunks.append(current)
        return chunks

    def _process_chunk(self, requests: list[_QueuedVisionRequest]) -> None:
        local_job_id = f"qwen-vlm-{time.strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
        job_dir = self.jobs_dir / local_job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        input_path = job_dir / "input.jsonl"
        input_path.write_bytes(b"".join(request.line for request in requests))
        self._create_local_job(local_job_id, input_path, requests)
        provider_batch_id: str | None = None
        try:
            input_file_id = self.api.upload_file(input_path)
            self._update_job(
                local_job_id,
                status="uploaded",
                input_file_id=input_file_id,
            )
            batch = self.api.create_batch(
                input_file_id=input_file_id,
                completion_window=self.batch_config.completion_window,
                name=local_job_id,
                description=f"OLED material VLM batch with {len(requests)} requests",
            )
            provider_batch_id = _required_string(batch, "id")
            self._update_job(
                local_job_id,
                status=str(batch.get("status") or "validating"),
                provider_batch_id=provider_batch_id,
                provider_payload=batch,
            )
            self.progress(
                f"[qwen-batch] submitted {local_job_id}: {len(requests)} requests, "
                f"{input_path.stat().st_size / 1_000_000:.1f} MB, provider={provider_batch_id}"
            )
            final_batch = self._poll_batch(local_job_id, provider_batch_id, batch)
            self._download_and_ingest(local_job_id, final_batch)
            for request in requests:
                response = self._cached_response(request.request_hash)
                if response is None:
                    error = self._cached_error(request.request_hash) or (
                        f"Qwen Batch job {provider_batch_id} returned no result for {request.custom_id}."
                    )
                    self._finish_request(request, error=RuntimeError(error))
                else:
                    self._finish_request(request, result=response)
        except BaseException as exc:
            if provider_batch_id:
                self._update_job(
                    local_job_id,
                    status="poll_interrupted",
                    error_message=str(exc),
                )
            else:
                self._fail_job_requests(local_job_id, str(exc))
            for request in requests:
                self._finish_request(request, error=exc)

    def _recover_job(self, job: dict[str, Any]) -> None:
        local_job_id = str(job["local_job_id"])
        provider_batch_id = str(job.get("provider_batch_id") or "")
        if not provider_batch_id:
            self._fail_job_requests(local_job_id, "Batch job was interrupted before submission.")
            return
        try:
            batch = self.api.retrieve_batch(provider_batch_id)
            final_batch = self._poll_batch(local_job_id, provider_batch_id, batch)
            self._download_and_ingest(local_job_id, final_batch)
        except Exception as exc:
            self._update_job(
                local_job_id,
                status="poll_interrupted",
                error_message=str(exc),
            )
            raise

    def _poll_batch(
        self,
        local_job_id: str,
        provider_batch_id: str,
        batch: dict[str, Any],
    ) -> dict[str, Any]:
        status = str(batch.get("status") or "validating")
        last_reported: tuple[str, str] | None = None
        while status not in _TERMINAL_BATCH_STATUSES:
            counts = batch.get("request_counts") if isinstance(batch.get("request_counts"), dict) else {}
            progress_key = (status, json.dumps(counts, sort_keys=True))
            if progress_key != last_reported:
                self.progress(
                    f"[qwen-batch] {provider_batch_id}: status={status}, counts={counts}"
                )
                last_reported = progress_key
            self._update_job(
                local_job_id,
                status=status,
                provider_payload=batch,
            )
            time.sleep(max(0.1, self.batch_config.poll_interval_seconds))
            batch = self.api.retrieve_batch(provider_batch_id)
            status = str(batch.get("status") or status)
        self._update_job(
            local_job_id,
            status=status,
            provider_payload=batch,
            output_file_id=_optional_string(batch.get("output_file_id")),
            error_file_id=_optional_string(batch.get("error_file_id")),
        )
        self.progress(
            f"[qwen-batch] {provider_batch_id}: terminal status={status}, "
            f"counts={batch.get('request_counts') or {}}"
        )
        return batch

    def _download_and_ingest(self, local_job_id: str, batch: dict[str, Any]) -> None:
        job_dir = self.jobs_dir / local_job_id
        output_file_id = _optional_string(batch.get("output_file_id"))
        error_file_id = _optional_string(batch.get("error_file_id"))
        output_path: Path | None = None
        error_path: Path | None = None
        if output_file_id:
            output_path = job_dir / "output.jsonl"
            self.api.download_file(output_file_id, output_path)
            self._ingest_output_file(local_job_id, output_path)
        if error_file_id:
            error_path = job_dir / "error.jsonl"
            self.api.download_file(error_file_id, error_path)
            self._ingest_error_file(local_job_id, error_path)
        terminal_status = str(batch.get("status") or "failed")
        missing_error = (
            f"Qwen Batch job ended with status={terminal_status} without a successful result."
        )
        self._mark_missing_requests_failed(local_job_id, missing_error)
        self._write_manifest(
            local_job_id,
            {
                "local_job_id": local_job_id,
                "provider_batch": batch,
                "input_path": (job_dir / "input.jsonl").as_posix(),
                "output_path": output_path.as_posix() if output_path else None,
                "error_path": error_path.as_posix() if error_path else None,
            },
        )

    def _ingest_output_file(self, local_job_id: str, path: Path) -> None:
        custom_ids = self._job_custom_id_map(local_job_id)
        for payload in _read_jsonl(path):
            custom_id = _optional_string(payload.get("custom_id"))
            request_hash = custom_ids.get(custom_id or "")
            if not request_hash:
                continue
            response = payload.get("response")
            response = response if isinstance(response, dict) else {}
            body = response.get("body")
            status_code = response.get("status_code")
            if status_code == 200 and isinstance(body, dict):
                self._store_request_result(request_hash, body)
            else:
                error = payload.get("error") or response
                self._store_request_error(request_hash, json.dumps(error, ensure_ascii=False))

    def _ingest_error_file(self, local_job_id: str, path: Path) -> None:
        custom_ids = self._job_custom_id_map(local_job_id)
        for payload in _read_jsonl(path):
            custom_id = _optional_string(payload.get("custom_id"))
            request_hash = custom_ids.get(custom_id or "")
            if not request_hash:
                continue
            error = payload.get("error") or payload
            self._store_request_error(request_hash, json.dumps(error, ensure_ascii=False))

    def _finish_request(
        self,
        request: _QueuedVisionRequest,
        *,
        result: LLMResponse | None = None,
        error: BaseException | None = None,
    ) -> None:
        request.result = result
        request.error = error
        request.event.set()
        with self._condition:
            self._inflight.pop(request.request_hash, None)

    def _build_body(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": deepcopy(messages),
            "temperature": (
                self.provider_config.temperature if temperature is None else temperature
            ),
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if self.provider_config.vision_enable_thinking is not None:
            body["enable_thinking"] = self.provider_config.vision_enable_thinking
        if self.provider_config.response_format_json:
            body["response_format"] = {"type": "json_object"}
        return body

    def _fit_body_to_line_limit(self, body: dict[str, Any]) -> dict[str, Any]:
        normalized = deepcopy(body)
        normalized["messages"] = _expand_tiny_data_urls(
            normalized.get("messages"),
            min_dimension=12,
        )
        body = normalized
        probe = _request_line("vlm-size-probe", body)
        if len(probe) <= self.batch_config.max_line_bytes:
            return body
        for dimension, quality in (
            (self.batch_config.image_max_dimension, self.batch_config.image_jpeg_quality),
            (1600, 88),
            (1280, 86),
            (1024, 84),
            (768, 82),
        ):
            candidate = deepcopy(body)
            candidate["messages"] = _compress_data_urls(
                candidate.get("messages"),
                max_dimension=max(256, dimension),
                quality=max(50, min(95, quality)),
            )
            if len(_request_line("vlm-size-probe", candidate)) <= self.batch_config.max_line_bytes:
                return candidate
        return body

    def _init_cache(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS batch_jobs (
                  local_job_id TEXT PRIMARY KEY,
                  provider_batch_id TEXT,
                  input_file_id TEXT,
                  status TEXT NOT NULL,
                  input_path TEXT NOT NULL,
                  output_file_id TEXT,
                  error_file_id TEXT,
                  provider_payload_json TEXT NOT NULL DEFAULT '{}',
                  error_message TEXT,
                  created_at REAL NOT NULL,
                  updated_at REAL NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_batch_jobs_provider_id
                  ON batch_jobs(provider_batch_id)
                  WHERE provider_batch_id IS NOT NULL;
                CREATE TABLE IF NOT EXISTS batch_requests (
                  request_hash TEXT PRIMARY KEY,
                  local_job_id TEXT NOT NULL,
                  custom_id TEXT NOT NULL,
                  status TEXT NOT NULL,
                  response_json TEXT,
                  error_message TEXT,
                  created_at REAL NOT NULL,
                  updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_batch_requests_job
                  ON batch_requests(local_job_id, status);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.cache_path, timeout=60.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _create_local_job(
        self,
        local_job_id: str,
        input_path: Path,
        requests: list[_QueuedVisionRequest],
    ) -> None:
        now = time.time()
        with self._database_lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO batch_jobs (
                  local_job_id, status, input_path, created_at, updated_at
                ) VALUES (?, 'prepared', ?, ?, ?)
                """,
                (local_job_id, input_path.as_posix(), now, now),
            )
            for request in requests:
                conn.execute(
                    """
                    INSERT INTO batch_requests (
                      request_hash, local_job_id, custom_id, status, created_at, updated_at
                    ) VALUES (?, ?, ?, 'pending', ?, ?)
                    ON CONFLICT(request_hash) DO UPDATE SET
                      local_job_id = excluded.local_job_id,
                      custom_id = excluded.custom_id,
                      status = CASE
                        WHEN batch_requests.status = 'completed' THEN 'completed'
                        ELSE 'pending'
                      END,
                      error_message = CASE
                        WHEN batch_requests.status = 'completed' THEN batch_requests.error_message
                        ELSE NULL
                      END,
                      updated_at = excluded.updated_at
                    """,
                    (request.request_hash, local_job_id, request.custom_id, now, now),
                )

    def _update_job(
        self,
        local_job_id: str,
        *,
        status: str,
        provider_batch_id: str | None = None,
        input_file_id: str | None = None,
        output_file_id: str | None = None,
        error_file_id: str | None = None,
        provider_payload: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> None:
        with self._database_lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE batch_jobs
                SET status = ?,
                    provider_batch_id = COALESCE(?, provider_batch_id),
                    input_file_id = COALESCE(?, input_file_id),
                    output_file_id = COALESCE(?, output_file_id),
                    error_file_id = COALESCE(?, error_file_id),
                    provider_payload_json = COALESCE(?, provider_payload_json),
                    error_message = ?,
                    updated_at = ?
                WHERE local_job_id = ?
                """,
                (
                    status,
                    provider_batch_id,
                    input_file_id,
                    output_file_id,
                    error_file_id,
                    json.dumps(provider_payload, ensure_ascii=False) if provider_payload else None,
                    error_message,
                    time.time(),
                    local_job_id,
                ),
            )

    def _cached_response(self, request_hash: str) -> LLMResponse | None:
        with self._database_lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT response_json
                FROM batch_requests
                WHERE request_hash = ? AND status = 'completed'
                """,
                (request_hash,),
            ).fetchone()
        if not row or not row["response_json"]:
            return None
        payload = json.loads(row["response_json"])
        if not isinstance(payload, dict):
            return None
        return _llm_response_from_body(payload)

    def _cached_error(self, request_hash: str) -> str | None:
        with self._database_lock, self._connect() as conn:
            row = conn.execute(
                "SELECT error_message FROM batch_requests WHERE request_hash = ?",
                (request_hash,),
            ).fetchone()
        return str(row["error_message"]) if row and row["error_message"] else None

    def _store_request_result(self, request_hash: str, body: dict[str, Any]) -> None:
        with self._database_lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE batch_requests
                SET status = 'completed', response_json = ?, error_message = NULL, updated_at = ?
                WHERE request_hash = ?
                """,
                (json.dumps(body, ensure_ascii=False), time.time(), request_hash),
            )

    def _store_request_error(self, request_hash: str, error: str) -> None:
        with self._database_lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE batch_requests
                SET status = 'failed', error_message = ?, updated_at = ?
                WHERE request_hash = ? AND status != 'completed'
                """,
                (error, time.time(), request_hash),
            )

    def _store_standalone_response(
        self,
        request_hash: str,
        custom_id: str,
        response: LLMResponse,
    ) -> None:
        now = time.time()
        response_body = response.raw_response
        if not isinstance(response_body.get("choices"), list):
            response_body = {
                **response.raw_response,
                "choices": [{"message": {"role": "assistant", "content": response.content}}],
                "usage": response.usage,
            }
        with self._database_lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO batch_requests (
                  request_hash, local_job_id, custom_id, status, response_json,
                  created_at, updated_at
                ) VALUES (?, 'realtime-fallback', ?, 'completed', ?, ?, ?)
                ON CONFLICT(request_hash) DO UPDATE SET
                  status = 'completed', response_json = excluded.response_json,
                  error_message = NULL, updated_at = excluded.updated_at
                """,
                (
                    request_hash,
                    custom_id,
                    json.dumps(response_body, ensure_ascii=False),
                    now,
                    now,
                ),
            )

    def _recoverable_jobs(self) -> list[dict[str, Any]]:
        with self._database_lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT jobs.*
                FROM batch_jobs AS jobs
                JOIN batch_requests AS requests
                  ON requests.local_job_id = jobs.local_job_id
                WHERE requests.status = 'pending'
                ORDER BY jobs.created_at
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def _job_custom_id_map(self, local_job_id: str) -> dict[str, str]:
        with self._database_lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT custom_id, request_hash FROM batch_requests WHERE local_job_id = ?",
                (local_job_id,),
            ).fetchall()
        return {str(row["custom_id"]): str(row["request_hash"]) for row in rows}

    def _mark_missing_requests_failed(self, local_job_id: str, error: str) -> None:
        with self._database_lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE batch_requests
                SET status = 'failed', error_message = ?, updated_at = ?
                WHERE local_job_id = ? AND status = 'pending'
                """,
                (error, time.time(), local_job_id),
            )

    def _fail_job_requests(self, local_job_id: str, error: str) -> None:
        self._update_job(local_job_id, status="failed", error_message=error)
        self._mark_missing_requests_failed(local_job_id, error)

    def _write_manifest(self, local_job_id: str, payload: dict[str, Any]) -> None:
        path = self.jobs_dir / local_job_id / "manifest.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)


def _request_hash(body: dict[str, Any]) -> str:
    serialized = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(serialized).hexdigest()


def _request_line(custom_id: str, body: dict[str, Any]) -> bytes:
    payload = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }
    return (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _compress_data_urls(value: Any, *, max_dimension: int, quality: int) -> Any:
    if isinstance(value, list):
        return [
            _compress_data_urls(item, max_dimension=max_dimension, quality=quality)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: _compress_data_urls(item, max_dimension=max_dimension, quality=quality)
            for key, item in value.items()
        }
    if not isinstance(value, str) or not value.startswith("data:image/") or ";base64," not in value:
        return value
    try:
        encoded = value.split(";base64,", 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
            image = image.convert("RGBA")
            canvas = Image.new("RGB", image.size, "white")
            canvas.paste(image, mask=image.getchannel("A"))
            canvas.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            output = io.BytesIO()
            canvas.save(output, format="JPEG", quality=quality, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")
    except Exception:
        return value


def _expand_tiny_data_urls(value: Any, *, min_dimension: int) -> Any:
    if isinstance(value, list):
        return [_expand_tiny_data_urls(item, min_dimension=min_dimension) for item in value]
    if isinstance(value, dict):
        return {
            key: _expand_tiny_data_urls(item, min_dimension=min_dimension)
            for key, item in value.items()
        }
    if not isinstance(value, str) or not value.startswith("data:image/") or ";base64," not in value:
        return value
    try:
        encoded = value.split(";base64,", 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
            width, height = image.size
            if width > 10 and height > 10:
                return value
            scale = max(min_dimension / max(1, width), min_dimension / max(1, height))
            target = (
                max(min_dimension, round(width * scale)),
                max(min_dimension, round(height * scale)),
            )
            image = image.convert("RGBA").resize(target, Image.Resampling.NEAREST)
            canvas = Image.new("RGB", target, "white")
            canvas.paste(image, mask=image.getchannel("A"))
            output = io.BytesIO()
            canvas.save(output, format="PNG", optimize=True)
        return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")
    except Exception:
        return value


def _llm_response_from_body(body: dict[str, Any]) -> LLMResponse:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("Qwen Batch response does not contain choices.")
    message = choices[0].get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ValueError("Qwen Batch response does not contain message.content.")
    content = str(message["content"])
    try:
        parsed = extract_json_object(content)
        parse_error = None
    except (json.JSONDecodeError, ValueError) as exc:
        parsed = None
        parse_error = str(exc)
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    return LLMResponse(
        content=content,
        parsed_json=parsed,
        raw_response=body,
        usage=usage,
        parse_error=parse_error,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Qwen Batch response is missing {key}.")
    return value


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None
