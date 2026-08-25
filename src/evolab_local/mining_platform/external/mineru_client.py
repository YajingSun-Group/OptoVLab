from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from evolab_local.mining_platform.core.config import MinerUConfig


@dataclass(frozen=True)
class MinerUParsedDocument:
    task_id: str | None
    backend: str | None
    version: str | None
    file_name: str
    md_content: str
    content_list: list[dict[str, Any]]
    images: dict[str, str] = field(default_factory=dict)
    raw_result: dict[str, Any] = field(default_factory=dict)


class MinerUClient:
    def __init__(self, config: MinerUConfig) -> None:
        self.config = config
        self.base_url = config.base_url.rstrip("/")

    def parse_pdf(
        self,
        pdf_path: Path,
        *,
        return_images: bool | None = None,
    ) -> MinerUParsedDocument:
        task = self.submit_task(pdf_path, return_images=return_images)
        task_id = str(task["task_id"])
        deadline = time.monotonic() + self.config.timeout_seconds
        while True:
            status = self.get_task(task_id)
            state = status.get("status")
            if state == "completed":
                result = self.get_result(task_id)
                return parse_mineru_result(result, task_id=task_id)
            if state == "failed":
                raise RuntimeError(str(status.get("error") or "MinerU task failed."))
            if time.monotonic() >= deadline:
                raise TimeoutError(f"MinerU task timed out: {task_id}")
            time.sleep(self.config.poll_interval_seconds)

    def submit_task(
        self,
        pdf_path: Path,
        *,
        return_images: bool | None = None,
    ) -> dict[str, Any]:
        data = {
            "backend": self.config.backend,
            "parse_method": self.config.parse_method,
            "lang_list": self.config.lang_list,
            "formula_enable": str(self.config.formula_enable).lower(),
            "table_enable": str(self.config.table_enable).lower(),
            "return_md": str(self.config.return_md).lower(),
            "return_content_list": str(self.config.return_content_list).lower(),
            "return_middle_json": str(self.config.return_middle_json).lower(),
            "return_images": str(
                self.config.return_images if return_images is None else return_images
            ).lower(),
        }
        with pdf_path.open("rb") as handle:
            files = {"files": (pdf_path.name, handle, "application/pdf")}
            with httpx.Client(timeout=self.config.timeout_seconds, trust_env=False) as client:
                response = client.post(f"{self.base_url}/tasks", data=data, files=files)
                response.raise_for_status()
                payload = response.json()
        if not isinstance(payload, dict) or not payload.get("task_id"):
            raise ValueError("MinerU /tasks response does not contain task_id.")
        return payload

    def get_task(self, task_id: str) -> dict[str, Any]:
        with httpx.Client(timeout=30, trust_env=False) as client:
            response = client.get(f"{self.base_url}/tasks/{task_id}")
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("MinerU task status response must be an object.")
        return payload

    def get_result(self, task_id: str) -> dict[str, Any]:
        with httpx.Client(timeout=self.config.timeout_seconds, trust_env=False) as client:
            response = client.get(f"{self.base_url}/tasks/{task_id}/result")
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("MinerU task result response must be an object.")
        return payload


def parse_mineru_result(result: dict[str, Any], task_id: str | None = None) -> MinerUParsedDocument:
    results = result.get("results")
    if not isinstance(results, dict) or not results:
        raise ValueError("MinerU result does not contain results.")
    file_name, item = next(iter(results.items()))
    if not isinstance(item, dict):
        raise ValueError("MinerU result item must be an object.")
    md_content = item.get("md_content") if isinstance(item.get("md_content"), str) else ""
    raw_content_list = item.get("content_list")
    content_list = _parse_content_list(raw_content_list)
    raw_images = item.get("images")
    images = (
        {
            str(name): value
            for name, value in raw_images.items()
            if isinstance(name, str) and isinstance(value, str)
        }
        if isinstance(raw_images, dict)
        else {}
    )
    return MinerUParsedDocument(
        task_id=task_id,
        backend=result.get("backend") if isinstance(result.get("backend"), str) else None,
        version=result.get("version") if isinstance(result.get("version"), str) else None,
        file_name=str(file_name),
        md_content=md_content,
        content_list=content_list,
        images=images,
        raw_result=result,
    )


def _parse_content_list(raw: object) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, str):
        parsed = json.loads(raw)
    else:
        parsed = raw
    if not isinstance(parsed, list):
        raise ValueError("MinerU content_list must be a list or a JSON-encoded list.")
    return [item for item in parsed if isinstance(item, dict)]
