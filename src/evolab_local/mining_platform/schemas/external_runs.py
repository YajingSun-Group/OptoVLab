from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MinerUParseRun(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    mineru_run_id: str
    paper_id: str
    task_id: str | None = None
    status: str
    service_base_url: str
    parser_version: str | None = None
    content_item_count: int = 0
    result_path: str | None = None
    content_list_path: str | None = None
    markdown_path: str | None = None
    error_message: str | None = None
    created_at: str
    completed_at: str | None = None


class LLMMiningRun(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    llm_run_id: str
    paper_id: str
    template_id: str
    provider: str
    model: str
    status: str
    source_parser: str
    input_item_count: int = 0
    prompt_path: str | None = None
    raw_response_path: str | None = None
    mining_result_path: str | None = None
    validation_report_path: str | None = None
    candidate_run_id: str | None = None
    error_message: str | None = None
    token_usage: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    completed_at: str | None = None


class LLMMiningResult(BaseModel):
    run: LLMMiningRun
    mining_result: dict[str, Any] = Field(default_factory=dict)
    validation_report: dict[str, Any] = Field(default_factory=dict)
    candidate_run_id: str | None = None
