from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


IssueSeverity = Literal["fatal", "repairable", "warning"]


class MiningResultIssue(BaseModel):
    severity: IssueSeverity
    code: str
    path: str
    message: str
    field_path: str | None = None


class MiningResultValidationReport(BaseModel):
    template_id: str
    valid: bool
    errors: list[MiningResultIssue] = Field(default_factory=list)
    repairable: list[MiningResultIssue] = Field(default_factory=list)
    warnings: list[MiningResultIssue] = Field(default_factory=list)
    fatal_count: int = 0
    repairable_count: int = 0
    warning_count: int = 0
    checked_field_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
