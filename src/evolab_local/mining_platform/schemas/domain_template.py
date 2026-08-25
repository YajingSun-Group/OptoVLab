from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PaperMetadataPolicy(BaseModel):
    model_config = ConfigDict(extra="allow")

    source: str
    llm_should_extract: bool = False
    notes: str | None = None


class TemplateEntity(BaseModel):
    model_config = ConfigDict(extra="allow")

    path: str
    label: str
    description: str | None = None


class TemplateField(BaseModel):
    model_config = ConfigDict(extra="allow")

    field_path: str
    label: str
    data_type: str
    required: bool = False
    entity: str
    enum_ref: str | None = None
    evidence_required: bool | None = None
    ui_group: str | None = None
    review_priority: str | None = None
    extraction_hint: str | None = None
    unit: str | None = None
    validation: dict[str, Any] = Field(default_factory=dict)


class DomainTemplate(BaseModel):
    model_config = ConfigDict(extra="allow")

    template_id: str
    domain: str
    template_name: str
    version: str
    status: str
    scope: str | None = None
    description: str | None = None
    paper_metadata_policy: PaperMetadataPolicy
    scope_filter: dict[str, list[str]] = Field(default_factory=dict)
    review_policy: dict[str, Any] = Field(default_factory=dict)
    entities: dict[str, TemplateEntity]
    vocabularies: dict[str, list[str]] = Field(default_factory=dict)
    ui: dict[str, Any] = Field(default_factory=dict)
    llm_output_schema: dict[str, Any]
    fields: list[TemplateField]
    current_version_scope: dict[str, Any] = Field(default_factory=dict)
    example_output: dict[str, Any] = Field(default_factory=dict)
    source_path: Path | None = None

    def field_by_path(self, field_path: str) -> TemplateField | None:
        for field in self.fields:
            if field.field_path == field_path:
                return field
        return None

    @property
    def required_output_keys(self) -> list[str]:
        keys = self.llm_output_schema.get("required_keys", [])
        return [str(key) for key in keys] if isinstance(keys, list) else []
