from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from evolab_local.mining_platform.core.config import MiningPlatformConfig
from evolab_local.mining_platform.mining_result_validator import MiningResultValidator
from evolab_local.mining_platform.schemas.domain_template import DomainTemplate, TemplateField
from evolab_local.mining_platform.schemas.mining_result import MiningResultValidationReport


ENTITY_ALIASES: dict[str, str] = {
    "evidence": "evidence",
    "material": "materials",
    "materials": "materials",
    "device": "devices",
    "devices": "devices",
    "layer": "layers",
    "layers": "layers",
    "component": "components",
    "components": "components",
    "performance": "performance",
}

FIELD_ROOT_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)")


class DomainTemplateValidationError(ValueError):
    def __init__(self, template_id: str, issues: list[str]) -> None:
        self.template_id = template_id
        self.issues = issues
        joined = "; ".join(issues)
        super().__init__(f"Domain template {template_id!r} is invalid: {joined}")


class DomainTemplateLoader:
    def __init__(self, template_dir: Path) -> None:
        self.template_dir = template_dir

    @classmethod
    def from_config(cls, config: MiningPlatformConfig) -> DomainTemplateLoader:
        return cls(config.project_root / "config" / "mining_platform" / "domains")

    def load(self, template_id: str) -> DomainTemplate:
        path = self._template_path(template_id)
        return load_domain_template(path)

    def list_templates(self) -> list[DomainTemplate]:
        if not self.template_dir.exists():
            return []
        templates: list[DomainTemplate] = []
        for path in sorted(self.template_dir.glob("*.yaml")):
            templates.append(load_domain_template(path))
        for path in sorted(self.template_dir.glob("*.yml")):
            templates.append(load_domain_template(path))
        return templates

    def _template_path(self, template_id: str) -> Path:
        candidate = Path(template_id)
        if candidate.suffix in {".yaml", ".yml"}:
            return candidate if candidate.is_absolute() else self.template_dir / candidate
        return self.template_dir / f"{template_id}.yaml"


class DomainTemplateService:
    def __init__(self, config: MiningPlatformConfig) -> None:
        self.config = config
        self.loader = DomainTemplateLoader.from_config(config)

    def get_template(self, template_id: str) -> DomainTemplate:
        return self.loader.load(template_id)

    def list_templates(self) -> list[DomainTemplate]:
        return self.loader.list_templates()

    def validate_mining_result(
        self,
        template_id: str,
        payload: object,
    ) -> MiningResultValidationReport:
        template = self.get_template(template_id)
        return MiningResultValidator(template).validate(payload)


def load_domain_template(path: Path) -> DomainTemplate:
    if not path.exists():
        raise FileNotFoundError(f"Domain template not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        template = DomainTemplate.model_validate({**raw, "source_path": path})
    except ValidationError as exc:
        template_id = str(raw.get("template_id") or path.stem)
        raise DomainTemplateValidationError(template_id, [str(exc)]) from exc
    validate_domain_template(template)
    return template


def validate_domain_template(template: DomainTemplate) -> None:
    issues: list[str] = []
    issues.extend(_validate_entities(template))
    issues.extend(_validate_llm_output_schema(template))
    issues.extend(_validate_fields(template))
    issues.extend(_validate_example_output(template))
    if issues:
        raise DomainTemplateValidationError(template.template_id, issues)


def _validate_entities(template: DomainTemplate) -> list[str]:
    issues: list[str] = []
    if not template.entities:
        issues.append("entities must not be empty")
        return issues
    for entity_name, entity in template.entities.items():
        if not entity.path:
            issues.append(f"entity {entity_name!r} is missing path")
    return issues


def _validate_llm_output_schema(template: DomainTemplate) -> list[str]:
    issues: list[str] = []
    required_keys = template.required_output_keys
    if not required_keys:
        issues.append("llm_output_schema.required_keys must not be empty")
    for key in required_keys:
        if key not in template.entities:
            issues.append(
                f"llm_output_schema.required_keys contains {key!r}, but entities does not define it"
            )
    excluded_keys = template.llm_output_schema.get("excluded_keys", [])
    if isinstance(excluded_keys, list) and "paper" in excluded_keys:
        policy = template.paper_metadata_policy
        if policy.llm_should_extract:
            issues.append(
                "paper metadata is excluded from LLM output, "
                "but paper_metadata_policy.llm_should_extract is true"
            )
    return issues


def _validate_fields(template: DomainTemplate) -> list[str]:
    issues: list[str] = []
    if not template.fields:
        return ["fields must not be empty"]

    seen_paths: set[str] = set()
    for field in template.fields:
        if field.field_path in seen_paths:
            issues.append(f"duplicate field_path {field.field_path!r}")
        seen_paths.add(field.field_path)
        issues.extend(_validate_field_entity(template, field))
        issues.extend(_validate_field_enum(template, field))
    return issues


def _validate_field_entity(template: DomainTemplate, field: TemplateField) -> list[str]:
    issues: list[str] = []
    canonical_entity = _canonical_entity(field.entity, template.entities)
    if canonical_entity is None:
        issues.append(f"field {field.field_path!r} uses unknown entity {field.entity!r}")
        return issues

    root_entity = _field_root(field.field_path)
    if root_entity is None:
        issues.append(f"field {field.field_path!r} has invalid field_path syntax")
        return issues
    if root_entity not in template.entities:
        issues.append(
            f"field {field.field_path!r} starts with {root_entity!r}, "
            "but entities does not define that root"
        )
        return issues

    entity_path = template.entities[canonical_entity].path
    if not _field_path_matches_entity_path(field.field_path, entity_path):
        issues.append(
            f"field {field.field_path!r} is tagged as entity {field.entity!r}, "
            f"but does not match entity path {entity_path!r}"
        )
    return issues


def _validate_field_enum(template: DomainTemplate, field: TemplateField) -> list[str]:
    issues: list[str] = []
    if field.enum_ref and field.enum_ref not in template.vocabularies:
        issues.append(f"field {field.field_path!r} references unknown enum_ref {field.enum_ref!r}")
    if field.data_type in {"enum", "list_enum"} and not field.enum_ref:
        issues.append(
            f"field {field.field_path!r} has data_type {field.data_type!r} without enum_ref"
        )
    return issues


def _validate_example_output(template: DomainTemplate) -> list[str]:
    issues: list[str] = []
    if not template.example_output:
        issues.append("example_output must not be empty")
        return issues
    for key in template.required_output_keys:
        if key not in template.example_output:
            issues.append(f"example_output is missing required output key {key!r}")
        elif not isinstance(template.example_output[key], list):
            issues.append(f"example_output key {key!r} must be a list")
    return issues


def _canonical_entity(entity: str, configured_entities: dict[str, Any]) -> str | None:
    if entity in configured_entities:
        return entity
    alias = ENTITY_ALIASES.get(entity)
    if alias in configured_entities:
        return alias
    return None


def _field_root(field_path: str) -> str | None:
    match = FIELD_ROOT_RE.match(field_path)
    return match.group(1) if match else None


def _field_path_matches_entity_path(field_path: str, entity_path: str) -> bool:
    normalized_field_path = _normalize_template_path(field_path)
    normalized_entity_path = _normalize_template_path(entity_path)
    return normalized_field_path == normalized_entity_path or normalized_field_path.startswith(
        f"{normalized_entity_path}."
    )


def _normalize_template_path(path: str) -> str:
    return path.replace("[]", "")
