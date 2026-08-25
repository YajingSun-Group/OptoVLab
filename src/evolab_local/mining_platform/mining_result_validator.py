from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from evolab_local.mining_platform.schemas.domain_template import DomainTemplate, TemplateField
from evolab_local.mining_platform.schemas.mining_result import (
    MiningResultIssue,
    MiningResultValidationReport,
)


@dataclass(frozen=True)
class PathSegment:
    name: str
    is_list: bool


@dataclass(frozen=True)
class PathValue:
    path: str
    value: Any
    exists: bool


@dataclass(frozen=True)
class TraversalIssue:
    path: str
    code: str
    message: str
    value_present: bool = True


class MiningResultValidator:
    def __init__(self, template: DomainTemplate) -> None:
        self.template = template

    def validate(self, payload: object) -> MiningResultValidationReport:
        fatal_errors: list[MiningResultIssue] = []
        repairable: list[MiningResultIssue] = []
        warnings: list[MiningResultIssue] = []

        if not isinstance(payload, Mapping):
            fatal_errors.append(
                _fatal(
                    "root_not_object",
                    "$",
                    "Mining result must be a JSON object.",
                )
            )
            return _report(self.template, fatal_errors, repairable, warnings)

        _bucket_issues(self._validate_top_level(payload), fatal_errors, repairable, warnings)
        evidence_ids = self._collect_unique_ids(
            payload,
            top_level_key="evidence",
            id_key="evidence_id",
            errors=fatal_errors,
        )
        material_ids = self._collect_unique_ids(
            payload,
            top_level_key="materials",
            id_key="paper_material_id",
            errors=fatal_errors,
        )
        _bucket_issues(self._validate_template_fields(payload), fatal_errors, repairable, warnings)
        _bucket_issues(
            self._validate_evidence_refs(payload, evidence_ids), fatal_errors, repairable, warnings
        )
        _bucket_issues(
            self._validate_material_refs(payload, material_ids), fatal_errors, repairable, warnings
        )

        return _report(
            self.template,
            fatal_errors,
            repairable,
            warnings,
            checked_field_count=len(self.template.fields),
            metadata={
                "evidence_count": len(evidence_ids),
                "material_count": len(material_ids),
                "device_count": _list_len(payload.get("devices")),
            },
        )

    def _validate_top_level(self, payload: Mapping[str, Any]) -> list[MiningResultIssue]:
        errors: list[MiningResultIssue] = []
        for key in self.template.required_output_keys:
            if key not in payload:
                errors.append(
                    _fatal("missing_top_level_key", f"$.{key}", f"Missing top-level key {key!r}.")
                )
                continue
            if not isinstance(payload[key], list):
                errors.append(
                    _fatal(
                        "top_level_key_not_list",
                        f"$.{key}",
                        f"Top-level key {key!r} must be a list.",
                    )
                )
        excluded_keys = self.template.llm_output_schema.get("excluded_keys", [])
        if isinstance(excluded_keys, list):
            for key in excluded_keys:
                if key in payload:
                    errors.append(
                        _warning(
                            "excluded_top_level_key",
                            f"$.{key}",
                            f"Top-level key {key!r} is excluded by template.",
                        )
                    )
        return errors

    def _collect_unique_ids(
        self,
        payload: Mapping[str, Any],
        *,
        top_level_key: str,
        id_key: str,
        errors: list[MiningResultIssue],
    ) -> set[str]:
        values = payload.get(top_level_key)
        if not isinstance(values, list):
            return set()
        ids: set[str] = set()
        for index, item in enumerate(values):
            item_path = f"$.{top_level_key}[{index}]"
            if not isinstance(item, Mapping):
                errors.append(
                    _fatal(
                        "top_level_item_not_object",
                        item_path,
                        f"Items in {top_level_key!r} must be objects.",
                    )
                )
                continue
            value = item.get(id_key)
            if _is_empty(value):
                continue
            if not isinstance(value, str):
                errors.append(
                    _fatal(
                        "id_not_string",
                        f"{item_path}.{id_key}",
                        f"{id_key!r} must be a string.",
                    )
                )
                continue
            if value in ids:
                errors.append(
                    _fatal(
                        "duplicate_id",
                        f"{item_path}.{id_key}",
                        f"Duplicate {id_key!r}: {value!r}.",
                    )
                )
            ids.add(value)
        return ids

    def _validate_template_fields(self, payload: Mapping[str, Any]) -> list[MiningResultIssue]:
        errors: list[MiningResultIssue] = []
        for field in self.template.fields:
            values, traversal_issues = resolve_field_values(payload, field.field_path)
            for issue in traversal_issues:
                if field.required or issue.value_present:
                    issue_factory = _fatal if field.required else _repairable
                    errors.append(
                        issue_factory(
                            issue.code,
                            issue.path,
                            issue.message,
                            field_path=field.field_path,
                        )
                    )

            if field.required:
                # No resolved values means an upstream wildcard collection is
                # empty. Required applies to items that exist, not to an
                # imaginary child inside an empty collection.
                if not values:
                    continue
                present_values = [
                    value for value in values if value.exists and not _is_empty(value.value)
                ]
                if not present_values:
                    errors.append(
                        _fatal(
                            "required_field_missing",
                            field.field_path,
                            f"Required field {field.field_path!r} is missing or empty.",
                            field_path=field.field_path,
                        )
                    )
                for value in values:
                    if not value.exists or _is_empty(value.value):
                        errors.append(
                            _fatal(
                                "required_field_missing",
                                value.path,
                                f"Required field {field.field_path!r} is missing or empty.",
                                field_path=field.field_path,
                            )
                        )

            for value in values:
                if not value.exists or _is_empty(value.value):
                    continue
                type_issue = self._validate_value_type(field, value)
                if type_issue:
                    errors.append(type_issue)
                range_issue = self._validate_numeric_range(field, value)
                if range_issue:
                    errors.append(range_issue)
        return _dedupe_issues(errors)

    def _validate_value_type(
        self,
        field: TemplateField,
        path_value: PathValue,
    ) -> MiningResultIssue | None:
        value = path_value.value
        data_type = field.data_type
        if data_type == "string":
            valid = isinstance(value, str)
        elif data_type == "number":
            valid = _is_number(value)
        elif data_type == "integer":
            valid = isinstance(value, int) and not isinstance(value, bool)
        elif data_type == "boolean":
            valid = isinstance(value, bool)
        elif data_type == "list_string":
            valid = isinstance(value, list) and all(isinstance(item, str) for item in value)
        elif data_type == "list_number":
            valid = isinstance(value, list) and all(_is_number(item) for item in value)
        elif data_type == "enum":
            valid = isinstance(value, str) and self._enum_value_allowed(field, value)
        elif data_type == "list_enum":
            valid = (
                isinstance(value, list)
                and all(isinstance(item, str) for item in value)
                and all(self._enum_value_allowed(field, item) for item in value)
            )
        else:
            valid = True

        if valid:
            return None
        return _warning(
            "invalid_field_value",
            path_value.path,
            f"Value does not match data_type {data_type!r} for {field.field_path!r}.",
            field_path=field.field_path,
        )

    def _enum_value_allowed(self, field: TemplateField, value: str) -> bool:
        if not field.enum_ref:
            return False
        return value in self.template.vocabularies.get(field.enum_ref, [])

    def _validate_numeric_range(
        self,
        field: TemplateField,
        path_value: PathValue,
    ) -> MiningResultIssue | None:
        if not _is_number(path_value.value):
            return None
        min_value = field.validation.get("min")
        max_value = field.validation.get("max")
        value = float(path_value.value)
        if min_value is not None and value < float(min_value):
            return _warning(
                "value_below_min",
                path_value.path,
                f"Value {value:g} is below minimum {float(min_value):g}.",
                field_path=field.field_path,
            )
        if max_value is not None and value > float(max_value):
            return _warning(
                "value_above_max",
                path_value.path,
                f"Value {value:g} is above maximum {float(max_value):g}.",
                field_path=field.field_path,
            )
        return None

    def _validate_evidence_refs(
        self,
        payload: Mapping[str, Any],
        evidence_ids: set[str],
    ) -> list[MiningResultIssue]:
        errors: list[MiningResultIssue] = []
        for path, refs in _find_key_values(payload, "evidence_refs"):
            if refs is None:
                continue
            if not isinstance(refs, list):
                errors.append(
                    _fatal("evidence_refs_not_list", path, "evidence_refs must be a list.")
                )
                continue
            for index, ref in enumerate(refs):
                ref_path = f"{path}[{index}]"
                if not isinstance(ref, str):
                    errors.append(
                        _fatal(
                            "evidence_ref_not_string", ref_path, "Evidence ref must be a string."
                        )
                    )
                elif ref not in evidence_ids:
                    errors.append(
                        _fatal(
                            "unknown_evidence_ref",
                            ref_path,
                            f"Unknown evidence ref {ref!r}.",
                        )
                    )
        return errors

    def _validate_material_refs(
        self,
        payload: Mapping[str, Any],
        material_ids: set[str],
    ) -> list[MiningResultIssue]:
        errors: list[MiningResultIssue] = []
        for path, material_id in _find_key_values(payload, "paper_material_id"):
            if path.startswith("$.materials["):
                continue
            if material_id is None or material_id == "":
                continue
            if not isinstance(material_id, str):
                errors.append(
                    _fatal(
                        "paper_material_id_not_string",
                        path,
                        "paper_material_id must be a string when provided.",
                    )
                )
            elif material_id not in material_ids:
                errors.append(
                    _fatal(
                        "unknown_paper_material_id",
                        path,
                        f"Unknown paper_material_id {material_id!r}.",
                    )
                )
        return errors


def resolve_field_values(
    payload: Mapping[str, Any],
    field_path: str,
) -> tuple[list[PathValue], list[TraversalIssue]]:
    segments = _parse_field_path(field_path)
    return _walk_path(payload, "$", segments)


def is_empty_value(value: Any) -> bool:
    return _is_empty(value)


def _parse_field_path(field_path: str) -> list[PathSegment]:
    segments: list[PathSegment] = []
    for raw in field_path.split("."):
        if raw.endswith("[]"):
            segments.append(PathSegment(name=raw[:-2], is_list=True))
        else:
            segments.append(PathSegment(name=raw, is_list=False))
    return segments


def _walk_path(
    current: Any,
    current_path: str,
    segments: list[PathSegment],
) -> tuple[list[PathValue], list[TraversalIssue]]:
    if not segments:
        return [PathValue(current_path, current, True)], []

    segment = segments[0]
    remaining = segments[1:]
    next_path = f"{current_path}.{segment.name}"
    if not isinstance(current, Mapping):
        if current is None:
            return [PathValue(next_path, None, False)], []
        return [], [
            TraversalIssue(
                current_path,
                "expected_object",
                f"Expected object before reading {segment.name!r}.",
            )
        ]

    if segment.name not in current:
        expected_path = f"{next_path}[]" if segment.is_list else next_path
        return [PathValue(expected_path, None, False)], []

    child = current[segment.name]
    if segment.is_list:
        if child is None:
            return [PathValue(f"{next_path}[]", None, False)], []
        if not isinstance(child, list):
            return [], [
                TraversalIssue(
                    next_path,
                    "expected_list",
                    f"Expected list at {next_path}.",
                )
            ]
        if not child:
            # A required field below a wildcard is required for every item that
            # exists, but an empty collection does not imply a missing child.
            # For example, performance=[] means that no performance was
            # reported; it must not require a fabricated metric_family value.
            return [], []
        values: list[PathValue] = []
        issues: list[TraversalIssue] = []
        for index, item in enumerate(child):
            item_values, item_issues = _walk_path(item, f"{next_path}[{index}]", remaining)
            values.extend(item_values)
            issues.extend(item_issues)
        return values, issues

    return _walk_path(child, next_path, remaining)


def _find_key_values(payload: Any, key: str, path: str = "$") -> list[tuple[str, Any]]:
    found: list[tuple[str, Any]] = []
    if isinstance(payload, Mapping):
        for item_key, value in payload.items():
            item_path = f"{path}.{item_key}"
            if item_key == key:
                found.append((item_path, value))
            found.extend(_find_key_values(value, key, item_path))
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            found.extend(_find_key_values(item, key, f"{path}[{index}]"))
    return found


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, list):
        return len(value) == 0
    return False


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _list_len(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _fatal(
    code: str,
    path: str,
    message: str,
    *,
    field_path: str | None = None,
) -> MiningResultIssue:
    return _issue("fatal", code, path, message, field_path=field_path)


def _repairable(
    code: str,
    path: str,
    message: str,
    *,
    field_path: str | None = None,
) -> MiningResultIssue:
    return _issue("repairable", code, path, message, field_path=field_path)


def _warning(
    code: str,
    path: str,
    message: str,
    *,
    field_path: str | None = None,
) -> MiningResultIssue:
    return _issue("warning", code, path, message, field_path=field_path)


def _issue(
    severity: str,
    code: str,
    path: str,
    message: str,
    *,
    field_path: str | None = None,
) -> MiningResultIssue:
    return MiningResultIssue(
        severity=severity,
        code=code,
        path=path,
        message=message,
        field_path=field_path,
    )


def _bucket_issues(
    issues: list[MiningResultIssue],
    fatal_errors: list[MiningResultIssue],
    repairable: list[MiningResultIssue],
    warnings: list[MiningResultIssue],
) -> None:
    for issue in issues:
        if issue.severity == "fatal":
            fatal_errors.append(issue)
        elif issue.severity == "repairable":
            repairable.append(issue)
        else:
            warnings.append(issue)


def _report(
    template: DomainTemplate,
    errors: list[MiningResultIssue],
    repairable: list[MiningResultIssue],
    warnings: list[MiningResultIssue],
    checked_field_count: int = 0,
    metadata: dict[str, Any] | None = None,
) -> MiningResultValidationReport:
    deduped_errors = _dedupe_issues(errors)
    deduped_repairable = _dedupe_issues(repairable)
    deduped_warnings = _dedupe_issues(warnings)
    return MiningResultValidationReport(
        template_id=template.template_id,
        valid=not deduped_errors,
        errors=deduped_errors,
        repairable=deduped_repairable,
        warnings=deduped_warnings,
        fatal_count=len(deduped_errors),
        repairable_count=len(deduped_repairable),
        warning_count=len(deduped_warnings),
        checked_field_count=checked_field_count,
        metadata=metadata or {},
    )


def _dedupe_issues(issues: list[MiningResultIssue]) -> list[MiningResultIssue]:
    seen: set[tuple[str, str, str | None, str]] = set()
    deduped: list[MiningResultIssue] = []
    for issue in issues:
        key = (issue.code, issue.path, issue.field_path, issue.message)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(issue)
    return deduped
