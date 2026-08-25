from __future__ import annotations

import gzip
import json
import threading
from collections import Counter
from pathlib import Path
from typing import Any


class OLEDDeviceCatalog:
    """Lazy, read-only access to the exported OLED device corpus."""

    def __init__(self, dataset_path: Path) -> None:
        self.dataset_path = dataset_path
        self._lock = threading.Lock()
        self._records: list[dict[str, Any]] | None = None

    def records(self) -> list[dict[str, Any]]:
        if self._records is None:
            with self._lock:
                if self._records is None:
                    if not self.dataset_path.exists():
                        raise FileNotFoundError(f"OLED dataset not found: {self.dataset_path}")
                    with gzip.open(self.dataset_path, "rt", encoding="utf-8") as handle:
                        payload = json.load(handle)
                    if not isinstance(payload, list):
                        raise ValueError("OLED dataset root must be a JSON array")
                    self._records = [item for item in payload if isinstance(item, dict)]
        return self._records

    def stats(self) -> dict[str, Any]:
        records = self.records()
        paper_ids = {record.get("paper_id") for record in records if record.get("paper_id")}
        emitters = {
            str(record.get("final_emitter")).strip()
            for record in records
            if record.get("final_emitter")
        }
        quality = Counter(str(record.get("quality_tier") or "unknown") for record in records)
        return {
            "devices": len(records),
            "papers": len(paper_ids),
            "final_emitters": len(emitters),
            "quality_tiers": dict(quality.most_common()),
            "dataset_path": str(self.dataset_path),
        }

    @staticmethod
    def searchable_text(record: dict[str, Any]) -> str:
        layer_parts: list[str] = []
        for layer in record.get("layers") or []:
            if not isinstance(layer, dict):
                continue
            layer_parts.extend(
                str(layer.get(key) or "")
                for key in ("layer_role", "layer_name", "material", "materials")
            )
            for component in layer.get("components") or []:
                if isinstance(component, dict):
                    layer_parts.extend(
                        str(component.get(key) or "")
                        for key in ("material_mention", "component_role", "full_name")
                    )
        material_parts: list[str] = []
        for material in record.get("materials") or []:
            if isinstance(material, dict):
                material_parts.extend(
                    str(material.get(key) or "")
                    for key in ("mention", "abbreviation", "full_name", "canonical_name")
                )
            else:
                material_parts.append(str(material))
        mechanisms = record.get("emission_mechanism") or []
        if not isinstance(mechanisms, list):
            mechanisms = [mechanisms]
        fields = [
            record.get("id"),
            record.get("doi"),
            record.get("title"),
            record.get("journal"),
            record.get("architecture"),
            record.get("device_label"),
            record.get("device_type"),
            record.get("emission_color"),
            record.get("fabrication_method"),
            record.get("final_emitter"),
            record.get("final_emitter_class"),
            " ".join(str(item) for item in mechanisms),
            " ".join(layer_parts),
            " ".join(material_parts),
        ]
        return " | ".join(str(value) for value in fields if value)
