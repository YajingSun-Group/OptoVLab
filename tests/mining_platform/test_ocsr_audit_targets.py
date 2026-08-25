from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "audit_ocsr_candidates.py"
_SPEC = importlib.util.spec_from_file_location("audit_ocsr_candidates", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
_load_target_pairs = _MODULE._load_target_pairs


def test_load_target_pairs_supports_single_and_grouped_materials(tmp_path) -> None:
    targets = tmp_path / "targets.jsonl"
    rows = [
        {"paper_id": "paper-1", "paper_material_id": "M001"},
        {"paper_id": "paper-2", "paper_material_ids": ["M002", "M003"]},
        {"paper_id": "paper-1", "paper_material_id": "M001"},
    ]
    targets.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    assert _load_target_pairs(targets) == {
        ("paper-1", "M001"),
        ("paper-2", "M002"),
        ("paper-2", "M003"),
    }


def test_load_target_pairs_rejects_rows_without_material_ids(tmp_path) -> None:
    targets = tmp_path / "targets.jsonl"
    targets.write_text('{"paper_id": "paper-1"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Missing paper material IDs"):
        _load_target_pairs(targets)
