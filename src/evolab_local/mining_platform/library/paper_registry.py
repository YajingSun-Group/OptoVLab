from __future__ import annotations

import json
from pathlib import Path

from evolab_local.mining_platform.schemas.paper import Paper


def write_paper_registry(path: Path, papers: list[Paper]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(paper.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        for paper in papers
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
