from __future__ import annotations

import re
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Protocol

import httpx

from evolab_local.mining_platform.core.config import AnySearchConfig


CAS_NUMBER_RE = re.compile(r"\b\d{2,7}-\d{2}-\d\b")


@dataclass(frozen=True)
class AnySearchResult:
    title: str
    url: str
    description: str = ""
    content: str = ""
    score: float | None = None
    raw_result: dict[str, Any] = field(default_factory=dict)

    def cas_numbers(self) -> list[str]:
        found = CAS_NUMBER_RE.findall(" ".join([self.title, self.description, self.content]))
        return list(dict.fromkeys(found))


class MaterialWebSearchClient(Protocol):
    def search(self, query: str, *, max_results: int | None = None) -> list[AnySearchResult]: ...


class AnySearchClient:
    def __init__(self, config: AnySearchConfig) -> None:
        self.config = config
        self._state_lock = Lock()
        self._disabled_status_code: int | None = None

    def search(self, query: str, *, max_results: int | None = None) -> list[AnySearchResult]:
        if not self.config.api_key.strip() or not query.strip():
            return []
        with self._state_lock:
            if self._disabled_status_code is not None:
                return []
        limit = max_results or self.config.max_results_per_query
        with httpx.Client(timeout=self.config.timeout_seconds, trust_env=False) as client:
            response = client.post(
                self.config.base_url,
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                json={"query": query, "max_results": limit},
            )
            if response.status_code in {401, 402, 403}:
                with self._state_lock:
                    self._disabled_status_code = response.status_code
                return []
            response.raise_for_status()
        return parse_anysearch_response(response.json())[:limit]


def parse_anysearch_response(payload: dict[str, Any]) -> list[AnySearchResult]:
    results = payload.get("data", {}).get("results")
    if not isinstance(results, list):
        return []
    parsed: list[AnySearchResult] = []
    for item in results:
        if not isinstance(item, dict) or not isinstance(item.get("url"), str):
            continue
        parsed.append(
            AnySearchResult(
                title=_text(item.get("title")),
                url=item["url"],
                description=_text(item.get("description")),
                content=_text(item.get("content")),
                score=_float(item.get("score")),
                raw_result=dict(item),
            )
        )
    return parsed


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None
