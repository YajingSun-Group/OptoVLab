from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import quote

import httpx

from evolab_local.mining_platform.core.config import OpsinConfig


@dataclass(frozen=True)
class OpsinCompound:
    query_text: str
    smiles: str
    inchi: str | None = None
    inchi_key: str | None = None
    raw_result: dict[str, Any] = field(default_factory=dict)


class SystematicNameResolverClient(Protocol):
    def resolve_name(self, name: str) -> OpsinCompound | None: ...


class OpsinClient:
    def __init__(self, config: OpsinConfig) -> None:
        self.config = config
        self.base_url = config.base_url.rstrip("/")

    def resolve_name(self, name: str) -> OpsinCompound | None:
        query = name.strip()
        if not query:
            return None
        url = f"{self.base_url}/{quote(query, safe='')}.json"
        with httpx.Client(
            timeout=self.config.timeout_seconds,
            trust_env=False,
            follow_redirects=True,
        ) as client:
            response = client.get(url)
        if response.status_code in {400, 404}:
            return None
        response.raise_for_status()
        return parse_opsin_response(response.json(), query_text=query)


def parse_opsin_response(payload: dict[str, Any], *, query_text: str) -> OpsinCompound | None:
    smiles = payload.get("smiles")
    if payload.get("status") != "SUCCESS" or not isinstance(smiles, str) or not smiles.strip():
        return None
    return OpsinCompound(
        query_text=query_text,
        smiles=smiles,
        inchi=_text(payload.get("stdinchi")) or _text(payload.get("inchi")),
        inchi_key=_text(payload.get("stdinchikey")),
        raw_result=dict(payload),
    )


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None
