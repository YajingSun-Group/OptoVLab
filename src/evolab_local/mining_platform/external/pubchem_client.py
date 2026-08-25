from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import quote

import httpx

from evolab_local.mining_platform.core.config import PubChemConfig


@dataclass(frozen=True)
class PubChemCompound:
    cid: str
    query_text: str
    iupac_name: str | None = None
    canonical_smiles: str | None = None
    isomeric_smiles: str | None = None
    inchi: str | None = None
    inchi_key: str | None = None
    formula: str | None = None
    molecular_weight: float | None = None
    synonyms: list[str] = field(default_factory=list)
    raw_property: dict[str, Any] = field(default_factory=dict)


class PublicCompoundResolverClient(Protocol):
    def resolve_name(
        self, name: str, *, max_results: int | None = None
    ) -> list[PubChemCompound]: ...


class PubChemClient:
    def __init__(self, config: PubChemConfig) -> None:
        self.config = config
        self.base_url = config.base_url.rstrip("/")

    def resolve_name(self, name: str, *, max_results: int | None = None) -> list[PubChemCompound]:
        query = name.strip()
        if not query:
            return []
        limit = max_results or self.config.max_results_per_query
        encoded = quote(query, safe="")
        properties = ",".join(
            [
                "IUPACName",
                "CanonicalSMILES",
                "IsomericSMILES",
                "InChI",
                "InChIKey",
                "MolecularFormula",
                "MolecularWeight",
            ]
        )
        url = f"{self.base_url}/rest/pug/compound/name/{encoded}/property/{properties}/JSON"
        with httpx.Client(timeout=self.config.timeout_seconds, trust_env=False) as client:
            response = client.get(url)
            if response.status_code == 404:
                return []
            response.raise_for_status()
            compounds = parse_pubchem_property_response(response.json(), query_text=query)[:limit]
            return [
                compound
                for compound in (self._with_synonyms(client, compound) for compound in compounds)
                if compound
            ]

    def _with_synonyms(
        self,
        client: httpx.Client,
        compound: PubChemCompound,
    ) -> PubChemCompound:
        url = f"{self.base_url}/rest/pug/compound/cid/{quote(compound.cid, safe='')}/synonyms/JSON"
        try:
            response = client.get(url)
            if response.status_code == 404:
                return compound
            response.raise_for_status()
            synonyms = parse_pubchem_synonyms_response(response.json())
        except httpx.HTTPError:
            return compound
        return PubChemCompound(
            **{
                **compound.__dict__,
                "synonyms": synonyms[: self.config.max_synonyms],
            }
        )


def parse_pubchem_property_response(
    payload: dict[str, Any],
    *,
    query_text: str,
) -> list[PubChemCompound]:
    properties = payload.get("PropertyTable", {}).get("Properties")
    if not isinstance(properties, list):
        return []
    compounds: list[PubChemCompound] = []
    for item in properties:
        if not isinstance(item, dict) or item.get("CID") is None:
            continue
        compounds.append(
            PubChemCompound(
                cid=str(item["CID"]),
                query_text=query_text,
                iupac_name=_string(item.get("IUPACName")),
                canonical_smiles=_first_string(
                    item.get("CanonicalSMILES"),
                    item.get("ConnectivitySMILES"),
                    item.get("SMILES"),
                ),
                isomeric_smiles=_first_string(
                    item.get("IsomericSMILES"),
                    item.get("SMILES"),
                    item.get("ConnectivitySMILES"),
                ),
                inchi=_string(item.get("InChI")),
                inchi_key=_string(item.get("InChIKey")),
                formula=_string(item.get("MolecularFormula")),
                molecular_weight=_float(item.get("MolecularWeight")),
                raw_property=dict(item),
            )
        )
    return compounds


def parse_pubchem_synonyms_response(payload: dict[str, Any]) -> list[str]:
    information = payload.get("InformationList", {}).get("Information")
    if not isinstance(information, list):
        return []
    synonyms: list[str] = []
    seen: set[str] = set()
    for item in information:
        if not isinstance(item, dict) or not isinstance(item.get("Synonym"), list):
            continue
        for synonym in item["Synonym"]:
            if not isinstance(synonym, str) or not synonym:
                continue
            normalized = synonym.casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            synonyms.append(synonym)
    return synonyms


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _first_string(*values: Any) -> str | None:
    return next((value for value in (_string(item) for item in values) if value), None)


def _float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None
