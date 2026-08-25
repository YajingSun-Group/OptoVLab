from __future__ import annotations

import json
import math
import re
import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from evolab_local.optovlab.config import RetrievalConfig
from evolab_local.optovlab.data_catalog import OLEDDeviceCatalog
from evolab_local.optovlab.schemas import RAGHit, RAGSearchResult


class OLEDDeviceRAGService:
    """Hybrid lexical and structured retrieval over exported OLED devices."""

    INDEX_VERSION = 1

    def __init__(
        self,
        catalog: OLEDDeviceCatalog,
        config: RetrievalConfig,
        cache_dir: Path | None = None,
    ) -> None:
        self.catalog = catalog
        self.config = config
        self.cache_dir = cache_dir
        self._lock = threading.Lock()
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix: sparse.csr_matrix | None = None

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> RAGSearchResult:
        self._ensure_index()
        assert self._vectorizer is not None
        assert self._matrix is not None
        records = self.catalog.records()
        requested = min(top_k or self.config.default_top_k, self.config.maximum_top_k)
        query_vector = self._vectorizer.transform([query])
        scores = (self._matrix @ query_vector.T).toarray().ravel()
        normalized_query = _normalize(query)
        query_terms = {term for term in re.split(r"[^a-z0-9+.-]+", normalized_query) if len(term) >= 2}
        mask = np.ones(len(records), dtype=bool)
        applied_filters = filters or {}
        for index, record in enumerate(records):
            if not _matches_filters(record, applied_filters):
                mask[index] = False
                continue
            emitter = _normalize(record.get("final_emitter"))
            architecture = _normalize(record.get("architecture"))
            exact_boost = 0.0
            for term in query_terms:
                if term == emitter:
                    exact_boost += 0.25
                elif term in emitter:
                    exact_boost += 0.10
                if term in architecture:
                    exact_boost += 0.025
            scores[index] += min(exact_boost, 0.45)
        scores[~mask] = -1.0
        candidate_count = min(len(records), max(requested * 8, requested))
        candidate_indexes = np.argpartition(scores, -candidate_count)[-candidate_count:]
        ordered = candidate_indexes[np.argsort(scores[candidate_indexes])[::-1]]
        hits: list[RAGHit] = []
        for index in ordered:
            if scores[index] < 0:
                continue
            record = records[int(index)]
            hits.append(
                RAGHit(
                    rank=len(hits) + 1,
                    score=round(float(scores[index]), 6),
                    device_id=str(record.get("id") or f"device-{index}"),
                    doi=record.get("doi"),
                    title=record.get("title"),
                    journal=record.get("journal"),
                    device_label=record.get("device_label"),
                    architecture=record.get("architecture"),
                    final_emitter=record.get("final_emitter"),
                    eqe_max=_float(record.get("eqe_max")),
                    record=_compact_record(record),
                )
            )
            if len(hits) >= requested:
                break
        return RAGSearchResult(query=query, total_devices=len(records), hits=hits)

    def _ensure_index(self) -> None:
        if self._matrix is not None:
            return
        with self._lock:
            if self._matrix is not None:
                return
            records = self.catalog.records()
            if self._load_cache(len(records)):
                return
            texts = [self.catalog.searchable_text(record) for record in records]
            vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 5),
                lowercase=True,
                max_features=self.config.maximum_features,
                min_df=1,
                sublinear_tf=True,
                dtype=np.float32,
            )
            self._matrix = vectorizer.fit_transform(texts).tocsr()
            self._vectorizer = vectorizer
            self._write_cache()

    def _fingerprint(self) -> dict[str, Any]:
        stat = self.catalog.dataset_path.stat()
        return {
            "version": self.INDEX_VERSION,
            "dataset_path": str(self.catalog.dataset_path.resolve()),
            "dataset_size": stat.st_size,
            "dataset_mtime_ns": stat.st_mtime_ns,
            "maximum_features": self.config.maximum_features,
        }

    def _load_cache(self, expected_rows: int) -> bool:
        if self.cache_dir is None:
            return False
        metadata_path = self.cache_dir / "metadata.json"
        vectorizer_path = self.cache_dir / "vectorizer.joblib"
        matrix_path = self.cache_dir / "matrix.npz"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata != self._fingerprint():
                return False
            vectorizer = joblib.load(vectorizer_path)
            matrix = sparse.load_npz(matrix_path).tocsr()
            if not isinstance(vectorizer, TfidfVectorizer) or matrix.shape[0] != expected_rows:
                return False
        except (OSError, ValueError, TypeError, json.JSONDecodeError, EOFError):
            return False
        self._vectorizer = vectorizer
        self._matrix = matrix
        return True

    def _write_cache(self) -> None:
        if self.cache_dir is None or self._vectorizer is None or self._matrix is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        vectorizer_path = self.cache_dir / "vectorizer.joblib"
        matrix_path = self.cache_dir / "matrix.npz"
        metadata_path = self.cache_dir / "metadata.json"
        vectorizer_tmp = self.cache_dir / "vectorizer.joblib.tmp"
        matrix_tmp = self.cache_dir / "matrix.tmp.npz"
        metadata_tmp = self.cache_dir / "metadata.json.tmp"
        try:
            joblib.dump(self._vectorizer, vectorizer_tmp)
            sparse.save_npz(matrix_tmp, self._matrix)
            metadata_tmp.write_text(json.dumps(self._fingerprint(), indent=2), encoding="utf-8")
            vectorizer_tmp.replace(vectorizer_path)
            matrix_tmp.replace(matrix_path)
            metadata_tmp.replace(metadata_path)
        except OSError:
            for path in (vectorizer_tmp, matrix_tmp, metadata_tmp):
                path.unlink(missing_ok=True)


def _normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _matches_filters(record: dict[str, Any], filters: dict[str, Any]) -> bool:
    for field in ("emission_color", "fabrication_method", "device_type", "final_emitter"):
        expected = filters.get(field)
        if expected and _normalize(expected) not in _normalize(record.get(field)):
            return False
    mechanism = filters.get("emission_mechanism")
    if mechanism:
        actual = record.get("emission_mechanism") or []
        if _normalize(mechanism) not in _normalize(" ".join(str(value) for value in actual)):
            return False
    eqe = _float(record.get("eqe_max"))
    if filters.get("minimum_eqe") is not None and (eqe is None or eqe < float(filters["minimum_eqe"])):
        return False
    if filters.get("maximum_eqe") is not None and (eqe is None or eqe > float(filters["maximum_eqe"])):
        return False
    return True


def _compact_record(record: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "id",
        "paper_id",
        "doi",
        "title",
        "journal",
        "year",
        "quality_tier",
        "device_label",
        "architecture",
        "emission_color",
        "emission_mechanism",
        "fabrication_method",
        "layer_count",
        "final_emitter",
        "final_emitter_class",
        "eqe_max",
        "ce_max",
        "pe_max",
        "turn_on_voltage",
        "luminance_max",
        "layers",
    )
    return {key: record.get(key) for key in keep if record.get(key) is not None}
