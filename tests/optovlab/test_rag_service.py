from __future__ import annotations

import gzip
import json
from pathlib import Path

from evolab_local.optovlab.config import RetrievalConfig
from evolab_local.optovlab.data_catalog import OLEDDeviceCatalog
from evolab_local.optovlab.rag_service import OLEDDeviceRAGService


def _catalog(tmp_path: Path) -> OLEDDeviceCatalog:
    records = [
        {
            "id": "device-4czipn",
            "doi": "10.1000/4czipn",
            "architecture": "ITO/PEDOT:PSS/mCP:4CzIPN/PPF/TmPyPB/LiF/Al",
            "final_emitter": "4CzIPN",
            "emission_color": "green",
            "fabrication_method": "solution_process",
            "eqe_max": 26.5,
        },
        {
            "id": "device-red",
            "doi": "10.1000/red",
            "architecture": "ITO/HTL/red emitter/ETL/Al",
            "final_emitter": "Ir(piq)3",
            "emission_color": "red",
            "fabrication_method": "vacuum_deposition",
            "eqe_max": 18.0,
        },
        {
            "id": "device-blue",
            "doi": "10.1000/blue",
            "architecture": "ITO/HTL/blue TADF emitter/ETL/Al",
            "final_emitter": "DMAC-DPS",
            "emission_color": "blue",
            "fabrication_method": "vacuum_deposition",
            "eqe_max": 22.5,
        },
    ]
    path = tmp_path / "oled.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(records, handle)
    return OLEDDeviceCatalog(path)


def test_rag_prioritizes_exact_material_and_architecture_match(tmp_path: Path) -> None:
    service = OLEDDeviceRAGService(
        _catalog(tmp_path),
        RetrievalConfig(maximum_features=5000, default_top_k=2, maximum_top_k=5),
    )

    result = service.search("4CzIPN PPF green solution processed device", top_k=2)

    assert result.total_devices == 3
    assert result.hits[0].device_id == "device-4czipn"
    assert result.hits[0].doi == "10.1000/4czipn"


def test_rag_applies_structured_filters(tmp_path: Path) -> None:
    service = OLEDDeviceRAGService(
        _catalog(tmp_path),
        RetrievalConfig(maximum_features=5000, default_top_k=3, maximum_top_k=5),
    )

    result = service.search("emitter device", filters={"emission_color": "blue", "minimum_eqe": 20})

    assert [hit.device_id for hit in result.hits] == ["device-blue"]


def test_rag_reuses_valid_disk_cache(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    config = RetrievalConfig(maximum_features=2000, default_top_k=2, maximum_top_k=5)
    cache_dir = tmp_path / "cache"
    first = OLEDDeviceRAGService(catalog, config, cache_dir=cache_dir)
    first.search("4CzIPN", top_k=1)

    second = OLEDDeviceRAGService(catalog, config, cache_dir=cache_dir)
    result = second.search("4CzIPN", top_k=1)

    assert result.hits[0].device_id == "device-4czipn"
    assert (cache_dir / "metadata.json").is_file()
    assert (cache_dir / "vectorizer.joblib").is_file()
    assert (cache_dir / "matrix.npz").is_file()
