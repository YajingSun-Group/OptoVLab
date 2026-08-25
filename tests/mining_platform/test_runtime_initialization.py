from __future__ import annotations

from evolab_local.mining_platform.core.config import load_config
from evolab_local.mining_platform.library.paper_service import PaperService
from evolab_local.mining_platform.storage.database import Database


def test_database_initialization_is_cached_per_process(tmp_path, monkeypatch) -> None:
    database = Database(tmp_path / "platform.sqlite")
    database.init_db()

    connect_calls = 0
    original_connect = database.connect

    def counted_connect():
        nonlocal connect_calls
        connect_calls += 1
        return original_connect()

    monkeypatch.setattr(database, "connect", counted_connect)
    database.init_db()

    assert connect_calls == 0


def test_paper_runtime_does_not_rewrite_existing_registry(
    mining_config_path,
    monkeypatch,
) -> None:
    from evolab_local.mining_platform.library import paper_service as paper_service_module

    service = PaperService(load_config(mining_config_path))
    service.init_runtime()
    writes = 0
    original_write = paper_service_module.write_paper_registry

    def counted_write(path, papers):
        nonlocal writes
        writes += 1
        return original_write(path, papers)

    monkeypatch.setattr(paper_service_module, "write_paper_registry", counted_write)
    service.init_runtime()

    assert writes == 0
