from __future__ import annotations

from evolab_local.mining_platform.core.config import load_config


def test_load_config_resolves_paths(mining_config_path) -> None:
    config = load_config(mining_config_path)

    assert config.paths.runtime_dir.is_absolute()
    assert config.paths.sqlite_path.name == "platform.sqlite"
    assert config.pdf_downloader.manifest_path.name == "download_manifest.json"
    assert config.paths.runtime_dir.exists()
    assert config.paths.logs_dir.exists()
