from __future__ import annotations

from fastapi.testclient import TestClient

from evolab_local.mining_platform.api.app import create_app
from evolab_local.mining_platform.core.config import load_config


def test_health_endpoint(mining_config_path) -> None:
    client = TestClient(create_app(config=load_config(mining_config_path)))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
