from __future__ import annotations

import httpx

from evolab_local.mining_platform.core.config import AnySearchConfig
from evolab_local.mining_platform.external.anysearch_client import AnySearchClient


def test_auth_or_balance_failure_disables_followup_requests(monkeypatch) -> None:
    request_count = 0

    def fake_post(self, url, **kwargs):
        del self, kwargs
        nonlocal request_count
        request_count += 1
        return httpx.Response(402, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    client = AnySearchClient(
        AnySearchConfig(
            api_key="test-key",
            base_url="https://example.test/search",
        )
    )

    assert client.search("first query") == []
    assert client.search("second query") == []
    assert request_count == 1
