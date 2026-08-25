from __future__ import annotations

from typing import Any

import httpx
import pytest

from evolab_local.mining_platform.core.config import LLMProviderConfig
from evolab_local.mining_platform.external import openai_compatible_client as client_module
from evolab_local.mining_platform.external.openai_compatible_client import (
    OpenAICompatibleLLMClient,
    OpenAICompatibleVisionClient,
)


def test_provider_specific_thinking_modes_are_sent_to_their_request_types(monkeypatch) -> None:
    payloads: list[dict[str, Any]] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "choices": [{"message": {"content": '{"accepted": true}'}}],
                "usage": {},
            }

    class FakeHTTPClient:
        def __init__(self, **_: Any) -> None:
            pass

        def __enter__(self) -> FakeHTTPClient:
            return self

        def __exit__(self, *_: Any) -> None:
            return None

        def post(self, _url: str, *, headers: dict[str, str], json: dict[str, Any]) -> FakeResponse:
            assert "Authorization" in headers
            payloads.append(json)
            return FakeResponse()

    monkeypatch.setattr(client_module.httpx, "Client", FakeHTTPClient)
    config = LLMProviderConfig(
        api_key="secret",
        base_url="https://example.test/compatible-mode/v1",
        default_model="qwen3.6-flash",
        vision_enable_thinking=False,
        thinking_mode="disabled",
    )

    OpenAICompatibleVisionClient(config).generate_json([{"role": "user", "content": []}])
    OpenAICompatibleLLMClient(config).generate_json([{"role": "user", "content": "extract"}])

    assert payloads[0]["enable_thinking"] is False
    assert "thinking" not in payloads[0]
    assert payloads[1]["thinking"] == {"type": "disabled"}
    assert "enable_thinking" not in payloads[1]


def test_llm_client_retries_transient_http_status(monkeypatch) -> None:
    attempts = 0

    class FakeHTTPClient:
        def __init__(self, **_: Any) -> None:
            pass

        def __enter__(self) -> FakeHTTPClient:
            return self

        def __exit__(self, *_: Any) -> None:
            return None

        def post(self, url: str, **_: Any) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            request = httpx.Request("POST", url)
            if attempts < 3:
                return httpx.Response(503, request=request)
            return httpx.Response(
                200,
                request=request,
                json={
                    "choices": [{"message": {"content": '{"accepted": true}'}}],
                    "usage": {},
                },
            )

    monkeypatch.setattr(client_module.httpx, "Client", FakeHTTPClient)
    config = LLMProviderConfig(
        api_key="secret",
        base_url="https://example.test/v1",
        default_model="test-model",
        request_max_attempts=3,
        retry_backoff_seconds=0,
    )

    response = OpenAICompatibleLLMClient(config).generate_json(
        [{"role": "user", "content": "judge"}]
    )

    assert attempts == 3
    assert response.parsed_json == {"accepted": True}


def test_llm_client_does_not_retry_non_transient_http_status(monkeypatch) -> None:
    attempts = 0

    class FakeHTTPClient:
        def __init__(self, **_: Any) -> None:
            pass

        def __enter__(self) -> FakeHTTPClient:
            return self

        def __exit__(self, *_: Any) -> None:
            return None

        def post(self, url: str, **_: Any) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            return httpx.Response(401, request=httpx.Request("POST", url))

    monkeypatch.setattr(client_module.httpx, "Client", FakeHTTPClient)
    config = LLMProviderConfig(
        api_key="secret",
        base_url="https://example.test/v1",
        default_model="test-model",
        request_max_attempts=3,
        retry_backoff_seconds=0,
    )

    with pytest.raises(httpx.HTTPStatusError):
        OpenAICompatibleLLMClient(config).generate_json(
            [{"role": "user", "content": "judge"}]
        )

    assert attempts == 1
