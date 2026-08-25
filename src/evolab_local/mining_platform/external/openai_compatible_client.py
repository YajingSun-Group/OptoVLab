from __future__ import annotations

import base64
import json
import mimetypes
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx

from evolab_local.mining_platform.core.config import LLMProviderConfig


@dataclass(frozen=True)
class LLMResponse:
    content: str
    parsed_json: dict[str, Any] | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    parse_error: str | None = None


class LLMClient(Protocol):
    def generate_json(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse: ...


class VisionClient(Protocol):
    def generate_json(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse: ...


class OpenAICompatibleLLMClient:
    def __init__(self, config: LLMProviderConfig) -> None:
        self.config = config

    def generate_json(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        if not self.config.api_key:
            raise ValueError("LLM provider API key is not configured.")
        payload: dict[str, Any] = {
            "model": model or self.config.default_model,
            "messages": messages,
            "temperature": self.config.temperature if temperature is None else temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if self.config.thinking_mode is not None:
            payload["thinking"] = {"type": self.config.thinking_mode}
        if self.config.response_format_json:
            payload["response_format"] = {"type": "json_object"}
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        base_url = self.config.base_url.rstrip("/")
        data = _post_chat_completion(
            self.config,
            f"{base_url}/chat/completions",
            headers=headers,
            payload=payload,
        )
        content = _extract_message_content(data)
        parsed_json, parse_error = _try_extract_json_object(content)
        return LLMResponse(
            content=content,
            parsed_json=parsed_json,
            raw_response=data,
            usage=data.get("usage") if isinstance(data.get("usage"), dict) else {},
            parse_error=parse_error,
        )


class OpenAICompatibleVisionClient:
    def __init__(self, config: LLMProviderConfig) -> None:
        self.config = config

    def generate_json(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        if not self.config.api_key:
            raise ValueError("Vision provider API key is not configured.")
        payload: dict[str, Any] = {
            "model": model or self.config.default_model,
            "messages": messages,
            "temperature": self.config.temperature if temperature is None else temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if self.config.vision_enable_thinking is not None:
            payload["enable_thinking"] = self.config.vision_enable_thinking
        if self.config.response_format_json:
            payload["response_format"] = {"type": "json_object"}
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        base_url = self.config.base_url.rstrip("/")
        data = _post_chat_completion(
            self.config,
            f"{base_url}/chat/completions",
            headers=headers,
            payload=payload,
        )
        content = _extract_message_content(data)
        parsed_json, parse_error = _try_extract_json_object(content)
        return LLMResponse(
            content=content,
            parsed_json=parsed_json,
            raw_response=data,
            usage=data.get("usage") if isinstance(data.get("usage"), dict) else {},
            parse_error=parse_error,
        )


class StaticJSONLLMClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def generate_json(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        del max_tokens
        content = json.dumps(self.payload, ensure_ascii=False, indent=2)
        return LLMResponse(
            content=content,
            parsed_json=self.payload,
            raw_response={"mock": True, "model": model},
            usage={},
        )


class StaticJSONVisionClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def generate_json(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        del max_tokens
        content = json.dumps(self.payload, ensure_ascii=False, indent=2)
        return LLMResponse(
            content=content,
            parsed_json=self.payload,
            raw_response={"mock": True, "model": model},
            usage={},
        )


def image_path_to_data_url(path: Path) -> str:
    media_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def extract_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("LLM response JSON must be an object.")
    return payload


def _try_extract_json_object(content: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return extract_json_object(content), None
    except (json.JSONDecodeError, ValueError) as exc:
        return None, str(exc)


def _extract_message_content(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("LLM response does not contain choices.")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("LLM response choice is invalid.")
    message = first.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ValueError("LLM response choice does not contain message.content.")
    return message["content"]


_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def _post_chat_completion(
    config: LLMProviderConfig,
    url: str,
    *,
    headers: dict[str, str],
    payload: dict[str, Any],
) -> dict[str, Any]:
    max_attempts = max(1, config.request_max_attempts)
    with httpx.Client(timeout=config.timeout_seconds, trust_env=False) as client:
        for attempt in range(max_attempts):
            try:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise ValueError("LLM provider response must be a JSON object.")
                return data
            except (httpx.RequestError, httpx.HTTPStatusError) as exc:
                if attempt + 1 >= max_attempts or not _retryable_request_error(exc):
                    raise
                delay = max(0.0, config.retry_backoff_seconds) * (2**attempt)
                if delay:
                    time.sleep(delay)
    raise RuntimeError("LLM request retry loop ended without a response.")


def _retryable_request_error(exc: httpx.RequestError | httpx.HTTPStatusError) -> bool:
    if isinstance(exc, httpx.RequestError):
        return True
    return exc.response.status_code in _RETRYABLE_STATUS_CODES
