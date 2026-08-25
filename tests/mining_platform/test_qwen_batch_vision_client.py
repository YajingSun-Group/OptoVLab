from __future__ import annotations

import json
import base64
import io
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image

from evolab_local.mining_platform.core.config import LLMProviderConfig, VisionBatchConfig
from evolab_local.mining_platform.external.openai_compatible_client import StaticJSONVisionClient
from evolab_local.mining_platform.external.qwen_batch_vision_client import (
    QwenBatchVisionClient,
)


class FakeBatchAPI:
    def __init__(self) -> None:
        self.upload_count = 0
        self.create_count = 0
        self.inputs: dict[str, list[dict[str, Any]]] = {}
        self.outputs: dict[str, str] = {}

    def upload_file(self, path: Path) -> str:
        self.upload_count += 1
        file_id = f"file-{self.upload_count}"
        self.inputs[file_id] = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return file_id

    def create_batch(
        self,
        *,
        input_file_id: str,
        completion_window: str,
        name: str,
        description: str,
        endpoint: str = "/v1/chat/completions",
    ) -> dict[str, Any]:
        del completion_window, name, description, endpoint
        self.create_count += 1
        output_file_id = f"output-{self.create_count}"
        records = []
        for request in self.inputs[input_file_id]:
            text = request["body"]["messages"][0]["content"]
            records.append(
                {
                    "custom_id": request["custom_id"],
                    "response": {
                        "status_code": 200,
                        "body": {
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "content": json.dumps({"echo": text}),
                                    }
                                }
                            ],
                            "usage": {
                                "prompt_tokens": 10,
                                "completion_tokens": 2,
                                "total_tokens": 12,
                            },
                        },
                    },
                    "error": None,
                }
            )
        self.outputs[output_file_id] = "\n".join(json.dumps(item) for item in records) + "\n"
        return {
            "id": f"batch-{self.create_count}",
            "status": "completed",
            "output_file_id": output_file_id,
            "request_counts": {"total": len(records), "completed": len(records), "failed": 0},
        }

    def retrieve_batch(self, batch_id: str) -> dict[str, Any]:
        raise AssertionError(f"Completed fake batch {batch_id} should not be polled.")

    def download_file(self, file_id: str, destination: Path) -> None:
        destination.write_text(self.outputs[file_id], encoding="utf-8")

    def close(self) -> None:
        return None


def _provider_config() -> LLMProviderConfig:
    return LLMProviderConfig(
        api_key="test-key",
        base_url="https://example.test/v1",
        default_model="qwen3.6-plus",
        vision_model="qwen3.6-flash",
        vision_enable_thinking=False,
        response_format_json=True,
    )


def _batch_config(**updates: Any) -> VisionBatchConfig:
    return VisionBatchConfig(
        flush_seconds=0.02,
        poll_interval_seconds=0.01,
        max_active_jobs=2,
        **updates,
    )


def test_concurrent_calls_share_batch_and_reuse_persistent_cache(tmp_path: Path) -> None:
    api = FakeBatchAPI()
    messages = [
        [{"role": "user", "content": "first"}],
        [{"role": "user", "content": "second"}],
    ]
    client = QwenBatchVisionClient(
        _provider_config(),
        _batch_config(),
        runtime_dir=tmp_path,
        batch_api=api,
        progress=lambda _message: None,
        recover_incomplete=False,
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(client.generate_json, messages[0])
        time.sleep(0.01)
        second_future = executor.submit(client.generate_json, messages[1])
        responses = [first_future.result(), second_future.result()]
    stats = client.statistics()
    client.close()

    assert [response.parsed_json for response in responses] == [
        {"echo": "first"},
        {"echo": "second"},
    ]
    assert api.upload_count == 1
    assert api.create_count == 1
    submitted = next(iter(api.inputs.values()))
    assert len(submitted) == 2
    assert all(item["body"]["enable_thinking"] is False for item in submitted)
    assert all("extra_body" not in item["body"] for item in submitted)
    assert stats["requests_by_status"] == {"completed": 2}
    assert stats["usage"]["total_tokens"] == 24

    cached_api = FakeBatchAPI()
    cached_client = QwenBatchVisionClient(
        _provider_config(),
        _batch_config(),
        runtime_dir=tmp_path,
        batch_api=cached_api,
        progress=lambda _message: None,
        recover_incomplete=False,
    )
    cached_response = cached_client.generate_json(messages[0])
    cached_client.close()
    assert cached_response.parsed_json == {"echo": "first"}
    assert cached_api.upload_count == 0


def test_oversized_request_uses_realtime_fallback_and_caches_result(tmp_path: Path) -> None:
    api = FakeBatchAPI()
    realtime = StaticJSONVisionClient({"decision": "fallback"})
    client = QwenBatchVisionClient(
        _provider_config(),
        _batch_config(max_line_bytes=100),
        runtime_dir=tmp_path,
        batch_api=api,
        realtime_client=realtime,
        progress=lambda _message: None,
        recover_incomplete=False,
    )
    messages = [{"role": "user", "content": "x" * 500}]
    first = client.generate_json(messages)
    second = client.generate_json(messages)
    stats = client.statistics()
    client.close()

    assert first.parsed_json == {"decision": "fallback"}
    assert second.parsed_json == {"decision": "fallback"}
    assert api.upload_count == 0
    assert stats["realtime_fallback_count"] == 1


def test_tiny_images_are_expanded_without_reencoding_normal_images(tmp_path: Path) -> None:
    tiny = io.BytesIO()
    Image.new("RGB", (26, 3), "black").save(tiny, format="PNG")
    tiny_url = "data:image/png;base64," + base64.b64encode(tiny.getvalue()).decode("ascii")
    normal = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(normal, format="PNG")
    normal_url = "data:image/png;base64," + base64.b64encode(normal.getvalue()).decode("ascii")
    api = FakeBatchAPI()
    client = QwenBatchVisionClient(
        _provider_config(),
        _batch_config(),
        runtime_dir=tmp_path,
        batch_api=api,
        progress=lambda _message: None,
        recover_incomplete=False,
    )
    client.generate_json(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": tiny_url}},
                    {"type": "image_url", "image_url": {"url": normal_url}},
                ],
            }
        ]
    )
    client.close()

    content = next(iter(api.inputs.values()))[0]["body"]["messages"][0]["content"]
    expanded_url = content[0]["image_url"]["url"]
    with Image.open(io.BytesIO(base64.b64decode(expanded_url.split(",", 1)[1]))) as expanded:
        assert expanded.width >= 12
        assert expanded.height >= 12
    assert content[1]["image_url"]["url"] == normal_url
