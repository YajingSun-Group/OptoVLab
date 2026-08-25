from __future__ import annotations

import base64
import json
from pathlib import Path

from evolab_local.mining_platform.external.mineru_client import parse_mineru_result
from evolab_local.mining_platform.mining.mineru_parse_service import write_mineru_outputs


def test_mineru_embedded_images_are_materialized_without_duplicate_base64(
    tmp_path: Path,
) -> None:
    image_bytes = b"fake-jpeg-payload"
    image_data_url = (
        "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("ascii")
    )
    raw_result = {
        "version": "3.1.0",
        "results": {
            "paper": {
                "md_content": "# Paper",
                "content_list": json.dumps(
                    [{"type": "image", "img_path": "images/figure.jpg"}]
                ),
                "images": {"figure.jpg": image_data_url},
            }
        },
    }
    parsed = parse_mineru_result(raw_result, task_id="task-1")

    paths = write_mineru_outputs(
        tmp_path,
        "run-1",
        parsed,
        images_requested=True,
    )

    assert (tmp_path / "run-1" / "images" / "figure.jpg").read_bytes() == image_bytes
    result_payload = json.loads(Path(paths["result_path"]).read_text(encoding="utf-8"))
    assert "images" not in result_payload["results"]["paper"]
    assert result_payload["results"]["paper"]["image_names"] == ["figure.jpg"]
    assert result_payload["_evolab_local"]["images_requested"] is True
    assert result_payload["_evolab_local"]["stored_image_count"] == 1
