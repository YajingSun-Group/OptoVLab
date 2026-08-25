from __future__ import annotations

from pathlib import Path


def project_root_for_config(config_path: Path) -> Path:
    resolved = config_path.resolve()
    if (
        resolved.name == "mining_platform.yaml"
        and resolved.parent.name == "mining_platform"
        and resolved.parent.parent.name == "config"
    ):
        return resolved.parent.parent.parent
    return resolved.parent


def resolve_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def display_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()
