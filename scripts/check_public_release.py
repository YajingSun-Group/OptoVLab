from __future__ import annotations

import gzip
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IGNORED_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    ".next",
    "runtime",
    "out",
    "dist",
}
TEXT_SUFFIXES = {
    "",
    ".cff",
    ".css",
    ".env",
    ".html",
    ".js",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
FORBIDDEN = {
    "private home path": re.compile(r"/home/[A-Za-z0-9._-]+/"),
    "server IP": re.compile(r"\b211\.81\.48\.70\b"),
    "private service hostname": re.compile(r"\bslurm0[12]\b", re.IGNORECASE),
    "DeepSeek-style secret": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "AnySearch-style secret": re.compile(r"\bas_sk_[A-Za-z0-9_-]{16,}\b"),
}
MAX_PUBLIC_FILE_BYTES = 10 * 1024 * 1024


def release_files() -> list[Path]:
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file() and not any(part in IGNORED_PARTS for part in path.parts)
    ]


def check_required_files(errors: list[str]) -> None:
    for relative in (
        "README.md",
        "LICENSE",
        "CITATION.cff",
        ".env.example",
        ".github/workflows/ci.yml",
        ".github/workflows/pages.yml",
        "site/index.html",
    ):
        if not (ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")


def check_files(files: list[Path], errors: list[str]) -> None:
    for path in files:
        relative = path.relative_to(ROOT)
        size = path.stat().st_size
        if size > MAX_PUBLIC_FILE_BYTES:
            errors.append(f"file exceeds 10 MiB: {relative} ({size} bytes)")
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {".gitignore"}:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in FORBIDDEN.items():
            if match := pattern.search(content):
                errors.append(f"{label} in {relative}: {match.group(0)}")


def check_demo_catalog(errors: list[str]) -> dict[str, int]:
    public_root = ROOT / "apps/database-web/public"
    catalog_path = public_root / "data/catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if not str(catalog.get("schema_version", "")).endswith("-demo"):
        errors.append("database catalog is not marked as demo")

    counts: dict[str, int] = {}
    for dataset in catalog.get("datasets", []):
        path = public_root / dataset["file"]
        if not path.is_file():
            errors.append(f"missing catalog dataset: {path.relative_to(ROOT)}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != dataset.get("sha256"):
            errors.append(f"catalog SHA-256 mismatch: {path.relative_to(ROOT)}")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            records = json.load(handle)
        count = len(records) if isinstance(records, list) else -1
        if count != dataset.get("record_count"):
            errors.append(f"catalog count mismatch: {path.relative_to(ROOT)}")
        counts[str(dataset.get("key"))] = count
    return counts


def check_site_assets(errors: list[str]) -> None:
    page = (ROOT / "site/index.html").read_text(encoding="utf-8")
    references = re.findall(r"(?:src|href|data-image)=\"([^\"]+)\"", page)
    for reference in references:
        if reference.startswith(("http://", "https://", "#", "mailto:")):
            continue
        target = ROOT / "site" / reference.split("#", maxsplit=1)[0]
        if not target.is_file():
            errors.append(f"missing site asset: {reference}")


def check_markdown_links(files: list[Path], errors: list[str]) -> None:
    for path in files:
        if path.suffix.lower() != ".md":
            continue
        content = path.read_text(encoding="utf-8")
        for reference in re.findall(r"\[[^\]]+\]\(([^)]+)\)", content):
            reference = reference.strip().strip("<>")
            if reference.startswith(("http://", "https://", "#", "mailto:")):
                continue
            relative = reference.split("#", maxsplit=1)[0]
            if not relative:
                continue
            target = (path.parent / relative).resolve()
            if not target.exists():
                errors.append(
                    f"broken Markdown link in {path.relative_to(ROOT)}: {reference}"
                )


def main() -> None:
    errors: list[str] = []
    files = release_files()
    check_required_files(errors)
    check_files(files, errors)
    counts = check_demo_catalog(errors)
    check_site_assets(errors)
    check_markdown_links(files, errors)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print(
        json.dumps(
            {
                "status": "ok",
                "files_checked": len(files),
                "demo_records": counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
