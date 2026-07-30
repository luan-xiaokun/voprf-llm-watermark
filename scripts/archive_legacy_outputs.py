#!/usr/bin/env python3
"""Copy legacy experiment outputs into a verified, append-only archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


MANIFEST_NAME = "archive-manifest.json"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, sort_keys=True, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("format_version") != 1 or not isinstance(
        value.get("files"), list
    ):
        raise ValueError(f"unsupported archive manifest: {path}")
    return value


def _copy_verified(
    source: Path,
    destination: Path,
    *,
    digest: str,
) -> str:
    if destination.exists():
        if (
            destination.stat().st_size == source.stat().st_size
            and sha256_file(destination) == digest
        ):
            return "verified"
        raise ValueError(
            f"archive file differs from source; refusing to overwrite: "
            f"{destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        temporary.unlink()
    shutil.copy2(source, temporary)
    with temporary.open("rb") as copied:
        os.fsync(copied.fileno())
    if (
        temporary.stat().st_size != source.stat().st_size
        or sha256_file(temporary) != digest
    ):
        raise OSError(f"verification failed while copying {source}")
    os.replace(temporary, destination)
    return "copied"


def archive(
    *,
    repository: Path,
    destination: Path,
    sources: list[Path],
) -> dict[str, Any]:
    repository = repository.expanduser().resolve()
    destination = destination.expanduser().resolve()
    resolved_sources = [
        (source if source.is_absolute() else repository / source).resolve()
        for source in sources
    ]
    for source in resolved_sources:
        try:
            source.relative_to(repository)
        except ValueError as error:
            raise ValueError(
                f"source must be inside repository: {source}"
            ) from error
        if not source.exists():
            raise FileNotFoundError(source)
        if source == destination or destination.is_relative_to(source):
            raise ValueError(
                f"archive destination cannot be inside source: {source}"
            )

    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / MANIFEST_NAME
    previous = _load_manifest(manifest_path)
    previous_entries = {
        entry["source_path"]: entry
        for entry in (previous or {}).get("files", [])
    }
    current_entries: dict[str, dict[str, Any]] = {}
    copied = 0
    verified = 0
    for source_root in resolved_sources:
        files = (
            [source_root]
            if source_root.is_file()
            else sorted(
                item for item in source_root.rglob("*") if item.is_file()
            )
        )
        for source in files:
            relative = source.relative_to(repository)
            source_key = relative.as_posix()
            archive_relative = Path("repository") / relative
            target = destination / archive_relative
            digest = sha256_file(source)
            disposition = _copy_verified(
                source, target, digest=digest
            )
            if disposition == "copied":
                copied += 1
            else:
                verified += 1
            stat = source.stat()
            current_entries[source_key] = {
                "source_path": source_key,
                "archive_path": archive_relative.as_posix(),
                "size": stat.st_size,
                "sha256": digest,
                "mtime_ns": stat.st_mtime_ns,
                "mode": stat.st_mode,
            }

    all_entries = {**previous_entries, **current_entries}
    manifest = {
        "format_version": 1,
        "repository": str(repository),
        "created_at": (
            previous["created_at"] if previous else _now()
        ),
        "verified_at": _now(),
        "sources": [
            source.relative_to(repository).as_posix()
            for source in resolved_sources
        ],
        "files": [
            all_entries[key] for key in sorted(all_entries)
        ],
        "summary": {
            "source_file_count": len(current_entries),
            "archived_file_count": len(all_entries),
            "copied_this_run": copied,
            "verified_this_run": verified,
            "source_bytes": sum(
                entry["size"] for entry in current_entries.values()
            ),
        },
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy legacy results into a checksum-verified archive without "
            "deleting or modifying the source."
        )
    )
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--repository", type=Path, default=Path.cwd()
    )
    parser.add_argument(
        "--source",
        type=Path,
        action="append",
        default=None,
        help=(
            "Source path relative to the repository; repeat as needed. "
            "Defaults to output and data/legacy."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = archive(
        repository=args.repository,
        destination=args.destination,
        sources=args.source or [Path("output"), Path("data/legacy")],
    )
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))
    print(f"Manifest: {args.destination.expanduser().resolve() / MANIFEST_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
