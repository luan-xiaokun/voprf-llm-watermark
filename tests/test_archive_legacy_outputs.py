from __future__ import annotations

import json
from pathlib import Path

from scripts.archive_legacy_outputs import MANIFEST_NAME, archive, sha256_file


def test_archive_is_verified_idempotent_and_never_deletes_source(tmp_path):
    repository = tmp_path / "repository"
    output = repository / "output" / "generation"
    legacy = repository / "data" / "legacy"
    output.mkdir(parents=True)
    legacy.mkdir(parents=True)
    first = output / "results.jsonl"
    second = legacy / "samples.json"
    first.write_text('{"sample": 1}\n', encoding="utf-8")
    second.write_text('{"sample": 2}\n', encoding="utf-8")
    destination = tmp_path / "archive"

    initial = archive(
        repository=repository,
        destination=destination,
        sources=[Path("output"), Path("data/legacy")],
    )
    repeated = archive(
        repository=repository,
        destination=destination,
        sources=[Path("output"), Path("data/legacy")],
    )

    assert first.exists()
    assert second.exists()
    assert initial["summary"]["copied_this_run"] == 2
    assert repeated["summary"]["copied_this_run"] == 0
    assert repeated["summary"]["verified_this_run"] == 2
    archived_first = (
        destination / "repository" / "output" / "generation" / first.name
    )
    assert sha256_file(archived_first) == sha256_file(first)
    saved = json.loads(
        (destination / MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert len(saved["files"]) == 2
    assert saved["created_at"] == initial["created_at"]


def test_archive_refuses_to_replace_different_existing_file(tmp_path):
    repository = tmp_path / "repository"
    source_dir = repository / "output"
    source_dir.mkdir(parents=True)
    source = source_dir / "result.txt"
    source.write_text("original", encoding="utf-8")
    destination = tmp_path / "archive"
    archived = destination / "repository" / "output" / "result.txt"
    archived.parent.mkdir(parents=True)
    archived.write_text("different", encoding="utf-8")

    try:
        archive(
            repository=repository,
            destination=destination,
            sources=[Path("output")],
        )
    except ValueError as error:
        assert "refusing to overwrite" in str(error)
    else:
        raise AssertionError("archive should reject a conflicting file")
    assert source.read_text(encoding="utf-8") == "original"
