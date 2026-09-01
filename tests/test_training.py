from __future__ import annotations

import hashlib
import json

import pytest

from oceanscribe_cleanup.config import load_config
from oceanscribe_cleanup.records import CleanupRecord
from oceanscribe_cleanup.training import (
    load_cleanup_records,
    validate_dataset_against_config,
    validate_dataset_manifest,
)


def record(identifier: str, *, split: str, source: str) -> dict[str, object]:
    return CleanupRecord(
        id=identifier,
        source_name=source,
        source_revision="a" * 40,
        source_record_id=identifier,
        license_spdx="MIT",
        split=split,
        transcript=f"raw transcript {identifier}",
        output=f"Clean transcript {identifier}.",
        language="en-US",
        commands=True,
        synthetic=False,
    ).model_dump(mode="json")


def test_records_and_manifest_are_validated_before_training(tmp_path) -> None:
    records_path = tmp_path / "records.jsonl"
    rows = [record("one", split="train", source="sotto")]
    records_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    digest = hashlib.sha256(records_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"sha256": digest, "sources": {}}))

    loaded = load_cleanup_records(records_path)
    assert loaded[0].id == "one"
    assert validate_dataset_manifest(records_path, manifest_path)["sha256"] == digest

    records_path.write_text(records_path.read_text() + json.dumps(rows[0]) + "\n")
    with pytest.raises(ValueError, match="duplicate record id"):
        load_cleanup_records(records_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_dataset_manifest(records_path, manifest_path)


def test_dataset_must_match_smoke_config() -> None:
    config = load_config("configs/runs/smoke-r16.yaml")
    rows = [
        CleanupRecord.model_validate(record("one", split="train", source="sotto")),
        CleanupRecord.model_validate(record("two", split="validation", source="aawaaz")),
    ]
    with pytest.raises(ValueError, match="expects 200 sotto"):
        validate_dataset_against_config(rows, config)  # type: ignore[arg-type]
