from __future__ import annotations

import csv
import json

import pytest
from test_data_prep import FakeTokenizer

from oceanscribe_cleanup.data_prep import AAWAAZ_REVISION, SOTTO_REVISION
from oceanscribe_cleanup.reviewed_csv import prepare_reviewed_smoke
from oceanscribe_cleanup.training import load_cleanup_records, validate_dataset_manifest


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_rows():
    return [
        {
            "record_id": f"{source}-{index}",
            "source_name": source,
            "source_revision": revision,
            "source_record_id": f"train:{index}",
            "license_spdx": "MIT",
            "split": "train",
            "commands": "true",
            "source_locale": "en-US",
            "english_review_status": "KEEP",
            "english_raw_transcript": f"um we should ship item {index} today for {source}",
            "english_reference_original": f"We should ship item {index} today for {source}.",
            "english_reference_reviewed": f"We should ship item {index} today for {source}.",
            "features": "external|filler-removal",
            "quality_flags": "upstream-synthetic",
        }
        for source, revision in (("sotto", SOTTO_REVISION), ("aawaaz", AAWAAZ_REVISION))
        for index in range(100)
    ]


def test_reviewed_subset_is_frozen_and_excludes_untrusted_and_oversized_rows(tmp_path):
    rows = source_rows()
    rows[0]["license_spdx"] = "unknown"
    rows[1]["english_review_status"] = "FIX"
    rows[2]["english_reference_reviewed"] = "An unsupported changed reference."
    rows[3]["english_raw_transcript"] = "word " * 3000
    source = tmp_path / "source.csv"
    write_csv(source, rows)
    records_path, manifest_path = prepare_reviewed_smoke(
        source, FakeTokenizer(), tmp_path / "first", per_source=10, validation_per_source=2
    )
    first = load_cleanup_records(records_path)
    assert len(first) == 20
    assert sum(r.split == "validation" for r in first) == 4
    assert not {r.id for r in first} & {r["record_id"] for r in rows[:4]}
    assert all(r.license_spdx == "MIT" and r.language == "en-US" for r in first)
    validate_dataset_manifest(records_path, manifest_path)
    selection = json.loads((tmp_path / "first" / "selection.json").read_text())
    assert "not truncated" in selection["excluded_records"][rows[3]["record_id"]]
    write_csv(source, list(reversed(rows)))
    second_path, _ = prepare_reviewed_smoke(
        source, FakeTokenizer(), tmp_path / "second", per_source=10, validation_per_source=2
    )
    assert records_path.read_bytes() == second_path.read_bytes()


def test_reviewed_subset_rejects_duplicate_ids(tmp_path):
    rows = source_rows()
    rows.append(rows[0])
    source = tmp_path / "source.csv"
    write_csv(source, rows)
    with pytest.raises(ValueError, match="duplicate CSV record IDs"):
        prepare_reviewed_smoke(source, FakeTokenizer(), tmp_path / "output")
