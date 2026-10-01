from __future__ import annotations

from collections import defaultdict

from test_data_prep import FakeTokenizer
from test_reviewed_csv import source_rows, write_csv

from oceanscribe_cleanup.bilingual_csv import prepare_bilingual
from oceanscribe_cleanup.training import load_cleanup_records, validate_dataset_manifest


def test_bilingual_selection_preserves_family_holdouts_and_generation_provenance(tmp_path):
    rows = source_rows()
    for index, row in enumerate(rows):
        row.update(german_translation_status="PASS", german_raw_transcript=f"äh Paket {index}",
                   german_reference=f"Paket {index}.")
    rows[0]["german_translation_status"] = "REVIEW"
    rows[1]["english_review_status"] = ""
    rows[2]["german_raw_transcript"] = "wort " * 3000
    rows[3]["english_review_status"] = "FIX"
    source = tmp_path / "source.csv"
    write_csv(source, rows)
    records = prepare_bilingual(source, FakeTokenizer(), tmp_path / "snapshot")
    reloaded = load_cleanup_records(tmp_path / "snapshot/records.jsonl")
    assert records == list(reloaded)
    validate_dataset_manifest(
        tmp_path / "snapshot/records.jsonl", tmp_path / "snapshot/manifest.json"
    )
    families = defaultdict(set)
    for record in records:
        families[(record.source_name, record.source_record_id)].add(record.split)
    assert all(len(splits) == 1 for splits in families.values())
    assert not any(r.parent_id in {row["record_id"] for row in rows[:3]} for r in records)
    german = [r for r in records if r.language == "de-DE" and r.parent_id == rows[3]["record_id"]]
    assert len(german) == 1
    assert german[0].generation_kind == "stochastic" and german[0].seed is None
    assert "generation-seed-unavailable" in german[0].quality_flags
    assert "LicenseRef" in german[0].license_spdx
    assert (tmp_path / "snapshot/source-snapshot.csv").read_bytes() == source.read_bytes()
