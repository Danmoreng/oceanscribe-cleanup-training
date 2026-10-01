from __future__ import annotations

import json

from oceanscribe_cleanup.quality import check_output
from oceanscribe_cleanup.targeted_data import generate_targeted, prepare_targeted
from oceanscribe_cleanup.training import load_cleanup_records, validate_dataset_manifest


def test_targeted_families_are_split_before_all_derivatives():
    train, challenge, checks = generate_targeted()
    assert len(train) == 200 and len(challenge) == 50
    splits = {}
    for record in (*train, *challenge):
        family = record.source_record_id
        assert family not in splits or splits[family] == record.split
        splits[family] = record.split
    assert {r.source_record_id for r in train}.isdisjoint(
        r.source_record_id for r in challenge
    )
    for split in ("train", "validation"):
        records = [r for r in train if r.split == split]
        assert sum("preserve" in r.features for r in records) / len(records) >= 0.2
        assert sum("hard-negative" in r.features for r in records) / len(records) >= 0.1
        assert sum(r.language == "de-DE" for r in records) == len(records) // 2
    assert all(check_output(r.output, r.output, checks[r.id])["passed"] for r in challenge)
    assert any(not r.commands for r in train)
    assert any(not r.terminology for r in train)


def test_targeted_artifacts_are_reproducible_and_loadable(tmp_path):
    for name in ("a", "b"):
        prepare_targeted(tmp_path / name)
    for subset in ("training", "challenge"):
        left = tmp_path / "a" / subset / "records.jsonl"
        right = tmp_path / "b" / subset / "records.jsonl"
        assert left.read_bytes() == right.read_bytes()
        records = load_cleanup_records(left)
        manifest = validate_dataset_manifest(left, left.with_name("manifest.json"))
        assert sum(manifest["counts_by_language"].values()) == len(records)
    assert len(json.loads((tmp_path / "a/challenge/checks.json").read_text())) == 50
