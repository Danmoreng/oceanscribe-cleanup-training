from __future__ import annotations

import hashlib
import json

import pytest

from oceanscribe_cleanup.config import load_config
from oceanscribe_cleanup.records import CleanupRecord
from oceanscribe_cleanup.training import (
    language_sampling_weights,
    load_cleanup_records,
    load_training_inputs,
    tokenize_records,
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


def test_token_statistics_count_microbatch_and_only_training_records() -> None:
    from test_data_prep import FakeTokenizer

    rows = [
        CleanupRecord.model_validate(record("train", split="train", source="sotto")),
        CleanupRecord.model_validate(record("validation", split="validation", source="aawaaz")),
    ]
    train, _, stats = tokenize_records(
        rows, FakeTokenizer(), max_sequence_length=2048,
        gradient_accumulation_steps=4, per_device_train_batch_size=4,
    )
    assert stats.non_padding_tokens_per_optimizer_step == train[0].sequence_length * 16


def test_hint_variants_are_distinct_but_family_split_leakage_is_rejected(tmp_path):
    first = record("one", split="train", source="sotto")
    second = {**first, "id": "one-hint", "terminology": ["OceanScribe"]}
    path = tmp_path / "records.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [first, second]) + "\n")
    assert len(load_cleanup_records(path)) == 2
    second.update(language="de-DE", split="test")
    path.write_text("\n".join(json.dumps(r) for r in [first, second]) + "\n")
    with pytest.raises(ValueError, match="family crosses split"):
        load_cleanup_records(path)


def test_token_balancing_equalizes_expected_language_tokens():
    from oceanscribe_cleanup.tokenization import TokenizedExample

    records = [CleanupRecord.model_validate(record(str(i), split="train", source="fixture"))
               for i in range(3)]
    records[2] = records[2].model_copy(update={"language": "de-DE"})
    examples = [TokenizedExample(tuple(range(n)), (), (), 0, n) for n in [10, 30, 100]]
    weights = language_sampling_weights(records, examples, "tokens")
    assert weights[0] * 10 + weights[1] * 30 == pytest.approx(weights[2] * 100)


@pytest.mark.parametrize("overlap", ["family", "text", "id", None])
def test_separate_development_has_frozen_hash_and_no_train_overlap(tmp_path, overlap):
    train = record("training", split="train", source="sotto")
    dev = record("development", split="validation", source="sotto")
    if overlap == "family":
        dev["source_record_id"] = train["source_record_id"]
    elif overlap == "text":
        dev["transcript"] = str(train["transcript"]).upper() + "  "
    elif overlap == "id":
        dev["id"] = train["id"]
    train_path = tmp_path / "train.jsonl"
    train_path.write_text(json.dumps(train) + "\n")
    dev_path = tmp_path / "dev.jsonl"
    dev_path.write_text(json.dumps(dev) + "\n")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "sha256": hashlib.sha256(dev_path.read_bytes()).hexdigest(),
        "human_reference_frozen": True,
    }))
    if overlap:
        with pytest.raises(ValueError, match="overlap"):
            load_training_inputs(train_path, dev_path)
    else:
        rows, manifest = load_training_inputs(train_path, dev_path)
        assert [r.split for r in rows] == ["train", "validation"]
        assert manifest["human_reference_frozen"]
        assert load_cleanup_records(train_path)[0].split == "train"
        manifest_path.write_text(json.dumps({
            "sha256": hashlib.sha256(dev_path.read_bytes()).hexdigest(),
            "human_reference_frozen": False,
        }))
        with pytest.raises(ValueError, match="human-confirmed"):
            load_training_inputs(train_path, dev_path)
