from __future__ import annotations

import pytest
from pydantic import ValidationError

from oceanscribe_cleanup.records import CleanupRecord


def valid_record(**overrides: object) -> CleanupRecord:
    values: dict[str, object] = {
        "id": "source-1",
        "source_name": "fixture",
        "source_revision": "abc123",
        "source_record_id": "1",
        "license_spdx": "Apache-2.0",
        "split": "train",
        "language": "de-DE",
        "commands": True,
        "transcript": "äh hallo",
        "output": "Hallo.",
        "terminology": [],
        "features": ["disfluency"],
        "edit_script": [],
        "synthetic": False,
        "generator_version": None,
        "seed": None,
        "parent_id": None,
        "quality_flags": [],
    }
    values.update(overrides)
    return CleanupRecord.model_validate(values)


def test_record_sequences_are_frozen_tuples() -> None:
    record = valid_record(features=[" disfluency "])
    assert record.features == ("disfluency",)
    assert record.edit_script == ()


def test_record_rejects_non_boolean_commands() -> None:
    with pytest.raises(ValidationError):
        valid_record(commands=1)


def test_record_accepts_empty_output() -> None:
    assert valid_record(output="").output == ""


def test_record_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        valid_record(surprise="nope")


def test_synthetic_records_require_generator_provenance() -> None:
    with pytest.raises(ValidationError):
        valid_record(synthetic=True)
    record = valid_record(synthetic=True, generator_version="generator-v1", seed=42)
    assert record.generator_version == "generator-v1"


def test_stochastic_generation_never_invents_an_unavailable_seed():
    with pytest.raises(ValidationError):
        valid_record(synthetic=True, generator_version="teacher-v1", generation_kind="stochastic")
    record = valid_record(synthetic=True, generator_version="teacher-v1",
                          generation_kind="stochastic",
                          quality_flags=["generation-seed-unavailable"])
    assert record.seed is None
