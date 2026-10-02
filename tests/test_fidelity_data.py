from __future__ import annotations

import re

import pytest

from oceanscribe_cleanup.fidelity_data import generate_fidelity, prepare_fidelity_mix
from oceanscribe_cleanup.quality import check_output
from oceanscribe_cleanup.targeted_data import prepare_targeted
from oceanscribe_cleanup.training import load_cleanup_records, validate_dataset_manifest


def test_fidelity_family_split_and_reference_invariants():
    mixed, challenge, checks = generate_fidelity()
    assert len(mixed) == 512 and len(challenge) == 64
    splits = {}
    for r in (*mixed, *challenge):
        assert splits.setdefault(r.source_record_id, r.split) == r.split
        assert check_output(r.output, r.output, checks[r.id])["passed"]
        # Sentence-start capitalization must not become a false preserve failure.
        assert r.output[0].isupper()
        # IDs remain exact even through cleanup and replacement of superseded quantities.
        identifier = re.search(r"(?:Auftrag|[Oo]rder) (\d+)", r.transcript)[1]
        assert re.search(r"(?:Auftrag|[Oo]rder) (\d+)", r.output)[1] == identifier
        assert r.seed == 73 and r.generation_kind == "deterministic"
    assert {r.source_record_id for r in mixed}.isdisjoint(r.source_record_id for r in challenge)
    train = [r for r in mixed if r.split == "train"]
    assert len(train) == 384
    assert sum("preserve" in r.features for r in train) / len(train) >= .2
    assert sum("hard-negative" in r.features for r in train) / len(train) >= .1
    assert len([r for r in train if r.language == "de-DE"]) == len(train) // 2


def test_fidelity_annotations_reject_observed_failure_types():
    mixed, _, checks = generate_fidelity()
    repetition = next(r for r in mixed if "number-repeat" in r.features)
    identifier = re.search(r"\d+", repetition.output)[0]
    broken = repetition.output.replace(identifier, identifier + "2")
    assert not check_output(broken, repetition.output, checks[repetition.id])["passed"]
    correction = next(r for r in mixed if "self-correction" in r.features)
    assert not check_output(
        correction.transcript, correction.output, checks[correction.id]
    )["passed"]
    units = next(r for r in mixed if "unit-absence" in r.features and r.language == "de-DE")
    assert not check_output(units.output + " 12 Meter", units.output, checks[units.id])["passed"]
    absent = next(r for r in mixed if r.terminology == ("NebulaForge",))
    assert not check_output(
        absent.output + " NebulaForge", absent.output, checks[absent.id]
    )["passed"]


def test_fidelity_mix_preserves_pilot_records_and_reproduces(tmp_path):
    prepare_targeted(tmp_path / "base")
    source = tmp_path / "base/training/records.jsonl"
    for name in ("one", "two"):
        prepare_fidelity_mix(source, tmp_path / name)
        path = tmp_path / name / "training/records.jsonl"
        records = load_cleanup_records(path)
        validate_dataset_manifest(path, path.with_name("manifest.json"))
        assert path.read_bytes().startswith(source.read_bytes())
        assert len(records) == 200 + 512
    assert (tmp_path / "one/training/records.jsonl").read_bytes() == (
        tmp_path / "two/training/records.jsonl"
    ).read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        prepare_fidelity_mix(source, tmp_path / "one")
