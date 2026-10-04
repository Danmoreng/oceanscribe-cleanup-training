from __future__ import annotations

from oceanscribe_cleanup.records import CleanupRecord
from oceanscribe_cleanup.terminology_evaluation import ablation_variants, literal_hint_metrics


def test_same_frozen_input_reference_and_split_for_all_hint_conditions():
    record = CleanupRecord.model_validate(
        {
            "id": "fixture",
            "source_name": "fixture",
            "source_record_id": "family",
            "split": "validation",
            "language": "de-DE",
            "commands": True,
            "transcript": "Prometeus meldet Fehler in PostgreSQL.",
            "output": "Prometheus meldet Fehler in PostgreSQL.",
            "terminology": ["Prometheus", "PostgreSQL"],
            "synthetic": False,
        }
    )
    variants = ablation_variants(record)
    assert set(variants) == {"with-hints", "without-hints", "irrelevant-hints"}
    for variant in variants.values():
        assert variant.transcript == record.transcript and variant.output == record.output
        assert variant.split == "validation" and variant.source_record_id == "family"
    assert not variants["without-hints"].terminology
    metrics = literal_hint_metrics([record], [record.output])
    assert metrics["correction_recall"] == {"corrected_terms": 1, "eligible_terms": 1}
    assert metrics["already_correct_retention"] == {"retained_terms": 1, "eligible_terms": 1}
    unrelated = variants["irrelevant-hints"]
    false_output = record.output + " " + unrelated.terminology[0]
    assert literal_hint_metrics([unrelated], [false_output])["false_insertion"] == {
        "inserted_terms": 1,
        "eligible_terms": 4,
    }
