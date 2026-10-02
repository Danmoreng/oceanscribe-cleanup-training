from oceanscribe_cleanup.review_studio.checks import assess_output


def test_missing_blank_line_is_reported_only_when_format_is_annotated():
    transcript = "First paragraph.\n\nSecond paragraph."
    output = "First paragraph.\nSecond paragraph."

    unchecked = assess_output(transcript, output)
    checked = assess_output(transcript, output, {"required_format": {"min_paragraphs": 2}})

    assert unchecked["automated_checks"]["status"] == "not_evaluated"
    assert checked["automated_checks"]["required_format"]["failed"] is True
    assert checked["automated_checks"]["failures"] == 1


def test_required_facts_bind_values_to_their_entities():
    annotations = {"required_facts": [["revenue", "$10"], ["cost", "$20"]]}
    correct = assess_output("", "Revenue is $10; cost is $20.", annotations)
    swapped = assess_output("", "Revenue is $20; cost is $10.", annotations)
    detached = assess_output("", "Revenue and cost were discussed. $10 and $20.", annotations)

    assert correct["automated_checks"]["required_facts"]["failures"] == []
    assert swapped["automated_checks"]["required_facts"]["failures"] == [
        {"entity": "revenue", "value": "$10"},
        {"entity": "cost", "value": "$20"},
    ]
    assert len(detached["automated_checks"]["required_facts"]["failures"]) == 2


def test_explicit_negation_and_forbidden_hint_checks():
    annotations = {
        "required_negations": ["not significant"],
        "forbidden_insertions": ["NebulaForge"],
    }
    okay = assess_output("", "The result was not significant.", annotations)
    lost_negation = assess_output("", "The result was significant.", annotations)
    extra_hint = assess_output("", "NebulaForge was significant.", annotations)

    assert okay["automated_checks"]["failures"] == 0
    assert okay["automated_checks"]["required_negations"]["missing"] == []
    assert lost_negation["automated_checks"]["required_negations"]["missing"] == ["not significant"]
    assert extra_hint["automated_checks"]["forbidden_insertions"]["present"] == ["NebulaForge"]


def test_preserve_is_scored_only_when_annotated_and_copy_baseline_is_separate():
    transcript = "Keep this exact text."
    edited = "Keep this edited text."
    unannotated = assess_output(transcript, edited)
    annotated = assess_output(transcript, edited, {"preserve_exact": True})

    assert unannotated["copy_only_baseline"] == {"is_copy": False, "denominator": 1}
    assert unannotated["automated_checks"]["status"] == "not_evaluated"
    assert unannotated["automated_checks"]["preserve_exact"]["failed"] is False
    assert annotated["automated_checks"]["preserve_exact"]["failed"] is True
    assert annotated["automated_checks"]["failures"] == 1


def test_meaning_failure_requires_human_assessment():
    automatic = assess_output("The value is 10.", "The value is 20.")
    human_error = assess_output(
        "The value is 10.", "The value is 20.", human_meaning_failure=True
    )

    assert automatic["meaning"] == {
        "status": "not_evaluated", "failures": None, "denominator": 0,
    }
    assert human_error["meaning"] == {
        "status": "human_assessed", "failures": 1, "denominator": 1,
    }
