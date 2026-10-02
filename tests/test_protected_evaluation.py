import pytest

from oceanscribe_cleanup.protected_evaluation import score_protected_outputs
from oceanscribe_cleanup.records import CleanupRecord


def record(identifier, text, *, split="validation", locale="de-DE"):
    return CleanupRecord(id=identifier, source_name="fixture", split=split, language=locale,
                         transcript=text, output=text, synthetic=False)


def test_asr_variants_do_not_inflate_independent_family_count_or_meaning_labels():
    records = [record("a-nemotron", "Nicht senden."), record("a-parakeet", "Nicht senden."),
               record("b", "Morgen prüfen.")]
    scores = score_protected_outputs(
        records, ["Senden.", "Senden.", "Morgen prüfen."], [True] * 3, ["a", "a", "b"]
    )
    assert scores["records"] == 3 and scores["families"] == 2
    assert scores["by_language"]["de-DE"]["family_mean_wer"] == 0.5
    assert scores["meaning"] == {"status": "not_evaluated", "denominator": 0, "failures": None}
    assert scores["preserve_exact"] == {"matches": 1, "denominator": 3}


def test_output_limit_and_control_failures_are_explicit():
    scores = score_protected_outputs([record("a", "Okay.")], ["<|assistant|>"],
                                     [False], ["a"])
    assert scores["generation_limit_hits"] == 1
    assert scores["forbidden_control_outputs"] == 1
    with pytest.raises(ValueError, match="Development only"):
        score_protected_outputs([record("test", "Okay.", split="test")], ["Okay."],
                                [True], ["test"])
