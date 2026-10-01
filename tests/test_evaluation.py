from __future__ import annotations

from dataclasses import replace

from oceanscribe_cleanup.evaluation import (
    GenerationResult,
    normalized_exact,
    output_has_forbidden_control,
    score_outputs,
)


def result(base: str, adapter: str) -> GenerationResult:
    return GenerationResult(
        id="fixture",
        source_name="fixture",
        features=(),
        transcript="um hello world",
        reference="Hello world.",
        base_output=base,
        adapter_output=adapter,
        base_stopped_at_eos=False,
        adapter_stopped_at_eos=True,
    )


def test_output_contract_helpers() -> None:
    assert normalized_exact("Hello\nworld.", "Hello world.")
    assert output_has_forbidden_control("Output:\nwrong")
    assert output_has_forbidden_control("text<|endoftext|>")
    assert not output_has_forbidden_control("Hello world.")


def test_adapter_metrics_improve_for_exact_fixture() -> None:
    scores = score_outputs([result("unrelated", "Hello world.")])
    assert scores["adapter"]["cer"] == 0
    assert scores["adapter"]["wer"] == 0
    assert scores["adapter"]["normalized_exact_matches"] == 1
    assert scores["adapter"]["eos_stop_rate"] == 1
    assert scores["delta"]["cer"] < 0


def test_quality_metrics_distinguish_missing_and_inserted_terms():
    first = replace(result("unrelated", "OceanScribe is ready."),
                    reference="OceanScribe is ready.",
                    checks={"corrected_terms": ["OceanScribe"], "absent_terms": ["NebulaForge"]})
    second = replace(result("unrelated", "NebulaForge is ready."), id="second",
                     reference="It is ready.", checks={"absent_terms": ["NebulaForge"]})
    quality = score_outputs([first, second])["adapter"]["quality_checks"]
    assert quality["term_correction_recall"] == 1
    assert quality["false_insertion_rate"] == 0.5
    assert quality["passed_records"] == 1
    assert quality["failures"][0]["id"] == "second"
