from __future__ import annotations

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
