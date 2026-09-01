from __future__ import annotations

from oceanscribe_cleanup.data_prep import (
    AAWAAZ_REVISION,
    SourcePair,
    is_conservative_aawaaz_pair,
    select_records,
)


class FakeTokenizer:
    eos_token = "<|endoftext|>"
    eos_token_id = 42
    pad_token = "<|endoftext|>"
    pad_token_id = 42
    model_max_length = 4096

    def __call__(self, text: str, **kwargs: object) -> dict[str, list[int]]:
        if text == self.eos_token:
            return {"input_ids": [self.eos_token_id]}
        return {"input_ids": list(range(len(text.split())))}


def aawaaz_pair(index: int, output: str = "We should ship the release today.") -> SourcePair:
    transcript = f"um we should ship the release today item {index}"
    if output == "We should ship the release today.":
        output = f"We should ship the release today, item {index}."
    return SourcePair(
        source_name="aawaaz",
        source_revision=AAWAAZ_REVISION,
        source_record_id=f"fixture:{index}",
        transcript=transcript,
        output=output,
        category="fixture",
    )


def test_conservative_filter_rejects_expansive_rewrite() -> None:
    assert is_conservative_aawaaz_pair(aawaaz_pair(1))
    assert not is_conservative_aawaaz_pair(
        aawaaz_pair(2, "A wholly unrelated and newly invented summary appears here.")
    )


def test_selection_is_deterministic_and_has_exact_split_counts() -> None:
    pairs = [aawaaz_pair(index) for index in range(200)]
    first, _ = select_records(
        pairs,
        FakeTokenizer(),
        source_name="aawaaz",
        train_count=12,
        validation_count=3,
        seed=42,
    )
    second, _ = select_records(
        list(reversed(pairs)),
        FakeTokenizer(),
        source_name="aawaaz",
        train_count=12,
        validation_count=3,
        seed=42,
    )
    assert [record.id for record in first] == [record.id for record in second]
    assert sum(record.split == "train" for record in first) == 12
    assert sum(record.split == "validation" for record in first) == 3
    assert all("upstream-synthetic" in record.features for record in first)
