from __future__ import annotations

import pytest

from oceanscribe_cleanup.collator import pad_features
from oceanscribe_cleanup.records import CleanupExample
from oceanscribe_cleanup.tokenization import (
    IGNORE_INDEX,
    OversizeExampleError,
    tokenize_example,
    validate_tokenizer_contract,
)


class FakeTokenizer:
    eos_token = "<|endoftext|>"
    eos_token_id = 42
    pad_token = "<|endoftext|>"
    pad_token_id = 42
    model_max_length = 4096

    def __call__(self, text: str, **kwargs: object) -> dict[str, list[int]]:
        assert kwargs["add_special_tokens"] is False
        assert kwargs["truncation"] is False
        if text == self.eos_token:
            return {"input_ids": [self.eos_token_id]}
        return {"input_ids": [ord(character) + 100 for character in text]}


def example(output: str = "Hallo.") -> CleanupExample:
    return CleanupExample(
        transcript="äh hallo",
        output=output,
        language="de-DE",
    )


def test_eos_is_exactly_one_expected_token() -> None:
    tokenizer = FakeTokenizer()
    validate_tokenizer_contract(tokenizer)
    tokenized = tokenize_example(example(), tokenizer, max_sequence_length=2048)
    assert tokenized.input_ids[-1] == tokenizer.eos_token_id
    assert tokenized.input_ids.count(tokenizer.eos_token_id) == 1


def test_prompt_is_masked_while_target_and_eos_are_supervised() -> None:
    tokenizer = FakeTokenizer()
    tokenized = tokenize_example(example("OK"), tokenizer, max_sequence_length=2048)
    assert tokenized.labels[: tokenized.prompt_length] == (IGNORE_INDEX,) * tokenized.prompt_length
    assert tokenized.labels[tokenized.prompt_length : -1] == (ord("O") + 100, ord("K") + 100)
    assert tokenized.labels[-1] == tokenizer.eos_token_id


def test_empty_target_still_supervises_eos() -> None:
    tokenized = tokenize_example(example(""), FakeTokenizer(), max_sequence_length=2048)
    assert tokenized.target_length == 0
    assert tokenized.labels[-1] == FakeTokenizer.eos_token_id


def test_padding_masks_by_position_when_pad_equals_eos() -> None:
    tokenizer = FakeTokenizer()
    short = tokenize_example(example(""), tokenizer, max_sequence_length=2048)
    long = tokenize_example(example("longer"), tokenizer, max_sequence_length=2048)
    batch = pad_features([short, long], pad_token_id=tokenizer.pad_token_id)
    assert batch["input_ids"][0][short.sequence_length - 1] == tokenizer.eos_token_id
    assert batch["labels"][0][short.sequence_length - 1] == tokenizer.eos_token_id
    assert all(label == IGNORE_INDEX for label in batch["labels"][0][short.sequence_length :])
    assert all(mask == 0 for mask in batch["attention_mask"][0][short.sequence_length :])


def test_oversize_examples_are_never_silently_truncated() -> None:
    with pytest.raises(OversizeExampleError) as caught:
        tokenize_example(example("target"), FakeTokenizer(), max_sequence_length=8)
    assert caught.value.actual_tokens > 8
    assert "not truncated" in str(caught.value)


def test_train_and_inference_prefix_are_identical() -> None:
    tokenizer = FakeTokenizer()
    sample = example("clean")
    tokenized = tokenize_example(sample, tokenizer, max_sequence_length=2048)
    from oceanscribe_cleanup.prompting import render_example_prompt

    prefix_ids = tokenizer(
        render_example_prompt(sample),
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
    )["input_ids"]
    assert tokenized.input_ids[: tokenized.prompt_length] == tuple(prefix_ids)
