"""Token-level construction of raw-v1 training examples."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .prompt_contracts import RAW_V1
from .prompting import render_example_prompt
from .records import CleanupExample

IGNORE_INDEX = -100


class TokenizerLike(Protocol):
    eos_token: str | None
    eos_token_id: int | None
    pad_token: str | None
    pad_token_id: int | None
    model_max_length: int

    def __call__(self, text: str, **kwargs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class TokenizedExample:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    attention_mask: tuple[int, ...]
    prompt_length: int
    target_length: int

    @property
    def sequence_length(self) -> int:
        return len(self.input_ids)


class OversizeExampleError(ValueError):
    def __init__(
        self,
        *,
        actual_tokens: int,
        max_sequence_length: int,
        prompt_tokens: int,
        target_tokens: int,
    ) -> None:
        self.actual_tokens = actual_tokens
        self.max_sequence_length = max_sequence_length
        self.prompt_tokens = prompt_tokens
        self.target_tokens = target_tokens
        super().__init__(
            "example has "
            f"{actual_tokens} tokens ({prompt_tokens} prompt, {target_tokens} target, "
            f"1 EOS), exceeding max_sequence_length={max_sequence_length}; "
            "the example was not truncated"
        )


def _input_ids(encoded: Any) -> list[int]:
    if isinstance(encoded, dict):
        ids = encoded.get("input_ids")
    else:
        ids = getattr(encoded, "input_ids", None)
    if ids is None:
        raise TypeError("tokenizer result does not contain input_ids")
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("expected a single tokenizer sequence")
        ids = ids[0]
    if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)):
        raise TypeError("tokenizer input_ids must be a sequence")
    if not all(isinstance(token_id, int) for token_id in ids):
        raise TypeError("tokenizer input_ids must contain integers")
    return list(ids)


def encode_without_special_tokens(tokenizer: TokenizerLike, text: str) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
    )
    return _input_ids(encoded)


def validate_tokenizer_contract(tokenizer: TokenizerLike) -> None:
    if tokenizer.eos_token != RAW_V1.expected_eos_token:
        raise ValueError(
            f"expected EOS token {RAW_V1.expected_eos_token!r}, got {tokenizer.eos_token!r}"
        )
    if not isinstance(tokenizer.eos_token_id, int):
        raise ValueError("tokenizer must define an integer eos_token_id")
    eos_ids = encode_without_special_tokens(tokenizer, RAW_V1.expected_eos_token)
    if eos_ids != [tokenizer.eos_token_id]:
        raise ValueError(
            f"raw-v1 EOS literal must encode to exactly tokenizer.eos_token_id; got {eos_ids!r}"
        )


def tokenize_example(
    example: CleanupExample,
    tokenizer: TokenizerLike,
    *,
    max_sequence_length: int,
) -> TokenizedExample:
    """Build prompt-masked labels without truncation or string-appended EOS."""
    if max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be positive")
    validate_tokenizer_contract(tokenizer)
    assert tokenizer.eos_token_id is not None

    prompt_ids = encode_without_special_tokens(tokenizer, render_example_prompt(example))
    target_ids = encode_without_special_tokens(tokenizer, example.output)
    input_ids = prompt_ids + target_ids + [tokenizer.eos_token_id]
    labels = [IGNORE_INDEX] * len(prompt_ids) + target_ids + [tokenizer.eos_token_id]

    if len(input_ids) > max_sequence_length:
        raise OversizeExampleError(
            actual_tokens=len(input_ids),
            max_sequence_length=max_sequence_length,
            prompt_tokens=len(prompt_ids),
            target_tokens=len(target_ids),
        )

    return TokenizedExample(
        input_ids=tuple(input_ids),
        labels=tuple(labels),
        attention_mask=(1,) * len(input_ids),
        prompt_length=len(prompt_ids),
        target_length=len(target_ids),
    )
