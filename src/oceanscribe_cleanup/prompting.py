"""Rendering helpers for the selected versioned prompt contract."""

from __future__ import annotations

from collections.abc import Sequence

from .prompt_contracts import RAW_V1
from .records import CleanupExample, PromptInput


def render_user_prompt(
    transcript: str,
    language: str,
    *,
    commands: bool = True,
    terminology: Sequence[str] = (),
) -> str:
    """Render the exact raw-v1 inference prefix, with validation."""
    prompt_input = PromptInput(
        transcript=transcript,
        language=language,
        commands=commands,
        terminology=tuple(terminology) if not isinstance(terminology, str) else terminology,
    )
    return RAW_V1.render(
        transcript=prompt_input.transcript,
        language=prompt_input.language,
        commands=prompt_input.commands,
        terminology=prompt_input.terminology,
    )


def render_example_prompt(example: PromptInput) -> str:
    return RAW_V1.render(
        transcript=example.transcript,
        language=example.language,
        commands=example.commands,
        terminology=example.terminology,
    )


def render_training_sequence(
    example: CleanupExample,
    *,
    eos_token: str = RAW_V1.expected_eos_token,
) -> str:
    """Render text for human inspection only.

    The actual training path is :func:`tokenize_example`, which appends EOS by
    token ID and builds completion-only labels. String rendering is retained as
    a backwards-compatible documentation/debugging helper.
    """
    if not eos_token:
        raise ValueError("eos_token must not be empty")
    return render_example_prompt(example) + example.output + eos_token
