"""Canonical raw-completion format used by training and later inference."""

from dataclasses import dataclass
from typing import Sequence


TASK_KEYWORD = "Cleanup"
DEFAULT_EOS_TOKEN = "<|endoftext|>"


def _normalize_terminology(terminology: Sequence[str]) -> tuple[str, ...]:
    if isinstance(terminology, str):
        raise TypeError("terminology must be a sequence of canonical terms")

    normalized: list[str] = []
    for term in terminology:
        if not isinstance(term, str):
            raise TypeError("every terminology entry must be a string")
        if "\n" in term or "\r" in term:
            raise ValueError("terminology entries must not contain newlines")
        term = term.strip()
        if not term:
            raise ValueError("terminology entries must not be empty")
        normalized.append(term)
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class CleanupExample:
    transcript: str
    output: str
    language: str
    commands: bool = True
    terminology: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.transcript.strip():
            raise ValueError("transcript must not be empty")
        if not self.output.strip():
            raise ValueError("output must not be empty")
        if not self.language.strip():
            raise ValueError("language must not be empty")
        _normalize_terminology(self.terminology)


def render_user_prompt(
    transcript: str,
    language: str,
    *,
    commands: bool = True,
    terminology: Sequence[str] = (),
) -> str:
    """Render the exact inference prefix without chat control tokens."""
    if not transcript.strip():
        raise ValueError("transcript must not be empty")
    if not language.strip():
        raise ValueError("language must not be empty")

    terms = _normalize_terminology(terminology)
    commands_value = "on" if commands else "off"
    terminology_lines = "".join(f"{term}\n" for term in terms)
    return (
        f"{TASK_KEYWORD}\n"
        f"Language: {language}\n"
        f"Commands: {commands_value}\n"
        "<terminology>\n"
        f"{terminology_lines}"
        "</terminology>\n"
        "<transcript>\n"
        f"{transcript}\n"
        "</transcript>\n"
        "Output:\n"
    )


def render_training_sequence(
    example: CleanupExample,
    *,
    eos_token: str = DEFAULT_EOS_TOKEN,
) -> str:
    """Render prompt + supervised completion + EOS as one raw token sequence."""
    if not eos_token:
        raise ValueError("eos_token must not be empty")
    return (
        render_user_prompt(
            example.transcript,
            example.language,
            commands=example.commands,
            terminology=example.terminology,
        )
        + example.output
        + eos_token
    )
