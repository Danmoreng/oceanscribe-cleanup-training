"""The immutable ``raw-v1`` OceanScribe prompt contract.

This module is the sole source of model-facing literals. Run configuration may
select this contract, but it cannot override individual delimiters.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

SUPPORTED_LOCALES: Final = ("de-DE", "en-US", "en-GB")
MODEL_SPECIAL_TOKEN_PATTERN: Final = re.compile(r"<\|[^\r\n<>]+\|>")


@dataclass(frozen=True, slots=True)
class RawV1PromptContract:
    version: str = "raw-v1"
    task_keyword: str = "Cleanup"
    terminology_open: str = "<terminology>"
    terminology_close: str = "</terminology>"
    transcript_open: str = "<transcript>"
    transcript_close: str = "</transcript>"
    output_separator: str = "Output:\n"
    expected_eos_token: str = "<|endoftext|>"

    @property
    def reserved_markers(self) -> tuple[str, ...]:
        return (
            self.terminology_open,
            self.terminology_close,
            self.transcript_open,
            self.transcript_close,
            self.output_separator,
        )

    def validate_payload(self, value: str, *, field_name: str) -> None:
        for marker in self.reserved_markers:
            if marker in value:
                raise ValueError(f"{field_name} contains reserved raw-v1 marker {marker!r}")
        special_token = MODEL_SPECIAL_TOKEN_PATTERN.search(value)
        if special_token:
            raise ValueError(
                f"{field_name} contains reserved model token {special_token.group(0)!r}"
            )

    def render(
        self,
        *,
        transcript: str,
        language: str,
        commands: bool,
        terminology: Sequence[str],
    ) -> str:
        commands_value = "enabled" if commands else "disabled"
        terminology_lines = "".join(f"{term}\n" for term in terminology)
        return (
            f"{self.task_keyword}\n"
            f"Language: {language}\n"
            f"Commands: {commands_value}\n"
            f"{self.terminology_open}\n"
            f"{terminology_lines}"
            f"{self.terminology_close}\n"
            f"{self.transcript_open}\n"
            f"{transcript}\n"
            f"{self.transcript_close}\n"
            f"{self.output_separator}"
        )


RAW_V1: Final = RawV1PromptContract()
