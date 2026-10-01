"""Validated, immutable cleanup records and prompt inputs."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

from .prompt_contracts.raw_v1 import RAW_V1, SUPPORTED_LOCALES

Locale = Literal["de-DE", "en-US", "en-GB"]
Split = Literal["train", "validation", "test"]
NonEmpty = Annotated[str, Field(min_length=1)]


def normalize_newlines(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


def normalize_terminology(value: object) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise TypeError("terminology must be a list or tuple of canonical terms")
    if len(value) > 64:
        raise ValueError("terminology must contain at most 64 entries")

    normalized: list[str] = []
    for term in value:
        if not isinstance(term, str):
            raise TypeError("every terminology entry must be a string")
        if "\n" in term or "\r" in term:
            raise ValueError("terminology entries must not contain newlines")
        clean = " ".join(term.split())
        if not clean:
            raise ValueError("terminology entries must not be blank")
        if len(clean) > 200:
            raise ValueError("terminology entries must contain at most 200 characters")
        RAW_V1.validate_payload(clean, field_name="terminology")
        normalized.append(clean)
    return tuple(normalized)


class StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class PromptInput(StrictFrozenModel):
    transcript: str
    language: Locale
    commands: StrictBool = True
    terminology: tuple[str, ...] = ()

    @field_validator("transcript", mode="before")
    @classmethod
    def validate_transcript(cls, value: object) -> str:
        if not isinstance(value, str):
            raise TypeError("transcript must be a string")
        value = normalize_newlines(value)
        if not value.strip():
            raise ValueError("transcript must not be blank")
        RAW_V1.validate_payload(value, field_name="transcript")
        return value

    @field_validator("language")
    @classmethod
    def validate_language(cls, value: str) -> str:
        if value not in SUPPORTED_LOCALES:
            raise ValueError(f"unsupported locale: {value}")
        return value

    @field_validator("terminology", mode="before")
    @classmethod
    def validate_terminology(cls, value: object) -> tuple[str, ...]:
        return normalize_terminology(value)


class CleanupExample(PromptInput):
    """Small prompt/completion value object used by format and token tests."""

    output: str

    @field_validator("output", mode="before")
    @classmethod
    def validate_output(cls, value: object) -> str:
        if not isinstance(value, str):
            raise TypeError("output must be a string")
        value = normalize_newlines(value)
        RAW_V1.validate_payload(value, field_name="output")
        return value


class EditOperation(StrictFrozenModel):
    operation: NonEmpty
    source: str = ""
    target: str = ""
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)


class CleanupRecord(CleanupExample):
    schema_version: Literal[1] = 1
    id: NonEmpty
    source_name: NonEmpty
    source_revision: str | None = None
    source_record_id: str | None = None
    license_spdx: str | None = None
    split: Split
    features: tuple[str, ...] = ()
    edit_script: tuple[EditOperation, ...] = ()
    synthetic: StrictBool
    generator_version: str | None = None
    generation_kind: Literal["deterministic", "stochastic"] | None = None
    seed: int | None = None
    parent_id: str | None = None
    quality_flags: tuple[str, ...] = ()

    @field_validator("features", "quality_flags", mode="before")
    @classmethod
    def freeze_string_sequence(cls, value: object) -> tuple[str, ...]:
        if isinstance(value, str) or not isinstance(value, (list, tuple)):
            raise TypeError("value must be a list or tuple of strings")
        if not all(isinstance(item, str) and item.strip() for item in value):
            raise ValueError("entries must be non-blank strings")
        return tuple(item.strip() for item in value)

    @field_validator("edit_script", mode="before")
    @classmethod
    def freeze_edit_script(cls, value: object) -> tuple[object, ...]:
        if not isinstance(value, (list, tuple)):
            raise TypeError("edit_script must be a list or tuple")
        return tuple(value)

    @model_validator(mode="after")
    def require_synthetic_provenance(self) -> CleanupRecord:
        if self.synthetic:
            if not self.generator_version:
                raise ValueError("synthetic records require generator_version")
            if self.seed is None and not (
                self.generation_kind == "stochastic"
                and "generation-seed-unavailable" in self.quality_flags
            ):
                raise ValueError("synthetic records require seed or explicit stochastic provenance")
        return self
