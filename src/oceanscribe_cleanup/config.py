"""Strict, versioned run and sweep configuration loading."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_validator,
)


class CommandsMode(StrEnum):
    ENABLED = "enabled"
    DISABLED = "disabled"
    MIXED = "mixed"


class StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ModelConfig(StrictConfigModel):
    id: Literal["Qwen/Qwen3.5-0.8B-Base"]
    revision: str | None


class DataConfig(StrictConfigModel):
    subset: str | None = None
    sotto_en: int | None = Field(default=None, ge=0)
    aawaaz_en: int | None = Field(default=None, ge=0)
    synthetic_en: int | None = Field(default=None, ge=0)
    synthetic_de: int | None = Field(default=None, ge=0)
    balance_by: Literal["tokens", "records"] | None = None
    preserve_minimum: float | None = Field(default=None, ge=0, le=1)
    hard_negative_minimum: float | None = Field(default=None, ge=0, le=1)
    use_real_contiguous_documents: StrictBool = False


class LoraConfig(StrictConfigModel):
    rank: int = Field(gt=0)
    alpha: int = Field(gt=0)
    dropout: float = Field(ge=0, lt=1)


class TrainingConfig(StrictConfigModel):
    precision: Literal["bf16"] = "bf16"
    optimizer: Literal["adamw_torch_fused"] = "adamw_torch_fused"
    learning_rate: float = Field(gt=0)
    scheduler: Literal["cosine"] = "cosine"
    warmup_ratio: float = Field(ge=0, lt=1)
    weight_decay: float = Field(ge=0)
    max_grad_norm: float = Field(gt=0)
    per_device_train_batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    gradient_checkpointing: StrictBool
    use_cache: StrictBool
    max_sequence_length: Literal[2048, 4096]
    packing: Literal[False]
    truncation: Literal[False]
    oversize_policy: Literal["reject", "route_long_context"]
    eval_strategy: Literal["steps"]
    eval_steps: int = Field(gt=0)
    save_steps: int = Field(gt=0)
    logging_steps: int = Field(gt=0)
    save_total_limit: int = Field(gt=0)
    seed: int
    data_seed: int
    max_steps: int | None = Field(default=None, gt=0)
    epochs: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def require_one_duration(self) -> TrainingConfig:
        if (self.max_steps is None) == (self.epochs is None):
            raise ValueError("exactly one of training.max_steps or training.epochs is required")
        return self


class ResumeConfig(StrictConfigModel):
    adapter_path: str = Field(min_length=1)
    expected_base_model_revision: str = Field(min_length=1)
    expected_prompt_contract: Literal["raw-v1"]


class RunConfig(StrictConfigModel):
    schema_version: Literal[1]
    config_type: Literal["run"]
    run_name: str = Field(min_length=1)
    prompt_contract_version: Literal["raw-v1"]
    commands_mode: CommandsMode
    model: ModelConfig
    data: DataConfig
    lora: LoraConfig
    training: TrainingConfig
    resume: ResumeConfig | None = None

    @field_validator("commands_mode", mode="before")
    @classmethod
    def parse_commands_mode(cls, value: object) -> CommandsMode:
        if not isinstance(value, str):
            raise ValueError("commands_mode must be enabled, disabled, or mixed")
        return CommandsMode(value)

    @model_validator(mode="after")
    def validate_resume(self) -> RunConfig:
        if self.resume and self.training.max_sequence_length != 4096:
            raise ValueError("continuation runs must use 4096 max_sequence_length")
        if self.resume and self.resume.expected_prompt_contract != self.prompt_contract_version:
            raise ValueError("resume prompt contract must match the run prompt contract")
        return self


class SweepConfig(StrictConfigModel):
    schema_version: Literal[1]
    config_type: Literal["sweep"]
    sweep_name: str = Field(min_length=1)
    base_config: str = Field(min_length=1)
    matrix: dict[str, tuple[int, ...]]

    @model_validator(mode="before")
    @classmethod
    def normalize_matrix(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        matrix = value.get("matrix")
        if isinstance(matrix, dict):
            value = dict(value)
            value["matrix"] = {
                key: tuple(items) if isinstance(items, list) else items
                for key, items in matrix.items()
            }
        return value

    @model_validator(mode="after")
    def validate_matrix(self) -> SweepConfig:
        if set(self.matrix) != {"lora.rank", "lora.alpha"}:
            raise ValueError("sweep matrix must contain lora.rank and lora.alpha only")
        ranks = self.matrix["lora.rank"]
        alphas = self.matrix["lora.alpha"]
        if not ranks or len(ranks) != len(alphas):
            raise ValueError("rank and alpha sweep values must be non-empty and aligned")
        if any(value <= 0 for value in (*ranks, *alphas)):
            raise ValueError("LoRA sweep values must be positive")
        return self


AnyConfig = Annotated[RunConfig | SweepConfig, Field(discriminator="config_type")]


def load_config(path: str | Path) -> RunConfig | SweepConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{config_path}: expected a YAML mapping")
    config_type = payload.get("config_type")
    if config_type == "run":
        return RunConfig.model_validate(payload)
    if config_type == "sweep":
        return SweepConfig.model_validate(payload)
    raise ValueError(f"{config_path}: config_type must be 'run' or 'sweep'")


def find_config_files(root: str | Path = "configs") -> tuple[Path, ...]:
    return tuple(sorted(Path(root).rglob("*.yaml")))


def validate_config_tree(root: str | Path = "configs") -> tuple[Path, ...]:
    paths = find_config_files(root)
    if not paths:
        raise ValueError(f"no YAML configs found below {Path(root)}")
    for path in paths:
        load_config(path)
    return paths
