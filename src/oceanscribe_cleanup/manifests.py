"""Reproducibility and generated-dataset manifests."""

from __future__ import annotations

import hashlib
import platform
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

FLOATING_REVISIONS = {"main", "master", "latest", "HEAD"}


class ManifestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SourceManifest(ManifestModel):
    revision: str | None
    license_spdx: str | None
    allowed_use: str
    redistribution: str
    record_count: int = Field(ge=0)


class DatasetManifest(ManifestModel):
    schema_version: Literal[1] = 1
    prompt_contract_version: Literal["raw-v1"]
    sources: dict[str, SourceManifest]
    counts_by_language: dict[str, int]
    counts_by_feature: dict[str, int]
    counts_by_length_bucket: dict[str, int]
    deduplication: dict[str, int | str]
    oversize_exclusions: dict[str, int]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RunManifest(ManifestModel):
    schema_version: Literal[1] = 1
    repository_git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_repository: str
    model_revision: str
    tokenizer_revision: str
    python_version: str
    torch_version: str
    cuda_version: str | None
    transformers_version: str
    peft_version: str
    gpu_name: str | None
    gpu_vram_bytes: int | None = Field(default=None, ge=0)
    prompt_contract_version: Literal["raw-v1"]
    dataset_revisions: dict[str, str]
    random_seeds: dict[str, int]

    @field_validator("model_revision", "tokenizer_revision")
    @classmethod
    def require_resolved_revision(cls, value: str) -> str:
        if value in FLOATING_REVISIONS or not value:
            raise ValueError("real runs require a resolved, non-floating revision")
        return value


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_commit(root: str | Path = ".") -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def installed_version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not-installed"


def runtime_versions() -> dict[str, str | None]:
    cuda_version: str | None = None
    try:
        import torch

        cuda_version = torch.version.cuda
    except ImportError:
        pass
    return {
        "python_version": platform.python_version(),
        "torch_version": installed_version("torch"),
        "cuda_version": cuda_version,
        "transformers_version": installed_version("transformers"),
        "peft_version": installed_version("peft"),
    }
