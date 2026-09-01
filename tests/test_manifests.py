from __future__ import annotations

import pytest
from pydantic import ValidationError

from oceanscribe_cleanup.manifests import RunManifest


def manifest(**overrides: object) -> RunManifest:
    values: dict[str, object] = {
        "repository_git_commit": "a" * 40,
        "config_sha256": "b" * 64,
        "model_repository": "Qwen/Qwen3.5-0.8B-Base",
        "model_revision": "c" * 40,
        "tokenizer_revision": "c" * 40,
        "python_version": "3.11.9",
        "torch_version": "2.10.0",
        "cuda_version": "13.0",
        "transformers_version": "5.2.0",
        "peft_version": "0.17.0",
        "gpu_name": "fixture",
        "gpu_vram_bytes": 1,
        "prompt_contract_version": "raw-v1",
        "dataset_revisions": {"fixture": "d" * 40},
        "random_seeds": {"training": 42, "data": 42},
    }
    values.update(overrides)
    return RunManifest.model_validate(values)


@pytest.mark.parametrize("revision", ["", "main", "master", "latest", "HEAD"])
def test_floating_model_revisions_are_rejected(revision: str) -> None:
    with pytest.raises(ValidationError):
        manifest(model_revision=revision)


def test_resolved_run_manifest_is_immutable() -> None:
    value = manifest()
    with pytest.raises(ValidationError):
        value.model_revision = "d" * 40  # type: ignore[misc]
