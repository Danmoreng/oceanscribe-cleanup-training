from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from oceanscribe_cleanup.config import CommandsMode, RunConfig, load_config, validate_config_tree


def test_all_repository_configs_validate() -> None:
    paths = validate_config_tree("configs")
    assert set(paths) == set(Path("configs").rglob("*.yaml"))


def test_enabled_remains_an_enum_not_a_yaml_bool() -> None:
    config = load_config("configs/runs/smoke-r16.yaml")
    assert isinstance(config, RunConfig)
    assert config.commands_mode is CommandsMode.ENABLED
    assert not isinstance(config.commands_mode, bool)


def test_unknown_config_fields_are_rejected(tmp_path: Path) -> None:
    source = Path("configs/runs/smoke-r16.yaml").read_text(encoding="utf-8")
    malformed = tmp_path / "unknown.yaml"
    malformed.write_text(source + "unknown_field: nope\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(malformed)


def test_yaml_on_is_not_a_valid_commands_mode(tmp_path: Path) -> None:
    source = Path("configs/runs/smoke-r16.yaml").read_text(encoding="utf-8")
    malformed = tmp_path / "boolean.yaml"
    malformed.write_text(
        source.replace("commands_mode: enabled", "commands_mode: on"),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_config(malformed)


def test_best_checkpoint_requires_aligned_save_and_eval():
    config = load_config("configs/runs/smoke-r16.yaml").model_dump(mode="json")
    config["training"].update(load_best_model_at_end=True, save_steps=3, eval_steps=2)
    with pytest.raises(ValidationError, match="save_steps divisible"):
        RunConfig.model_validate(config)


def test_stage_a_adapter_continuation_checks_revision():
    config = load_config("configs/runs/smoke-r16.yaml").model_dump(mode="json")
    config["resume"] = {"adapter_path": "local/adapter-final",
                        "expected_base_model_revision": config["model"]["revision"],
                        "expected_prompt_contract": "raw-v1"}
    assert RunConfig.model_validate(config).training.max_sequence_length == 2048
    config["resume"]["expected_base_model_revision"] = "a" * 40
    with pytest.raises(ValidationError, match="resume base revision"):
        RunConfig.model_validate(config)
