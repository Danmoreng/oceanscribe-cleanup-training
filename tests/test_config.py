from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from oceanscribe_cleanup.config import CommandsMode, RunConfig, load_config, validate_config_tree


def test_all_repository_configs_validate() -> None:
    paths = validate_config_tree("configs")
    assert len(paths) == 8


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
