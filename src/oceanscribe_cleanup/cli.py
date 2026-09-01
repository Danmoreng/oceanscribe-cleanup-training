"""Preflight commands that do not start data generation or training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from .config import validate_config_tree

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.command("validate-configs")
def validate_configs(
    root: Annotated[Path, typer.Option("--root", help="Configuration tree to validate")] = Path(
        "configs"
    ),
) -> None:
    """Load every YAML config through its strict versioned schema."""
    try:
        paths = validate_config_tree(root)
    except Exception as error:
        typer.echo(f"Configuration validation failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    for path in paths:
        typer.echo(f"ok  {path}")
    typer.echo(f"Validated {len(paths)} configuration(s).")


@app.command("tokenizer-preflight")
def tokenizer_preflight(
    revision: Annotated[str, typer.Option("--revision", help="Pinned tag or commit SHA")],
    model_repository: Annotated[
        str, typer.Option("--model", help="Hugging Face model repository")
    ] = "Qwen/Qwen3.5-0.8B-Base",
) -> None:
    """Download the pinned tokenizer and report its raw-v1 token contract."""
    from .model_inspection import load_tokenizer_report, tokenizer_report_dict

    _, report = load_tokenizer_report(model_repository, revision)
    typer.echo(json.dumps(tokenizer_report_dict(report), indent=2, sort_keys=True))


@app.command("inspect-model")
def inspect_model(
    revision: Annotated[str, typer.Option("--revision", help="Pinned tag or commit SHA")],
    model_repository: Annotated[
        str, typer.Option("--model", help="Hugging Face model repository")
    ] = "Qwen/Qwen3.5-0.8B-Base",
) -> None:
    """Load text-only Qwen first, then report module groups and LoRA allowlist."""
    from .model_inspection import load_model_for_inspection

    _, report = load_model_for_inspection(model_repository, revision)
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
