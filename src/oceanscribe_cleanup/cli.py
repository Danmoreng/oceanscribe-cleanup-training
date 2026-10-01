"""Local data preparation, model inspection, training, evaluation and export CLI."""

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


@app.command("prepare-external-smoke")
def prepare_external_smoke(
    revision: Annotated[str, typer.Option("--revision", help="Pinned model/tokenizer revision")],
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path(
        "data/processed/smoke-en-400"
    ),
    per_source: Annotated[int, typer.Option("--per-source", min=1)] = 200,
    validation_per_source: Annotated[
        int, typer.Option("--validation-per-source", min=1)
    ] = 20,
    seed: Annotated[int, typer.Option("--seed")] = 42,
) -> None:
    """Select pinned, deduplicated Sotto and conservative Aawaaz smoke records."""
    from transformers import AutoTokenizer

    from .data_prep import (
        load_aawaaz_pairs,
        load_sotto_pairs,
        select_records,
        write_dataset,
    )

    if validation_per_source >= per_source:
        raise typer.BadParameter("validation-per-source must be smaller than per-source")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3.5-0.8B-Base", revision=revision
    )
    train_count = per_source - validation_per_source
    records = []
    stats = {}
    for name, pairs in (
        ("sotto", load_sotto_pairs(Path("data/raw/sotto"))),
        ("aawaaz", load_aawaaz_pairs(Path("data/raw/aawaaz"))),
    ):
        selected, source_stats = select_records(
            pairs,
            tokenizer,
            source_name=name,
            train_count=train_count,
            validation_count=validation_per_source,
            seed=seed,
        )
        records.extend(selected)
        stats[name] = source_stats
    records.sort(key=lambda record: (record.split, record.source_name, record.id))
    records_path, manifest_path = write_dataset(records, stats, output_dir)
    typer.echo(f"Wrote {len(records)} records to {records_path}")
    typer.echo(f"Wrote manifest to {manifest_path}")


@app.command("prepare-reviewed-smoke")
def prepare_reviewed_smoke_command(
    csv_path: Annotated[Path, typer.Option("--csv", exists=True, dir_okay=False)],
    revision: Annotated[str, typer.Option("--revision")],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    per_source: Annotated[int, typer.Option("--per-source", min=1)] = 200,
    validation_per_source: Annotated[int, typer.Option("--validation-per-source", min=1)] = 20,
    seed: Annotated[int, typer.Option("--seed")] = 42,
) -> None:
    """Freeze unchanged English KEEP references with source-family holdouts."""
    from transformers import AutoTokenizer

    from .reviewed_csv import prepare_reviewed_smoke

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3.5-0.8B-Base", revision=revision, trust_remote_code=False
    )
    records, manifest = prepare_reviewed_smoke(
        csv_path, tokenizer, output_dir, per_source=per_source,
        validation_per_source=validation_per_source, seed=seed,
    )
    typer.echo(f"Wrote records to {records}; manifest to {manifest}")


@app.command("train")
def train(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    dataset: Annotated[Path, typer.Option("--dataset", exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
) -> None:
    """Run one pinned BF16 LoRA training configuration and verify adapter reload."""
    from .training import train_run

    try:
        report = train_run(config, dataset, output_dir)
    except Exception as error:
        typer.echo(f"Training failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


@app.command("evaluate-adapter")
def evaluate_adapter_command(
    revision: Annotated[str, typer.Option("--revision")],
    adapter: Annotated[Path, typer.Option("--adapter", exists=True, file_okay=False)],
    dataset: Annotated[Path, typer.Option("--dataset", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output")],
    limit: Annotated[int, typer.Option("--limit", min=1)] = 40,
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 4,
    max_new_tokens: Annotated[int, typer.Option("--max-new-tokens", min=1)] = 256,
    split: Annotated[str, typer.Option("--split")] = "validation",
    adapter_only: Annotated[bool, typer.Option("--adapter-only")] = False,
) -> None:
    """Greedily compare the pinned base model and one adapter on validation records."""
    from .evaluation import evaluate_adapter

    payload = evaluate_adapter(
        model_repository="Qwen/Qwen3.5-0.8B-Base",
        revision=revision,
        adapter_path=adapter,
        records_path=dataset,
        output_path=output,
        limit=limit,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        split=split,
        compare_base=not adapter_only,
    )
    typer.echo(json.dumps(payload["scores"], indent=2, sort_keys=True))


@app.command("merge-adapter")
def merge_adapter_command(
    adapter: Annotated[Path, typer.Option("--adapter", exists=True, file_okay=False)],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
) -> None:
    """Merge Base + LoRA into HF and native-engine BF16 weight directories."""
    from .native_export import merge_adapter

    report = merge_adapter(adapter, output_dir)
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


@app.command("prepare-targeted")
def prepare_targeted_command(
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    seed: Annotated[int, typer.Option("--seed")] = 42,
) -> None:
    """Generate original bilingual templates with separate challenge families."""
    from .targeted_data import prepare_targeted

    typer.echo(json.dumps(prepare_targeted(output_dir, seed), indent=2))


if __name__ == "__main__":
    app()
