"""Small, explicit BF16 LoRA training pipeline for raw-v1 cleanup records."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from .collator import CompletionOnlyCollator
from .config import RunConfig, load_config
from .manifests import RunManifest, repository_commit, runtime_versions, sha256_file
from .model_inspection import (
    assert_expected_trainable_parameters,
    discover_language_linear_modules,
    resolve_hugging_face_revision,
)
from .records import CleanupRecord
from .tokenization import TokenizedExample, tokenize_example, validate_tokenizer_contract


@dataclass(frozen=True, slots=True)
class DatasetStatistics:
    total_records: int
    train_records: int
    validation_records: int
    records_by_source: dict[str, int]
    average_prompt_tokens: float
    average_target_tokens: float
    average_sequence_tokens: float
    maximum_sequence_tokens: int
    non_padding_tokens_per_optimizer_step: float


class TokenizedCleanupDataset:
    def __init__(self, examples: Sequence[TokenizedExample]) -> None:
        self._examples = tuple(examples)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> TokenizedExample:
        return self._examples[index]


def load_cleanup_records(path: str | Path) -> tuple[CleanupRecord, ...]:
    records: list[CleanupRecord] = []
    seen_ids: set[str] = set()
    seen_pairs: set[tuple[Any, ...]] = set()
    family_splits: dict[tuple[str, str | None, str], str] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = CleanupRecord.model_validate(json.loads(line))
            except (json.JSONDecodeError, ValueError, TypeError) as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid cleanup record: {error}"
                ) from error
            if record.id in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate record id {record.id!r}")
            pair = (
                record.language, record.commands, record.terminology,
                record.transcript.casefold(), record.output.casefold(),
            )
            if pair in seen_pairs:
                raise ValueError(f"{path}:{line_number}: duplicate transcript/output pair")
            family = (record.source_name, record.source_revision,
                      record.source_record_id or record.parent_id or record.id)
            if family in family_splits and family_splits[family] != record.split:
                raise ValueError(f"{path}:{line_number}: source family crosses split boundaries")
            family_splits[family] = record.split
            seen_ids.add(record.id)
            seen_pairs.add(pair)
            records.append(record)
    if not records:
        raise ValueError(f"{path}: dataset is empty")
    return tuple(records)


def validate_dataset_manifest(records_path: Path, manifest_path: Path) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = payload.get("sha256")
    actual = sha256_file(records_path)
    if expected != actual:
        raise ValueError(
            f"dataset SHA-256 mismatch: manifest has {expected!r}, records have {actual!r}"
        )
    return payload


def validate_dataset_against_config(records: Sequence[CleanupRecord], config: RunConfig) -> None:
    expected = {
        "sotto": config.data.sotto_en or 0,
        "aawaaz": config.data.aawaaz_en or 0,
    }
    actual = Counter(record.source_name for record in records if not record.synthetic)
    for source, count in expected.items():
        if actual[source] != count:
            raise ValueError(
                f"config expects {count} {source} records, dataset contains {actual[source]}"
            )
    configured_total = sum(
        value or 0
        for value in (
            config.data.sotto_en,
            config.data.aawaaz_en,
            config.data.synthetic_en,
            config.data.synthetic_de,
        )
    )
    if len(records) != configured_total:
        raise ValueError(
            f"config expects {configured_total} total records, dataset contains {len(records)}"
        )
    for locale, expected_count in (("en", config.data.synthetic_en),
                                   ("de", config.data.synthetic_de)):
        actual_count = sum(r.synthetic and r.language.startswith(locale) for r in records)
        if expected_count is not None and actual_count != expected_count:
            raise ValueError(
                f"config expects {expected_count} synthetic {locale}, got {actual_count}"
            )
    if config.commands_mode.value == "disabled" and any(record.commands for record in records):
        raise ValueError("disabled commands_mode requires commands=false on every record")
    if config.commands_mode.value == "enabled" and not all(record.commands for record in records):
        raise ValueError("enabled commands_mode requires commands=true on every record")
    if not any(record.split == "train" for record in records):
        raise ValueError("dataset has no training records")
    if not any(record.split == "validation" for record in records):
        raise ValueError("dataset has no validation records")
    if any(record.split == "test" for record in records):
        raise ValueError("gold test records must not enter a training run")
    train = [r for r in records if r.split == "train"]
    for feature, minimum in (("preserve", config.data.preserve_minimum),
                             ("hard-negative", config.data.hard_negative_minimum)):
        if minimum is not None and sum(feature in r.features for r in train) / len(train) < minimum:
            raise ValueError(f"training records do not meet the {feature} minimum of {minimum}")


def language_sampling_weights(
    records: Sequence[CleanupRecord], examples: Sequence[TokenizedExample], mode: str,
) -> list[float]:
    """Equalize expected language tokens or records using replacement sampling."""
    if len(records) != len(examples) or not records:
        raise ValueError("sampling needs matching nonempty records and tokenized examples")
    totals: Counter[str] = Counter()
    for record, example in zip(records, examples, strict=True):
        totals[record.language.split("-")[0]] += example.sequence_length if mode == "tokens" else 1
    return [1 / totals[r.language.split("-")[0]] for r in records]


def tokenize_records(
    records: Sequence[CleanupRecord],
    tokenizer: Any,
    *,
    max_sequence_length: int,
    gradient_accumulation_steps: int,
    per_device_train_batch_size: int = 1,
) -> tuple[TokenizedCleanupDataset, TokenizedCleanupDataset, DatasetStatistics]:
    validate_tokenizer_contract(tokenizer)
    train: list[TokenizedExample] = []
    validation: list[TokenizedExample] = []
    all_examples: list[TokenizedExample] = []
    for record in records:
        example = tokenize_example(
            record,
            tokenizer,
            max_sequence_length=max_sequence_length,
        )
        all_examples.append(example)
        if record.split == "train":
            train.append(example)
        elif record.split == "validation":
            validation.append(example)

    average_sequence = mean(example.sequence_length for example in all_examples)
    statistics = DatasetStatistics(
        total_records=len(records),
        train_records=len(train),
        validation_records=len(validation),
        records_by_source=dict(sorted(Counter(record.source_name for record in records).items())),
        average_prompt_tokens=mean(example.prompt_length for example in all_examples),
        average_target_tokens=mean(example.target_length for example in all_examples),
        average_sequence_tokens=average_sequence,
        maximum_sequence_tokens=max(example.sequence_length for example in all_examples),
        non_padding_tokens_per_optimizer_step=(
            mean(example.sequence_length for example in train)
            * gradient_accumulation_steps
            * per_device_train_batch_size
        ),
    )
    return TokenizedCleanupDataset(train), TokenizedCleanupDataset(validation), statistics


def build_run_manifest(
    config: RunConfig,
    *,
    config_path: Path,
    resolved_revision: str,
    dataset_manifest: dict[str, Any],
) -> RunManifest:
    import torch

    versions = runtime_versions()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    gpu_vram = (
        torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
    )
    dataset_revisions = {
        source: details["revision"] for source, details in dataset_manifest["sources"].items()
    }
    return RunManifest(
        repository_git_commit=repository_commit(),
        config_sha256=sha256_file(config_path),
        model_repository=config.model.id,
        model_revision=resolved_revision,
        tokenizer_revision=resolved_revision,
        python_version=str(versions["python_version"]),
        torch_version=str(versions["torch_version"]),
        cuda_version=versions["cuda_version"],
        transformers_version=str(versions["transformers_version"]),
        peft_version=str(versions["peft_version"]),
        gpu_name=gpu_name,
        gpu_vram_bytes=gpu_vram,
        prompt_contract_version=config.prompt_contract_version,
        dataset_revisions=dataset_revisions,
        random_seeds={"training": config.training.seed, "data": config.training.data_seed},
    )


def _load_text_model(model_repository: str, revision: str) -> Any:
    import torch
    import transformers

    model_class = getattr(transformers, "Qwen3_5ForCausalLM", None)
    if model_class is None:
        raise RuntimeError(
            "Qwen3_5ForCausalLM is unavailable; refusing multimodal training fallback"
        )
    return model_class.from_pretrained(
        model_repository,
        revision=revision,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )


def _inject_lora(model: Any, config: RunConfig) -> tuple[Any, tuple[str, ...]]:
    from peft import LoraConfig, TaskType, get_peft_model

    model.requires_grad_(False)
    allowlist = discover_language_linear_modules(model)
    if not allowlist:
        raise RuntimeError("model inspection found no language Linear modules for LoRA")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=list(allowlist),
        bias="none",
        modules_to_save=None,
    )
    model = get_peft_model(model, lora_config)
    assert_expected_trainable_parameters(model, allowlist)
    return model, allowlist


def _load_training_adapter(
    model: Any, config: RunConfig, resolved_revision: str,
) -> tuple[Any, tuple[str, ...]]:
    """Continue a saved adapter as a new run, with a fresh optimizer/scheduler."""
    from peft import PeftConfig, PeftModel

    assert config.resume is not None
    resume = config.resume
    if resume.expected_base_model_revision != resolved_revision:
        raise ValueError("continuation base model revision mismatch")
    adapter_path = Path(resume.adapter_path)
    adapter_config = PeftConfig.from_pretrained(adapter_path)
    if adapter_config.base_model_name_or_path != config.model.id:
        raise ValueError("saved adapter base model repository mismatch")
    if (adapter_config.r, adapter_config.lora_alpha) != (config.lora.rank, config.lora.alpha):
        raise ValueError("saved adapter rank/alpha mismatch")
    if adapter_config.lora_dropout != config.lora.dropout:
        raise ValueError("saved adapter dropout mismatch")
    parent_manifest = adapter_path.parent / "run-manifest.json"
    if not parent_manifest.is_file():
        raise ValueError("continuation requires the parent run-manifest.json beside the adapter")
    parent = RunManifest.model_validate(json.loads(parent_manifest.read_text()))
    if parent.model_revision != resolved_revision:
        raise ValueError("saved adapter parent model revision mismatch")
    if parent.prompt_contract_version != resume.expected_prompt_contract:
        raise ValueError("saved adapter prompt contract mismatch")
    model.requires_grad_(False)
    allowlist = discover_language_linear_modules(model)
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    assert_expected_trainable_parameters(model, allowlist)
    return model, allowlist


def train_run(
    config_path: str | Path,
    records_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed

    config_file = Path(config_path)
    records_file = Path(records_path)
    run_dir = Path(output_dir)
    loaded_config = load_config(config_file)
    if not isinstance(loaded_config, RunConfig):
        raise TypeError("training requires a concrete run config, not a sweep config")
    config = loaded_config
    if config.model.revision is None:
        raise ValueError("training requires a pinned model revision in the run config")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the BF16 Stage A training path")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected CUDA device does not support BF16")

    resolved_revision = resolve_hugging_face_revision(config.model.id, config.model.revision)
    records = load_cleanup_records(records_file)
    validate_dataset_against_config(records, config)
    dataset_manifest_path = records_file.with_name("manifest.json")
    dataset_manifest = validate_dataset_manifest(records_file, dataset_manifest_path)
    set_seed(config.training.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.id,
        revision=resolved_revision,
        trust_remote_code=False,
    )
    train_dataset, eval_dataset, dataset_statistics = tokenize_records(
        records,
        tokenizer,
        max_sequence_length=config.training.max_sequence_length,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
    )

    model = _load_text_model(config.model.id, resolved_revision)
    model.config.use_cache = config.training.use_cache
    model, allowlist = (
        _load_training_adapter(model, config, resolved_revision)
        if config.resume else _inject_lora(model, config)
    )
    if config.training.gradient_checkpointing:
        model.enable_input_require_grads()

    run_dir.mkdir(parents=True, exist_ok=False)
    run_manifest = build_run_manifest(
        config,
        config_path=config_file,
        resolved_revision=resolved_revision,
        dataset_manifest=dataset_manifest,
    )
    (run_dir / "run-manifest.json").write_text(
        json.dumps(run_manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "effective-config.yaml").write_bytes(config_file.read_bytes())
    (run_dir / "dataset-manifest.json").write_text(
        json.dumps(dataset_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_dir / "dataset-statistics.json").write_text(
        json.dumps(asdict(dataset_statistics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "lora-targets.json").write_text(
        json.dumps(list(allowlist), indent=2) + "\n", encoding="utf-8"
    )

    training = config.training
    arguments = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        per_device_train_batch_size=training.per_device_train_batch_size,
        per_device_eval_batch_size=training.per_device_train_batch_size,
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        max_steps=training.max_steps if training.max_steps is not None else -1,
        num_train_epochs=float(training.epochs or 1),
        learning_rate=training.learning_rate,
        lr_scheduler_type=training.scheduler,
        warmup_steps=training.warmup_ratio,
        optim=training.optimizer,
        weight_decay=training.weight_decay,
        max_grad_norm=training.max_grad_norm,
        bf16=True,
        bf16_full_eval=True,
        tf32=True,
        gradient_checkpointing=training.gradient_checkpointing,
        use_cache=training.use_cache,
        eval_strategy=training.eval_strategy,
        eval_steps=training.eval_steps,
        save_strategy="steps",
        save_steps=training.save_steps,
        load_best_model_at_end=training.load_best_model_at_end,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_strategy="steps",
        logging_steps=training.logging_steps,
        logging_first_step=True,
        save_total_limit=training.save_total_limit,
        seed=training.seed,
        data_seed=training.data_seed,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        include_num_input_tokens_seen="non_padding",
    )
    assert tokenizer.pad_token_id is not None
    trainer_class = Trainer
    if config.data.balance_by is not None:
        if arguments.world_size != 1:
            raise ValueError("language balancing currently supports a single training rank")
        train_records = [r for r in records if r.split == "train"]
        weights = language_sampling_weights(
            train_records, train_dataset._examples, config.data.balance_by
        )
        (run_dir / "sampling.json").write_text(json.dumps({
            "mode": config.data.balance_by, "replacement": True,
            "seed": training.data_seed,
            "weights": dict(zip((r.id for r in train_records), weights, strict=True)),
        }, indent=2) + "\n", encoding="utf-8")

        class BalancedTrainer(Trainer):
            def _get_train_sampler(self, train_dataset=None):
                from torch.utils.data import WeightedRandomSampler

                generator = torch.Generator().manual_seed(
                    training.data_seed + int(self.state.epoch or 0)
                )
                return WeightedRandomSampler(weights, len(weights), replacement=True,
                                             generator=generator)

        trainer_class = BalancedTrainer
    trainer = trainer_class(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CompletionOnlyCollator(
            pad_token_id=tokenizer.pad_token_id,
            pad_to_multiple_of=8,
        ),
        processing_class=tokenizer,
    )
    initial_metrics = trainer.evaluate(metric_key_prefix="initial_eval")
    train_result = trainer.train()
    final_metrics = trainer.evaluate(metric_key_prefix="final_eval")

    adapter_dir = run_dir / "adapter-final"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    trainable_parameters = model.get_nb_trainable_parameters()
    base_model = model.unload()
    reloaded = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=False)
    if any(parameter.requires_grad for parameter in reloaded.parameters()):
        raise RuntimeError("reloaded inference adapter unexpectedly has trainable parameters")

    tokens_seen = train_result.metrics.get("num_input_tokens_seen", 0)
    train_runtime = train_result.metrics.get("train_runtime", 0)
    report = {
        "initial_metrics": initial_metrics,
        "train_metrics": train_result.metrics,
        "final_metrics": final_metrics,
        "adapter_dir": str(adapter_dir),
        "adapter_reload_verified": True,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "adapter_is_best_checkpoint": training.load_best_model_at_end,
        "continued_from_adapter": config.resume.adapter_path if config.resume else None,
        "continuation_optimizer": "fresh" if config.resume else None,
        "trainable_parameters": trainable_parameters,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "observed_non_padding_tokens_per_second": (
            tokens_seen / train_runtime if train_runtime else None
        ),
    }
    (run_dir / "training-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
