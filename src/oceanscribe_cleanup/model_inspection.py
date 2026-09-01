"""Manual Qwen preflight and LoRA-scope inspection helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from .prompt_contracts import RAW_V1
from .tokenization import validate_tokenizer_contract

ModuleGroup = Literal["language_backbone", "vision", "projector", "embeddings", "lm_head", "mtp"]

FORBIDDEN_NAME_PARTS = {
    "vision": ("vision", "visual"),
    "projector": ("projector", "multi_modal_projector", "merger"),
    "embeddings": ("embed_tokens", "embedding"),
    "lm_head": ("lm_head",),
    "mtp": ("mtp", "multi_token", "draft_model"),
}


@dataclass(frozen=True, slots=True)
class TokenizerReport:
    model_repository: str
    resolved_revision: str
    tokenizer_class: str
    eos_token: str | None
    eos_token_id: int | None
    pad_token: str | None
    pad_token_id: int | None
    model_max_length: int
    prompt_contract_version: str


def resolve_hugging_face_revision(model_repository: str, revision: str) -> str:
    if not revision or revision in {"main", "master", "latest", "HEAD"}:
        raise ValueError("--revision must be a pinned tag or commit, not a floating ref")
    from huggingface_hub import HfApi

    info = HfApi().model_info(model_repository, revision=revision)
    if not info.sha:
        raise RuntimeError("Hugging Face did not return a resolved repository SHA")
    return info.sha


def load_tokenizer_report(model_repository: str, revision: str) -> tuple[Any, TokenizerReport]:
    from transformers import AutoTokenizer

    resolved_revision = resolve_hugging_face_revision(model_repository, revision)
    tokenizer = AutoTokenizer.from_pretrained(
        model_repository,
        revision=resolved_revision,
        trust_remote_code=False,
    )
    validate_tokenizer_contract(tokenizer)
    report = TokenizerReport(
        model_repository=model_repository,
        resolved_revision=resolved_revision,
        tokenizer_class=type(tokenizer).__name__,
        eos_token=tokenizer.eos_token,
        eos_token_id=tokenizer.eos_token_id,
        pad_token=tokenizer.pad_token,
        pad_token_id=tokenizer.pad_token_id,
        model_max_length=tokenizer.model_max_length,
        prompt_contract_version=RAW_V1.version,
    )
    return tokenizer, report


def classify_module(name: str) -> ModuleGroup:
    lowered = name.lower()
    for group in ("projector", "vision", "embeddings", "lm_head", "mtp"):
        if any(part in lowered for part in FORBIDDEN_NAME_PARTS[group]):
            return group  # type: ignore[return-value]
    return "language_backbone"


def module_parameter_counts(model: Any) -> dict[str, int]:
    counts = {group: 0 for group in (*FORBIDDEN_NAME_PARTS, "language_backbone")}
    for name, parameter in model.named_parameters():
        counts[classify_module(name)] += parameter.numel()
    return counts


def discover_language_linear_modules(model: Any) -> tuple[str, ...]:
    import torch

    names = (
        name
        for name, module in model.named_modules()
        if name
        and isinstance(module, torch.nn.Linear)
        and classify_module(name) == "language_backbone"
    )
    return tuple(sorted(names))


def assert_expected_trainable_parameters(
    model: Any,
    allowed_linear_modules: tuple[str, ...],
) -> None:
    unexpected: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = classify_module(name)
        belongs_to_allowed_module = any(
            f"{module_name}." in name or name.startswith(f"{module_name}.")
            for module_name in allowed_linear_modules
        )
        is_lora_parameter = ".lora_" in name or name.startswith("lora_")
        if group != "language_backbone" or not belongs_to_allowed_module or not is_lora_parameter:
            unexpected.append(name)
    if unexpected:
        details = "\n".join(f"- {name}" for name in unexpected)
        raise RuntimeError(f"unexpected trainable parameters:\n{details}")


def load_model_for_inspection(model_repository: str, revision: str) -> tuple[Any, dict[str, Any]]:
    import transformers

    resolved_revision = resolve_hugging_face_revision(model_repository, revision)
    common = {
        "revision": resolved_revision,
        "trust_remote_code": False,
        "torch_dtype": "auto",
        "low_cpu_mem_usage": True,
    }
    text_error: str | None = None
    text_class = getattr(transformers, "Qwen3_5ForCausalLM", None)
    if text_class is not None:
        try:
            model = text_class.from_pretrained(model_repository, **common)
            load_path = "Qwen3_5ForCausalLM"
        except (OSError, ValueError) as error:
            text_error = f"{type(error).__name__}: {error}"
            model = None
    else:
        text_error = "Qwen3_5ForCausalLM is unavailable in this Transformers build"
        model = None

    if model is None:
        multimodal_class = getattr(transformers, "Qwen3_5ForConditionalGeneration", None)
        if multimodal_class is None:
            raise RuntimeError(
                f"text-only load failed ({text_error}); multimodal class is unavailable"
            )
        model = multimodal_class.from_pretrained(model_repository, **common)
        load_path = "Qwen3_5ForConditionalGeneration"

    allowlist = discover_language_linear_modules(model)
    report = {
        "model_repository": model_repository,
        "resolved_revision": resolved_revision,
        "load_path": load_path,
        "text_only_load_error": text_error,
        "parameter_counts": module_parameter_counts(model),
        "language_linear_lora_allowlist": allowlist,
    }
    return model, report


def tokenizer_report_dict(report: TokenizerReport) -> dict[str, Any]:
    return asdict(report)
