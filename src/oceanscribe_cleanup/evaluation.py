"""Greedy base-versus-adapter evaluation for held-out raw-v1 cleanup records."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from .manifests import sha256_file
from .prompt_contracts import RAW_V1
from .prompting import render_example_prompt
from .quality import score_checks
from .records import CleanupRecord
from .tokenization import validate_tokenizer_contract
from .training import _load_text_model, load_cleanup_records


@dataclass(frozen=True, slots=True)
class GenerationResult:
    id: str
    source_name: str
    features: tuple[str, ...]
    transcript: str
    reference: str
    base_output: str
    adapter_output: str
    base_stopped_at_eos: bool
    adapter_stopped_at_eos: bool
    language: str = "en-US"
    terminology: tuple[str, ...] = ()
    checks: dict[str, Any] | None = None


def normalized_exact(reference: str, hypothesis: str) -> bool:
    return " ".join(reference.split()) == " ".join(hypothesis.split())


def output_has_forbidden_control(output: str) -> bool:
    return any(marker in output for marker in (*RAW_V1.reserved_markers, "<|"))


def score_outputs(results: Sequence[GenerationResult]) -> dict[str, Any]:
    from jiwer import cer, wer
    from rapidfuzz.fuzz import ratio

    if not results:
        raise ValueError("cannot score an empty result set")
    references = [result.reference for result in results]
    base_outputs = [result.base_output for result in results]
    adapter_outputs = [result.adapter_output for result in results]

    def metrics(outputs: list[str], *, adapter: bool) -> dict[str, Any]:
        eos_values = [
            result.adapter_stopped_at_eos if adapter else result.base_stopped_at_eos
            for result in results
        ]
        return {
            "cer": cer(references, outputs),
            "wer": wer(references, outputs),
            "normalized_exact_matches": sum(
                normalized_exact(reference, output)
                for reference, output in zip(references, outputs, strict=True)
            ),
            "mean_character_similarity": mean(
                ratio(reference, output) / 100
                for reference, output in zip(references, outputs, strict=True)
            ),
            "eos_stop_rate": mean(eos_values),
            "forbidden_control_outputs": sum(output_has_forbidden_control(o) for o in outputs),
            "empty_outputs": sum(not output.strip() for output in outputs),
            "quality_checks": score_checks(results, adapter=adapter),
        }

    base = metrics(base_outputs, adapter=False)
    adapter = metrics(adapter_outputs, adapter=True)
    return {
        "record_count": len(results),
        "base": base,
        "adapter": adapter,
        "delta": {
            "cer": adapter["cer"] - base["cer"],
            "wer": adapter["wer"] - base["wer"],
            "mean_character_similarity": (
                adapter["mean_character_similarity"] - base["mean_character_similarity"]
            ),
        },
    }


def _generate(
    model: Any,
    tokenizer: Any,
    records: Sequence[CleanupRecord],
    *,
    batch_size: int,
    max_new_tokens: int,
) -> tuple[list[str], list[bool]]:
    import torch

    outputs: list[str] = []
    eos_stops: list[bool] = []
    model.eval()
    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        prompts = [render_example_prompt(record) for record in batch_records]
        encoded = tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            truncation=False,
            return_tensors="pt",
        ).to(model.device)
        input_width = encoded.input_ids.shape[1]
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        new_tokens = generated[:, input_width:]
        for token_ids in new_tokens:
            ids = token_ids.tolist()
            eos_stops.append(tokenizer.eos_token_id in ids)
            if tokenizer.eos_token_id in ids:
                ids = ids[: ids.index(tokenizer.eos_token_id)]
            outputs.append(tokenizer.decode(ids, skip_special_tokens=False).strip())
    return outputs, eos_stops


def _write_markdown(results: Sequence[GenerationResult], scores: dict[str, Any]) -> str:
    lines = [
        "# Qualitative adapter evaluation",
        "",
        f"Records: {len(results)}",
        "",
        "| Model | CER | WER | Exact | Similarity | EOS | Controls |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("base", "adapter"):
        metric = scores[name]
        if metric is None:
            continue
        lines.append(
            f"| {name} | {metric['cer']:.4f} | {metric['wer']:.4f} | "
            f"{metric['normalized_exact_matches']} | "
            f"{metric['mean_character_similarity']:.4f} | "
            f"{metric['eos_stop_rate']:.1%} | {metric['forbidden_control_outputs']} |"
        )
    lines.extend(["", "## Outputs", ""])
    for index, result in enumerate(results, start=1):
        lines.extend(
            [
                f"### {index}. {result.source_name} / {result.id}",
                "",
                f"Features: {', '.join(result.features)}",
                "",
                "**Transcript**",
                "",
                result.transcript,
                "",
                "**Reference**",
                "",
                result.reference,
                "",
                *(["**Base**", "", result.base_output or "*(empty)*", ""]
                  if scores["base"] is not None else []),
                "**Adapter**",
                "",
                result.adapter_output or "*(empty)*",
                "",
            ]
        )
    return "\n".join(lines)


def evaluate_adapter(
    *,
    model_repository: str,
    revision: str,
    adapter_path: str | Path,
    records_path: str | Path,
    output_path: str | Path,
    limit: int = 40,
    batch_size: int = 4,
    max_new_tokens: int = 256,
    split: str = "validation",
    compare_base: bool = True,
) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_repository, revision=revision, trust_remote_code=False
    )
    validate_tokenizer_contract(tokenizer)
    tokenizer.padding_side = "left"
    from collections import defaultdict

    if split not in {"validation", "test"}:
        raise ValueError("evaluation split must be validation or test")
    groups = defaultdict(list)
    for record in sorted(load_cleanup_records(records_path), key=lambda r: r.id):
        if record.split == split:
            groups[(record.language, record.source_name)].append(record)
    validation = []
    while len(validation) < limit and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(validation) < limit:
                validation.append(groups[key].pop(0))
    if not validation:
        raise ValueError(f"dataset has no {split} records")
    checks_path = Path(records_path).with_name("checks.json")
    checks = json.loads(checks_path.read_text()) if checks_path.exists() else {}

    base_model = _load_text_model(model_repository, revision).to("cuda")
    base_outputs, base_eos = _generate(
        base_model,
        tokenizer,
        validation,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    ) if compare_base else ([""] * len(validation), [False] * len(validation))
    adapter_model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
    adapter_outputs, adapter_eos = _generate(
        adapter_model,
        tokenizer,
        validation,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
    results = [
        GenerationResult(
            id=record.id,
            source_name=record.source_name,
            features=record.features,
            transcript=record.transcript,
            reference=record.output,
            base_output=base_output,
            adapter_output=adapter_output,
            base_stopped_at_eos=base_stop,
            adapter_stopped_at_eos=adapter_stop,
            language=record.language,
            terminology=record.terminology,
            checks=checks.get(record.id),
        )
        for record, base_output, adapter_output, base_stop, adapter_stop in zip(
            validation, base_outputs, adapter_outputs, base_eos, adapter_eos, strict=True
        )
    ]
    scores = score_outputs(results)
    by_language = {
        language: score_outputs([r for r in results if r.language == language])
        for language in sorted({r.language for r in results})
    }
    if not compare_base:
        for score in [scores, *by_language.values()]:
            score["base"] = None
            score["delta"] = None
    payload = {
        "model_repository": model_repository,
        "revision": revision,
        "adapter_path": str(adapter_path),
        "dataset_sha256": sha256_file(records_path),
        "checks_sha256": sha256_file(checks_path) if checks_path.exists() else None,
        "split": split,
        "base_evaluated": compare_base,
        "decoding": {
            "do_sample": False,
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "chat_template": False,
        },
        "scores": scores,
        "scores_by_language": by_language,
        "results": [asdict(result) for result in results],
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    destination.with_suffix(".md").write_text(_write_markdown(results, scores) + "\n")
    return payload
