"""Merge a pinned cleanup adapter and prepare weights for the native engines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .manifests import RunManifest, runtime_versions, sha256_file
from .tokenization import validate_tokenizer_contract


def native_tensor_name(name: str) -> str:
    """Translate text-only Transformers keys to the engines' language prefix."""
    if name.startswith(("model.layers.", "model.embed_tokens.", "model.norm.")):
        if "lora_" in name:
            raise ValueError("native export requires merged weights")
        return "model.language_model." + name.removeprefix("model.")
    raise ValueError(f"unexpected text-model tensor: {name}")


def native_state_dict(state: dict[str, Any]) -> dict[str, Any]:
    import torch

    embedding = state.get("model.embed_tokens.weight")
    head = state.get("lm_head.weight")
    if embedding is None or head is None or not torch.equal(embedding, head):
        raise ValueError("native 0.8B export requires a tied embedding/output matrix")
    return {
        native_tensor_name(name): tensor.detach().cpu().contiguous()
        for name, tensor in state.items() if name != "lm_head.weight"
    }


def write_native_bpe_files(directory: Path) -> None:
    """Expose existing BPE IDs/merge ranks to the native legacy tokenizer loader."""
    payload = json.loads((directory / "tokenizer.json").read_text())
    model = payload["model"]
    if model["type"] != "BPE":
        raise ValueError("native tokenizer export requires BPE")
    vocab = dict(model["vocab"])
    decoder = {}
    for token in payload["added_tokens"]:
        content, token_id = token["content"], token["id"]
        if content in vocab and vocab[content] != token_id:
            raise ValueError("added token ID conflicts with BPE vocabulary")
        vocab[content] = token_id
        decoder[str(token_id)] = {k: v for k, v in token.items() if k != "id"}
    if len(set(vocab.values())) != len(vocab):
        raise ValueError("native vocabulary contains duplicate token IDs")
    merges = []
    for merge in model["merges"]:
        parts = merge.split(" ") if isinstance(merge, str) else merge
        if len(parts) != 2 or any(not part or any(c.isspace() for c in part) for part in parts):
            raise ValueError("BPE merge cannot be represented in native merges.txt")
        merges.append(" ".join(parts))
    (directory / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False) + "\n")
    (directory / "merges.txt").write_text("#version: 0.2\n" + "\n".join(merges) + "\n")
    config_path = directory / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config["added_tokens_decoder"] = decoder
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")


def merge_adapter(adapter_path: Path, output_dir: Path) -> dict[str, Any]:
    """Write reloadable HF weights plus a distinct native-loader tensor layout."""
    from peft import PeftConfig, PeftModel
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    from .training import _load_text_model

    parent = RunManifest.model_validate_json(
        (adapter_path.parent / "run-manifest.json").read_text(encoding="utf-8")
    )
    repository = "Qwen/Qwen3.5-0.8B-Base"
    if parent.model_repository != repository or parent.prompt_contract_version != "raw-v1":
        raise ValueError("export requires the pinned Qwen 0.8B Base raw-v1 training run")
    adapter_config = PeftConfig.from_pretrained(adapter_path)
    if adapter_config.base_model_name_or_path != repository:
        raise ValueError("adapter repository differs from its run manifest")
    if output_dir.exists():
        raise ValueError("export destination already exists; choose a new artifact directory")
    tokenizer = AutoTokenizer.from_pretrained(
        repository, revision=parent.tokenizer_revision, trust_remote_code=False,
    )
    validate_tokenizer_contract(tokenizer)
    base = _load_text_model(repository, parent.model_revision)
    adapter = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    merged = adapter.merge_and_unload(safe_merge=True)
    if not merged.config.tie_word_embeddings:
        raise ValueError("native engine requires tied word embeddings")
    state = native_state_dict(merged.state_dict())
    hf_dir, native_dir = output_dir / "hf", output_dir / "native"
    hf_dir.mkdir(parents=True)
    native_dir.mkdir()
    merged.save_pretrained(hf_dir, safe_serialization=True)
    tokenizer.save_pretrained(hf_dir)
    # This second directory is for the C++ engines, not a Transformers checkpoint.
    config = merged.config.to_dict()
    config["_name_or_path"] = repository
    (native_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    tokenizer.save_pretrained(native_dir)
    write_native_bpe_files(native_dir)
    save_file(state, native_dir / "model.safetensors", metadata={"format": "pt"})
    report = {
        "schema_version": 1,
        "model_repository": repository,
        "model_revision": parent.model_revision,
        "tokenizer_revision": parent.tokenizer_revision,
        "prompt_contract_version": "raw-v1",
        "eos_token_id": tokenizer.eos_token_id,
        "adapter_path": str(adapter_path.resolve()),
        "adapter_sha256": sha256_file(adapter_path / "adapter_model.safetensors"),
        "parent_manifest_sha256": sha256_file(adapter_path.parent / "run-manifest.json"),
        "versions": runtime_versions(),
        "hf_directory": str(hf_dir.resolve()),
        "native_directory": str(native_dir.resolve()),
        "native_tensor_count": len(state),
        "native_layout": "model.* -> model.language_model.*; tied lm_head omitted",
        "quantization": None,
        "parity_verified": False,
        "files_sha256": {
            str(path.relative_to(output_dir)): sha256_file(path)
            for path in sorted(output_dir.rglob("*")) if path.is_file()
        },
    }
    (output_dir / "export-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
