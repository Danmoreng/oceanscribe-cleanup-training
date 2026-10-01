from __future__ import annotations

import json

import pytest

from oceanscribe_cleanup.native_export import (
    native_state_dict,
    native_tensor_name,
    write_native_bpe_files,
)


def test_native_export_only_accepts_merged_language_tensors():
    assert native_tensor_name("model.layers.1.mlp.up_proj.weight") == (
        "model.language_model.layers.1.mlp.up_proj.weight"
    )
    for name in ("model.visual.weight", "lm_head.weight",
                 "model.layers.0.mlp.up_proj.lora_A.default.weight"):
        with pytest.raises(ValueError):
            native_tensor_name(name)


def test_native_export_preserves_values_and_rejects_untied_head():
    torch = pytest.importorskip("torch", reason="optional native tensor export integration")

    embedding = torch.tensor([[1., 2.], [3., 4.]])
    state = {"model.embed_tokens.weight": embedding,
             "lm_head.weight": embedding.clone(),
             "model.norm.weight": torch.tensor([5., 6.])}
    mapped = native_state_dict(state)
    assert set(mapped) == {"model.language_model.embed_tokens.weight",
                           "model.language_model.norm.weight"}
    assert torch.equal(mapped["model.language_model.embed_tokens.weight"], embedding)
    state["lm_head.weight"][0, 0] = 9
    with pytest.raises(ValueError, match="tied embedding"):
        native_state_dict(state)


def test_native_bpe_preserves_special_ids_and_merge_order(tmp_path):
    payload = {"model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2},
                         "merges": [["a", "b"]]},
               "added_tokens": [{"content": "<|endoftext|>", "id": 3, "special": True}]}
    (tmp_path / "tokenizer.json").write_text(json.dumps(payload))
    (tmp_path / "tokenizer_config.json").write_text('{}')
    write_native_bpe_files(tmp_path)
    assert json.loads((tmp_path / "vocab.json").read_text()) == {
        "a": 0, "b": 1, "ab": 2, "<|endoftext|>": 3,
    }
    assert (tmp_path / "merges.txt").read_text().splitlines() == ["#version: 0.2", "a b"]
    config = json.loads((tmp_path / "tokenizer_config.json").read_text())
    assert config["added_tokens_decoder"]["3"]["content"] == "<|endoftext|>"
    payload["model"]["vocab"]["<|endoftext|>"] = 2
    (tmp_path / "tokenizer.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="conflicts"):
        write_native_bpe_files(tmp_path)
