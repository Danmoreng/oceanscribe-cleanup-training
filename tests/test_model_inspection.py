from __future__ import annotations

import pytest

from oceanscribe_cleanup.model_inspection import (
    assert_expected_trainable_parameters,
    classify_module,
)


class FakeParameter:
    def __init__(self, requires_grad: bool) -> None:
        self.requires_grad = requires_grad


class FakeModel:
    def __init__(self, parameters: dict[str, bool]) -> None:
        self.parameters = parameters

    def named_parameters(self):  # type: ignore[no-untyped-def]
        return [(name, FakeParameter(trainable)) for name, trainable in self.parameters.items()]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("model.layers.0.self_attn.q_proj", "language_backbone"),
        ("visual.blocks.0.attn", "vision"),
        ("visual.merger.proj", "projector"),
        ("model.embed_tokens", "embeddings"),
        ("lm_head", "lm_head"),
        ("model.mtp.layers.0", "mtp"),
    ],
)
def test_module_groups(name: str, expected: str) -> None:
    assert classify_module(name) == expected


def test_only_lora_parameters_in_allowed_language_modules_may_train() -> None:
    allowed = ("model.layers.0.self_attn.q_proj",)
    valid = FakeModel(
        {
            "base_model.model.layers.0.self_attn.q_proj.base_layer.weight": False,
            "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight": True,
            "base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight": True,
        }
    )
    assert_expected_trainable_parameters(valid, allowed)

    invalid_base_weight = FakeModel(
        {"base_model.model.layers.0.self_attn.q_proj.base_layer.weight": True}
    )
    with pytest.raises(RuntimeError, match="unexpected trainable"):
        assert_expected_trainable_parameters(invalid_base_weight, allowed)

    invalid_vision = FakeModel({"visual.blocks.0.attn.q_proj.lora_A.weight": True})
    with pytest.raises(RuntimeError, match="unexpected trainable"):
        assert_expected_trainable_parameters(invalid_vision, allowed)
