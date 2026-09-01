"""Position-aware padding for completion-only language-model training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .tokenization import IGNORE_INDEX, TokenizedExample


def _as_feature(feature: TokenizedExample | Mapping[str, Sequence[int]]) -> dict[str, list[int]]:
    if isinstance(feature, TokenizedExample):
        return {
            "input_ids": list(feature.input_ids),
            "labels": list(feature.labels),
            "attention_mask": list(feature.attention_mask),
        }
    return {
        "input_ids": list(feature["input_ids"]),
        "labels": list(feature["labels"]),
        "attention_mask": list(feature.get("attention_mask", [1] * len(feature["input_ids"]))),
    }


def pad_features(
    features: Sequence[TokenizedExample | Mapping[str, Sequence[int]]],
    *,
    pad_token_id: int,
    pad_to_multiple_of: int | None = None,
) -> dict[str, list[list[int]]]:
    if not features:
        raise ValueError("cannot collate an empty batch")
    rows = [_as_feature(feature) for feature in features]
    for row in rows:
        if not (len(row["input_ids"]) == len(row["labels"]) == len(row["attention_mask"])):
            raise ValueError("input_ids, labels and attention_mask lengths must match")

    target_length = max(len(row["input_ids"]) for row in rows)
    if pad_to_multiple_of is not None:
        if pad_to_multiple_of <= 0:
            raise ValueError("pad_to_multiple_of must be positive")
        remainder = target_length % pad_to_multiple_of
        if remainder:
            target_length += pad_to_multiple_of - remainder

    batch = {"input_ids": [], "labels": [], "attention_mask": []}
    for row in rows:
        padding = target_length - len(row["input_ids"])
        batch["input_ids"].append(row["input_ids"] + [pad_token_id] * padding)
        # Padding is masked by position. It is intentionally not inferred by
        # comparing IDs because Qwen may use one ID for both PAD and EOS.
        batch["labels"].append(row["labels"] + [IGNORE_INDEX] * padding)
        batch["attention_mask"].append(row["attention_mask"] + [0] * padding)
    return batch


@dataclass(frozen=True, slots=True)
class CompletionOnlyCollator:
    pad_token_id: int
    pad_to_multiple_of: int | None = None
    return_tensors: str | None = "pt"

    def __call__(
        self,
        features: Sequence[TokenizedExample | Mapping[str, Sequence[int]]],
    ) -> dict[str, Any]:
        batch = pad_features(
            features,
            pad_token_id=self.pad_token_id,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        if self.return_tensors is None:
            return batch
        if self.return_tensors != "pt":
            raise ValueError("return_tensors must be 'pt' or None")
        try:
            import torch
        except ImportError as error:  # pragma: no cover - train extra supplies torch
            raise RuntimeError("PyTorch is required for tensor collation") from error
        return {name: torch.tensor(values, dtype=torch.long) for name, values in batch.items()}
