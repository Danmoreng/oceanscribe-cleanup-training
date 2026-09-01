from __future__ import annotations

import os

import pytest

from oceanscribe_cleanup.model_inspection import load_tokenizer_report
from oceanscribe_cleanup.prompt_contracts import RAW_V1


@pytest.mark.model_download
def test_pinned_qwen_tokenizer_contract() -> None:
    revision = os.environ.get("OCEANSCRIBE_MODEL_REVISION")
    if not revision:
        pytest.skip("set OCEANSCRIBE_MODEL_REVISION to a pinned tag or commit")
    tokenizer, report = load_tokenizer_report(
        "Qwen/Qwen3.5-0.8B-Base",
        revision,
    )
    assert report.resolved_revision
    assert report.eos_token == RAW_V1.expected_eos_token
    assert report.eos_token_id == tokenizer.eos_token_id
    assert tokenizer(
        RAW_V1.expected_eos_token,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
    ).input_ids == [tokenizer.eos_token_id]
