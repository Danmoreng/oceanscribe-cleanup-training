"""Compatibility imports for the original prompt module."""

from .prompt_contracts import RAW_V1
from .prompting import render_training_sequence, render_user_prompt
from .records import CleanupExample

DEFAULT_EOS_TOKEN = RAW_V1.expected_eos_token
TASK_KEYWORD = RAW_V1.task_keyword

__all__ = [
    "CleanupExample",
    "DEFAULT_EOS_TOKEN",
    "TASK_KEYWORD",
    "render_training_sequence",
    "render_user_prompt",
]
