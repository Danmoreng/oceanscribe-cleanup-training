"""Canonical plain-text prompt used by training and later inference."""

from dataclasses import dataclass


TASK_KEYWORD = "Cleanup"


@dataclass(frozen=True, slots=True)
class CleanupExample:
    transcript: str
    output: str
    language: str
    commands: bool = True

    def __post_init__(self) -> None:
        if not self.transcript.strip():
            raise ValueError("transcript must not be empty")
        if not self.output.strip():
            raise ValueError("output must not be empty")
        if not self.language.strip():
            raise ValueError("language must not be empty")


def render_user_prompt(
    transcript: str,
    language: str,
    *,
    commands: bool = True,
) -> str:
    """Render the stable model-facing request without chat control tokens."""
    if not transcript.strip():
        raise ValueError("transcript must not be empty")
    if not language.strip():
        raise ValueError("language must not be empty")

    commands_value = "on" if commands else "off"
    return (
        f"{TASK_KEYWORD}\n"
        f"Language: {language}\n"
        f"Commands: {commands_value}\n"
        "<transcript>\n"
        f"{transcript}\n"
        "</transcript>\n"
        "Output:"
    )


def build_messages(example: CleanupExample) -> list[dict[str, str]]:
    """Return messages ready for the pinned official Qwen chat template."""
    return [
        {
            "role": "user",
            "content": render_user_prompt(
                example.transcript,
                example.language,
                commands=example.commands,
            ),
        },
        {"role": "assistant", "content": example.output},
    ]
