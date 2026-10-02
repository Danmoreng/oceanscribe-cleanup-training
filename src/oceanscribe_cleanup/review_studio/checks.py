"""Deterministic, annotation-scoped checks for Review Studio outputs.

Supported annotations are deliberately literal and explicit:

* ``required_format.min_paragraphs``: minimum paragraphs separated by one or
  more blank lines (for example, ``{"min_paragraphs": 2}``).
* ``required_facts``: pairs ``[entity, value]``. Both phrases must be locally
  bound in one clause, with at most six intervening words and no other
  annotated entity between them. This catches value swaps in simple records;
  it is not a general semantic relation parser.
* ``required_negations`` and ``forbidden_insertions``: explicit literal
  phrases to require or prohibit. They are never inferred from transcript.
* ``preserve_exact``: when true, require output to equal transcript exactly.

Meaning is never inferred by these rules. It is counted only when a human
passes ``human_meaning_failure`` as a boolean.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

_CLAUSE_BREAK = re.compile(r"[,;.!?\n]+|\b(?:but|while|whereas)\b", re.IGNORECASE)
_WORD = re.compile(r"\b[\w]+(?:['’][\w]+)*\b", re.UNICODE)


def _phrase_present(text: str, phrase: str) -> bool:
    return bool(re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text, re.IGNORECASE))


def _fact_bound(text: str, entity: str, value: str, entities: Sequence[str]) -> bool:
    """Return whether entity and value have a local, unambiguous textual binding."""
    for clause in _CLAUSE_BREAK.split(text):
        entity_spans = [
            match.span() for candidate in entities
            for match in re.finditer(r"(?<!\w)" + re.escape(candidate) + r"(?!\w)",
                                     clause, re.IGNORECASE)
        ]
        target_spans = [
            match.span() for match in re.finditer(r"(?<!\w)" + re.escape(entity) + r"(?!\w)",
                                                  clause, re.IGNORECASE)
        ]
        value_spans = [
            match.span() for match in re.finditer(r"(?<!\w)" + re.escape(value) + r"(?!\w)",
                                                  clause, re.IGNORECASE)
        ]
        for start, end in target_spans:
            for value_start, value_end in value_spans:
                if end <= value_start:
                    gap = clause[end:value_start]
                    other_entity_between = any(
                        start < other_start < value_start
                        for other_start, other_end in entity_spans
                        if (other_start, other_end) != (start, end)
                    )
                elif value_end <= start:
                    gap = clause[value_end:start]
                    other_entity_between = any(
                        value_start < other_start < start
                        for other_start, other_end in entity_spans
                        if (other_start, other_end) != (start, end)
                    )
                else:
                    continue
                if len(_WORD.findall(gap)) <= 6 and not other_entity_between:
                    return True
    return False


def _string_list(value: Any, key: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"annotations.{key} must be a list of strings")
    return value


def assess_output(
    transcript: str,
    output: str,
    annotations: dict[str, Any] | None = None,
    human_meaning_failure: bool | None = None,
) -> dict[str, Any]:
    """Measure copy behavior and only deterministic checks explicitly annotated.

    Unannotated meaning remains ``not_evaluated`` with denominator zero. A
    caller may supply a human judgment as ``human_meaning_failure``; no text
    heuristic creates or fills that judgment.
    """
    if not isinstance(transcript, str) or not isinstance(output, str):
        raise TypeError("transcript and output must be strings")
    if annotations is not None and not isinstance(annotations, dict):
        raise TypeError("annotations must be a dictionary or None")
    if human_meaning_failure is not None and not isinstance(human_meaning_failure, bool):
        raise TypeError("human_meaning_failure must be a boolean or None")

    annotations = annotations or {}
    format_rules = annotations.get("required_format", {})
    if not isinstance(format_rules, dict):
        raise ValueError("annotations.required_format must be a dictionary")
    min_paragraphs = format_rules.get("min_paragraphs")
    if min_paragraphs is not None and (not isinstance(min_paragraphs, int) or min_paragraphs < 1):
        raise ValueError("required_format.min_paragraphs must be a positive integer")
    paragraphs = [p for p in re.split(r"\n\s*\n", output.strip()) if p.strip()]
    paragraph_failed = min_paragraphs is not None and len(paragraphs) < min_paragraphs

    facts = annotations.get("required_facts", [])
    if not isinstance(facts, list):
        raise ValueError("annotations.required_facts must be a list of [entity, value] pairs")
    parsed_facts: list[tuple[str, str]] = []
    for fact in facts:
        if (not isinstance(fact, (list, tuple)) or len(fact) != 2
                or any(not isinstance(part, str) or not part for part in fact)):
            raise ValueError("each required_facts item must be a non-empty [entity, value] pair")
        parsed_facts.append((fact[0], fact[1]))
    entities = [entity for entity, _ in parsed_facts]
    missing_facts = [
        {"entity": entity, "value": value}
        for entity, value in parsed_facts
        if not _fact_bound(output, entity, value, entities)
    ]

    required_negations = _string_list(annotations.get("required_negations"), "required_negations")
    missing_negations = [
        phrase for phrase in required_negations if not _phrase_present(output, phrase)
    ]
    forbidden = _string_list(annotations.get("forbidden_insertions"), "forbidden_insertions")
    inserted = [phrase for phrase in forbidden if _phrase_present(output, phrase)]

    preserve_exact = annotations.get("preserve_exact", False)
    if not isinstance(preserve_exact, bool):
        raise ValueError("annotations.preserve_exact must be a boolean")
    preserve_failure = preserve_exact and output != transcript

    meaning = (
        {"status": "not_evaluated", "failures": None, "denominator": 0}
        if human_meaning_failure is None
        else {"status": "human_assessed", "failures": int(human_meaning_failure), "denominator": 1}
    )

    evaluated = bool(min_paragraphs is not None or parsed_facts or required_negations
                     or forbidden or preserve_exact)
    failures = (
        int(paragraph_failed) + len(missing_facts) + len(missing_negations)
        + len(inserted) + int(preserve_failure)
    )
    return {
        "copy_only_baseline": {"is_copy": output == transcript, "denominator": 1},
        "automated_checks": {
            "status": "evaluated" if evaluated else "not_evaluated",
            "failures": failures if evaluated else None,
            "denominator": (
                int(min_paragraphs is not None) + len(parsed_facts) + len(required_negations)
                + len(forbidden) + int(preserve_exact)
            ),
            "required_format": {
                "paragraph_count": len(paragraphs),
                "min_paragraphs": min_paragraphs,
                "failed": paragraph_failed,
            },
            "required_facts": {
                "opportunities": len(parsed_facts), "failures": missing_facts,
            },
            "required_negations": {
                "opportunities": len(required_negations), "missing": missing_negations,
            },
            "forbidden_insertions": {
                "opportunities": len(forbidden), "present": inserted,
            },
            "preserve_exact": {
                "annotated": preserve_exact, "failed": preserve_failure,
            },
        },
        "meaning": meaning,
    }
