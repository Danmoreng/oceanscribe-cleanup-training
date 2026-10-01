"""Explicit annotated checks; these do not replace general semantic evaluation."""

from __future__ import annotations

import re
from typing import Any


def contains_phrase(text: str, phrase: str, *, canonical: bool = False) -> bool:
    text = " ".join(text.split())
    phrase = " ".join(phrase.split())
    return bool(re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text,
                          flags=0 if canonical else re.IGNORECASE))


def check_output(output: str, reference: str, checks: dict[str, Any]) -> dict[str, Any]:
    missing = [group for group in checks.get("required_groups", [])
               if not any(contains_phrase(output, phrase) for phrase in group)]
    forbidden = [phrase for phrase in checks.get("forbidden", [])
                 if contains_phrase(output, phrase)]
    corrected = checks.get("corrected_terms", [])
    absent = checks.get("absent_terms", [])
    missing_terms = [
        term for term in corrected if not contains_phrase(output, term, canonical=True)
    ]
    inserted_terms = [term for term in absent if contains_phrase(output, term)]
    preserve_failed = bool(checks.get("preserve_exact")) and output != reference
    return {
        "passed": not (missing or forbidden or missing_terms or inserted_terms or preserve_failed),
        "missing_required_groups": missing, "forbidden_phrases": forbidden,
        "missing_corrected_terms": missing_terms, "false_inserted_terms": inserted_terms,
        "preserve_failed": preserve_failed,
        "correction_opportunities": len(corrected), "absent_term_opportunities": len(absent),
    }


def score_checks(results, *, adapter: bool) -> dict[str, Any]:
    checked = [r for r in results if r.checks is not None]
    outputs = [check_output(r.adapter_output if adapter else r.base_output, r.reference, r.checks)
               for r in checked]
    positives = sum(x["correction_opportunities"] for x in outputs)
    negatives = sum(x["absent_term_opportunities"] for x in outputs)
    return {
        "annotated_records": len(outputs),
        "passed_records": sum(x["passed"] for x in outputs),
        "missing_required_records": sum(bool(x["missing_required_groups"]) for x in outputs),
        "forbidden_phrase_records": sum(bool(x["forbidden_phrases"]) for x in outputs),
        "preserve_failures": sum(x["preserve_failed"] for x in outputs),
        "term_correction_opportunities": positives,
        "term_correction_recall": (
            1 - sum(len(x["missing_corrected_terms"]) for x in outputs) / positives
            if positives else None
        ),
        "absent_term_opportunities": negatives,
        "false_insertion_rate": (
            sum(len(x["false_inserted_terms"]) for x in outputs) / negatives if negatives else None
        ),
        "failures": [{"id": r.id, **x} for r, x in zip(checked, outputs, strict=True)
                     if not x["passed"]],
    }
