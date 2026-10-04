"""Three cleanup prompts for the same frozen ASR input and human-confirmed target."""

from __future__ import annotations

from collections.abc import Sequence

from .records import CleanupRecord

IRRELEVANT_TERMS = (
    "Zeeman-Effekt",
    "Hochdruckhomogenisierung",
    "Stirlingmotor",
    "Kryolith",
    "Aphel",
    "Mykorrhiza",
    "Bathythermograph",
    "Kreuzpolarisation",
)


def ablation_variants(record: CleanupRecord) -> dict[str, CleanupRecord]:
    """Change only the canonical hint list; never alter raw text, target, family or split."""
    absent = tuple(
        term
        for term in IRRELEVANT_TERMS
        if term.casefold() not in record.transcript.casefold()
        and term.casefold() not in record.output.casefold()
    )[:4]
    if len(absent) < 4:
        raise ValueError("Not enough irrelevant control terms absent from this reference")
    return {
        mode: CleanupRecord.model_validate({**record.model_dump(mode="json"), "terminology": terms})
        for mode, terms in (
            ("with-hints", record.terminology),
            ("without-hints", ()),
            ("irrelevant-hints", absent),
        )
    }


def literal_hint_metrics(
    records: Sequence[CleanupRecord],
    outputs: Sequence[str],
) -> dict:
    """Count canonical spelling and false insertion, without inferring semantic correctness."""
    if len(records) != len(outputs):
        raise ValueError("Each record needs one output")
    required = corrected = unrelated = inserted = present = retained = 0
    for record, output in zip(records, outputs, strict=True):
        for term in record.terminology:
            canonical = term.casefold()
            in_raw = canonical in record.transcript.casefold()
            in_reference = canonical in record.output.casefold()
            in_output = canonical in output.casefold()
            if in_reference and not in_raw:
                required += 1
                corrected += int(in_output)
            elif in_reference and in_raw:
                present += 1
                retained += int(in_output)
            elif not in_reference and not in_raw:
                unrelated += 1
                inserted += int(in_output)
    return {
        "correction_recall": {"corrected_terms": corrected, "eligible_terms": required},
        "already_correct_retention": {"retained_terms": retained, "eligible_terms": present},
        "false_insertion": {"inserted_terms": inserted, "eligible_terms": unrelated},
        "method": "literal casefolded canonical substrings; no semantic correctness claim",
    }
