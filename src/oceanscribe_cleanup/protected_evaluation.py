"""Local checkpoint measurements; no heuristic supplies a human meaning label."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from statistics import mean

from .evaluation import output_has_forbidden_control
from .records import CleanupRecord


def score_protected_outputs(
    records: Sequence[CleanupRecord], outputs: Sequence[str], eos: Sequence[bool],
    family_ids: Sequence[str],
) -> dict:
    from jiwer import cer, wer

    if not records or not (len(records) == len(outputs) == len(eos) == len(family_ids)):
        raise ValueError("protected scoring needs matching nonempty records and outputs")
    if any(r.split != "validation" for r in records):
        raise ValueError("checkpoint selection measurements require Development only")
    if any(not isinstance(value, bool) for value in eos):
        raise ValueError("EOS measurements must be explicit booleans")
    locales = {}
    for locale in sorted({r.language for r in records}):
        positions = [i for i, r in enumerate(records) if r.language == locale]
        references = [" ".join(records[i].output.split()) for i in positions]
        hypotheses = [" ".join(outputs[i].split()) for i in positions]
        by_family = defaultdict(list)
        for i, ref, hyp in zip(positions, references, hypotheses, strict=True):
            by_family[family_ids[i]].append(wer(ref, hyp))
        locales[locale] = {
            "records": len(positions), "families": len(by_family),
            "wer": wer(references, hypotheses), "cer": cer(references, hypotheses),
            "family_mean_wer": mean(mean(values) for values in by_family.values()),
        }
    preserve = [i for i, r in enumerate(records) if r.output == r.transcript]
    paragraphs = [i for i, r in enumerate(records) if "\n\n" in r.output]
    return {
        "schema_version": "protected-checkpoint-metrics-v1",
        "normalization": "whitespace-fold-v1; punctuation-and-case-preserved",
        "records": len(records), "families": len(set(family_ids)),
        "by_language": locales,
        "eos_stops": sum(eos), "generation_limit_hits": len(eos) - sum(eos),
        "forbidden_control_outputs": sum(output_has_forbidden_control(o) for o in outputs),
        "empty_outputs": sum(not o.strip() for o in outputs),
        "copy_only_outputs": sum(o == r.transcript for o, r in zip(outputs, records, strict=True)),
        "preserve_exact": {
            "matches": sum(outputs[i] == records[i].transcript for i in preserve),
            "denominator": len(preserve),
        },
        "reference_blank_line_cases": {
            "with_blank_line": sum("\n\n" in outputs[i] for i in paragraphs),
            "denominator": len(paragraphs),
        },
        "meaning": {"status": "not_evaluated", "denominator": 0, "failures": None},
        "independent_human_gold": False,
    }
