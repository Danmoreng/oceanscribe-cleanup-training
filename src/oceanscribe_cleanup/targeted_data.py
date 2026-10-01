"""Original bilingual templates for training diagnostics and a synthetic challenge suite."""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from .manifests import DatasetManifest, SourceManifest, sha256_file
from .quality import check_output
from .records import CleanupRecord, EditOperation

GENERATOR_VERSION = "targeted-v1"
CATEGORIES = (
    "preserve", "already-correct-term", "filler", "repetition", "self-correction",
    "negation", "limit", "paragraph-command", "literal-command", "terminology",
)


def _template(category: str, index: int, language: str):
    de = language == "de-DE"
    number = 100 + index
    document = f"Bericht {number}" if de else f"report {number}"
    checks = {}
    commands = category != "literal-command"
    features = [category]
    if category == "preserve":
        raw = clean = (f"Mira hat {document} bereits geprüft." if de
                       else f"Mira has already reviewed {document}.")
        features.append("preserve")
        checks = {"preserve_exact": True}
    elif category == "already-correct-term":
        raw = clean = (f"OceanScribe speichert {document} lokal." if de
                       else f"OceanScribe stores {document} locally.")
        features.append("preserve")
        checks = {"preserve_exact": True}
    elif category == "filler":
        raw = (f"Ähm bitte sende mir also {document} bis Freitag." if de
               else f"Um please send me uh {document} by Friday.")
        clean = (f"Bitte sende mir {document} bis Freitag." if de
                 else f"Please send me {document} by Friday.")
        checks = {"required_groups": [[document], ["Freitag" if de else "Friday"]]}
    elif category == "repetition":
        raw = (f"Wir prüfen wir prüfen {document} gemeinsam." if de
               else f"We review we review {document} together.")
        clean = (f"Wir prüfen {document} gemeinsam." if de
                 else f"We review {document} together.")
        checks = {"required_groups": [[document], ["gemeinsam" if de else "together"]]}
    elif category == "self-correction":
        raw = (f"Für {document} brauchen wir sechs nein neun Kopien." if de
               else f"For {document} we need six no nine copies.")
        clean = (f"Für {document} brauchen wir neun Kopien." if de
                 else f"For {document} we need nine copies.")
        checks = {"required_groups": [[document], ["neun", "9"] if de else ["nine", "9"]],
                  "forbidden": ["sechs", "6"] if de else ["six", "6"]}
    elif category == "negation":
        raw = clean = (f"Bitte veröffentliche {document} nicht vor Montag." if de
                       else f"Please do not publish {document} before Monday.")
        features.append("hard-negative")
        checks = {"required_groups": [[document], ["nicht" if de else "not"],
                                      ["Montag" if de else "Monday"]], "preserve_exact": True}
    elif category == "limit":
        raw = clean = (f"Für {document} sind zweihundert Euro die Obergrenze." if de
                       else f"For {document}, two hundred euros is the limit.")
        features.append("hard-negative")
        checks = {"required_groups": [[document], ["Obergrenze" if de else "limit"]],
                  "forbidden": ["überweisen", "senden"] if de else ["send", "transfer"]}
    elif category == "paragraph-command":
        raw = (f"Bitte prüfe {document} neuer Absatz Die Frist endet morgen." if de
               else f"Please review {document} new paragraph The deadline is tomorrow.")
        clean = (f"Bitte prüfe {document}.\n\nDie Frist endet morgen." if de
                 else f"Please review {document}.\n\nThe deadline is tomorrow.")
        checks = {"required_groups": [[document], ["morgen" if de else "tomorrow"]],
                  "forbidden": ["neuer Absatz" if de else "new paragraph"]}
    elif category == "literal-command":
        raw = clean = (f"Für {document} lautet das Stichwort neuer Absatz." if de
                       else f"For {document}, the keyword is new paragraph.")
        checks = {"required_groups": [[document], ["neuer Absatz" if de else "new paragraph"]],
                  "preserve_exact": True}
    else:
        raw = (f"Ich öffne {document} in ocean scribe." if de
               else f"I open {document} in ocean scribe.")
        clean = (f"Ich öffne {document} in OceanScribe." if de
                 else f"I open {document} in OceanScribe.")
        checks = {"required_groups": [[document]], "corrected_terms": ["OceanScribe"]}
    return raw, clean, commands, tuple(dict.fromkeys(features)), checks


def generate_targeted(seed: int = 42):
    """Split original families first, then derive locales and hint variants."""
    revision = sha256_file(__file__)
    rng = random.Random(seed)
    training, challenge, annotations = [], [], {}
    for category_index, category in enumerate(CATEGORIES):
        validation_index = rng.randrange(5)
        # Test families use distinct lexical IDs; they still share these templates.
        # This is a diagnostic challenge set, not independent human gold.
        family_indices = [(i, "validation" if i == validation_index else "train")
                          for i in range(5)]
        family_indices += [(100 + i, "test") for i in range(3 if category_index < 5 else 2)]
        for index, split in family_indices:
            family = f"{GENERATOR_VERSION}/{category}/{index}"
            for language in ("en-US", "de-DE"):
                raw, clean, commands, features, checks = _template(category, index, language)
                supported = category in {"terminology", "already-correct-term"}
                hint = "OceanScribe" if supported else "NebulaForge"
                variants = ((), (hint,)) if split != "test" else ((hint,),)
                for variant_index, terms in enumerate(variants):
                    record = CleanupRecord(
                        id=f"{family}/{language}/{variant_index}",
                        source_name="oceanscribe-targeted", source_revision=revision,
                        source_record_id=family, license_spdx="Apache-2.0", split=split,
                        transcript=raw, output=clean, commands=commands, language=language,
                        terminology=terms, features=features,
                        synthetic=True, generator_version=GENERATOR_VERSION, seed=seed,
                        parent_id=family,
                        edit_script=(EditOperation(operation="template-cleanup", source=raw,
                                                   target=clean),),
                    )
                    record_checks = dict(checks)
                    if terms and not supported:
                        record_checks["absent_terms"] = list(terms)
                    if not check_output(clean, clean, record_checks)["passed"]:
                        raise ValueError(f"reference fails its own checks: {record.id}")
                    (challenge if split == "test" else training).append(record)
                    if split == "test":
                        annotations[record.id] = record_checks
    return tuple(training), tuple(challenge), annotations


def prepare_targeted(output_dir: Path, seed: int = 42):
    if output_dir.exists():
        raise ValueError("targeted output directory already exists")
    training, challenge, checks = generate_targeted(seed)
    for name, records in (("training", training), ("challenge", challenge)):
        directory = output_dir / name
        directory.mkdir(parents=True)
        path = directory / "records.jsonl"
        path.write_text("".join(r.model_dump_json() + "\n" for r in records))
        manifest = DatasetManifest(
            prompt_contract_version="raw-v1",
            sources={"oceanscribe-targeted": SourceManifest(
                revision=records[0].source_revision, license_spdx="Apache-2.0",
                allowed_use="Local original-template training/evaluation diagnostics",
                redistribution="Not redistributed in this repository", record_count=len(records),
            )},
            counts_by_language=dict(Counter(r.language for r in records)),
            counts_by_feature=dict(Counter(f for r in records for f in r.features)),
            counts_by_length_bucket={"not_tokenized": len(records)},
            deduplication={"method": "locale/command/hints/transcript/output exact key",
                           "family_split": "before locale/hint augmentation"},
            oversize_exclusions={}, sha256=sha256_file(path),
        )
        (directory / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
        (directory / "provenance.json").write_text(json.dumps({
            "generator_version": GENERATOR_VERSION, "seed": seed,
            "generator_sha256": records[0].source_revision,
            "source": "Original project-authored templates; no teacher generation or corpus input",
            "limitation": "Shared template diagnostics; not independent human gold",
            "counts_by_split": dict(Counter(r.split for r in records)),
        }, indent=2) + "\n")
    (output_dir / "challenge" / "checks.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2) + "\n"
    )
    return {"training_records": len(training), "challenge_records": len(challenge),
            "directory": str(output_dir)}
