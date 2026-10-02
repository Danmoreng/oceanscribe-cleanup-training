"""Original diagnostic templates for conservative cleanup; never human gold."""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from .manifests import DatasetManifest, SourceManifest, sha256_file
from .quality import check_output
from .records import CleanupRecord, EditOperation
from .training import load_cleanup_records, validate_dataset_manifest

VERSION = "fidelity-v2"
CATEGORIES = (
    "number-repeat", "number-preserve", "self-correction", "unit-absence",
    "list-associations", "paragraph-command", "literal-command", "terminology",
)


def _example(category: str, number: int, de: bool):
    document = f"Auftrag {number}" if de else f"order {number}"
    features = [category, "fidelity"]
    checks = {"required_groups": [[document]]}
    commands = category != "literal-command"
    if category == "number-repeat":
        raw = (f"Wir prüfen wir prüfen {document} am Dienstag." if de
               else f"We review we review {document} on Tuesday.")
        clean = (f"Wir prüfen {document} am Dienstag." if de
                 else f"We review {document} on Tuesday.")
    elif category == "number-preserve":
        raw = clean = (f"{document} kostet 73 Euro und umfasst 18 Pakete." if de
                       else f"{document.capitalize()} costs 73 euros and includes 18 packages.")
        features.extend(("preserve", "hard-negative"))
        checks.update(preserve_exact=True)
    elif category == "self-correction":
        raw = (f"Für {document} brauchen wir 5 nein 9 Kopien. Abholung um 7 nein um 11."
               if de else f"For {document} we need 5 no 9 copies. Pickup at 7 no at 11.")
        clean = (f"Für {document} brauchen wir 9 Kopien. Abholung um 11." if de
                 else f"For {document} we need 9 copies. Pickup at 11.")
        checks.update(required_groups=[[document], ["9"], ["11"]], forbidden=["5", "7"])
    elif category == "unit-absence":
        raw = (f"Ähm {document} betrifft einen Raum mit 12 mal 14. Die Einheit fehlt." if de
               else f"Um {document} concerns a room measuring 12 by 14. The unit is missing.")
        clean = (f"{document} betrifft einen Raum mit 12 mal 14. Die Einheit fehlt." if de
                 else f"{document.capitalize()} concerns a room measuring 12 by 14. "
                      "The unit is missing.")
        checks.update(required_groups=[[document], ["12"], ["14"],
                                       ["Einheit fehlt" if de else "unit is missing"]],
                      forbidden=["Meter", "m", "Quadratmeter"] if de
                      else ["meters", "m", "feet", "square"])
    elif category == "list-associations":
        raw = (f"Für {document} brauchen wir äh Suppe Cracker Kaffee und Zucker." if de
               else f"For {document} we need uh soup crackers coffee and sugar.")
        clean = (f"Für {document} brauchen wir Suppe, Cracker, Kaffee und Zucker." if de
                 else f"For {document} we need soup, crackers, coffee and sugar.")
        checks.update(required_groups=[[document], ["Suppe" if de else "soup"],
                                       ["Cracker" if de else "crackers"],
                                       ["Kaffee" if de else "coffee"],
                                       ["Zucker" if de else "sugar"]],
                      forbidden=["Suppe: Cracker", "Kaffee: Zucker"] if de
                      else ["soup: crackers", "coffee: sugar"])
    elif category == "paragraph-command":
        raw = (f"{document} ist bestätigt neuer Absatz Die Lieferung kommt am Donnerstag."
               if de else f"{document} is confirmed new paragraph Delivery arrives on Thursday.")
        clean = (f"{document} ist bestätigt.\n\nDie Lieferung kommt am Donnerstag." if de
                 else f"{document.capitalize()} is confirmed.\n\nDelivery arrives on Thursday.")
        checks.update(forbidden=["neuer Absatz" if de else "new paragraph"],
                      required_groups=[[document], ["Donnerstag" if de else "Thursday"]])
    elif category == "literal-command":
        raw = clean = (f"Bei {document} steht im Kommentar neuer Absatz." if de
                       else f"For {document}, the comment says new paragraph.")
        features.append("preserve")
        checks.update(preserve_exact=True)
    else:
        raw = (f"Ich speichere {document} in ocean scribe." if de
               else f"I save {document} in ocean scribe.")
        clean = (f"Ich speichere {document} in OceanScribe." if de
                 else f"I save {document} in OceanScribe.")
        checks.update(corrected_terms=["OceanScribe"])
    return raw, clean, commands, tuple(features), checks


def generate_fidelity(seed: int = 73):
    """Choose family splits before producing locale and terminology derivatives."""
    revision = sha256_file(__file__)
    rng = random.Random(seed)
    # Nonoverlapping numeric identifiers within this generator, independent of old cases.
    numbers = rng.sample(range(6000, 9500), len(CATEGORIES) * 20)
    mixed, challenge, checks = [], [], {}
    for ci, category in enumerate(CATEGORIES):
        validation = set(rng.sample(range(16), 4))
        for index in range(20):
            split = "test" if index >= 16 else "validation" if index in validation else "train"
            family = f"{VERSION}/{category}/{index}"
            for language in ("en-US", "de-DE"):
                raw, clean, commands, features, annotation = _example(
                    category, numbers[ci * 20 + index], language == "de-DE"
                )
                supported = category == "terminology"
                hint = "OceanScribe" if supported else "NebulaForge"
                variants = ((hint,),) if split == "test" else ((), (hint,))
                for variant, terminology in enumerate(variants):
                    flags = (
                        (*features, "hard-negative") if terminology and not supported else features
                    )
                    record = CleanupRecord(
                        id=f"{family}/{language}/{variant}", source_name="oceanscribe-fidelity",
                        source_revision=revision, source_record_id=family,
                        license_spdx="Apache-2.0",
                        split=split, transcript=raw, output=clean, language=language,
                        commands=commands, terminology=terminology,
                        features=tuple(dict.fromkeys(flags)), synthetic=True,
                        generator_version=VERSION, generation_kind="deterministic", seed=seed,
                        parent_id=family,
                        edit_script=(EditOperation(operation="original-fidelity-template",
                                                   source=raw, target=clean),),
                        quality_flags=("synthetic-template-diagnostic-not-human-gold",),
                    )
                    annotation = dict(annotation)
                    if terminology and not supported:
                        annotation["absent_terms"] = list(terminology)
                    if not check_output(clean, clean, annotation)["passed"]:
                        raise ValueError(f"reference fails annotated checks: {record.id}")
                    (challenge if split == "test" else mixed).append(record)
                    checks[record.id] = annotation
    return tuple(mixed), tuple(challenge), checks


def prepare_fidelity_mix(records_path: Path, output_dir: Path, seed: int = 73):
    """Append original templates to an immutable pilot snapshot without changing old rows."""
    if output_dir.exists():
        raise ValueError("fidelity destination already exists")
    base_manifest = validate_dataset_manifest(records_path, records_path.with_name("manifest.json"))
    original = load_cleanup_records(records_path)
    templates, challenge, checks = generate_fidelity(seed)
    if any(r.source_name == "oceanscribe-fidelity" for r in original):
        raise ValueError("pilot snapshot already contains fidelity templates")
    for name, records in (("training", (*original, *templates)), ("challenge", challenge)):
        directory = output_dir / name
        directory.mkdir(parents=True)
        path = directory / "records.jsonl"
        path.write_text("".join(r.model_dump_json() + "\n" for r in records))
        # Reuse the strict loader to enforce IDs, family isolation and exact deduplication.
        load_cleanup_records(path)
        sources = dict(base_manifest["sources"]) if name == "training" else {}
        sources["oceanscribe-fidelity"] = SourceManifest(
            revision=templates[0].source_revision, license_spdx="Apache-2.0",
            allowed_use="Original local synthetic template training and diagnostic evaluation",
            redistribution="Not redistributed in this repository",
            record_count=sum(r.source_name == "oceanscribe-fidelity" for r in records),
        ).model_dump()
        manifest = DatasetManifest.model_validate({
            "prompt_contract_version": "raw-v1", "sources": sources,
            "counts_by_language": dict(Counter(r.language for r in records)),
            "counts_by_feature": dict(Counter(f for r in records for f in r.features)),
            "counts_by_length_bucket": {"not_tokenized": len(records)},
            "deduplication": {"method": "strict cleanup loader",
                              "family_split": "unchanged pilot plus template family split"},
            "oversize_exclusions": {}, "sha256": sha256_file(path),
        })
        (directory / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
        (directory / "checks.json").write_text(json.dumps(
            {r.id: checks[r.id] for r in records if r.id in checks}, ensure_ascii=False, indent=2
        ) + "\n")
    provenance = {
        "generator_version": VERSION, "generator_sha256": sha256_file(__file__), "seed": seed,
        "base_dataset_sha256": sha256_file(records_path),
        "base_manifest_sha256": sha256_file(records_path.with_name("manifest.json")),
        "base_dataset_path": str(records_path.resolve()),
        "limitation": "Original shared-template diagnostics, not independent human gold",
        "added_counts_by_split": dict(Counter(r.split for r in (*templates, *challenge))),
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance
