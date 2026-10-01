"""Freeze reviewed user-provided bilingual pairs while translations continue."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path

from .data_prep import SourcePair, is_conservative_aawaaz_pair
from .manifests import DatasetManifest, SourceManifest, sha256_file
from .records import CleanupRecord, EditOperation
from .reviewed_csv import REVISIONS
from .reviewed_csv import VERSION as SPLIT_VERSION
from .targeted_data import generate_targeted
from .tokenization import OversizeExampleError, tokenize_example

VERSION = "reviewed-bilingual-pilot-v1"
LOCAL_LICENSE = "MIT AND LicenseRef-UserProvided-LocalTraining"


def prepare_bilingual(csv_path: Path, tokenizer, output_dir: Path, *, seed: int = 42):
    """Use reviewed KEEP/FIX English paired with German PASS, plus original templates."""
    if output_dir.exists():
        raise ValueError("bilingual snapshot destination already exists")
    source_bytes = csv_path.read_bytes()
    rows = list(csv.DictReader(io.StringIO(source_bytes.decode("utf-8-sig"))))
    if len({r["record_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate CSV record IDs")
    records, excluded, seen = [], {}, set()
    for row in sorted(rows, key=lambda r: r["record_id"]):
        identifier = row["record_id"]
        if row["source_name"] not in REVISIONS or (
            row["source_revision"] != REVISIONS[row["source_name"]]
            or row["license_spdx"] != "MIT" or row["commands"] != "true"
            or row["source_locale"] != "en-US" or row["split"] != "train"
        ):
            excluded[identifier] = "unregistered or inconsistent source metadata"
            continue
        if row["english_review_status"] not in {"KEEP", "FIX"}:
            excluded[identifier] = "English reference review incomplete or REVIEW"
            continue
        if row["german_translation_status"] != "PASS":
            excluded[identifier] = "German pair incomplete or REVIEW"
            continue
        if not all(row[k].strip() for k in ("english_reference_reviewed",
                                            "german_raw_transcript", "german_reference")):
            excluded[identifier] = "missing paired text"
            continue
        if row["english_review_status"] == "KEEP" and (
            row["english_reference_reviewed"] != row["english_reference_original"]
        ):
            excluded[identifier] = "KEEP reference was changed"
            continue
        pair = SourcePair(source_name=row["source_name"], source_revision=row["source_revision"],
                          source_record_id=row["source_record_id"],
                          transcript=row["english_raw_transcript"],
                          output=row["english_reference_reviewed"])
        if pair.source_name == "aawaaz" and not is_conservative_aawaaz_pair(pair):
            excluded[identifier] = "non-conservative English pair"
            continue
        family = f"{pair.source_name}\0{pair.source_revision}\0{pair.source_record_id}"
        score = hashlib.sha256(f"{SPLIT_VERSION}\0{seed}\0{family}".encode()).hexdigest()
        split = "validation" if int(score[:8], 16) % 10 == 0 else "train"
        family_records = []
        try:
            for language, raw, output in (
                ("en-US", pair.transcript, pair.output),
                ("de-DE", row["german_raw_transcript"], row["german_reference"]),
            ):
                generated = language == "de-DE" or row["english_review_status"] == "FIX"
                edits = []
                if generated:
                    edits = [EditOperation(operation="translation" if language == "de-DE"
                                           else "reference-review", source=pair.transcript,
                                           target=raw),
                             EditOperation(operation="translation" if language == "de-DE"
                                           else "reference-review",
                                           source=row["english_reference_original"], target=output)]
                record = CleanupRecord(
                    id=f"{identifier}/{language}", source_name=pair.source_name,
                    source_revision=pair.source_revision, source_record_id=pair.source_record_id,
                    license_spdx=LOCAL_LICENSE if generated else "MIT", split=split,
                    transcript=raw, output=output, language=language, commands=True,
                    features=tuple(filter(None, row["features"].split("|"))) + ("bilingual-pilot",),
                    synthetic=generated,
                    generator_version="chatgpt-pro-user-provided-v1" if generated else None,
                    generation_kind="stochastic" if generated else None, seed=None,
                    parent_id=identifier if generated else None, edit_script=tuple(edits),
                    quality_flags=tuple(filter(None, row["quality_flags"].split("|")))
                    + (f"csv-review:{row['english_review_status']}", "csv-german:PASS")
                    + (("generation-seed-unavailable", "generation-model-id-unavailable")
                       if generated else ()),
                )
                tokenize_example(record, tokenizer, max_sequence_length=2048)
                key = (language, " ".join(raw.split()).casefold(),
                       " ".join(output.split()).casefold())
                if key in seen:
                    raise ValueError("duplicate pair")
                family_records.append((record, key))
        except OversizeExampleError:
            excluded[identifier] = "one language exceeds 2048 tokens; whole family excluded"
            continue
        except (ValueError, TypeError) as error:
            excluded[identifier] = f"invalid pair: {error}"
            continue
        records.extend(r for r, _ in family_records)
        seen.update(k for _, k in family_records)
    templates, _, _ = generate_targeted(seed)
    derivative_seen = {
        (r.language, r.commands, r.terminology, r.transcript.casefold(), r.output.casefold())
        for r in records
    }
    # Keep augmentation derivatives in their already assigned source-family split.
    # Add clean-input preserve examples and unsupported-hint hard negatives.
    for split in ("train", "validation"):
        families = sorted({
            (r.source_name, r.source_record_id) for r in records if r.split == split
        })
        preserve = set(families[:math.ceil(len(families) / 3)])
        absent = set(families[-math.ceil(len(families) / 4):]) if families else set()
        for record in tuple(records):
            if record.split != split:
                continue
            family = (record.source_name, record.source_record_id)
            for kind, selected in (("preserve", preserve), ("absent-hint", absent)):
                if family not in selected:
                    continue
                if kind == "absent-hint" and "nebulaforge" in (
                    record.transcript + record.output
                ).casefold():
                    continue
                values = record.model_dump(mode="json")
                values.update(
                    id=record.id + "/" + kind, parent_id=record.id, synthetic=True,
                    generator_version=VERSION + "-augmentation", generation_kind="deterministic",
                    seed=seed, features=list(dict.fromkeys([*record.features,
                        "preserve" if kind == "preserve" else "hard-negative", kind,
                    ])),
                    transcript=record.output if kind == "preserve" else record.transcript,
                    terminology=[] if kind == "preserve" else ["NebulaForge"],
                    edit_script=[EditOperation(
                        operation=kind, source=record.transcript,
                        target=record.output if kind == "preserve" else record.transcript,
                    ).model_dump(mode="json")],
                )
                derivative = CleanupRecord.model_validate(values)
                key = (derivative.language, derivative.commands, derivative.terminology,
                       derivative.transcript.casefold(), derivative.output.casefold())
                if key in derivative_seen:
                    continue
                try:
                    tokenize_example(derivative, tokenizer, max_sequence_length=2048)
                except OversizeExampleError:
                    excluded[derivative.id] = "augmentation exceeds 2048 tokens; not truncated"
                    continue
                records.append(derivative)
                derivative_seen.add(key)
    records.extend(templates)
    records.sort(key=lambda r: (r.split, r.source_name, r.source_record_id or r.id, r.language))
    if not any(r.language == "de-DE" and not r.source_name.startswith("oceanscribe")
               for r in records):
        raise ValueError("no usable reviewed German corpus records")
    output_dir.mkdir(parents=True)
    (output_dir / "source-snapshot.csv").write_bytes(source_bytes)
    records_path = output_dir / "records.jsonl"
    records_path.write_text("".join(r.model_dump_json() + "\n" for r in records))
    sources = {}
    for source in sorted({r.source_name for r in records}):
        subset = [r for r in records if r.source_name == source]
        sources[source] = SourceManifest(
            revision=subset[0].source_revision,
            license_spdx="Apache-2.0" if source == "oceanscribe-targeted" else LOCAL_LICENSE,
            allowed_use="Local transcript-cleanup training explicitly requested by data owner; "
                        "upstream MIT attribution retained; derivatives separately marked",
            redistribution="No dataset redistribution", record_count=len(subset),
        )
    manifest = DatasetManifest(
        prompt_contract_version="raw-v1", sources=sources,
        counts_by_language=dict(Counter(r.language for r in records)),
        counts_by_feature=dict(Counter(f for r in records for f in r.features)),
        counts_by_length_bucket={"at-most-2048": len(records)},
        deduplication={"method": "normalized locale/raw/target; paired family selection",
                       "family_split": f"{SPLIT_VERSION}, seed {seed}, before locale derivatives"},
        oversize_exclusions={"families": sum("exceeds 2048" in v for v in excluded.values())},
        sha256=sha256_file(records_path),
    )
    (output_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
    selection = {
        "importer_version": VERSION, "importer_sha256": sha256_file(__file__),
        "source_csv_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "selection_seed": seed, "generation_seed": None,
        "generation_source": "User-provided ChatGPT Pro CSV (subscription, exact model unknown)",
        "policy": "English KEEP/FIX plus German PASS; incomplete/REVIEW references excluded",
        "excluded_families": excluded,
        "counts_by_split": dict(Counter(r.split for r in records)),
        "family_splits": {f"{r.source_name}/{r.source_record_id}": r.split for r in records},
        "corpus_german_records": sum(r.language == "de-DE" and r.source_name !=
                                     "oceanscribe-targeted" for r in records),
        "reviewed_pair_families": sum(
            r.language == "de-DE" and r.generator_version == "chatgpt-pro-user-provided-v1"
            for r in records
        ),
    }
    (output_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    return records
