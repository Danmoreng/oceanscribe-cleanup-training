"""Freeze an English smoke subset from unchanged, reviewed upstream references."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from .data_prep import (
    AAWAAZ_REVISION,
    SOTTO_REVISION,
    SelectionStats,
    SourcePair,
    is_conservative_aawaaz_pair,
    write_dataset,
)
from .manifests import sha256_file
from .records import CleanupRecord
from .tokenization import OversizeExampleError, TokenizerLike, tokenize_example

VERSION = "reviewed-keep-smoke-v1"
REVISIONS = {"sotto": SOTTO_REVISION, "aawaaz": AAWAAZ_REVISION}


def prepare_reviewed_smoke(
    csv_path: Path,
    tokenizer: TokenizerLike,
    output_dir: Path,
    *,
    per_source: int = 200,
    validation_per_source: int = 20,
    seed: int = 42,
) -> tuple[Path, Path]:
    if not 0 < validation_per_source < per_source:
        raise ValueError("validation_per_source must be positive and smaller than per_source")
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len({row["record_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate CSV record IDs")
    candidates: dict[tuple[str, str], list[tuple[str, CleanupRecord]]] = {}
    excluded: dict[str, str] = {}
    counts: dict[str, Counter[str]] = {name: Counter() for name in REVISIONS}
    seen_pairs: set[tuple[str, str]] = set()
    for row in sorted(rows, key=lambda row: row["record_id"]):
        source = row["source_name"]
        if source not in REVISIONS:
            excluded[row["record_id"]] = "unregistered source"
            continue
        counts[source]["input"] += 1
        if row["english_review_status"] != "KEEP":
            excluded[row["record_id"]] = "not an unchanged KEEP reference"
            continue
        if (
            row["source_revision"] != REVISIONS[source]
            or row["license_spdx"] != "MIT"
            or row["english_reference_reviewed"] != row["english_reference_original"]
            or row["commands"] != "true"
            or row["source_locale"] != "en-US"
            or row["split"] != "train"
        ):
            excluded[row["record_id"]] = "metadata or unchanged-reference check failed"
            counts[source]["invalid"] += 1
            continue
        family = f"{source}\0{row['source_revision']}\0{row['source_record_id']}"
        score = hashlib.sha256(f"{VERSION}\0{seed}\0{family}".encode()).hexdigest()
        split = "validation" if int(score[:8], 16) % 10 == 0 else "train"
        pair = SourcePair(
            source_name=source,
            source_revision=row["source_revision"],
            source_record_id=row["source_record_id"],
            transcript=row["english_raw_transcript"],
            output=row["english_reference_original"],
        )
        if source == "aawaaz" and not is_conservative_aawaaz_pair(pair):
            excluded[row["record_id"]] = "non-conservative upstream pair"
            counts[source]["conservative"] += 1
            continue
        key = tuple(" ".join(text.split()).casefold() for text in (pair.transcript, pair.output))
        if key in seen_pairs:
            excluded[row["record_id"]] = "duplicate pair"
            counts[source]["duplicate"] += 1
            continue
        try:
            record = CleanupRecord(
                id=row["record_id"],
                source_name=source,
                source_revision=pair.source_revision,
                source_record_id=pair.source_record_id,
                license_spdx="MIT",
                split=split,
                language="en-US",
                commands=True,
                transcript=pair.transcript,
                output=pair.output,
                synthetic=False,
                features=tuple(row["features"].split("|")),
                quality_flags=tuple(filter(None, row["quality_flags"].split("|")))
                + ("csv-review:KEEP", "csv-original-split:train"),
            )
            tokenize_example(record, tokenizer, max_sequence_length=2048)
        except OversizeExampleError:
            excluded[row["record_id"]] = "over 2048 tokens; not truncated"
            counts[source]["oversize"] += 1
            continue
        except (ValueError, TypeError):
            excluded[row["record_id"]] = "invalid raw-v1 record"
            counts[source]["invalid"] += 1
            continue
        seen_pairs.add(key)
        candidates.setdefault((source, split), []).append((score, record))
    selected = []
    for source in REVISIONS:
        for split, count in (
            ("train", per_source - validation_per_source),
            ("validation", validation_per_source),
        ):
            available = sorted(candidates.get((source, split), []), key=lambda item: item[0])
            if len(available) < count:
                raise ValueError(f"only {len(available)} eligible {source}/{split}; need {count}")
            selected.extend(record for _, record in available[:count])
    selected.sort(key=lambda record: (record.split, record.source_name, record.id))
    stats = {
        source: SelectionStats(
            c["input"], c["duplicate"], c["conservative"], c["invalid"], c["oversize"]
        )
        for source, c in counts.items()
    }
    paths = write_dataset(selected, stats, output_dir)
    (output_dir / "selection.json").write_text(
        json.dumps(
            {
                "generator_version": VERSION,
                "selection_seed": seed,
                "source_csv": str(csv_path.resolve()),
                "source_csv_sha256": sha256_file(csv_path),
                "selection": (
                    "English KEEP; original target unchanged; registered upstream MIT revisions"
                ),
                "upstream_generator_seed": None,
                "edit_script": [],
                "derivative_rule": (
                    "Translations and other derivatives inherit their source family split"
                ),
                "family_splits": [
                    {
                        "source": r.source_name,
                        "revision": r.source_revision,
                        "source_record_id": r.source_record_id,
                        "record_id": r.id,
                        "split": r.split,
                    }
                    for r in selected
                ],
                "excluded_records": excluded,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths
