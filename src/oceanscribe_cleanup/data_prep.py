"""Deterministic ingestion of pinned external transcript-cleanup datasets."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .manifests import DatasetManifest, SourceManifest, sha256_file
from .records import CleanupRecord
from .tokenization import OversizeExampleError, TokenizerLike, tokenize_example

SOTTO_REVISION = "183cc8fd58532f13fa192980185214de1bcd5acc"
AAWAAZ_REVISION = "5a8e62a11cebea4832057f304f12f8401d387b43"
SELECTION_VERSION = "external-smoke-v1"
WORD_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
FILLERS = frozenset({"ah", "eh", "er", "hmm", "like", "uh", "um"})


@dataclass(frozen=True, slots=True)
class SourcePair:
    source_name: Literal["sotto", "aawaaz"]
    source_revision: str
    source_record_id: str
    transcript: str
    output: str
    upstream_split: str | None = None
    category: str | None = None


@dataclass(frozen=True, slots=True)
class SelectionStats:
    input_records: int
    duplicate_records: int
    conservative_filter_exclusions: int
    invalid_record_exclusions: int
    oversize_exclusions: int


def normalized_pair_key(pair: SourcePair) -> tuple[str, str]:
    return (" ".join(pair.transcript.split()).casefold(), " ".join(pair.output.split()).casefold())


def stable_score(pair: SourcePair, *, seed: int) -> str:
    payload = (
        f"{SELECTION_VERSION}\0{seed}\0{pair.source_name}\0"
        f"{pair.source_record_id}\0{pair.transcript}\0{pair.output}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _words(text: str) -> list[str]:
    return WORD_PATTERN.findall(text.casefold())


def is_conservative_aawaaz_pair(pair: SourcePair) -> bool:
    """Keep high-overlap pairs and reject summaries or expansive rewrites."""
    input_words = _words(pair.transcript)
    output_words = _words(pair.output)
    if len(input_words) < 4 or not output_words:
        return False
    supported = sum((Counter(input_words) & Counter(output_words)).values())
    support_ratio = supported / len(output_words)
    length_ratio = len(output_words) / len(input_words)
    return support_ratio >= 0.85 and 0.40 <= length_ratio <= 1.05


def infer_features(pair: SourcePair) -> tuple[str, ...]:
    words = _words(pair.transcript)
    features: set[str] = {"external", "upstream-synthetic"}
    if any(word in FILLERS for word in words):
        features.add("filler-removal")
    if any(left == right for left, right in zip(words, words[1:], strict=False)):
        features.add("repetition")
    lowered = pair.transcript.casefold()
    if any(marker in lowered for marker in ("i mean", "wait no", "sorry", "rather")):
        features.add("self-correction")
    if normalized_pair_key(pair)[0] == normalized_pair_key(pair)[1]:
        features.add("preserve")
    return tuple(sorted(features))


def _deduplicate(pairs: Iterable[SourcePair]) -> tuple[list[SourcePair], int]:
    unique: list[SourcePair] = []
    seen: set[tuple[str, str]] = set()
    duplicates = 0
    for pair in pairs:
        key = normalized_pair_key(pair)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        unique.append(pair)
    return unique, duplicates


def _record(pair: SourcePair, *, split: Literal["train", "validation"]) -> CleanupRecord:
    digest = hashlib.sha256(
        f"{pair.source_name}\0{pair.source_record_id}\0{pair.source_revision}".encode()
    ).hexdigest()[:20]
    flags = ["upstream-generation-seed-unavailable"]
    if pair.category:
        flags.append(f"category:{pair.category}")
    if pair.upstream_split:
        flags.append(f"upstream-split:{pair.upstream_split}")
    return CleanupRecord(
        id=f"{pair.source_name}-{digest}",
        source_name=pair.source_name,
        source_revision=pair.source_revision,
        source_record_id=pair.source_record_id,
        license_spdx="MIT",
        split=split,
        transcript=pair.transcript,
        output=pair.output,
        language="en-US",
        commands=True,
        terminology=(),
        features=infer_features(pair),
        edit_script=(),
        synthetic=False,
        generator_version=None,
        seed=None,
        parent_id=None,
        quality_flags=tuple(flags),
    )


def select_records(
    pairs: Sequence[SourcePair],
    tokenizer: TokenizerLike,
    *,
    source_name: Literal["sotto", "aawaaz"],
    train_count: int,
    validation_count: int,
    seed: int,
    max_sequence_length: int = 2048,
) -> tuple[list[CleanupRecord], SelectionStats]:
    source_pairs = [pair for pair in pairs if pair.source_name == source_name]
    unique, duplicate_count = _deduplicate(source_pairs)
    conservative_exclusions = 0
    if source_name == "aawaaz":
        conservative: list[SourcePair] = []
        for pair in unique:
            if is_conservative_aawaaz_pair(pair):
                conservative.append(pair)
            else:
                conservative_exclusions += 1
        unique = conservative

    if source_name == "sotto":
        split_pairs = {
            "train": [pair for pair in unique if pair.upstream_split == "train"],
            "validation": [pair for pair in unique if pair.upstream_split == "validation"],
        }
    else:
        split_pairs = {"train": [], "validation": []}
        for pair in unique:
            bucket = int(stable_score(pair, seed=seed)[:8], 16) % 10
            split_pairs["validation" if bucket == 0 else "train"].append(pair)

    records: list[CleanupRecord] = []
    invalid = 0
    oversize = 0
    for split, count in (("train", train_count), ("validation", validation_count)):
        selected = 0
        candidates = sorted(split_pairs[split], key=lambda pair: stable_score(pair, seed=seed))
        for pair in candidates:
            try:
                record = _record(pair, split=split)
            except (TypeError, ValueError):
                invalid += 1
                continue
            try:
                tokenize_example(record, tokenizer, max_sequence_length=max_sequence_length)
            except OversizeExampleError:
                oversize += 1
                continue
            records.append(record)
            selected += 1
            if selected == count:
                break
        if selected != count:
            raise ValueError(
                f"{source_name} has only {selected} eligible {split} records; requested {count}"
            )

    return records, SelectionStats(
        input_records=len(source_pairs),
        duplicate_records=duplicate_count,
        conservative_filter_exclusions=conservative_exclusions,
        invalid_record_exclusions=invalid,
        oversize_exclusions=oversize,
    )


def load_sotto_pairs(root: Path) -> list[SourcePair]:
    import pyarrow.parquet as pq

    pairs: list[SourcePair] = []
    for split, filename in (
        ("train", "train-00000-of-00001.parquet"),
        ("validation", "validation-00000-of-00001.parquet"),
    ):
        path = root / "data" / filename
        table = pq.read_table(path, columns=["input", "output"])
        for index, row in enumerate(table.to_pylist()):
            pairs.append(
                SourcePair(
                    source_name="sotto",
                    source_revision=SOTTO_REVISION,
                    source_record_id=f"{split}:{index}",
                    transcript=row["input"],
                    output=row["output"],
                    upstream_split=split,
                )
            )
    return pairs


def load_aawaaz_pairs(root: Path) -> list[SourcePair]:
    pairs: list[SourcePair] = []
    for path in sorted(root.glob("*.jsonl")):
        category = path.stem
        with path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                row = json.loads(line)
                pairs.append(
                    SourcePair(
                        source_name="aawaaz",
                        source_revision=AAWAAZ_REVISION,
                        source_record_id=f"{category}:{index}",
                        transcript=row["input"],
                        output=row["output"],
                        category=category,
                    )
                )
    return pairs


def write_dataset(
    records: Sequence[CleanupRecord],
    stats: dict[str, SelectionStats],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n")

    feature_counts = Counter(feature for record in records for feature in record.features)
    length_counts: Counter[str] = Counter()
    for record in records:
        characters = len(record.transcript) + len(record.output)
        if characters <= 512:
            length_counts["characters_0000_0512"] += 1
        elif characters <= 2048:
            length_counts["characters_0513_2048"] += 1
        else:
            length_counts["characters_2049_plus"] += 1

    manifest = DatasetManifest(
        prompt_contract_version="raw-v1",
        sources={
            name: SourceManifest(
                revision=SOTTO_REVISION if name == "sotto" else AAWAAZ_REVISION,
                license_spdx="MIT",
                allowed_use="Local model training and evaluation under the declared MIT license",
                redistribution=(
                    "Do not redistribute from this repository; retain upstream attribution"
                ),
                record_count=sum(record.source_name == name for record in records),
            )
            for name in ("sotto", "aawaaz")
        },
        counts_by_language=dict(Counter(record.language for record in records)),
        counts_by_feature=dict(sorted(feature_counts.items())),
        counts_by_length_bucket=dict(sorted(length_counts.items())),
        deduplication={
            "method": "normalized exact prompt/completion pair within each source",
            **{
                f"{name}_duplicates": source_stats.duplicate_records
                for name, source_stats in stats.items()
            },
            **{
                f"{name}_conservative_exclusions": source_stats.conservative_filter_exclusions
                for name, source_stats in stats.items()
            },
        },
        oversize_exclusions={
            name: source_stats.oversize_exclusions for name, source_stats in stats.items()
        },
        sha256=sha256_file(records_path),
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return records_path, manifest_path
