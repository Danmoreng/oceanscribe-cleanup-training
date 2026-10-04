"""Append immutable original candidates to the explicitly expanded manual queue."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..records import CleanupRecord
from .store import ReviewStore, StudioError


def import_original_batches(store: ReviewStore, directory: Path) -> int:
    """Import every ready candidate, including FIX/REVIEW; do not approve any label."""
    imported = 0
    for folder in sorted(Path(directory).glob("batch-*")):
        if not (folder / "candidates.ready").is_file():
            continue
        task = json.loads((folder / "task.json").read_text())
        if not task["generation_policy"].get("original_only"):
            raise StudioError("Nur neue Original-Szenarien dürfen in diese Queue.")
        lines = (folder / "candidates.jsonl").read_bytes().splitlines()
        assignments = task["assignments"]
        if len(lines) != len(assignments):
            raise StudioError("Bereiter Batch ist unvollständig.")
        revision = hashlib.sha256((folder / "generator-prompt.txt").read_bytes()).hexdigest()
        for line, assignment in zip(lines, assignments, strict=True):
            item = json.loads(line)
            for field in ("family_id", "language", "split", "commands", "primary_category"):
                if item[field] != assignment[field]:
                    raise StudioError("Kandidat widerspricht unveränderlicher Batch-Zuordnung.")
            record = CleanupRecord.model_validate(
                {
                    "id": "new-" + item["family_id"],
                    "source_name": "original-model-generated-v3",
                    "source_revision": revision,
                    "source_record_id": item["family_id"],
                    "split": item["split"],
                    "language": item["language"],
                    "commands": item["commands"],
                    "terminology": item["terminology"],
                    "transcript": item["transcript"],
                    "output": item["output"],
                    "synthetic": True,
                    "generator_version": "original-bounded-v3",
                    "generation_kind": "stochastic",
                    "seed": None,
                    "features": ["original-synthetic-v3", item["primary_category"]],
                    "quality_flags": [
                        "generation-seed-unavailable",
                        "model-assisted-not-human-gold",
                    ],
                }
            )
            store.add_task(
                record.model_dump(mode="json"),
                kind="synthetic_review",
                task_id="synthetic_review:" + record.id,
                family_id=item["family_id"],
                question=(
                    "Modellgenerierter Originalfall: Ziel am Rohtext prüfen; Zahlen, Negationen, "
                    "Begründungen und Beziehungen erhalten. Nur belegbare Änderungen übernehmen. "
                    "Unlösbares ausschließen, Unsicheres zurückstellen. "
                    + (
                        "Synthetischer Development-Fall; bleibt außerhalb des Trainings. "
                        if item["split"] != "train"
                        else "Trainingskandidat. "
                    )
                    + f"Prüfschwerpunkt: {item['primary_category']}."
                ),
                origin={
                    "candidate_line_sha256": hashlib.sha256(line).hexdigest(),
                    "generator": item.get("provenance", {}),
                    "source_kind": "original-model-generated-v3",
                    "reference_origin": "model_assisted_human",
                    "independent_human_gold": False,
                    "provenance_checked": True,
                    "external_review_allowed": False,
                    "public_release_clearance": "not-established",
                },
            )
            imported += 1
    return imported


def import_recording_packet(store: ReviewStore, path: Path) -> int:
    """Append scripted tasks; protected test must remain spontaneous."""
    tasks = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise StudioError("Doppelte Aufnahmeaufgaben im Paket.")
    for task in tasks:
        if task["split"] == "test" or not task.get("script"):
            raise StudioError("Dieses Paket ist ausschließlich für synthetische Train/Dev-Diktate.")
        # Validate the complete prompt payload and canonical terminology before exposing it.
        CleanupRecord.model_validate(
            {
                "id": task["task_id"],
                "source_name": "scripted-packet-check",
                "split": task["split"],
                "language": task["language"],
                "transcript": task["script"],
                "output": task["intended_output"],
                "commands": task["commands"],
                "terminology": task.get("terminology", []),
                "synthetic": True,
                "generator_version": "scripted-original-v1",
                "generation_kind": "stochastic",
                "seed": None,
                "quality_flags": ["generation-seed-unavailable"],
            }
        )
        store.add_recording_task(task)
    return len(tasks)
