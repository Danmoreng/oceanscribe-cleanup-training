"""Idempotently seed Review Studio from the frozen fidelity-v2 snapshot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..records import CleanupRecord
from .store import ReviewStore, canonical

TRAIN_PATH = Path("data/processed/bilingual-fidelity-mix-v2/training/records.jsonl")
EXPECTED_TRAIN_RECORDS = 3478
SEED = 20261002
LOCAL_AUTHORIZATION = "existing-frozen-pilot-user-authorized"

BAD_FAMILY_IDS = frozenset(
    {
        "sotto-1bf095910f47dc7f342d",
        "sotto-1fea9b598d9f6a14d3a1",
        "sotto-280bf86a4a807b481aca",
    }
)

# The 47 first-pass KEEP families are a bounded regression-audit pool.
AUDIT_KEEP_IDS = frozenset(
    {
        "aawaaz-2e3d47f4d0eab72f818b",
        "sotto-4635ec91e51669de8c92",
        "aawaaz-3845de5c7c99261298cc",
        "sotto-43db5adee39c40192d47",
        "sotto-36a6f7530a3cbcfc399d",
        "aawaaz-1e6f1e475bc7cef012ed",
        "sotto-1727865baa0c1c06239e",
        "aawaaz-153300fa4274a2517151",
        "aawaaz-0eb00a472cbfb632a736",
        "sotto-4662ba976eeee2d933a9",
        "sotto-1507f49813a91c77e424",
        "aawaaz-4ea8ddb5101a72c0ffc5",
        "sotto-1f7c12195b62c21311f2",
        "sotto-13a5b8466695c0827051",
        "aawaaz-3e0394fb4935132912f7",
        "sotto-0c82f904d28a78089df3",
        "sotto-0fdc7f6bbf9f353c330c",
        "sotto-0f0005f74740959023d4",
        "aawaaz-0ae30fc60d56ca1e3f09",
        "aawaaz-26ee82d7f4c1710e33ef",
        "aawaaz-2ca9a110e7923583bfdd",
        "sotto-3c77fb28111aa89ab7e2",
        "aawaaz-27c3526c039934650c74",
        "sotto-2d6dfb2f3b6d6ab9ae5e",
        "sotto-3272009815975dae7fb7",
        "sotto-1ca2b770997647e2fb29",
        "aawaaz-3cf35e905390f44ffdee",
        "aawaaz-2c1413e9bd003134032b",
        "aawaaz-30b377a94cdcba73cf92",
        "sotto-0bc8caf01752016a57b0",
        "sotto-2f5a1608ac3be1d6640d",
        "aawaaz-1c89237731571dcd1d3e",
        "sotto-331b5e2825fd3db31134",
        "aawaaz-204cfee51374ba6da7ce",
        "aawaaz-1f3422a99f822add6e91",
        "sotto-4bf77eb52b66c4c20ea4",
        "aawaaz-03871ca123cbc411bd3e",
        "sotto-1dc3da85f378127bfcab",
        "aawaaz-0af85740a6b8cce89855",
        "sotto-3424d24af1a82b246faa",
        "sotto-0119dda7f81e9eae5705",
        "aawaaz-02296b4fdb473f1d077c",
        "aawaaz-199739c4ceee33fc5745",
        "aawaaz-0f909cf045e87fef9a11",
        "aawaaz-0e0fac65754f8ba8bd86",
        "aawaaz-0265177017f59d889477",
        "aawaaz-4ad799ced54297d9ceb9",
    }
)

PROPOSED_OUTPUTS = {
    "sotto-1bf095910f47dc7f342d": {
        "en-US": "We should update to 5.1 because it has the security patch.",
        "de-DE": (
            "Wir sollten auf 5.1 aktualisieren, weil diese Version den Sicherheitspatch enthält."
        ),
    },
    "sotto-1fea9b598d9f6a14d3a1": {
        "en-US": "The IRR target needs to be 10% for the board.",
        "de-DE": "Das IRR-Ziel muss für den Vorstand 10 % betragen.",
    },
}

REVIEW_QUESTIONS = {
    "sotto-1bf095910f47dc7f342d": (
        "Luna-Vorschlag: Begründung zum Sicherheitspatch erhalten; bitte prüfen."
    ),
    "sotto-1fea9b598d9f6a14d3a1": (
        "Luna-Vorschlag: IRR-Ziel als Bezugsgröße erhalten; bitte prüfen."
    ),
    "sotto-280bf86a4a807b481aca": (
        "Unklar: Der Rohtext sagt, Version 2.3 habe den Patch, aber nicht eindeutig, dass auf "
        "2.3 zurückgerollt werden soll. Die vorhandene Referenz ist nur ein Kandidat. Bitte "
        "Ziel festlegen oder ausschließen."
    ),
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _family_key(record: CleanupRecord) -> str:
    return canonical([record.source_name, record.source_revision, record.source_record_id])


def _base_id(record_id: str) -> str:
    return record_id.split("/", maxsplit=1)[0]


def _read_snapshot(path: Path) -> tuple[list[tuple[CleanupRecord, str]], str]:
    content = path.read_bytes()
    file_hash = _sha256(content)
    rows: list[tuple[CleanupRecord, str]] = []
    seen: set[str] = set()
    for line in content.splitlines(keepends=True):
        if not line.strip():
            continue
        source_line = line.removesuffix(b"\n")
        record = CleanupRecord.model_validate_json(source_line)
        if record.id in seen:
            raise ValueError(f"Duplicate snapshot record id: {record.id}")
        seen.add(record.id)
        rows.append((record, _sha256(source_line)))
    return rows, file_hash


def _legacy_origin(record: CleanupRecord, line_hash: str, snapshot_hash: str) -> dict[str, Any]:
    return {
        "provenance_checked": True,
        "local_authorization": LOCAL_AUTHORIZATION,
        "human_verified": False,
        "external_review_allowed": True,
        "legacy_retained": True,
        "source_kind": "synthetic-template"
        if record.source_name.startswith("oceanscribe-")
        else "external-dataset-record",
        "source_name": record.source_name,
        "source_revision": record.source_revision,
        "source_record_id": record.source_record_id,
        "license_assertion": record.license_spdx,
        "public_redistribution_clearance": "not-established",
        "dataset_snapshot_sha256": snapshot_hash,
        "source_line_sha256": line_hash,
        "record_sha256": _sha256(canonical(record.model_dump(mode="json")).encode()),
        "field_provenance": {
            "raw_transcript_sha256": _sha256(record.transcript.encode()),
            "cleanup_reference_sha256": _sha256(record.output.encode()),
            "historical_author": None,
            "historical_teacher_model_id": None,
            "historical_review_actor": None,
            "historical_review_time": None,
            "generator_version_recorded": record.generator_version,
            "generation_kind_recorded": record.generation_kind,
            "seed_recorded": record.seed,
            "parent_id_recorded": record.parent_id,
            "edit_script_sha256": _sha256(
                canonical([edit.model_dump(mode="json") for edit in record.edit_script]).encode()
            ),
        },
        "record_facts": {
            "synthetic": record.synthetic,
            "features": list(record.features),
            "quality_flags": list(record.quality_flags),
        },
    }


def _audit_selection(records: list[CleanupRecord]) -> set[str]:
    candidates = [
        record
        for record in records
        if _base_id(record.id) in AUDIT_KEEP_IDS
        and record.id == f"{_base_id(record.id)}/{record.language}"
    ]
    quotas = {
        ("sotto", "en-US"): 3,
        ("sotto", "de-DE"): 2,
        ("aawaaz", "en-US"): 2,
        ("aawaaz", "de-DE"): 3,
    }
    selected: set[str] = set()
    selected_families: set[str] = set()
    for (source, language), quota in quotas.items():
        pool = [
            record
            for record in candidates
            if record.source_name == source and record.language == language
        ]
        pool.sort(key=lambda record: _sha256(f"{SEED}\0{record.id}".encode()))
        added = 0
        for record in pool:
            family = _family_key(record)
            if family in selected_families:
                continue
            selected.add(record.id)
            selected_families.add(family)
            added += 1
            if added == quota:
                break
        if added != quota:
            raise ValueError(f"Audit pool cannot fill source/language stratum {source}/{language}")
    if len(selected) != 10:
        raise ValueError("Audit selection must contain exactly 10 cards")
    return selected


def _reference_selection(records: list[CleanupRecord]) -> list[CleanupRecord]:
    candidates = [
        record
        for record in records
        if record.split == "validation" and record.source_name in {"sotto", "aawaaz"}
    ]
    selected: list[CleanupRecord] = []
    families: set[str] = set()
    for language in ("en-US", "de-DE"):
        pool = [r for r in candidates if r.language == language]
        pool.sort(key=lambda r: _sha256(f"reference\0{SEED}\0{r.id}".encode()))
        count = 0
        for record in pool:
            family = _family_key(record)
            if family in families:
                continue
            selected.append(record)
            families.add(family)
            count += 1
            if count == 5:
                break
        if count != 5:
            raise ValueError(f"Validation set cannot fill five {language} references")
    return selected


def _add_task_idempotent(
    store: ReviewStore,
    record: CleanupRecord,
    *,
    kind: str,
    family_id: str,
    task_id: str,
    origin: dict[str, Any],
    status: str,
    proposed_output: str | None = None,
    question: str = "",
) -> str:
    normalized = record.model_dump(mode="json")
    if kind == "reference":
        normalized["output"] = ""
    elif proposed_output is not None:
        normalized["output"] = proposed_output
    expected_payload = {
        "record": normalized,
        "question": question,
        "origin": origin,
        "reference_frozen": False,
        "meaning_status": "not_evaluated",
    }
    expected_hash = _sha256(canonical(expected_payload).encode())
    with store._transaction() as db:
        prior = db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if prior:
            if prior["input_hash"] != expected_hash or prior["family"] != family_id:
                raise ValueError(f"Existing Studio task has different frozen input: {task_id}")
            return task_id
    return store.add_task(
        record,
        kind=kind,
        family_id=family_id,
        task_id=task_id,
        origin=origin,
        status=status,
        proposed_output=proposed_output,
        question=question,
    )


def _recording_tasks() -> list[dict[str, Any]]:
    # A single seeded plan reserves splits before the user makes any recording.
    definitions = [
        (
            "de-DE",
            "train",
            "scripted",
            "Zahlen mit führenden Nullen",
            "Bitte lies eine kurze Paketnotiz natürlich ein: "
            "Paket null acht bleibt gesperrt. Nein, Paket null sieben ist gemeint, "
            "weil dort das Siegel fehlt.",
            "Paket 07 bleibt gesperrt, weil dort das Siegel fehlt.",
        ),
        (
            "en-US",
            "train",
            "scripted",
            "Correction with rationale",
            "Dictate this short note naturally: Move the review to Friday. Actually, "
            "move it to Thursday because the release starts Friday.",
            "Move the review to Thursday because the release starts Friday.",
        ),
        (
            "de-DE",
            "train",
            "scripted",
            "Negation and preserve; commands disabled",
            "Sprich die kurze Notiz natürlich ein: Die Freigabe ist noch nicht erfolgt. "
            "Ähm, Status unverändert lassen.",
            "Die Freigabe ist noch nicht erfolgt. Status unverändert lassen.",
        ),
        (
            "en-US",
            "train",
            "scripted",
            "Short action list",
            "Dictate this list naturally: First, call Mina. Second, send the revised estimate. "
            "No, send it after legal review.",
            "First, call Mina. Second, send the revised estimate after legal review.",
        ),
        (
            "de-DE",
            "train",
            "scripted",
            "Code-switch",
            "Sprich natürlich: Das Meeting startet um neun, äh, actually make that half past nine, "
            "weil der Zug später ankommt.",
            "Das Meeting startet um half past nine, weil der Zug später ankommt.",
        ),
        (
            "de-DE",
            "validation",
            "scripted",
            "Terminänderung",
            "Sprich natürlich: Der Termin ist am Dienstag. Nein, am Mittwoch um zehn, "
            "weil der Raum dann frei ist.",
            "Der Termin ist am Mittwoch um zehn, weil der Raum dann frei ist.",
        ),
        (
            "en-US",
            "validation",
            "scripted",
            "Preserve and negation",
            "Dictate this short update naturally: No access was granted. The setting is unchanged, "
            "and the test can wait.",
            "No access was granted. The setting is unchanged, and the test can wait.",
        ),
        (
            "en-US",
            "validation",
            "scripted",
            "Code-switch",
            "Dictate naturally: Please keep the draft, äh, don't publish it yet, "
            "bis die Zahlen geprüft sind.",
            "Please keep the draft; don't publish it yet, bis die Zahlen geprüft sind.",
        ),
        ("de-DE", "test", "free", "Freie Situation", None, None),
        ("en-US", "test", "free", "Free situation", None, None),
    ]
    extension = [
        (
            "en-US",
            "train",
            "scripted",
            "Inventory correction",
            "Dictate naturally: We need twelve labels. Wait, make that twenty, "
            "because two boxes arrived late.",
            "We need twenty labels because two boxes arrived late.",
        ),
        (
            "de-DE",
            "train",
            "scripted",
            "Rückrufnotiz",
            "Sprich natürlich: Bitte ruf Frau Kern zurück. Ähm, nicht heute, "
            "sondern morgen nach dem Mittagessen.",
            "Bitte ruf Frau Kern morgen nach dem Mittagessen zurück.",
        ),
        (
            "en-US",
            "train",
            "scripted",
            "Unchanged status; commands disabled",
            "Dictate naturally: Keep the access disabled. Nothing has changed since yesterday.",
            "Keep the access disabled. Nothing has changed since yesterday.",
        ),
        (
            "de-DE",
            "train",
            "scripted",
            "Three action items",
            "Sprich natürlich: Erstens, die Liste prüfen. Zweitens, den Entwurf speichern. "
            "Nein, erst nach der Rückmeldung senden.",
            "Erstens, die Liste prüfen. Zweitens, den Entwurf speichern und erst nach der "
            "Rückmeldung senden.",
        ),
        (
            "en-US",
            "train",
            "scripted",
            "Code-switch",
            "Dictate naturally: The pickup is at eight, äh, um acht Uhr dreißig, "
            "because the office opens later.",
            "The pickup is um 8:30 because the office opens later.",
        ),
        (
            "de-DE",
            "validation",
            "scripted",
            "Mengen und Grund",
            "Sprich natürlich: Es fehlen vier Kartons. Nein, es fehlen drei, "
            "weil einer schon geliefert wurde.",
            "Es fehlen drei Kartons, weil einer schon geliefert wurde.",
        ),
        (
            "en-US",
            "validation",
            "scripted",
            "No insertion",
            "Dictate naturally: Leave the project name blank. We have not chosen one.",
            "Leave the project name blank. We have not chosen one.",
        ),
        (
            "de-DE",
            "validation",
            "scripted",
            "Code-switch",
            "Sprich natürlich: Der build ist fertig, äh, but don't deploy, bis der Test durch ist.",
            "Der Build ist fertig, but don't deploy until der Test durch ist.",
        ),
        ("de-DE", "test", "free", "Freie Situation", None, None),
        ("en-US", "test", "free", "Free situation", None, None),
    ]
    result = []
    for cohort, entries in (("00-main", definitions), ("01-extension", extension)):
        for index, (language, split, mode, title, script, target) in enumerate(entries, start=1):
            task_id = f"{cohort}-{index:02d}"
            payload: dict[str, Any] = {
                "task_id": task_id,
                "family_id": f"own-audio/{task_id}",
                "split": split,
                "language": language,
                "category": title,
                "prompt": (
                    f"Sprich diese Aufgabe natürlich ein: {title}. "
                    "Du kannst normal sprechen, dich korrigieren und kurze Füllwörter verwenden."
                    if language == "de-DE"
                    else (
                        f"Speak this naturally: {title}. "
                        "You may use normal fillers and self-correct."
                    )
                )
                if mode == "scripted"
                else (
                    "Erzähle frei von einer kurzen Alltagsaufgabe oder Nachricht."
                    if language == "de-DE"
                    else "Describe a short everyday task or message in your own words."
                ),
                "mode": mode,
                "cohort": cohort,
                "extension_only": cohort == "01-extension",
                "commands": title
                not in {
                    "Negation and preserve; commands disabled",
                    "Unchanged status; commands disabled",
                },
                "language_mix": ["en-US", "de-DE"] if title == "Code-switch" else [language],
                "script_origin": "agent",
                "script_generator_model_id": "gpt-6-luna",
                "script_generator_reasoning_effort": "unknown",
                "intended_output_origin": "agent_candidate" if target is not None else None,
                "speech_languages": ["en-US", "de-DE"] if title == "Code-switch" else [language],
                "code_switching": title == "Code-switch",
            }
            if split != "test" and script is not None:
                payload["script"] = script
                payload["intended_output"] = target
            result.append(payload)
    return result


def initialize_studio(store: ReviewStore, repo: Path) -> dict[str, Any]:
    """Seed bounded queues from the frozen snapshot without opening any task.

    Only the training snapshot is used to populate train tasks. Validation
    rows are read by this local bootstrap solely to choose ten protected
    reference cards; no validation text is returned or written to shared logs.
    Calling this again is idempotent and does not consume review/recording
    budgets because it never calls ``next_card`` or ``start_recording``.
    """
    snapshot_path = Path(repo) / TRAIN_PATH
    rows, snapshot_hash = _read_snapshot(snapshot_path)
    records = [record for record, _ in rows]
    train = [record for record in records if record.split == "train"]
    validation = [record for record in records if record.split == "validation"]
    if len(train) != EXPECTED_TRAIN_RECORDS:
        raise ValueError(
            f"Expected {EXPECTED_TRAIN_RECORDS} frozen training records; found {len(train)}"
        )
    if not BAD_FAMILY_IDS <= {_base_id(record.id) for record in train}:
        raise ValueError("Frozen training snapshot is missing a known audit family")

    by_id = {record.id: record for record in train}
    audit_ids = _audit_selection(train)
    references = _reference_selection(validation)
    # Verify the bilingual parent inputs exist before any persistent writes.
    for family in BAD_FAMILY_IDS:
        for language in ("en-US", "de-DE"):
            if f"{family}/{language}" not in by_id:
                raise ValueError(f"Known audit parent is missing: {family}/{language}")

    imports: list[tuple[CleanupRecord, str, str, str | None, str, dict[str, Any]]] = []
    line_hashes = {record.id: line_hash for record, line_hash in rows}
    for record in train:
        family_id = _base_id(record.id)
        base_record = record.id == f"{family_id}/{record.language}"
        origin = _legacy_origin(record, line_hashes[record.id], snapshot_hash)
        if family_id in BAD_FAMILY_IDS:
            if base_record:
                status = "needs_human"
                question = REVIEW_QUESTIONS[family_id]
                proposed = PROPOSED_OUTPUTS.get(family_id, {}).get(record.language)
                task_kind = "repair"
            else:
                status, question, proposed, task_kind = "quarantined", "", None, "repair"
        else:
            question = ""
            proposed = None
            status = "auto_accepted"
            task_kind = "audit" if record.id in audit_ids else "repair"
        task_id = f"{task_kind}:{record.id}"
        origin["seed_task_id"] = task_id
        origin["audit_seed"] = SEED if task_kind == "audit" else None
        # Stash metadata in the import plan; records are frozen models.
        imports.append((record, task_kind, status, proposed, question, origin))

    imported_ids = []
    for record, kind, status, proposed, question, origin in imports:
        imported_ids.append(
            _add_task_idempotent(
                store,
                record,
                kind=kind,
                family_id=_family_key(record),
                task_id=origin["seed_task_id"],
                origin=origin,
                status=status,
                proposed_output=proposed,
                question=question,
            )
        )

    reference_ids = []
    for record in references:
        row_hash = line_hashes.get(record.id)
        if row_hash is None:
            # Validation rows share the input file but are never rendered into this return value.
            row_hash = next(line_hash for candidate, line_hash in rows if candidate.id == record.id)
        origin = {
            "source_kind": "external-validation-candidate",
            "provenance_checked": False,
            "human_verified": False,
            "external_review_allowed": False,
            "source_name": record.source_name,
            "source_revision": record.source_revision,
            "source_record_id": record.source_record_id,
            "source_line_sha256": row_hash,
            "dataset_snapshot_sha256": snapshot_hash,
            "candidate_reference_sha256": _sha256(record.output.encode()),
            "candidate_reference_retained": False,
            "public_redistribution_clearance": "not-established",
        }
        reference_ids.append(
            _add_task_idempotent(
                store,
                record,
                kind="reference",
                family_id=_family_key(record),
                task_id=f"reference:{record.id}",
                origin=origin,
                status="needs_human",
                question="Rohtext zuerst prüfen; Referenz neu aus dem Rohtext festlegen.",
            )
        )

    recording_ids = []
    for task in _recording_tasks():
        expected_hash = _sha256(canonical(task).encode())
        with store._transaction() as db:
            prior = db.execute(
                "SELECT payload FROM recording_tasks WHERE id=?", (task["task_id"],)
            ).fetchone()
        if prior:
            pointer = json.loads(prior["payload"])
            if pointer.get("sha256") != expected_hash:
                raise ValueError(
                    f"Existing recording task has different frozen input: {task['task_id']}"
                )
        else:
            store.add_recording_task(task)
        recording_ids.append(task["task_id"])

    counts = {
        "frozen_train_records": len(train),
        "legacy_retained": sum(
            record.id.split("/", maxsplit=1)[0] not in BAD_FAMILY_IDS for record in train
        ),
        "known_bad_family_records": sum(
            record.id.split("/", maxsplit=1)[0] in BAD_FAMILY_IDS for record in train
        ),
        "human_repair_cards": sum(status == "needs_human" for _, _, status, _, _, _ in imports),
        "quarantined_descendants": sum(
            status == "quarantined" for _, _, status, _, _, _ in imports
        ),
        "audit_cards": len(audit_ids),
        "protected_reference_cards": len(reference_ids),
        "recording_tasks": len(recording_ids),
        "recording_tasks_main": 10,
        "recording_tasks_extension": 10,
    }
    return {
        "snapshot_sha256": snapshot_hash,
        "seed": SEED,
        "counts": counts,
        "queue_ids": {
            "repair": [task_id for task_id in imported_ids if task_id.startswith("repair:")],
            "audit": [task_id for task_id in imported_ids if task_id.startswith("audit:")],
            "reference": reference_ids,
        },
        "recording_task_ids": recording_ids,
    }
