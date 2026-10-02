from __future__ import annotations

import json
from pathlib import Path

from oceanscribe_cleanup.review_studio import bootstrap
from oceanscribe_cleanup.review_studio.store import ReviewStore


def _record(record_id: str, *, source: str, split: str, language: str, parent_id=None):
    base = record_id.split("/", maxsplit=1)[0]
    return {
        "id": record_id,
        "source_name": source,
        "source_revision": "source-rev-fixture",
        "source_record_id": base,
        "license_spdx": "MIT AND LicenseRef-UserProvided-LocalTraining",
        "split": split,
        "language": language,
        "commands": True,
        "terminology": [],
        "transcript": f"Fixture transcript {base}.",
        "output": f"Fixture cleanup {base}.",
        "synthetic": False,
        "features": ["external"],
        "parent_id": parent_id,
    }


def _write_fixture(repo: Path, *, include_descendant: bool = True) -> int:
    path = repo / bootstrap.TRAIN_PATH
    path.parent.mkdir(parents=True)
    lines = []
    for family in sorted(bootstrap.AUDIT_KEEP_IDS):
        source = family.split("-", maxsplit=1)[0]
        for language in ("en-US", "de-DE"):
            lines.append(
                _record(f"{family}/{language}", source=source, split="train", language=language)
            )
    for family in sorted(bootstrap.BAD_FAMILY_IDS):
        for language in ("en-US", "de-DE"):
            lines.append(
                _record(f"{family}/{language}", source="sotto", split="train", language=language)
            )
    if include_descendant:
        family = sorted(bootstrap.BAD_FAMILY_IDS)[0]
        lines.append(
            _record(
                f"{family}/en-US/preserve",
                source="sotto",
                split="train",
                language="en-US",
                parent_id=f"{family}/en-US",
            )
        )
    # Validation data in the fixture is synthetic and never contains real held-out text.
    for i in range(5):
        for source in ("sotto", "aawaaz"):
            for language in ("en-US", "de-DE"):
                family = f"validation-{source}-{language}-{i}"
                lines.append(
                    _record(
                        f"{family}/{language}", source=source, split="validation", language=language
                    )
                )
    path.write_text("".join(json.dumps(row) + "\n" for row in lines), encoding="utf-8")
    return sum(row["split"] == "train" for row in lines)


def test_bootstrap_is_seeded_idempotent_and_never_starts_budget(tmp_path, monkeypatch):
    train_count = _write_fixture(tmp_path)
    monkeypatch.setattr(bootstrap, "EXPECTED_TRAIN_RECORDS", train_count)
    store = ReviewStore(tmp_path / "studio")

    first = bootstrap.initialize_studio(store, tmp_path)
    task_count = store.dashboard()["counts"]
    second = bootstrap.initialize_studio(store, tmp_path)

    assert first == second
    assert first["counts"]["frozen_train_records"] == train_count
    assert first["counts"]["human_repair_cards"] == 6
    assert first["counts"]["quarantined_descendants"] == 1
    assert first["counts"]["audit_cards"] == 10
    assert first["counts"]["protected_reference_cards"] == 10
    assert first["counts"]["recording_tasks"] == 20
    assert store.dashboard()["counts"] == task_count
    assert store.dashboard()["budgets"]["reviews"]["used"] == 0
    assert store.dashboard()["budgets"]["recordings"]["used"] == 0

    audit_rows = store._transaction()
    with audit_rows as db:
        audits = db.execute("SELECT * FROM tasks WHERE kind='audit'").fetchall()
        assert len(audits) == 10
        details = [store._read(row["current_path"]) for row in audits]
    assert sum(item["record"]["source_name"] == "sotto" for item in details) == 5
    assert sum(item["record"]["language"] == "en-US" for item in details) == 5
    assert all(item["origin"]["audit_seed"] == 20261002 for item in details)


def test_bootstrap_quarantines_bad_family_and_keeps_luna_suggestions_as_candidates(
    tmp_path, monkeypatch
):
    train_count = _write_fixture(tmp_path)
    monkeypatch.setattr(bootstrap, "EXPECTED_TRAIN_RECORDS", train_count)
    store = ReviewStore(tmp_path / "studio")
    bootstrap.initialize_studio(store, tmp_path)

    with store._transaction() as db:
        all_rows = db.execute("SELECT * FROM tasks").fetchall()
        bad_rows = [
            row
            for row in all_rows
            if row["id"].split(":", maxsplit=1)[-1].split("/", maxsplit=1)[0]
            in bootstrap.BAD_FAMILY_IDS
        ]
        rows = [store._read(row["current_path"]) | {"status": row["status"]} for row in bad_rows]
    proposals = [row for row in rows if row["status"] == "needs_human"]
    descendants = [row for row in rows if row["status"] == "quarantined"]
    assert len(proposals) == 6
    assert len(descendants) == 1
    by_id = {row["record"]["id"]: row for row in proposals}
    en = by_id["sotto-1bf095910f47dc7f342d/en-US"]
    assert en["record"]["transcript"] == "Fixture transcript sotto-1bf095910f47dc7f342d."
    assert (
        en["record"]["output"] == bootstrap.PROPOSED_OUTPUTS["sotto-1bf095910f47dc7f342d"]["en-US"]
    )
    ambiguous = by_id["sotto-280bf86a4a807b481aca/en-US"]
    assert "Unklar" in ambiguous["question"]
    assert "Kandidat" in ambiguous["question"]
    assert ambiguous["record"]["output"] == "Fixture cleanup sotto-280bf86a4a807b481aca."


def test_protected_references_hide_candidate_target_and_are_external_validation_only(
    tmp_path, monkeypatch
):
    train_count = _write_fixture(tmp_path)
    monkeypatch.setattr(bootstrap, "EXPECTED_TRAIN_RECORDS", train_count)
    store = ReviewStore(tmp_path / "studio")
    result = bootstrap.initialize_studio(store, tmp_path)

    with store._transaction() as db:
        rows = db.execute("SELECT * FROM tasks WHERE kind='reference'").fetchall()
        cards = [store._card(row) for row in rows]
        payloads = [store._read(row["current_path"]) for row in rows]
    assert len(cards) == 10
    assert sum(card["language"] == "en-US" for card in cards) == 5
    assert sum(card["language"] == "de-DE" for card in cards) == 5
    assert len({payload["record"]["source_record_id"] for payload in payloads}) == 10
    assert all(card["output"] == "" for card in cards)
    assert all(payload["origin"]["candidate_reference_retained"] is False for payload in payloads)
    assert all(
        payload["origin"]["public_redistribution_clearance"] == "not-established"
        for payload in payloads
    )
    assert all("HIDDEN" not in json.dumps(card) for card in cards)
    assert len(result["queue_ids"]["reference"]) == 10


def test_recording_plan_reserves_main_and_extension_splits_without_test_targets():
    tasks = bootstrap._recording_tasks()
    main = [task for task in tasks if task["cohort"] == "00-main"]
    extension = [task for task in tasks if task["cohort"] == "01-extension"]
    assert len(main) == len(extension) == 10
    for cohort in (main, extension):
        assert {
            split: sum(task["split"] == split for task in cohort)
            for split in ("train", "validation", "test")
        } == {
            "train": 5,
            "validation": 3,
            "test": 2,
        }
        single_locale = [task for task in cohort if len(task["language_mix"]) == 1]
        assert sum(task["language"] == "de-DE" for task in single_locale) == 4
        assert sum(task["language"] == "en-US" for task in single_locale) == 4
        assert sum(len(task["language_mix"]) == 2 for task in cohort) == 2
        assert all(
            "script" not in task and "intended_output" not in task
            for task in cohort
            if task["split"] == "test"
        )
        assert all(
            task.get("script") and task.get("intended_output")
            for task in cohort
            if task["split"] != "test"
        )
    assert all(task["extension_only"] is False for task in main)
    assert all(task["extension_only"] is True for task in extension)
    assert (
        len(
            {task["script"] for task in main if "script" in task}
            & {task["script"] for task in extension if "script" in task}
        )
        == 0
    )
    assert all(task["script_origin"] == "agent" for task in tasks)
    assert all(task["script_generator_model_id"] == "gpt-6-luna" for task in tasks)
    assert all(task["script_generator_reasoning_effort"] == "unknown" for task in tasks)
    assert all(task["code_switching"] == (len(task["speech_languages"]) == 2) for task in tasks)
    commands_disabled = [task for task in tasks if not task["commands"]]
    assert len(commands_disabled) == 2
    assert all(
        "preserve" in task["category"].lower() or "unchanged" in task["category"].lower()
        for task in commands_disabled
    )
    code_switch_targets = [
        task["intended_output"] for task in tasks if task["category"] == "Code-switch"
    ]
    assert any("half past nine" in target for target in code_switch_targets)
    assert any("bis die Zahlen geprüft sind" in target for target in code_switch_targets)
    assert any("um 8:30" in target for target in code_switch_targets)
    assert any("but don't deploy" in target for target in code_switch_targets)
