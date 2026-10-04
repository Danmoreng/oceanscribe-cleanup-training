"""Budget, revision, family, privacy and export regressions for the local studio."""

from __future__ import annotations

import json

import pytest
from test_data_prep import FakeTokenizer

from oceanscribe_cleanup.review_studio.store import (
    BudgetError,
    ConflictError,
    ReviewStore,
    StudioError,
)


def record(identifier, *, split="train", output="Paket 07 bleibt gesperrt.", **extra):
    return {
        "id": identifier,
        "source_name": "unit-fixture",
        "source_revision": "fixture-v1",
        "source_record_id": identifier,
        "license_spdx": "Apache-2.0",
        "split": split,
        "language": "de-DE",
        "transcript": "äh Paket 07 bleibt gesperrt",
        "output": output,
        "commands": True,
        "terminology": [],
        "synthetic": False,
        **extra,
    }


def decide(store, card, *, action="accept", output=None, request="save", **extra):
    return store.decide(
        card["task_id"],
        card["revision"],
        card["input_sha256"],
        action,
        output=output,
        request_id=request,
        **extra,
    )


def test_revision_cas_retry_and_persistent_raw_input(tmp_path):
    store = ReviewStore(tmp_path)
    store.add_task(record("a"), origin={"provenance_checked": True})
    card = store.next_card()["card"]
    result = decide(store, card, action="edit", output="Paket 07 bleibt gesperrt.")
    assert decide(store, card, action="edit", output="Paket 07 bleibt gesperrt.") == result
    with pytest.raises(ConflictError):
        decide(store, card, request="stale")
    restarted = ReviewStore(tmp_path)
    assert restarted.dashboard()["budgets"]["reviews"]["used"] == 1
    exported = restarted.export_snapshot(FakeTokenizer())
    data = json.loads(
        (tmp_path / "snapshots" / exported["path"].split("/")[-1] / "records.jsonl").read_text()
    )
    assert data["transcript"] == "äh Paket 07 bleibt gesperrt"
    from oceanscribe_cleanup.training import validate_dataset_manifest

    snapshot = tmp_path / "snapshots" / exported["path"].split("/")[-1]
    validate_dataset_manifest(snapshot / "records.jsonl", snapshot / "manifest.json")


def test_all_fifty_tasks_including_deferred_are_counted_and_extension_is_bounded(tmp_path):
    store = ReviewStore(tmp_path)
    for kind, count in [("repair", 20), ("audit", 10), ("reference", 10), ("comparison", 10)]:
        for index in range(count):
            store.add_task(
                record(
                    f"{kind}-{index}",
                    split="train" if kind in {"repair", "audit"} else "validation",
                ),
                kind=kind,
            )
            card = store.next_card(kind)["card"]
            decide(store, card, action="defer", request=f"skip-{kind}-{index}")
    assert store.dashboard()["budgets"]["reviews"]["used"] == 50
    store.add_task(record("fifty-one"))
    with pytest.raises(BudgetError):
        store.next_card()
    assert ReviewStore(tmp_path).dashboard()["budgets"]["reviews"]["limit"] == 50
    with pytest.raises(StudioError):
        store.extend(False)
    store.extend(True)
    assert store.next_card()["card"]
    store.extend(True)
    assert store.dashboard()["budgets"]["reviews"]["limit"] == 100


def test_private_reference_has_no_teacher_target_and_never_enters_train_export(tmp_path):
    store = ReviewStore(tmp_path)
    store.add_task(
        record("gold", split="validation", output="HIDDEN TEACHER TARGET"), kind="reference"
    )
    card = store.next_card("reference")["card"]
    assert card["output"] == "" and "HIDDEN" not in json.dumps(card)
    decide(store, card, output="Paket 07 bleibt gesperrt.")
    store.add_task(
        record("train"),
        origin={"provenance_checked": True, "external_review_allowed": True},
        status="auto_accepted",
    )
    original_read = store._read

    def guard(path):
        assert not path.startswith("protected/")
        return original_read(path)

    store._read = guard
    for teacher in (False, True):
        export = store.export_snapshot(FakeTokenizer(), teacher=teacher)
        assert export["records"] == 1
        assert "gold" not in (tmp_path / export["path"] / "records.jsonl").read_text()


def test_explicit_expansion_persists_and_keeps_synthetic_dev_out_of_exports(tmp_path):
    store = ReviewStore(tmp_path)
    with pytest.raises(StudioError):
        store.authorize_workload(unlimited_reviews=True, recording_limit=60)
    store.authorize_workload(unlimited_reviews=True, recording_limit=60, confirm=True)
    for index in range(105):
        store.add_task(record(str(index)))
        decide(store, store.next_card()["card"], request=f"accept-{index}")
    dev = record(
        "synthetic-dev",
        split="validation",
        synthetic=True,
        source_name="original-model-generated-v3",
        generator_version="fixture-v1",
        seed=7,
    )
    store.add_task(dev, kind="synthetic_review", origin={"independent_human_gold": False})
    decide(store, store.next_card("synthetic_review")["card"], request="accept-dev")
    restarted = ReviewStore(tmp_path)
    assert restarted.dashboard()["budgets"]["reviews"]["limit"] is None
    assert restarted.dashboard()["budgets"]["recordings"]["limit"] == 60
    exported = restarted.export_snapshot(FakeTokenizer())
    assert "synthetic-dev" not in (tmp_path / exported["path"] / "records.jsonl").read_text()
    with pytest.raises(StudioError):
        store.add_task({**dev, "split": "test"}, kind="synthetic_review")
    with pytest.raises(StudioError):
        store.add_task(record("real-dev", split="validation"), kind="synthetic_review")


def test_expanded_recording_list_exposes_more_than_twenty_tasks(tmp_path):
    store = ReviewStore(tmp_path)
    for index in range(44):
        store.add_recording_task({**recording_task(), "task_id": f"clip-{index:02}"})
    assert len(store.recording_tasks()) == 10
    store.authorize_workload(unlimited_reviews=True, recording_limit=60, confirm=True)
    assert len(store.recording_tasks()) == 44


def test_free_tasks_are_hidden_and_cannot_start_but_history_is_preserved(tmp_path):
    store = ReviewStore(tmp_path)
    free = {**recording_task(), "task_id": "free-task", "split": "test", "mode": "free"}
    free.pop("script")
    store.add_recording_task(free)
    store.add_recording_task(recording_task())
    assert [task["task_id"] for task in store.recording_tasks()] == ["clip"]
    with pytest.raises(StudioError, match="Freie Sprechaufgaben"):
        store.start_recording("free-task", "start-free", consent_local=True)
    assert store.dashboard()["budgets"]["recordings"]["used"] == 0
    with store._transaction() as db:
        assert db.execute("SELECT COUNT(*) FROM recording_tasks").fetchone()[0] == 2


def test_luna_reading_keeps_family_split_hints_and_source_unapproved(tmp_path):
    store = ReviewStore(tmp_path)
    store.authorize_workload(unlimited_reviews=True, recording_limit=60, confirm=True)
    source = record(
        "luna-reading",
        split="validation",
        synthetic=True,
        source_name="original-model-generated-v3",
        generator_version="fixture",
        seed=7,
        terminology=["PostgreSQL"],
    )
    store.add_task(source, kind="synthetic_review", family_id="original-family")
    card = store.next_card("synthetic_review")["card"]
    result = store.recording_task_from_card(
        card["task_id"], card["revision"], card["input_sha256"], output="Paket 07 bleibt gesperrt."
    )
    assert result == store.recording_task_from_card(
        card["task_id"], card["revision"], card["input_sha256"], output="Paket 07 bleibt gesperrt."
    )
    task = result["task"]
    assert task["family_id"] == "original-family" and task["split"] == "validation"
    assert task["terminology"] == ["PostgreSQL"] and task["script"] == source["transcript"]
    assert result["source_review_approved"] is False
    assert store.dashboard()["pending_by_kind"]["synthetic_review"] == 1
    assert store.dashboard()["budgets"]["recordings"]["used"] == 0
    # Large available task lists do not reserve attempts or hide chosen Luna readings.
    for index in range(65):
        store.add_recording_task({**recording_task(), "task_id": f"extra-{index}"})
    assert len(store.recording_tasks()) == 66
    rid = store.start_recording(task["task_id"], "start", consent_local=True)["recording_id"]
    store.save_audio(rid, b"RIFF0000WAVEfixture", "audio/wav", "upload", engine="parakeet")
    store.complete_asr(rid, result={"engine": "parakeet", "transcript": "Paket 07 gesperrt."})
    store.confirm_recording(rid, output="Paket 07 gesperrt.", request_id="confirm")
    with store._transaction() as db:
        child = db.execute("SELECT * FROM tasks WHERE id LIKE 'own-asr/%'").fetchone()
        assert child["family"] == "original-family" and child["split"] == "validation"
        payload = store._read(child["current_path"])
        assert payload["record"]["transcript"] == "Paket 07 gesperrt."
        assert payload["record"]["terminology"] == ["PostgreSQL"]
        assert payload["origin"]["independent_human_gold"] is False
    # Later source edits reopen the real ASR review; never reconstruct it from a script.
    decide(store, card, action="edit", output="Paket 07 bleibt gesperrt", request="revise-source")
    with store._transaction() as db:
        child = db.execute("SELECT * FROM tasks WHERE id LIKE 'own-asr/%'").fetchone()
        payload = store._read(child["current_path"])
        assert child["status"] == "needs_human"
        assert payload["origin"]["human_verified"] is False
        assert payload["record"]["transcript"] == "Paket 07 gesperrt."
        assert payload["record"]["output"] == "Paket 07 gesperrt."
        assert payload["reference_frozen"] is False
    with pytest.raises(ConflictError):
        store.recording_task_from_card(card["task_id"], 999, card["input_sha256"])


def test_new_comparison_round_keeps_old_blinded_answers_immutable(tmp_path):
    store = ReviewStore(tmp_path)
    reference = store.add_task(record("reference", split="validation"), kind="reference")
    decide(store, store.next_card("reference")["card"], output="Paket 07 bleibt gesperrt.")
    answers = [{"model_id": "a", "output": "A"}, {"model_id": "b", "output": "B"}]
    first = store.add_comparison(reference, answers)
    second_answers = [{"model_id": "a", "output": "A"}, {"model_id": "c", "output": "C"}]
    second = store.add_comparison(reference, second_answers, comparison_run_id="round-2")
    assert first != second
    assert store.add_comparison(reference, answers) == first
    with pytest.raises(ConflictError):
        store.add_comparison(reference, second_answers)
    with store._transaction() as db:
        row = db.execute("SELECT * FROM tasks WHERE id=?", (first,)).fetchone()
        assert {a["model_id"] for a in store._read(row["current_path"])["answers"]} == {"a", "b"}


def test_parent_edit_regenerates_only_deterministic_children_not_other_language(tmp_path):
    store = ReviewStore(tmp_path)
    origin = {"provenance_checked": True}
    store.add_task(record("base"), family_id="family", origin=origin)
    store.add_task(
        record("de-preserve", parent_id="base", features=["preserve"]),
        family_id="family",
        origin=origin,
        status="auto_accepted",
    )
    store.add_task(
        record("en", language="en-US"), family_id="family", origin=origin, status="auto_accepted"
    )
    card = store.next_card()["card"]
    decide(store, card, action="edit", output="Paket 07 bleibt weiterhin gesperrt.")
    export = store.export_snapshot(FakeTokenizer())
    rows = [
        json.loads(line)
        for line in (tmp_path / export["path"] / "records.jsonl").read_text().splitlines()
    ]
    child = next(r for r in rows if r["id"] == "de-preserve")
    assert child["output"] == child["transcript"] == "Paket 07 bleibt weiterhin gesperrt."
    assert len(rows) == 3
    provenance = [
        json.loads(line)
        for line in (tmp_path / export["path"] / "provenance.jsonl").read_text().splitlines()
    ]
    assert not next(r for r in provenance if r["record_id"] == "de-preserve")["origin"][
        "human_verified"
    ]


@pytest.mark.parametrize("bad_gate", ["permission", "provenance", "oversize", "open"])
def test_whole_family_export_gate(tmp_path, bad_gate):
    store = ReviewStore(tmp_path)
    origin = {"provenance_checked": True, "external_review_allowed": True}
    store.add_task(record("a"), family_id="f", origin=origin, status="auto_accepted")
    other_origin = dict(origin)
    if bad_gate == "permission":
        other_origin["external_review_allowed"] = False
    if bad_gate == "provenance":
        other_origin["provenance_checked"] = False
    store.add_task(
        record("z", transcript="word " * 3000 if bad_gate == "oversize" else "Text"),
        family_id="f",
        origin=other_origin,
        status="needs_human" if bad_gate == "open" else "auto_accepted",
    )
    result = store.export_snapshot(FakeTokenizer(), teacher=bad_gate == "permission")
    assert result["records"] == 0 and result["families"] == 0


def recording_task():
    return {
        "task_id": "clip",
        "family_id": "scenario",
        "split": "train",
        "language": "de-DE",
        "script": "Bitte freigeben, nein, bitte nicht freigeben.",
        "question": "Natürlich sprechen",
    }


def test_recording_attempts_retry_dual_asr_and_deletion_propagate(tmp_path):
    store = ReviewStore(tmp_path)
    store.add_recording_task(recording_task())
    first = store.start_recording("clip", "start", consent_local=True)
    assert store.start_recording("clip", "start", consent_local=True) == first
    for i in range(9):
        store.start_recording("clip", f"start-{i}", consent_local=True)
    with pytest.raises(BudgetError):
        store.start_recording("clip", "eleven", consent_local=True)
    rid = first["recording_id"]
    wav = b"RIFF" + b"0000" + b"WAVE" + b"unit fixture payload"
    store.save_audio(rid, wav, "audio/wav", "upload", engine="both")
    assert ReviewStore(tmp_path).pending_asr_jobs()[0]["engine"] == "both"
    store.complete_asr(
        rid,
        result={
            "variants": [
                {
                    "engine": "nemotron",
                    "transcript": "nicht freigeben",
                    "provenance": {"unit_test": True},
                },
                {
                    "engine": "parakeet",
                    "transcript": "Nicht freigeben.",
                    "provenance": {"unit_test": True},
                },
            ]
        },
    )
    before = store.recording_status(rid)
    store.confirm_recording(
        rid,
        outputs={"nemotron": "Nicht freigeben.", "parakeet": "Nicht freigeben."},
        request_id="confirm",
    )
    export = store.export_snapshot(FakeTokenizer())
    assert export["records"] == 2 and export["families"] == 1
    assert store.recording_status(rid)["variants"] == before["variants"]
    assert store.export_snapshot(FakeTokenizer(), teacher=True)["records"] == 0
    store.delete_recording(rid, "delete")
    assert not (tmp_path / export["path"] / "records.jsonl").exists()
    assert (tmp_path / export["path"] / "BLOCKED.json").exists()
    assert store.export_snapshot(FakeTokenizer())["records"] == 0
    with pytest.raises(StudioError):
        store.audio_path(rid)


def test_asr_lost_negation_can_be_excluded_without_inventing_raw(tmp_path):
    store = ReviewStore(tmp_path)
    store.add_recording_task(recording_task())
    rid = store.start_recording("clip", "start", consent_local=True)["recording_id"]
    store.save_audio(rid, b"RIFF0000WAVEfixture", "audio/wav", "upload")
    store.complete_asr(
        rid,
        result={"engine": "nemotron", "transcript": "freigeben", "provenance": {"unit_test": True}},
    )
    store.confirm_recording(rid, output=None, unrecoverable=True, request_id="exclude")
    assert store.recording_status(rid)["raw_transcript"] == "freigeben"
    assert store.export_snapshot(FakeTokenizer())["records"] == 0


def test_human_can_mark_real_asr_loss_without_restoring_from_script(tmp_path):
    store = ReviewStore(tmp_path)
    store.add_recording_task(recording_task())
    rid = store.start_recording("clip", "start", consent_local=True)["recording_id"]
    store.save_audio(rid, b"RIFF0000WAVEfixture", "audio/wav", "upload")
    store.complete_asr(
        rid,
        result={
            "engine": "nemotron",
            "transcript": "Paket prüfen.",
            "provenance": {"unit_test": True},
        },
    )
    store.confirm_recording(rid, unrecoverable=True, request_id="exclude-number")
    assert store.recording_status(rid)["raw_transcript"] == "Paket prüfen."


def test_technical_asr_retry_keeps_audio_hash_attempt_budget_and_is_bounded(tmp_path):
    store = ReviewStore(tmp_path)
    store.add_recording_task(recording_task())
    rid = store.start_recording("clip", "start", consent_local=True)["recording_id"]
    store.save_audio(rid, b"RIFF0000WAVEfixture", "audio/wav", "upload", engine="both")
    audio = store.audio_path(rid).read_bytes()
    for i in range(3):
        store.complete_asr(rid, error="technical failure")
        result = store.retry_asr(rid, f"retry-{i}")
        assert store.retry_asr(rid, f"retry-{i}") == result
        assert result["engine"] == "both"
        assert store.dashboard()["budgets"]["recordings"]["used"] == 1
        assert store.audio_path(rid).read_bytes() == audio
    store.complete_asr(rid, error="technical failure")
    with pytest.raises(StudioError, match="Drei"):
        store.retry_asr(rid, "fourth")
    tasks = store.recording_tasks()
    assert tasks[0]["latest_recording_id"] == rid
    assert tasks[0]["recording_status"] == "asr_failed"
    assert tasks[0]["attempt_count"] == 1


def test_discarded_attempt_keeps_audio_but_cannot_be_requeued_or_confirmed(tmp_path):
    store = ReviewStore(tmp_path)
    store.add_recording_task(recording_task())
    rid = store.start_recording("clip", "start", consent_local=True)["recording_id"]
    store.save_audio(rid, b"RIFF0000WAVEfixture", "audio/wav", "upload")
    assert store.discard_recording(rid, "discard")["status"] == "discarded"
    assert store.discard_recording(rid, "discard")["status"] == "discarded"
    assert store.pending_asr_jobs() == []
    assert store.audio_path(rid).read_bytes() == b"RIFF0000WAVEfixture"
    with pytest.raises(ConflictError):
        store.complete_asr(rid, result={"transcript": "late result"})
    assert store.dashboard()["budgets"]["recordings"]["used"] == 1
    assert store.export_snapshot(FakeTokenizer())["records"] == 0


def test_comparison_cannot_mutate_and_is_budgeted_after_reference(tmp_path):
    store = ReviewStore(tmp_path)
    tid = store.add_task(record("gold", split="validation"), kind="reference")
    answers = [
        {"model_id": "A", "output": "Paket 07 bleibt gesperrt."},
        {"model_id": "B", "output": "Paket 70 bleibt gesperrt."},
    ]
    with pytest.raises(StudioError):
        store.add_comparison(tid, answers)
    card = store.next_card("reference")["card"]
    decide(store, card, output="Paket 07 bleibt gesperrt.")
    store.add_comparison(tid, answers)
    store.add_comparison(tid, answers)
    with pytest.raises(ConflictError):
        store.add_comparison(tid, [{**answers[0], "output": "changed"}, answers[1]])
    comparison = store.next_card("comparison")["card"]
    assert len(comparison["answers"]) == 2
    assert "model_id" not in json.dumps(comparison)
    assert store.dashboard()["budgets"]["reviews"]["used"] == 2
    with pytest.raises(StudioError):
        decide(store, comparison, request="missing-assessment")


@pytest.mark.parametrize(
    "raw,target,expected_warning",
    [
        (
            "Erstens die Liste prüfen, zweitens den Entwurf speichern.",
            "1. Die Liste prüfen\n2. Den Entwurf speichern",
            None,
        ),
        (
            "First, check the list. Second, save the draft.",
            "1. Check the list\n2. Save the draft",
            None,
        ),
        ("Paket prüfen.", "Paket 007 prüfen.", "number-not-detected-in-raw"),
        ("freigeben", "Nicht freigeben.", "negation-not-detected-in-raw"),
    ],
)
def test_human_number_and_negation_decisions_are_not_blocked(
    tmp_path, raw, target, expected_warning
):
    store = ReviewStore(tmp_path)
    store.add_recording_task(recording_task())
    rid = store.start_recording("clip", "start", consent_local=True)["recording_id"]
    store.save_audio(rid, b"RIFF0000WAVEfixture", "audio/wav", "upload")
    store.complete_asr(
        rid, result={"engine": "nemotron", "transcript": raw, "provenance": {"unit_test": True}}
    )
    store.confirm_recording(rid, unrecoverable=True, request_id="mistaken-loss")
    assert store.reopen_asr_loss(rid)["status"] == "needs_confirmation"
    result = store.confirm_recording(rid, output=target, request_id="human-review")
    assert result["status"] == "confirmed"
    assert [w["code"] for w in result["warnings"]] == (
        [expected_warning] if expected_warning else []
    )
    assert store.recording_status(rid)["raw_transcript"] == raw
    exported = store.export_snapshot(FakeTokenizer())
    rows = [
        json.loads(line)
        for line in (tmp_path / exported["path"] / "records.jsonl").read_text().splitlines()
    ]
    assert rows[0]["output"] == target and rows[0]["transcript"] == raw
    assert store.dashboard()["budgets"]["recordings"]["used"] == 1
