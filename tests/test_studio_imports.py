from __future__ import annotations

import json

import pytest
from test_studio_store import decide

from oceanscribe_cleanup.review_studio.imports import import_original_batches
from oceanscribe_cleanup.review_studio.store import ConflictError, ReviewStore


def test_original_import_preserves_reviewed_targets_and_fixed_split(tmp_path):
    folder = tmp_path / "inputs" / "batch-01"
    folder.mkdir(parents=True)
    assignment = {
        "family_id": "new-family",
        "language": "de-DE",
        "split": "validation",
        "commands": False,
        "primary_category": "negation",
    }
    candidate = {
        **assignment,
        "transcript": "äh nicht freigeben",
        "output": "Nicht freigeben.",
        "terminology": ["PostgreSQL"],
    }
    (folder / "task.json").write_text(
        json.dumps(
            {
                "generation_policy": {"original_only": True},
                "assignments": [assignment],
            }
        )
    )
    (folder / "generator-prompt.txt").write_text("fixture prompt")
    source = folder / "candidates.jsonl"
    source.write_text(json.dumps(candidate) + "\n")
    store = ReviewStore(tmp_path / "studio")
    store.authorize_workload(unlimited_reviews=True, confirm=True)
    assert import_original_batches(store, folder.parent) == 0
    (folder / "candidates.ready").touch()
    # No model review file required: unresolved candidates are also offered to the human.
    assert import_original_batches(store, folder.parent) == 1
    card = store.next_card("synthetic_review")["card"]
    assert card["split"] == "validation" and card["terminology"] == ["PostgreSQL"]
    decide(store, card, action="edit", output="nicht freigeben", request="review")
    assert import_original_batches(store, folder.parent) == 1
    with store._transaction() as db:
        row = db.execute("SELECT * FROM tasks").fetchone()
        assert row["status"] == "human_accepted" and row["revision"] == 1
        payload = store._read(row["current_path"])
        assert payload["record"]["output"] == "nicht freigeben"
        assert payload["origin"]["independent_human_gold"] is False
    source.write_text(json.dumps({**candidate, "transcript": "andere Eingabe"}) + "\n")
    with pytest.raises(ConflictError):
        import_original_batches(store, folder.parent)
