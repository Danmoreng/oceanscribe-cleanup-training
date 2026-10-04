"""Local GPU comparison of identical confirmed ASR inputs with three hint conditions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from oceanscribe_cleanup.config import load_config
from oceanscribe_cleanup.protected_evaluation import score_protected_outputs
from oceanscribe_cleanup.records import CleanupRecord
from oceanscribe_cleanup.review_studio.store import ReviewStore
from oceanscribe_cleanup.terminology_evaluation import ablation_variants, literal_hint_metrics
from oceanscribe_cleanup.training import load_cleanup_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studio", type=Path, default=Path("data/review-studio/current"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--training-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    store = ReviewStore(args.studio)
    destination = args.output.resolve()
    if not destination.is_relative_to(store.root / "protected"):
        parser.error("Private ASR evaluations must stay inside the studio protected directory")
    if destination.exists():
        parser.error("Frozen output already exists; choose a new directory")
    entries = []
    with store._transaction() as db:
        event_cut = db.execute("SELECT COALESCE(MAX(seq),0) FROM events").fetchone()[0]
        for row in db.execute(
            "SELECT * FROM tasks WHERE split='validation' AND status='human_accepted' ORDER BY id"
        ).fetchall():
            payload = store._read(row["current_path"])
            origin = payload["origin"]
            if not origin.get("recording_task", "").startswith(
                ("02-technical-", "03-technical-long-", "04-luna-")
            ):
                continue
            record = CleanupRecord.model_validate(payload["record"])
            if record.source_name != "own-local-asr" or not payload.get("reference_frozen"):
                continue
            if not any(term.casefold() in record.output.casefold() for term in record.terminology):
                continue
            entries.append(
                (
                    record,
                    {
                        "family_id": row["family"],
                        "studio_revision": row["revision"],
                        "current_sha256": hashlib.sha256(
                            (store.root / row["current_path"]).read_bytes()
                        ).hexdigest(),
                        "recording_id": origin["recording_id"],
                        "independent_human_gold": False,
                    },
                )
            )
    if not entries:
        print("Noch keine bestätigten Development-ASR-Paare mit passenden Fachbegriffen.")
        return
    train = load_cleanup_records(args.training_dataset)
    train_families = {record.source_record_id for record in train}
    train_raw = {" ".join(record.transcript.casefold().split()) for record in train}
    for record, provenance in entries:
        if provenance["family_id"] in train_families:
            raise ValueError("Evaluation family also occurs in supplied training snapshot")
        if " ".join(record.transcript.casefold().split()) in train_raw:
            raise ValueError("Evaluation transcript also occurs in supplied training snapshot")

    # Read-only collection completes before any model/GPU work. Nothing enters training.
    from peft import PeftModel
    from transformers import AutoTokenizer

    from oceanscribe_cleanup.evaluation import _generate, _load_text_model
    from oceanscribe_cleanup.tokenization import tokenize_example

    config = load_config(args.config)
    records = [record for record, _ in entries]
    families = [provenance["family_id"] for _, provenance in entries]
    variants = [ablation_variants(record) for record in records]
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.id, revision=config.model.revision, local_files_only=True
    )
    tokenizer.padding_side = "left"
    for conditions in variants:
        for record in conditions.values():
            tokenize_example(record, tokenizer, max_sequence_length=2048)
    destination.mkdir(parents=True, mode=0o700)
    data = "".join(
        json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n" for record in records
    )
    (destination / "records.jsonl").write_text(data)
    manifest = {
        "dataset_sha256": hashlib.sha256(data.encode()).hexdigest(),
        "studio_event_cut": event_cut,
        "records": len(records),
        "families": len(set(families)),
        "independent_human_gold": False,
        "human_reference_frozen": True,
        "meaning_status": "not_evaluated",
        "source_revisions": [p for _, p in entries],
        "adapter": str(args.adapter.resolve()),
        "model_revision": config.model.revision,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    model = PeftModel.from_pretrained(
        _load_text_model(config.model.id, config.model.revision).to("cuda"),
        args.adapter,
        is_trainable=False,
    )
    reports = {}
    for mode in ("with-hints", "without-hints", "irrelevant-hints"):
        prompts = [conditions[mode] for conditions in variants]
        outputs, eos = _generate(model, tokenizer, prompts, batch_size=1, max_new_tokens=1024)
        reports[mode] = {
            "scores": score_protected_outputs(prompts, outputs, eos, families),
            "canonical_terms": literal_hint_metrics(records, outputs),
            "presented_hint_false_insertion": literal_hint_metrics(prompts, outputs)[
                "false_insertion"
            ],
            "outputs": [
                {"record_id": record.id, "output": output, "stopped_at_eos": stopped}
                for record, output, stopped in zip(records, outputs, eos, strict=True)
            ],
        }
    (destination / "results.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                "records": len(records),
                "families": len(set(families)),
                "output": str(destination),
                "independent_human_gold": False,
            }
        )
    )


if __name__ == "__main__":
    main()
