"""Persistent bounded human work, immutable inputs, and gated snapshot exports."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import uuid
from collections import Counter
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from ..records import CleanupRecord

KINDS = {"repair": 20, "audit": 10, "reference": 10, "comparison": 10}
ACCEPTED = {"auto_accepted", "human_accepted"}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: bytes):
    return hashlib.sha256(value).hexdigest()


def now():
    return datetime.now(UTC).isoformat()


class StudioError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


class ConflictError(StudioError):
    def __init__(self, message="Stand wurde inzwischen geändert; bitte neu laden."):
        super().__init__(message, 409)


class BudgetError(StudioError):
    def __init__(self, message="Geplanter Beitrag abgeschlossen. Keine automatische Erweiterung."):
        super().__init__(message, 409)


class ReviewStore:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)
        for name in ("train", "protected", "audio", "snapshots", "jobs"):
            (self.root / name).mkdir(exist_ok=True)
            os.chmod(self.root / name, 0o700)
        self.db = self.root / "work.sqlite3"
        with self._transaction() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS tasks(
                  id TEXT PRIMARY KEY,family TEXT NOT NULL,kind TEXT NOT NULL,split TEXT NOT NULL,
                  language TEXT NOT NULL,status TEXT NOT NULL,revision INTEGER NOT NULL,
                  input_hash TEXT NOT NULL,input_path TEXT NOT NULL,current_path TEXT NOT NULL,
                  started INTEGER NOT NULL DEFAULT 0,created TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events(
                  seq INTEGER PRIMARY KEY AUTOINCREMENT,task_id TEXT,action TEXT NOT NULL,
                  revision INTEGER,metadata TEXT NOT NULL,created TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS requests(
                  id TEXT PRIMARY KEY,input_hash TEXT NOT NULL,response TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS recording_tasks(
                  id TEXT PRIMARY KEY,payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS recordings(
                  id TEXT PRIMARY KEY,task_id TEXT NOT NULL,status TEXT NOT NULL,
                  payload TEXT NOT NULL,created TEXT NOT NULL);
            """)
            db.execute("INSERT OR IGNORE INTO settings VALUES('extended','false')")
            db.execute("INSERT OR IGNORE INTO settings VALUES('recording_attempts','0')")

    @contextmanager
    def _transaction(self):
        db = sqlite3.connect(self.db, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _file(self, area, payload):
        data = canonical(payload).encode()
        relative = Path(area) / f"{uuid.uuid4().hex}.json"
        path = self.root / relative
        with path.open("xb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        return str(relative), digest(data)

    def _read(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise StudioError("Ungültiger lokaler Pfad.")
        return json.loads(path.read_text())

    def _event(self, db, task, action, revision=None, **metadata):
        # Never put raw transcript/reference/audio content in the shared event log.
        db.execute(
            "INSERT INTO events(task_id,action,revision,metadata,created) VALUES(?,?,?,?,?)",
            (task, action, revision, canonical(metadata), now()),
        )

    def _budgets(self, db):
        extended = json.loads(
            db.execute("SELECT value FROM settings WHERE key='extended'").fetchone()[0]
        )
        used = dict(Counter(row[0] for row in db.execute("SELECT kind FROM tasks WHERE started=1")))
        attempts = int(
            db.execute("SELECT value FROM settings WHERE key='recording_attempts'").fetchone()[0]
        )
        factor = 2 if extended else 1
        return {
            "reviews": {
                "used": sum(used.values()),
                "limit": 50 * factor,
                "maximum": 100,
                "by_kind": {
                    k: {"used": used.get(k, 0), "limit": n * factor} for k, n in KINDS.items()
                },
            },
            "recordings": {"used": attempts, "limit": 10 * factor, "maximum": 20},
        }

    def _cached(self, db, request_id, payload):
        if not isinstance(request_id, str) or not request_id or len(request_id) > 200:
            raise StudioError("Ein eindeutiger request_id ist erforderlich.")
        key = digest(canonical(payload).encode())
        prior = db.execute("SELECT * FROM requests WHERE id=?", (request_id,)).fetchone()
        if prior:
            if prior["input_hash"] != key:
                raise ConflictError("Request-ID wurde für eine andere Aktion verwendet.")
            return key, json.loads(prior["response"])
        return key, None

    def _cache(self, db, request_id, key, result):
        db.execute("INSERT INTO requests VALUES(?,?,?)", (request_id, key, canonical(result)))
        return result

    def add_task(
        self,
        record,
        *,
        kind="repair",
        family_id=None,
        question="",
        origin=None,
        status="needs_human",
        task_id=None,
        proposed_output=None,
    ):
        record = CleanupRecord.model_validate(record).model_dump(mode="json")
        if kind not in KINDS:
            raise StudioError("Unbekannte Aufgabenart.")
        if kind in {"repair", "audit"} and record["split"] != "train":
            raise StudioError("Dev/Test wird nicht in Reparatur- oder Agentenqueues aufgenommen.")
        if kind in {"reference", "comparison"} and record["split"] == "train":
            raise StudioError("Bewertungsanker darf kein Trainingsfall sein.")
        family = family_id or canonical(
            [record["source_name"], record["source_revision"], record["source_record_id"]]
        )
        identifier = task_id or f"{kind}:{record['id']}"
        area = "train" if record["split"] == "train" else "protected"
        payload = {
            "record": record,
            "question": question,
            "origin": origin or {},
            "reference_frozen": False,
            "meaning_status": "not_evaluated",
        }
        if kind == "reference":
            # Do not retain teacher targets in the human reference task input.
            payload["record"]["output"] = ""
        elif proposed_output is not None:
            payload["record"]["output"] = proposed_output
            CleanupRecord.model_validate(payload["record"])
        with self._transaction() as db:
            prior = db.execute("SELECT * FROM tasks WHERE id=?", (identifier,)).fetchone()
            if prior:
                old = self._read(prior["input_path"])
                if prior["family"] != family or old["record"] != payload["record"]:
                    raise ConflictError(
                        "Aufgaben-ID gehört bereits zu anderen unveränderlichen Eingaben."
                    )
                return identifier
            path, record_hash = self._file(area, payload)
            db.execute(
                "INSERT INTO tasks VALUES(?,?,?,?,?,?,?,?,?,?,0,?)",
                (
                    identifier,
                    family,
                    kind,
                    record["split"],
                    record["language"],
                    status,
                    0,
                    record_hash,
                    path,
                    path,
                    now(),
                ),
            )
            self._event(
                db, identifier, "import", 0, input_sha256=record_hash, split=record["split"]
            )
        return identifier

    def dashboard(self):
        with self._transaction() as db:
            counts = dict(Counter(row[0] for row in db.execute("SELECT status FROM tasks")))
            families = dict(
                Counter(
                    row[0] for row in db.execute("SELECT split FROM tasks GROUP BY family,split")
                )
            )
            recording_states = dict(
                Counter(row[0] for row in db.execute("SELECT status FROM recordings"))
            )
            corrected = db.execute(
                "SELECT COUNT(DISTINCT task_id) FROM events WHERE action='edit'"
            ).fetchone()[0]
            return {
                "budgets": self._budgets(db),
                "counts": counts,
                "families": families,
                "recordings": recording_states,
                "human_corrected_tasks": corrected,
                "session_size": 5,
                "meaning_status": "not_evaluated",
                "saving_starts_training": False,
            }

    def extend(self, confirm=False):
        if confirm is not True:
            raise StudioError("Die freiwillige Erweiterung muss bewusst bestätigt werden.")
        with self._transaction() as db:
            db.execute("UPDATE settings SET value='true' WHERE key='extended'")
            self._event(db, None, "budget_extension", review_limit=100, recording_limit=20)
        return self.dashboard()

    def _card(self, row):
        if digest((self.root / row["input_path"]).read_bytes()) != row["input_hash"]:
            raise ConflictError(
                "Unveränderliche Eingabedatei stimmt nicht mehr mit ihrem Hash überein."
            )
        payload = self._read(row["current_path"])
        record = payload["record"]
        result = {
            "task_id": row["id"],
            "kind": row["kind"],
            "split": row["split"],
            "language": row["language"],
            "raw_transcript": record["transcript"],
            "output": record["output"],
            "question": payload["question"],
            "revision": row["revision"],
            "input_sha256": row["input_hash"],
            "commands": record["commands"],
            "terminology": record["terminology"],
            "meaning_status": payload.get("meaning_status", "not_evaluated"),
        }
        if row["kind"] == "comparison":
            result["answers"] = [
                {"label": f"Antwort {i + 1}", "output": a["output"]}
                for i, a in enumerate(payload.get("answers", []))
            ]
            result["reference"] = record["output"]
        elif row["kind"] == "reference" and not payload.get("reference_frozen"):
            result["output"] = ""
        # Automatic confidence, origin labels and previous teacher refs stay hidden in audit.
        return result

    def next_card(self, kind="repair", task_id=None):
        if kind not in KINDS:
            raise StudioError("Unbekannte Aufgabenart.")
        with self._transaction() as db:
            if task_id:
                row = db.execute(
                    "SELECT * FROM tasks WHERE id=? AND kind=?", (task_id, kind)
                ).fetchone()
            else:
                statuses = (
                    "('needs_human','auto_accepted')" if kind == "audit" else "('needs_human')"
                )
                row = db.execute(
                    f"SELECT * FROM tasks WHERE kind=? AND status IN {statuses} "
                    "ORDER BY started DESC,created,id LIMIT 1",
                    (kind,),
                ).fetchone()
            if not row:
                return {"card": None, "budgets": self._budgets(db)}
            if row["status"] not in {"needs_human", "auto_accepted"}:
                raise ConflictError(
                    "Abgeschlossene Karte benötigt einen neuen budgetierten Reviewauftrag."
                )
            if not row["started"]:
                budgets = self._budgets(db)["reviews"]
                category = budgets["by_kind"][kind]
                if budgets["used"] >= budgets["limit"] or category["used"] >= category["limit"]:
                    raise BudgetError()
                db.execute("UPDATE tasks SET started=1 WHERE id=?", (row["id"],))
                self._event(db, row["id"], "view", row["revision"], kind=kind)
            return {"card": self._card(row), "budgets": self._budgets(db)}

    def decide(
        self,
        task_id,
        expected_revision,
        input_sha256,
        action,
        output=None,
        request_id="",
        error_codes=None,
        semantic_failure=None,
        **extra,
    ):
        if action not in {"accept", "edit", "defer", "exclude", "undo"}:
            raise StudioError("Unbekannte Entscheidung.")
        data = {
            "task_id": task_id,
            "expected_revision": expected_revision,
            "input_sha256": input_sha256,
            "action": action,
            "output": output,
            "error_codes": error_codes,
            "semantic_failure": semantic_failure,
            "extra": extra,
        }
        with self._transaction() as db:
            key, cached = self._cached(db, request_id, data)
            if cached is not None:
                return cached
            row = db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
            if not row or not row["started"]:
                raise StudioError("Aufgabe muss zuerst geöffnet werden.")
            if row["revision"] != expected_revision or row["input_hash"] != input_sha256:
                raise ConflictError()
            if digest((self.root / row["input_path"]).read_bytes()) != input_sha256:
                raise ConflictError("Gespeicherte Eingabe wurde verändert.")
            payload = self._read(row["current_path"])
            previous_output = payload["record"]["output"]
            if row["kind"] not in {"audit", "comparison"}:
                semantic_failure = None
            if row["status"] not in {"needs_human", "auto_accepted"} and action != "undo":
                raise ConflictError("Karte ist bereits abgeschlossen.")
            status = {
                "accept": "human_accepted",
                "edit": "human_accepted",
                "defer": "deferred",
                "exclude": "quarantined",
                "undo": "needs_human",
            }[action]
            if action in {"accept", "edit"}:
                if row["kind"] == "reference" and output is None:
                    raise StudioError("Bitte die Referenz vom Rohtext aus festlegen.")
                if output is not None:
                    if not isinstance(output, str):
                        raise StudioError("Ziel muss Text sein; leere Completion ist erlaubt.")
                    payload["record"]["output"] = output
                CleanupRecord.model_validate(payload["record"])
                payload["reference_frozen"] = True
                payload["origin"]["human_verified"] = True
                if row["kind"] == "audit" and isinstance(semantic_failure, bool):
                    payload["meaning_status"] = "failure" if semantic_failure else "passed"
                if row["kind"] == "comparison":
                    if payload["record"]["output"] != previous_output:
                        raise StudioError(
                            "Die festgeschriebene Referenz darf im Vergleich nicht geändert werden."
                        )
                    if not isinstance(semantic_failure, bool):
                        raise StudioError(
                            "Modellvergleich benötigt eine ausdrückliche Bedeutungsbewertung."
                        )
                    payload["meaning_status"] = "failure" if semantic_failure else "passed"
                    payload["assessment"] = extra
                    reviews = extra.get("answer_reviews")
                    if (
                        not isinstance(reviews, list)
                        or len(reviews) != 2
                        or {r.get("answer_index") for r in reviews} != {0, 1}
                        or any(not isinstance(r.get("semantic_failure"), bool) for r in reviews)
                    ):
                        raise StudioError(
                            "Beide verblindeten Antworten benötigen eine eigene Bewertung."
                        )
            elif action == "undo":
                # Undo approval, retain revision history; a changed target is never silently lost.
                payload["reference_frozen"] = False
                payload["origin"]["human_verified"] = False
                payload["meaning_status"] = "not_evaluated"
            payload["error_codes"] = error_codes or []
            path, revision_hash = self._file(
                "train" if row["split"] == "train" else "protected", payload
            )
            revision = row["revision"] + 1
            db.execute(
                "UPDATE tasks SET status=?,revision=?,current_path=? WHERE id=?",
                (status, revision, path, task_id),
            )
            # Exclusion blocks the family. Target corrections recheck its descendants.
            if (
                action in {"edit", "exclude", "undo"}
                or payload["record"]["output"] != previous_output
            ):
                if action == "exclude":
                    db.execute(
                        "UPDATE tasks SET status='quarantined',revision=revision+1 "
                        "WHERE family=? AND id!=?",
                        (row["family"], task_id),
                    )
                else:
                    self._refresh_descendants(db, row, payload, accepted=status == "human_accepted")
                self._event(db, task_id, "invalidate_family", revision, family=row["family"])
            if row["kind"] == "audit" and semantic_failure is True:
                cluster = payload["origin"].get("cluster")
                for candidate in db.execute("SELECT * FROM tasks WHERE split='train'").fetchall():
                    if (
                        cluster
                        and self._read(candidate["current_path"])["origin"].get("cluster")
                        == cluster
                    ):
                        db.execute(
                            "UPDATE tasks SET status='quarantined',revision=revision+1 WHERE id=?",
                            (candidate["id"],),
                        )
                self._event(db, task_id, "severe_audit_failure", cluster=cluster)
            self._event(
                db,
                task_id,
                action,
                revision,
                revision_sha256=revision_hash,
                error_codes=error_codes or [],
                meaning_evaluated=isinstance(semantic_failure, bool),
            )
            result = {
                "task_id": task_id,
                "status": status,
                "revision": revision,
                "input_sha256": row["input_hash"],
                "budgets": self._budgets(db),
            }
            return self._cache(db, request_id, key, result)

    def _refresh_descendants(self, db, row, parent, accepted):
        parent_record = parent["record"]
        for child in db.execute(
            "SELECT * FROM tasks WHERE family=? AND id!=?", (row["family"], row["id"])
        ).fetchall():
            content = self._read(child["current_path"])
            record = content["record"]
            if record.get("parent_id") != parent_record["id"]:
                if content["origin"].get("reference_task_id") == row["id"]:
                    db.execute(
                        "UPDATE tasks SET status='quarantined',revision=revision+1 WHERE id=?",
                        (child["id"],),
                    )
                continue
            features = record.get("features", [])
            status = "quarantined"
            if accepted and ("preserve" in features or "absent-hint" in features):
                record["output"] = parent_record["output"]
                if "preserve" in features:
                    record["transcript"] = parent_record["output"]
                record["quality_flags"] = [*record["quality_flags"], "studio-derived-rechecked"]
                content["origin"].update(
                    human_verified=False,
                    derived_from=parent_record["id"],
                    parent_revision=row["revision"] + 1,
                )
                CleanupRecord.model_validate(record)
                status = "auto_accepted"
            path, file_hash = self._file(
                "train" if row["split"] == "train" else "protected", content
            )
            db.execute(
                "UPDATE tasks SET status=?,revision=revision+1,current_path=? WHERE id=?",
                (status, path, child["id"]),
            )
            self._event(
                db,
                child["id"],
                "derived_recheck",
                child["revision"] + 1,
                status=status,
                revision_sha256=file_hash,
                human_verified=False,
            )

    def add_comparison(self, reference_task_id, answers):
        if len(answers) != 2 or any(set(a) != {"model_id", "output"} for a in answers):
            raise StudioError("Ein Vergleich enthält genau zwei Modellantworten.")
        with self._transaction() as db:
            row = db.execute("SELECT * FROM tasks WHERE id=?", (reference_task_id,)).fetchone()
            if not row or row["kind"] != "reference" or row["status"] != "human_accepted":
                raise StudioError("Referenz muss vor jeder Modellantwort festgeschrieben sein.")
            payload = self._read(row["current_path"])
            if not payload["reference_frozen"]:
                raise StudioError("Referenz ist nicht festgeschrieben.")
            frozen_hash = digest((self.root / row["current_path"]).read_bytes())
            record, family = payload["record"], row["family"]
        task_id = self.add_task(
            record,
            kind="comparison",
            family_id=family,
            task_id=f"comparison:{reference_task_id}:{frozen_hash}",
            question="Welche Antwort bewahrt die Bedeutung und bereinigt sinnvoll?",
            origin={"reference_sha256": frozen_hash, "reference_task_id": reference_task_id},
        )
        with self._transaction() as db:
            row = db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
            payload = self._read(row["current_path"])
            if int(digest(task_id.encode())[:2], 16) % 2:
                answers = list(reversed(answers))
            if "answers" in payload:
                if payload["answers"] != answers:
                    raise ConflictError("Bestehender Vergleich ist unveränderlich.")
                return task_id
            payload["answers"] = answers
            path, _ = self._file("protected", payload)
            db.execute("UPDATE tasks SET current_path=? WHERE id=?", (path, task_id))
        return task_id

    def add_recording_task(self, payload):
        if payload["split"] not in {"train", "validation", "test"}:
            raise StudioError("Ungültiger Split.")
        if payload["split"] == "test" and (payload.get("script") or payload.get("intended_output")):
            raise StudioError("Verschlossener Test hat nur eine freie Situation.")
        # Protected script content stays outside shared SQLite/agent manifests.
        area = "train" if payload["split"] == "train" else "protected"
        with self._transaction() as db:
            prior = db.execute(
                "SELECT * FROM recording_tasks WHERE id=?", (payload["task_id"],)
            ).fetchone()
            if prior:
                if self._read(json.loads(prior["payload"])["path"]) != payload:
                    raise ConflictError(
                        "Aufnahmeaufgabe hat bereits eine andere unveränderliche Vorlage."
                    )
                return
            path, file_hash = self._file(area, payload)
            pointer = {"path": path, "sha256": file_hash, "split": payload["split"]}
            db.execute(
                "INSERT OR IGNORE INTO recording_tasks VALUES(?,?)",
                (payload["task_id"], canonical(pointer)),
            )

    def recording_tasks(self):
        with self._transaction() as db:
            rows = db.execute("SELECT * FROM recording_tasks ORDER BY id").fetchall()
            budget = self._budgets(db)["recordings"]
            tasks = []
            for row in rows:
                task = self._read(json.loads(row["payload"])["path"])
                attempts = db.execute(
                    "SELECT id,status FROM recordings WHERE task_id=? "
                    "ORDER BY created DESC,id DESC",
                    (row["id"],),
                ).fetchall()
                task.update(
                    latest_recording_id=attempts[0]["id"] if attempts else None,
                    recording_status=attempts[0]["status"] if attempts else None,
                    attempt_count=len(attempts),
                )
                tasks.append(task)
            return tasks[: 10 if budget["limit"] == 10 else 20]

    def start_recording(self, task_id, request_id, consent_local=False):
        if consent_local is not True:
            raise StudioError("Lokale Audioverarbeitung muss bewusst bestätigt werden.")
        with self._transaction() as db:
            key, cached = self._cached(db, request_id, {"start": task_id, "consent": consent_local})
            if cached is not None:
                return cached
            budget = self._budgets(db)["recordings"]
            if budget["used"] >= budget["limit"]:
                raise BudgetError()
            row = db.execute("SELECT * FROM recording_tasks WHERE id=?", (task_id,)).fetchone()
            if not row:
                raise StudioError("Unbekannte Aufnahmeaufgabe.")
            task = self._read(json.loads(row["payload"])["path"])
            if task.get("extension_only") and budget["limit"] == 10:
                raise BudgetError()
            rid = uuid.uuid4().hex
            payload = {
                "task_pointer": json.loads(row["payload"]),
                "split": task["split"],
                "family_id": task["family_id"],
                "language": task["language"],
                "consent_local": True,
                "external_review_allowed": False,
                "speaker_id": "local-speaker-1",
                "created": now(),
            }
            db.execute(
                "INSERT INTO recordings VALUES(?,?,?,?,?)",
                (rid, task_id, "recording", canonical(payload), now()),
            )
            db.execute(
                "UPDATE settings SET value=? WHERE key='recording_attempts'",
                (str(budget["used"] + 1),),
            )
            self._event(db, rid, "recording_attempt", split=task["split"], family=task["family_id"])
            return self._cache(db, request_id, key, {"recording_id": rid, "status": "recording"})

    def save_audio(
        self,
        recording_id,
        data,
        mime,
        request_id,
        engine="nemotron",
        language=None,
        browser_metadata=None,
    ):
        if engine not in {"nemotron", "parakeet", "both"}:
            raise StudioError("Ungültige ASR-Auswahl.")
        supported = {
            "audio/webm": ".webm",
            "audio/ogg": ".ogg",
            "audio/wav": ".wav",
            "audio/x-wav": ".wav",
            "audio/mp4": ".m4a",
        }
        if not isinstance(data, bytes) or not 12 <= len(data) <= 25 * 1024 * 1024:
            raise StudioError("Audio muss zwischen 12 Bytes und 25 MiB groß sein.")
        base_mime = mime.split(";")[0].strip().lower()
        if base_mime not in supported:
            raise StudioError("Nicht unterstützter Audiocodec.")
        valid = (
            (
                base_mime in {"audio/wav", "audio/x-wav"}
                and data[:4] == b"RIFF"
                and data[8:12] == b"WAVE"
            )
            or (base_mime == "audio/webm" and data[:4] == b"\x1aE\xdf\xa3")
            or (base_mime == "audio/ogg" and data[:4] == b"OggS")
            or (base_mime == "audio/mp4" and data[4:8] == b"ftyp")
        )
        if not valid:
            raise StudioError("Audioinhalt passt nicht zum angegebenen Format.")
        with self._transaction() as db:
            key, cached = self._cached(
                db,
                request_id,
                {
                    "audio": recording_id,
                    "sha256": digest(data),
                    "mime": mime,
                    "engine": engine,
                    "language": language,
                    "browser_metadata": browser_metadata,
                },
            )
            if cached is not None:
                return cached
            row = db.execute("SELECT * FROM recordings WHERE id=?", (recording_id,)).fetchone()
            if not row or row["status"] != "recording":
                raise ConflictError("Aufnahme ist nicht mehr offen.")
            payload = json.loads(row["payload"])
            if language not in {None, "auto", payload["language"]}:
                raise StudioError("ASR-Sprache stimmt nicht mit der Aufnahmeaufgabe überein.")
            path = self.root / "audio" / (recording_id + supported[base_mime])
            with path.open("xb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            payload.update(
                audio_path=str(path.relative_to(self.root)),
                audio_sha256=digest(data),
                mime=mime,
                engine=engine,
                asr_language=payload["language"] if language is None else language,
                browser_metadata=browser_metadata or {},
            )
            db.execute(
                "UPDATE recordings SET status='asr_pending',payload=? WHERE id=?",
                (canonical(payload), recording_id),
            )
            self._event(db, recording_id, "audio_saved", audio_sha256=digest(data))
            return self._cache(
                db,
                request_id,
                key,
                {
                    "recording_id": recording_id,
                    "audio_path": str(path),
                    "status": "asr_pending",
                    "engine": engine,
                    "language": payload["asr_language"],
                },
            )

    def discard_recording(self, recording_id, request_id, reason="unsuitable-recording"):
        """Exclude an attempt without deleting audio or renewing its budget."""
        with self._transaction() as db:
            key, cached = self._cached(db, request_id, {"discard": recording_id, "reason": reason})
            if cached is not None:
                return cached
            row = db.execute("SELECT * FROM recordings WHERE id=?", (recording_id,)).fetchone()
            if not row or row["status"] in {"confirmed", "deleted"}:
                raise ConflictError("Diese Aufnahme kann nicht so verworfen werden.")
            payload = json.loads(row["payload"])
            payload["discard_reason"] = reason
            db.execute(
                "UPDATE recordings SET status='discarded',payload=? WHERE id=?",
                (canonical(payload), recording_id),
            )
            self._event(db, recording_id, "recording_discarded", reason=reason)
            return self._cache(
                db, request_id, key, {"recording_id": recording_id, "status": "discarded"}
            )

    def retry_asr(self, recording_id, request_id):
        """Technical retry of the same immutable audio; never a new recording attempt."""
        with self._transaction() as db:
            key, cached = self._cached(db, request_id, {"retry_asr": recording_id})
            if cached is not None:
                return cached
            row = db.execute("SELECT * FROM recordings WHERE id=?", (recording_id,)).fetchone()
            if not row or row["status"] != "asr_failed":
                raise ConflictError("Nur fehlgeschlagene Transkriptionen können erneut starten.")
            payload = json.loads(row["payload"])
            if payload.get("asr_retries", 0) >= 3:
                raise StudioError(
                    "Drei technische Wiederholungen ausgeschöpft. Audio bleibt erhalten."
                )
            audio = self.root / payload.get("audio_path", "missing")
            if not audio.is_file() or digest(audio.read_bytes()) != payload.get("audio_sha256"):
                raise StudioError("Gespeichertes Audio fehlt oder sein Hash stimmt nicht überein.")
            payload["asr_retries"] = payload.get("asr_retries", 0) + 1
            payload.pop("error", None)
            db.execute(
                "UPDATE recordings SET status='asr_pending',payload=? WHERE id=?",
                (canonical(payload), recording_id),
            )
            self._event(db, recording_id, "asr_retried", retry=payload["asr_retries"])
            return self._cache(
                db,
                request_id,
                key,
                {
                    "recording_id": recording_id,
                    "status": "asr_pending",
                    "engine": payload.get("engine", "nemotron"),
                    "language": payload.get("asr_language", payload["language"]),
                },
            )

    def pending_asr_jobs(self):
        with self._transaction() as db:
            pending = []
            for row in db.execute("SELECT * FROM recordings WHERE status='asr_pending'"):
                payload = json.loads(row["payload"])
                pending.append(
                    {
                        "recording_id": row["id"],
                        "engine": payload.get("engine", "nemotron"),
                        "language": payload.get("asr_language", payload["language"]),
                    }
                )
            return pending

    def complete_asr(self, recording_id, result=None, error=None):
        with self._transaction() as db:
            row = db.execute("SELECT * FROM recordings WHERE id=?", (recording_id,)).fetchone()
            if not row or row["status"] != "asr_pending":
                raise ConflictError("ASR-Job ist nicht offen.")
            payload = json.loads(row["payload"])
            if error:
                payload["error"] = str(error)[:1000]
                status, raw = "asr_failed", None
            else:
                variants = result.get("variants", [result]) if isinstance(result, dict) else []
                valid = [
                    v
                    for v in variants
                    if isinstance(v, dict)
                    and isinstance(v.get("transcript"), str)
                    and v["transcript"].strip()
                ]
                raw = valid[0]["transcript"] if valid else None
                if not valid:
                    status, raw = "asr_unrecoverable", None
                else:
                    # Raw ASR remains byte-exact in its own immutable file.
                    asr = {
                        "variants": [
                            {**v, "raw_sha256": digest(v["transcript"].encode())} for v in valid
                        ],
                        "audio_sha256": payload["audio_sha256"],
                        "errors": result.get("errors", []),
                    }
                    path, asr_hash = self._file(
                        "train" if payload["split"] == "train" else "protected", asr
                    )
                    payload.update(asr_path=path, asr_sha256=asr_hash)
                    status = "needs_confirmation"
            db.execute(
                "UPDATE recordings SET status=?,payload=? WHERE id=?",
                (status, canonical(payload), recording_id),
            )
            self._event(db, recording_id, "asr_finished", status=status)
            return {
                "recording_id": recording_id,
                "status": status,
                "raw_transcript": raw,
                "error": payload.get("error"),
            }

    def recording_status(self, recording_id):
        with self._transaction() as db:
            row = db.execute("SELECT * FROM recordings WHERE id=?", (recording_id,)).fetchone()
            if not row:
                raise StudioError("Aufnahme nicht gefunden.", 404)
            payload = json.loads(row["payload"])
            asr = (
                self._read(payload["asr_path"])
                if payload.get("asr_path") and row["status"] != "deleted"
                else {}
            )
            variants = [
                {
                    "engine": v.get("engine", "local"),
                    "raw_transcript": v["transcript"],
                    "raw_sha256": v["raw_sha256"],
                }
                for v in asr.get("variants", [])
            ]
            return {
                "recording_id": recording_id,
                "status": row["status"],
                "raw_transcript": variants[0]["raw_transcript"] if variants else None,
                "variants": variants,
                "errors": asr.get("errors", []),
                "error": payload.get("error"),
                "language": payload["language"],
                "split": payload["split"],
            }

    def confirm_recording(
        self,
        recording_id,
        output=None,
        unrecoverable=False,
        request_id="",
        outputs=None,
        unrecoverable_engines=None,
    ):
        with self._transaction() as db:
            key, cached = self._cached(
                db,
                request_id,
                {
                    "confirm": recording_id,
                    "output": output,
                    "unrecoverable": unrecoverable,
                    "outputs": outputs,
                    "unrecoverable_engines": unrecoverable_engines,
                },
            )
            if cached is not None:
                return cached
            row = db.execute("SELECT * FROM recordings WHERE id=?", (recording_id,)).fetchone()
            if not row or row["status"] != "needs_confirmation":
                raise ConflictError("Diese Aufnahme hat keinen bestätigbaren ASR-Rohtext.")
            payload = json.loads(row["payload"])
            status = "asr_unrecoverable" if unrecoverable else "confirmed"
            if not unrecoverable:
                asr = self._read(payload["asr_path"])
                for variant in asr["variants"]:
                    engine = variant.get("engine", "local")
                    if engine in (unrecoverable_engines or []):
                        continue
                    final_output = outputs.get(engine) if isinstance(outputs, dict) else output
                    if not isinstance(final_output, str):
                        raise StudioError(
                            "Jede ASR-Variante braucht ein textlich lösbares Ziel oder ASR-Verlust."
                        )
                    self._confirm_variant(db, row, payload, variant, final_output)
                if not payload.get("record_ids"):
                    status = "asr_unrecoverable"
            db.execute(
                "UPDATE recordings SET status=?,payload=? WHERE id=?",
                (status, canonical(payload), recording_id),
            )
            self._event(db, recording_id, "recording_confirmed", status=status)
            return self._cache(
                db, request_id, key, {"recording_id": recording_id, "status": status}
            )

    def _confirm_variant(self, db, row, payload, variant, output):
        recording_id = row["id"]
        engine = variant.get("engine", "local")
        if engine not in {"local", "nemotron", "parakeet"}:
            raise StudioError("Unbekannte ASR-Variante.")
        task = self._read(payload["task_pointer"]["path"])
        scripted = bool(task.get("script"))
        raw = variant["transcript"].casefold()
        target = output.casefold()
        negation = (
            r"\b(?:nicht|kein\w*|nie|niemals|nein|ohne|not|no|never|without|cannot|can't|don't)\b"
        )
        if re.search(negation, target) and not re.search(negation, raw):
            raise StudioError(
                "Das Ziel ergänzt eine im ASR-Text fehlende Negation. Als ASR-Verlust markieren."
            )
        # Conservative absence guard; spoken-number formatting remains allowed.
        # Other numeric changes still require the human text-solvability review.
        spoken_number = (
            r"\b(?:null|eins|ein|eine|einen|zwei|drei|vier|fünf|sechs|sieben|acht|neun|zehn|"
            r"elf|zwölf|dreizehn|vierzehn|fünfzehn|sechzehn|siebzehn|achtzehn|neunzehn|"
            r"zwanzig|dreißig|vierzig|fünfzig|sechzig|siebzig|achtzig|neunzig|hundert|tausend|"
            r"zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
            r"fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|"
            r"sixty|seventy|eighty|ninety|hundred|thousand|\w*(?:zig|ßig|hundert|tausend))\b"
        )
        if re.search(r"\d", target) and not (
            re.search(r"\d", raw) or re.search(spoken_number, raw)
        ):
            raise StudioError(
                "Das Ziel ergänzt eine im ASR-Text fehlende Zahl. Als ASR-Verlust markieren."
            )
        record = {
            "id": f"own-asr/{recording_id}/{engine}",
            "source_name": "own-local-asr",
            "source_revision": payload["asr_sha256"],
            "source_record_id": payload["family_id"],
            "license_spdx": "LicenseRef-Private-LocalTraining",
            "split": payload["split"],
            "language": payload["language"],
            "commands": task.get("commands", True),
            "terminology": task.get("terminology", []),
            "transcript": variant["transcript"],
            "output": output,
            "synthetic": scripted,
            "generator_version": "studio-human-audio-scripted-v1" if scripted else None,
            "generation_kind": "stochastic" if scripted else None,
            "seed": None,
            "features": ["real-asr"],
            "quality_flags": ["generation-seed-unavailable"] if scripted else [],
        }
        record = CleanupRecord.model_validate(record).model_dump(mode="json")
        origin = {
            "source_kind": "human_audio_scripted",
            "reference_origin": "model_assisted_human",
            "asr_origin": variant.get("provenance"),
            "human_verified": True,
            "asr_variant": engine,
            "external_review_allowed": False,
            "recording_id": recording_id,
            "recording_task": row["task_id"],
            "asr_raw_immutable": True,
        }
        origin["source_kind"] = "human_audio_scripted" if scripted else "human_audio_spontaneous"
        origin.update(
            speech_languages=task.get("speech_languages", [task["language"]]),
            code_switching=task.get("code_switching", False),
            script_origin=task.get("script_origin", "agent") if scripted else None,
            script_generator_model_id=task.get("script_generator_model_id") if scripted else None,
            script_generator_reasoning_effort=task.get("script_generator_reasoning_effort")
            if scripted
            else None,
            reference_origin="model_assisted_human" if scripted else "human_text-confirmed",
            accent="natural German accent (user supplied)",
        )
        final = {
            "record": record,
            "question": "",
            "origin": origin,
            "reference_frozen": True,
            "meaning_status": "not_evaluated",
        }
        path, record_hash = self._file(
            "train" if payload["split"] == "train" else "protected", final
        )
        kind = "repair" if payload["split"] == "train" else "reference"
        db.execute(
            "INSERT INTO tasks VALUES(?,?,?,?,?,?,?,?,?,?,0,?)",
            (
                record["id"],
                payload["family_id"],
                kind,
                record["split"],
                record["language"],
                "human_accepted",
                1,
                record_hash,
                path,
                path,
                now(),
            ),
        )
        payload.setdefault("record_ids", []).append(record["id"])

    def audio_path(self, recording_id):
        with self._transaction() as db:
            row = db.execute("SELECT * FROM recordings WHERE id=?", (recording_id,)).fetchone()
            if not row or row["status"] == "deleted":
                raise StudioError("Audio nicht verfügbar.", 404)
            payload = json.loads(row["payload"])
            path = (self.root / payload.get("audio_path", "missing")).resolve()
            if not path.is_relative_to(self.root / "audio") or not path.is_file():
                raise StudioError("Audio nicht verfügbar.", 404)
            return path

    def delete_recording(self, recording_id, request_id):
        with self._transaction() as db:
            key, cached = self._cached(db, request_id, {"delete": recording_id})
            if cached is not None:
                return cached
            row = db.execute("SELECT * FROM recordings WHERE id=?", (recording_id,)).fetchone()
            if not row:
                raise StudioError("Unbekannte Aufnahme.", 404)
            payload = json.loads(row["payload"])
            if payload.get("audio_path"):
                (self.root / payload["audio_path"]).unlink(missing_ok=True)
            if payload.get("asr_path"):
                (self.root / payload["asr_path"]).unlink(missing_ok=True)
            db.execute("UPDATE recordings SET status='deleted' WHERE id=?", (recording_id,))
            db.execute(
                "UPDATE tasks SET status='quarantined',revision=revision+1 WHERE family=?",
                (payload["family_id"],),
            )
            # Block existing exports containing the deleted recording.
            for path in (self.root / "snapshots").glob("*/provenance.jsonl"):
                if recording_id in path.read_text():
                    hashes = {p.name: digest(p.read_bytes()) for p in path.parent.glob("*.json*")}
                    for item in path.parent.glob("*.json*"):
                        item.unlink()
                    (path.parent / "BLOCKED.json").write_text(
                        canonical(
                            {
                                "reason": "audio-deleted",
                                "recording_id": recording_id,
                                "previous_hashes": hashes,
                            }
                        )
                    )
            for area in ("train", "protected"):
                for path in (self.root / area).glob("*.json"):
                    content = json.loads(path.read_text())
                    if content.get("origin", {}).get("recording_id") == recording_id:
                        path.unlink()
            self._event(db, recording_id, "audio_deleted", family=payload["family_id"])
            return self._cache(
                db, request_id, key, {"recording_id": recording_id, "status": "deleted"}
            )

    def export_snapshot(self, tokenizer=None, max_sequence_length=2048, teacher=False):
        from ..tokenization import OversizeExampleError, tokenize_example

        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3.5-0.8B-Base",
                revision="dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
                local_files_only=True,
            )
        # Fixed revision cut, held while reading immutable version files; no protected data is read.
        with self._transaction() as db:
            rows = db.execute(
                "SELECT * FROM tasks WHERE split='train' ORDER BY family,id"
            ).fetchall()
            cut = db.execute("SELECT COALESCE(MAX(seq),0) FROM events").fetchone()[0]
            blocked = {r["family"] for r in rows if r["status"] not in ACCEPTED}
            grouped = {}
            reasons = []
            for row in rows:
                if row["family"] in blocked:
                    reasons.append({"task_id": row["id"], "reason": "unresolved-family"})
                    continue
                payload = self._read(row["current_path"])
                origin = payload["origin"]
                if teacher and not origin.get("external_review_allowed", False):
                    blocked.add(row["family"])
                    reasons.append(
                        {"task_id": row["id"], "reason": "external-review-not-authorized"}
                    )
                    continue
                if not origin.get("provenance_checked", False) and origin.get(
                    "source_kind"
                ) not in {"human_audio_scripted", "human_audio_spontaneous"}:
                    blocked.add(row["family"])
                    reasons.append({"task_id": row["id"], "reason": "provenance-unchecked"})
                    continue
                record = CleanupRecord.model_validate(payload["record"])
                try:
                    tokenize_example(record, tokenizer, max_sequence_length=max_sequence_length)
                except OversizeExampleError:
                    blocked.add(row["family"])
                    reasons.append({"task_id": row["id"], "reason": "oversize-whole-family"})
                    continue
                grouped.setdefault(row["family"], []).append((row, record, origin))
            selected = [
                item for family, items in grouped.items() if family not in blocked for item in items
            ]
            unique = []
            seen = set()
            for row, record, origin in selected:
                key = (
                    record.language,
                    record.commands,
                    record.terminology,
                    " ".join(record.transcript.split()).casefold(),
                    " ".join(record.output.split()).casefold(),
                )
                if key in seen:
                    reasons.append({"task_id": row["id"], "reason": "normalized-duplicate"})
                    continue
                seen.add(key)
                unique.append((row, record, origin))
            selected = unique
        identifier = f"{'teacher' if teacher else 'train'}-{cut}-{uuid.uuid4().hex[:8]}"
        tmp = self.root / "snapshots" / (".tmp-" + identifier)
        target = self.root / "snapshots" / identifier
        tmp.mkdir()
        try:
            data = "".join(
                canonical(record.model_dump(mode="json")) + "\n" for _, record, _ in selected
            )
            sidecars = [
                {
                    "record_id": record.id,
                    "record_sha256": digest(canonical(record.model_dump(mode="json")).encode()),
                    "family_id": row["family"],
                    "revision": row["revision"],
                    "origin": origin,
                    "input_sha256": row["input_hash"],
                }
                for row, record, origin in selected
            ]
            (tmp / "records.jsonl").write_text(data)
            (tmp / "provenance.jsonl").write_text("".join(canonical(p) + "\n" for p in sidecars))
            manifest = {
                "schema_version": "studio-export-v1",
                "prompt_contract": "raw-v1",
                "revision_cut": cut,
                "records": len(selected),
                "families": len({r["family"] for r, _, _ in selected}),
                "split": "train",
                "teacher_export": teacher,
                "sha256": digest(data.encode()),
                "created": now(),
                "excluded": reasons,
                "human_verified_records": sum(
                    o.get("human_verified", False) for _, _, o in selected
                ),
            }
            from ..manifests import DatasetManifest, SourceManifest

            sources = {}
            for name in sorted({record.source_name for _, record, _ in selected}):
                source_records = [r for _, r, _ in selected if r.source_name == name]
                revisions = sorted({r.source_revision for r in source_records if r.source_revision})
                licenses = sorted({r.license_spdx for r in source_records if r.license_spdx})
                sources[name] = SourceManifest(
                    revision=revisions[0]
                    if len(revisions) == 1
                    else digest(canonical(revisions).encode()),
                    license_spdx=" AND ".join(licenses) if licenses else None,
                    allowed_use=(
                        "User-authorized local cleanup experiment; per-record provenance in sidecar"
                    ),
                    redistribution="Not publicly cleared; no release by Studio",
                    record_count=len(source_records),
                )
            lengths = Counter()
            for _, record, _ in selected:
                size = tokenize_example(
                    record, tokenizer, max_sequence_length=max_sequence_length
                ).sequence_length
                lengths[
                    "0-512"
                    if size <= 512
                    else "513-1024"
                    if size <= 1024
                    else "1025-2048"
                    if size <= 2048
                    else "2049-4096"
                    if size <= 4096
                    else "4097+"
                ] += 1
            compatible = DatasetManifest(
                prompt_contract_version="raw-v1",
                sources=sources,
                counts_by_language=dict(Counter(r.language for _, r, _ in selected)),
                counts_by_feature=dict(Counter(f for _, r, _ in selected for f in r.features)),
                counts_by_length_bucket=dict(lengths),
                deduplication={
                    "policy": "locale/commands/terms/normalized input/output",
                    "excluded": sum(p["reason"] == "normalized-duplicate" for p in reasons),
                },
                oversize_exclusions={
                    "families": len(
                        {
                            row["family"]
                            for row in rows
                            if any(
                                p["task_id"] == row["id"] and p["reason"] == "oversize-whole-family"
                                for p in reasons
                            )
                        }
                    )
                },
                sha256=manifest["sha256"],
            )
            (tmp / "manifest.json").write_text(compatible.model_dump_json(indent=2))
            (tmp / "studio-manifest.json").write_text(canonical(manifest))
            with self._transaction() as db:
                for row, _, origin in selected:
                    live = db.execute(
                        "SELECT status,revision FROM tasks WHERE id=?", (row["id"],)
                    ).fetchone()
                    if live["status"] not in ACCEPTED or live["revision"] != row["revision"]:
                        raise ConflictError("Entscheidungen änderten sich während des Exports.")
                    if origin.get("recording_id"):
                        clip = db.execute(
                            "SELECT status FROM recordings WHERE id=?", (origin["recording_id"],)
                        ).fetchone()
                        if not clip or clip[0] == "deleted":
                            raise ConflictError("Audio wurde während des Exports gelöscht.")
                os.rename(tmp, target)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return {"path": str(target), **manifest}
