from __future__ import annotations

import http.cookiejar
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from oceanscribe_cleanup.review_studio.server import MAX_BODY, StudioHTTPServer
from oceanscribe_cleanup.review_studio.store import ReviewStore


class ConflictError(Exception):
    pass


class FakeStore:
    def __init__(self, root: Path):
        self.root = root
        self.decisions: list[dict] = []
        self.audio: dict[str, bytes] = {}

    def dashboard(self):
        return {
            "budgets": {
                "reviews": {"used": len(self.decisions), "limit": 50},
                "recordings": {"used": 0, "limit": 10},
            },
            "counts": {},
        }

    def next_card(self, kind="repair", task_id=None):
        return {
            "card": {
                "task_id": "t1",
                "kind": kind,
                "raw_transcript": "raw",
                "output": "clean",
                "revision": 1,
                "input_sha256": "abc",
                "question": "Check",
            }
        }

    def decide(self, **kwargs):
        if kwargs["expected_revision"] != 1:
            raise ConflictError("Revision changed")
        self.decisions.append(kwargs)
        return {"saved": True, **self.dashboard()}

    def extend(self, confirm):
        assert confirm
        return self.dashboard()

    def recording_tasks(self):
        return []

    def start_recording(self, task_id, request_id, consent_local):
        assert consent_local
        return {"recording_id": "r1"}

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
        self.audio[recording_id] = data
        path = self.root / f"{recording_id}.audio"
        path.write_bytes(data)
        return {"recording_id": recording_id}

    def recording_status(self, recording_id):
        return {"recording_id": recording_id, "status": "asr_failed", "language": "de-DE"}

    def audio_path(self, recording_id):
        return self.root / f"{recording_id}.audio"

    def complete_asr(self, recording_id, result=None, error=None):
        return {"error": error}

    def confirm_recording(self, **kwargs):
        return kwargs

    def delete_recording(self, **kwargs):
        return {"deleted": True}

    def export_snapshot(self):
        return {"snapshot_id": "snap1"}


@pytest.fixture
def server(tmp_path):
    instance = StudioHTTPServer(("127.0.0.1", 0), FakeStore(tmp_path))
    instance.get_asr_status = lambda: {"engines": {}}
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    yield instance
    instance.shutdown()
    instance.server_close()
    thread.join(timeout=2)


def origin(server):
    return f"http://127.0.0.1:{server.server_port}"


def open_url(server, path, *, method="GET", data=None, headers=None):
    request = urllib.request.Request(
        origin(server) + path, data=data, method=method, headers=headers or {}
    )
    return urllib.request.urlopen(request)


def session(server):
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
    )
    response = opener.open(origin(server) + "/api/status")
    payload = json.loads(response.read())
    return opener, payload["csrf_token"]


def test_static_app_and_status_are_local_and_secure(server):
    response = open_url(server, "/")
    assert response.status == 200
    assert "Prüfen".encode() in response.read()
    status = open_url(server, "/api/status")
    assert "HttpOnly" in status.headers["Set-Cookie"]
    assert "SameSite=Strict" in status.headers["Set-Cookie"]
    assert json.loads(status.read())["status"]["budgets"]["reviews"]["limit"] == 50


def test_decision_requires_csrf_and_is_persisted_idempotency_key_forwarded(server):
    body = json.dumps(
        {
            "task_id": "t1",
            "expected_revision": 1,
            "input_sha256": "abc",
            "action": "edit",
            "output": "corrected",
            "request_id": "req1",
        }
    ).encode()
    with pytest.raises(urllib.error.HTTPError) as err:
        open_url(
            server,
            "/api/decision",
            method="POST",
            data=body,
            headers={"Content-Type": "application/json"},
        )
    assert err.value.code == 403
    opener, token = session(server)
    request = urllib.request.Request(
        origin(server) + "/api/decision",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "X-CSRF-Token": token},
    )
    result = json.loads(opener.open(request).read())
    assert result["saved"] is True
    assert server.store.decisions[0]["request_id"] == "req1"


def test_revision_conflict_maps_to_http_409(server):
    opener, token = session(server)
    body = json.dumps(
        {
            "task_id": "t1",
            "expected_revision": 8,
            "input_sha256": "abc",
            "action": "accept",
            "request_id": "r",
        }
    ).encode()
    request = urllib.request.Request(
        origin(server) + "/api/decision",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "X-CSRF-Token": token},
    )
    with pytest.raises(urllib.error.HTTPError) as err:
        opener.open(request)
    assert err.value.code == 409
    assert json.loads(err.value.read())["error"] == "revision_conflict"


def test_cookie_free_browser_requires_matching_session_and_csrf(server):
    payload = json.loads(open_url(server, "/api/status").read())
    headers = {
        "Content-Type": "application/json",
        "Origin": origin(server),
        "X-Local-Session": payload["session_id"],
        "X-CSRF-Token": payload["csrf_token"],
    }
    for changed in (
        {"X-CSRF-Token": "invalid"},
        {"X-Local-Session": "invalid"},
        {"Origin": "http://example.com"},
    ):
        with pytest.raises(urllib.error.HTTPError) as err:
            open_url(
                server, "/api/next-card", method="POST", data=b"{}", headers={**headers, **changed}
            )
        assert err.value.code == 403
    result = json.loads(
        open_url(server, "/api/next-card", method="POST", data=b"{}", headers=headers).read()
    )
    assert result["card"]["task_id"] == "t1"


def test_origin_path_and_body_limits(server):
    with pytest.raises(urllib.error.HTTPError) as err:
        open_url(server, "/", headers={"Host": "example.com"})
    assert err.value.code == 403
    with pytest.raises(urllib.error.HTTPError) as err:
        open_url(server, "/%2e%2e/server.py")
    assert err.value.code == 404
    opener, token = session(server)
    request = urllib.request.Request(
        origin(server) + "/api/extend",
        data=b"{}",
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-CSRF-Token": token,
            "Content-Length": str(MAX_BODY + 1),
        },
    )
    with pytest.raises(urllib.error.HTTPError) as err:
        opener.open(request)
    assert err.value.code == 413


def test_recording_audio_saved_and_read_back_locally(server):
    opener, token = session(server)
    payload = {
        "recording_id": "r1",
        "request_id": "req-audio",
        "mime": "audio/webm",
        "base64": "YXVkaW8=",
    }
    request = urllib.request.Request(
        origin(server) + "/api/recording/upload",
        data=json.dumps(payload).encode(),
        method="POST",
        headers={"Content-Type": "application/json", "X-CSRF-Token": token},
    )
    result = json.loads(opener.open(request).read())
    assert result["recording_id"] == "r1"
    assert server.store.audio["r1"] == b"audio"
    audio = opener.open(origin(server) + "/api/audio/r1")
    assert audio.read() == b"audio"


def test_http_review_roundtrip_persists_in_real_store(tmp_path, monkeypatch):
    monkeypatch.setattr(StudioHTTPServer, "get_asr_status", lambda _self: {"engines": {}})
    store = ReviewStore(tmp_path / "studio")
    store.add_task(
        {
            "id": "source-1",
            "source_name": "fixture",
            "source_revision": "rev1",
            "source_record_id": "1",
            "license_spdx": "Apache-2.0",
            "split": "train",
            "language": "de-DE",
            "commands": True,
            "transcript": "äh schick den Bericht",
            "output": "Schick den Bericht.",
            "terminology": [],
            "features": ["disfluency"],
            "edit_script": [],
            "synthetic": False,
            "generator_version": None,
            "seed": None,
            "parent_id": None,
            "quality_flags": [],
        },
        task_id="repair:fixture-1",
        question="Ist das Ergebnis bedeutungstreu?",
    )
    live = StudioHTTPServer(("127.0.0.1", 0), store)
    thread = threading.Thread(target=live.serve_forever, daemon=True)
    thread.start()
    try:
        opener, token = session(live)
        next_request = urllib.request.Request(
            origin(live) + "/api/next-card",
            data=json.dumps({"kind": "repair"}).encode(),
            method="POST",
            headers={"Content-Type": "application/json", "X-CSRF-Token": token},
        )
        card = json.loads(opener.open(next_request).read())["card"]
        decision = {
            "task_id": card["task_id"],
            "expected_revision": card["revision"],
            "input_sha256": card["input_sha256"],
            "action": "edit",
            "output": "Schick den Bericht morgen.",
            "request_id": "same-decision",
        }
        request = urllib.request.Request(
            origin(live) + "/api/decision",
            data=json.dumps(decision).encode(),
            method="POST",
            headers={"Content-Type": "application/json", "X-CSRF-Token": token},
        )
        first = json.loads(opener.open(request).read())
        request = urllib.request.Request(
            origin(live) + "/api/decision",
            data=json.dumps(decision).encode(),
            method="POST",
            headers={"Content-Type": "application/json", "X-CSRF-Token": token},
        )
        second = json.loads(opener.open(request).read())
        assert first == second
        reopened = ReviewStore(tmp_path / "studio")
        assert reopened.dashboard()["counts"]["human_accepted"] == 1
        assert reopened.dashboard()["budgets"]["reviews"]["used"] == 1
    finally:
        live.shutdown()
        live.server_close()
        thread.join(timeout=2)


def test_server_requeues_durable_pending_asr_jobs(tmp_path, monkeypatch):
    store = FakeStore(tmp_path)
    store.pending_asr_jobs = lambda: [
        {"recording_id": "r-pending", "engine": "both", "language": "en-US"}
    ]
    scheduled = []
    monkeypatch.setattr(
        StudioHTTPServer,
        "schedule_asr",
        lambda _server, *args: scheduled.append(args),
    )
    server = StudioHTTPServer(("127.0.0.1", 0), store)
    try:
        server.recover_pending_asr()
        assert scheduled == [("r-pending", "both", "en-US")]
    finally:
        server.server_close()


def test_http_retry_and_discard_same_clip_preserve_attempt_budget(tmp_path, monkeypatch):
    store = ReviewStore(tmp_path)
    store.add_recording_task(
        {"task_id": "clip", "family_id": "fixture/clip", "split": "train", "language": "de-DE"}
    )
    rid = store.start_recording("clip", "start", consent_local=True)["recording_id"]
    store.save_audio(rid, b"RIFF0000WAVEfixture", "audio/wav", "upload")
    store.complete_asr(rid, error="fixture error")
    scheduled = []
    monkeypatch.setattr(StudioHTTPServer, "get_asr_status", lambda _self: {"engines": {}})
    monkeypatch.setattr(
        StudioHTTPServer, "schedule_asr", lambda _self, *args: scheduled.append(args)
    )
    live = StudioHTTPServer(("127.0.0.1", 0), store)
    thread = threading.Thread(target=live.serve_forever, daemon=True)
    thread.start()
    try:
        opener, token = session(live)
        for endpoint in ("retry", "discard"):
            request = urllib.request.Request(
                origin(live) + "/api/recording/" + endpoint,
                data=json.dumps({"recording_id": rid, "request_id": endpoint}).encode(),
                headers={"Content-Type": "application/json", "X-CSRF-Token": token},
            )
            assert json.loads(opener.open(request).read())["recording_id"] == rid
        assert scheduled == [(rid, "nemotron", "de-DE")]
        assert store.dashboard()["budgets"]["recordings"]["used"] == 1
        assert store.pending_asr_jobs() == []
        assert store.recording_status(rid)["status"] == "discarded"
    finally:
        live.shutdown()
        live.server_close()
        thread.join(timeout=2)
