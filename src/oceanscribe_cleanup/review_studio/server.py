"""Loopback-only HTTP server for the local OceanScribe review studio."""

from __future__ import annotations

import base64
import json
import mimetypes
import secrets
import threading
import time
import urllib.parse
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

MAX_BODY = 24 * 1024 * 1024
MAX_AUDIO = 16 * 1024 * 1024
STATIC_DIR = Path(__file__).with_name("static")
CSP = (
    "default-src 'self'; img-src 'self' data:; media-src 'self' blob:; "
    "object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
)


class StudioHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], store: Any, static_dir: Path = STATIC_DIR):
        if address[0] not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Review studio may bind to loopback addresses only")
        self.store = store
        self.static_dir = static_dir.resolve()
        self.sessions: dict[str, str] = {}
        self.sessions_lock = threading.Lock()
        self.asr_lock = threading.Lock()
        self.active_asr: set[str] = set()
        self.active_asr_lock = threading.Lock()
        self.asr_status_cache: tuple[float, dict[str, Any]] | None = None
        super().__init__(address, StudioHandler)

    def schedule_asr(self, recording_id: str, engine: str, language: str) -> None:
        with self.active_asr_lock:
            if recording_id in self.active_asr:
                return
            self.active_asr.add(recording_id)
        threading.Thread(
            target=_run_asr_job,
            args=(self, recording_id, engine, language),
            daemon=True,
        ).start()

    def recover_pending_asr(self) -> None:
        pending = getattr(self.store, "pending_asr_jobs", None)
        if pending is None:
            return
        for job in pending():
            self.schedule_asr(
                job["recording_id"], job.get("engine", "nemotron"), job.get("language", "auto")
            )

    def get_asr_status(self) -> dict[str, Any]:
        cached = self.asr_status_cache
        if cached and time.monotonic() - cached[0] < 30:
            return cached[1]
        from .asr import LocalASRRunner

        result = LocalASRRunner.discover().status()
        self.asr_status_cache = (time.monotonic(), result)
        return result


class StudioHandler(BaseHTTPRequestHandler):
    server: StudioHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *_args: Any) -> None:
        # Transcript contents, request bodies and audio identifiers are private.
        return

    def _json(self, value: Any, status: int = 200, headers: dict[str, str] | None = None) -> None:
        raw = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            CSP,
        )
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(raw)

    def _error(self, message: str, status: int = 400, code: str = "invalid_request") -> None:
        self._json({"error": code, "message": message}, status)

    def _local_request(self) -> bool:
        host = self.headers.get("Host", "")
        # Reject DNS rebinding/foreign Host values; allow normal ephemeral ports.
        try:
            parsed = urllib.parse.urlsplit("http://" + host)
            host_name = parsed.hostname
            port = parsed.port
        except ValueError:
            return False
        if (
            host_name not in {"127.0.0.1", "localhost", "::1"}
            or port != self.server.server_address[1]
        ):
            return False
        origin = self.headers.get("Origin")
        if origin:
            try:
                o = urllib.parse.urlsplit(origin)
                if (
                    o.scheme != "http"
                    or o.hostname not in {"127.0.0.1", "localhost", "::1"}
                    or o.port != self.server.server_address[1]
                ):
                    return False
            except ValueError:
                return False
        return True

    def _session(self) -> tuple[str, str, bool]:
        cookie = self.headers.get("Cookie", "")
        cookie_sid = next(
            (
                item.partition("=")[2]
                for item in cookie.split(";")
                if item.strip().startswith("os_review=")
            ),
            "",
        )
        cookie_sid = cookie_sid.strip()
        header_sid = self.headers.get("X-Local-Session", "")
        with self.server.sessions_lock:
            if cookie_sid in self.server.sessions:
                return cookie_sid, self.server.sessions[cookie_sid], False
            if header_sid in self.server.sessions:
                return header_sid, self.server.sessions[header_sid], False
            sid, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
            self.server.sessions[sid] = csrf
            return sid, csrf, True

    def _request_json(self) -> dict[str, Any]:
        length_s = self.headers.get("Content-Length")
        if length_s is None:
            raise ValueError("Content-Length is required")
        try:
            length = int(length_s)
        except ValueError as exc:
            raise ValueError("Invalid Content-Length") from exc
        if length < 0 or length > MAX_BODY:
            raise OverflowError("Request body exceeds the 24 MiB limit")
        try:
            data = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Body must be valid UTF-8 JSON") from exc
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")
        return data

    def _call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        method: Callable[..., Any] | None = getattr(self.server.store, name, None)
        if method is None:
            raise NotImplementedError(f"Local store operation {name} is not available")
        return method(*args, **kwargs)

    def do_GET(self) -> None:  # noqa: N802
        if not self._local_request():
            self._error("Only the local studio origin is accepted", 403, "origin_rejected")
            return
        sid, csrf, fresh = self._session()
        path = urllib.parse.urlsplit(self.path).path
        if path == "/api/status":
            try:
                state = self._call("dashboard")
                try:
                    asr_state = self.server.get_asr_status()
                except Exception as asr_exc:
                    asr_state = {
                        "available": False,
                        "requested_models": ["OceanScribe production Nemotron", "Parakeet TDT"],
                        "engines": {},
                        "integration_complete": False,
                        "error": str(asr_exc),
                    }
                self._json(
                    {"csrf_token": csrf, "session_id": sid, "status": state, "asr": asr_state},
                    headers={
                        "Set-Cookie": (
                            f"os_review={sid}; HttpOnly; SameSite=Strict; Path=/; Max-Age=86400"
                        )
                    },
                )
            except Exception as exc:
                self._store_error(exc)
            return
        if path == "/api/recording-tasks":
            try:
                self._json(self._call("recording_tasks"))
            except Exception as exc:
                self._store_error(exc)
            return
        if path.startswith("/api/recording/"):
            recording_id = path.removeprefix("/api/recording/")
            if not recording_id or any(
                c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                for c in recording_id
            ):
                self._error("Invalid recording identifier", 404, "not_found")
                return
            try:
                self._json(self._call("recording_status", recording_id))
            except Exception as exc:
                self._store_error(exc)
            return
        if path.startswith("/api/audio/"):
            audio_id = path[len("/api/audio/") :]
            if not audio_id or any(
                c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                for c in audio_id
            ):
                self._error("Invalid audio identifier", 404, "not_found")
                return
            try:
                audio_path = self._call("audio_path", audio_id)
                body = Path(audio_path).read_bytes()
                mime = mimetypes.guess_type(str(audio_path))[0] or "application/octet-stream"
                if not isinstance(body, bytes):
                    raise ValueError("Audio store returned invalid bytes")
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                self._store_error(exc)
            return
        if path.startswith("/api/"):
            self._error("Unknown API endpoint", 404, "not_found")
            return
        candidate = (
            self.server.static_dir / ("index.html" if path == "/" else path.lstrip("/"))
        ).resolve()
        if self.server.static_dir not in candidate.parents and candidate != self.server.static_dir:
            self._error("Not found", 404, "not_found")
            return
        if not candidate.is_file():
            self._error("Not found", 404, "not_found")
            return
        content = candidate.read_bytes()
        mime = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        if mime.startswith("text/") or mime in {"application/javascript", "application/json"}:
            mime += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            CSP,
        )
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self) -> None:  # noqa: N802
        if not self._local_request():
            self._error("Only the local studio origin is accepted", 403, "origin_rejected")
            return
        sid, csrf, _ = self._session()
        has_session = self.headers.get("Cookie", "").find(
            f"os_review={sid}"
        ) >= 0 or secrets.compare_digest(self.headers.get("X-Local-Session", ""), sid)
        if not has_session or not secrets.compare_digest(
            self.headers.get("X-CSRF-Token", ""), csrf
        ):
            self._error("Missing or invalid local session token", 403, "csrf_rejected")
            return
        path = urllib.parse.urlsplit(self.path).path
        try:
            data = self._request_json()
            if path == "/api/next-card":
                result = self._call(
                    "next_card", kind=data.get("kind", "repair"), task_id=data.get("task_id")
                )
            elif path == "/api/decision":
                result = self._call(
                    "decide",
                    **{
                        k: data[k]
                        for k in (
                            "task_id",
                            "expected_revision",
                            "input_sha256",
                            "action",
                            "output",
                            "request_id",
                        )
                        if k in data
                    },
                    semantic_failure=data.get("semantic_failure"),
                    error_codes=data.get("error_codes", []),
                    answer_reviews=data.get("answer_reviews", []),
                )
            elif path == "/api/extend":
                if data.get("confirm") is not True:
                    self._error("Explicit confirmation is required", 400, "confirmation_required")
                    return
                result = self._call("extend", confirm=True)
            elif path == "/api/recording/start":
                if data.get("consent_local") is not True:
                    self._error("Local processing consent is required", 400, "consent_required")
                    return
                result = self._call("start_recording", **data)
            elif path == "/api/recording/upload":
                encoded = data.pop("base64", None)
                if not isinstance(encoded, str) or len(encoded) > (MAX_AUDIO * 4 // 3 + 8):
                    raise ValueError("Audio payload is missing or exceeds 16 MiB")
                try:
                    raw = base64.b64decode(encoded, validate=True)
                except ValueError as exc:
                    raise ValueError("Audio payload is not valid base64") from exc
                if not raw or len(raw) > MAX_AUDIO:
                    raise ValueError("Audio must be between 1 byte and 16 MiB")
                mime = data.get("mime", "")
                if not isinstance(mime, str) or not mime.startswith("audio/") or len(mime) > 100:
                    raise ValueError("Unsupported audio MIME type")
                engine = data.get("engine", "nemotron")
                if engine not in {"nemotron", "parakeet", "both"}:
                    raise ValueError("ASR engine must be nemotron, parakeet, or both")
                result = self._call(
                    "save_audio",
                    data=raw,
                    recording_id=data.get("recording_id"),
                    request_id=data.get("request_id"),
                    mime=mime,
                    engine=engine,
                    language=data.get("language"),
                    browser_metadata=data.get("browser_metadata"),
                )
                status = self._call("recording_status", result["recording_id"])
                if status.get("status") == "asr_pending":
                    self.server.schedule_asr(
                        result["recording_id"],
                        engine,
                        status.get("language", data.get("language", "auto")),
                    )
                result.update(status="asr_pending", engine=engine)
            elif path == "/api/recording/retry":
                result = self._call("retry_asr", data["recording_id"], data["request_id"])
                self.server.schedule_asr(
                    result["recording_id"], result["engine"], result["language"]
                )
            elif path == "/api/recording/discard":
                result = self._call("discard_recording", data["recording_id"], data["request_id"])
            elif path == "/api/recording/confirm":
                result = self._call("confirm_recording", **data)
            elif path == "/api/recording/delete":
                result = self._call("delete_recording", **data)
            elif path == "/api/export":
                result = self._call("export_snapshot")
            else:
                self._error("Unknown API endpoint", 404, "not_found")
                return
            self._json(result if result is not None else {"ok": True})
        except OverflowError as exc:
            self._error(str(exc), 413, "body_too_large")
        except NotImplementedError as exc:
            self._error(str(exc), 501, "not_implemented")
        except Exception as exc:
            self._store_error(exc)

    def _store_error(self, exc: Exception) -> None:
        name = type(exc).__name__.lower()
        status, code = 500, "store_error"
        if hasattr(exc, "status"):
            status = int(exc.status)
            code = "store_error" if status >= 500 else "invalid_request"
        if "conflict" in name:
            status, code = 409, "revision_conflict"
        elif "budget" in name:
            status, code = 409, "budget_exhausted"
        elif isinstance(exc, (ValueError, TypeError, KeyError)):
            status, code = 400, "invalid_request"
        self._error(str(exc), status, code)


def _run_asr_job(server: StudioHTTPServer, recording_id: str, engine: str, language: str) -> None:
    """Run a single local ASR job at a time and persist its outcome."""
    try:
        from .asr import LocalASRRunner

        with server.asr_lock:
            status = server.store.recording_status(recording_id)
            if status.get("status") != "asr_pending":
                return
            runner = LocalASRRunner.discover()
            audio = Path(server.store.audio_path(recording_id))
            if engine == "both":
                result = runner.transcribe_many(
                    audio, language=language, engines=("nemotron", "parakeet")
                )
            else:
                result = runner.transcribe(audio, language=language, engine=engine)
            server.store.complete_asr(recording_id, result=result)
    except Exception as exc:
        try:
            server.store.complete_asr(recording_id, error=str(exc))
        except Exception:
            # A retried/removed job may have already moved out of asr_pending.
            pass
    finally:
        with server.active_asr_lock:
            server.active_asr.discard(recording_id)


def serve(store: Any, host: str = "127.0.0.1", port: int = 0) -> StudioHTTPServer:
    """Create and run the server; intended for a CLI wrapper to call serve_forever."""
    server = StudioHTTPServer((host, port), store)
    server.recover_pending_asr()
    return server
