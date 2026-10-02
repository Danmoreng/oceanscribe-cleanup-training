"""Offline transcription through OceanScribe's installed NeMo-Speech.cpp SDK."""

from __future__ import annotations

import array
import ctypes
import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

PRODUCTION_ROOT = Path("/home/sebastian/Development/OceanScribe")
DEFAULT_MAX_AUDIO_BYTES = 50 * 1024 * 1024
DEFAULT_MAX_AUDIO_SECONDS = 120.0


class ASRUnavailable(RuntimeError):
    """The configured local OceanScribe ASR runtime cannot be used."""


class AudioInputError(ValueError):
    """The supplied audio is invalid or exceeds the local processing limits."""


class _BackendConfig(ctypes.Structure):
    _fields_ = [("size", ctypes.c_size_t), ("gpu", ctypes.c_int32)]


class _ModelConfig(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("path", ctypes.c_char_p),
        ("name", ctypes.c_char_p),
    ]


class _StreamingConfig(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("chunk_size", ctypes.c_float),
        ("ctc_left_padding", ctypes.c_float),
        ("ctc_right_padding", ctypes.c_float),
        ("rnnt_right_context", ctypes.c_int32),
    ]


class _RecognizerConfig(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("backend", ctypes.POINTER(_BackendConfig)),
        ("model", ctypes.POINTER(_ModelConfig)),
        ("streaming", ctypes.POINTER(_StreamingConfig)),
        ("decoder", ctypes.c_void_p),
        ("vad", ctypes.c_void_p),
        ("endpointing", ctypes.c_void_p),
        ("postproc", ctypes.c_void_p),
        ("diar", ctypes.c_void_p),
        ("batching", ctypes.c_void_p),
    ]


class _SpeechContext(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("phrases", ctypes.POINTER(ctypes.c_char_p)),
        ("phrase_count", ctypes.c_size_t),
        ("boost", ctypes.c_float),
    ]


class _RecognitionOptions(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("request_id", ctypes.c_char_p),
        ("language_code", ctypes.c_char_p),
        ("interim_results", ctypes.c_bool),
        ("enable_word_time_offsets", ctypes.c_bool),
        ("enable_automatic_punctuation", ctypes.c_bool),
        ("verbatim_transcripts", ctypes.c_bool),
        ("profanity_filter", ctypes.c_bool),
        ("stop_history_eou_ms", ctypes.c_int32),
        ("speech_contexts", ctypes.POINTER(_SpeechContext)),
        ("speech_context_count", ctypes.c_size_t),
        ("max_alternatives", ctypes.c_int32),
        ("enable_speaker_diarization", ctypes.c_bool),
        ("max_speaker_count", ctypes.c_int32),
    ]


@lru_cache(maxsize=64)
def _sha256_cached(
    path: str, device: int, inode: int, size: int, modified_ns: int, changed_ns: int
) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    stat = path.stat()
    return _sha256_cached(
        str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
    )


def _json_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def _hf_hub_cache() -> Path:
    cache = _env_path("HF_HUB_CACHE")
    if cache is None:
        hf_home = _env_path("HF_HOME")
        if hf_home:
            cache = hf_home / "hub"
        else:
            xdg = _env_path("XDG_CACHE_HOME")
            cache = (xdg / "huggingface/hub") if xdg else Path.home() / ".cache/huggingface/hub"
    return cache


def _model_candidates(repo: str, filename: str, override_name: str) -> list[Path]:
    override = _env_path(override_name)
    if override:
        return [override]
    snapshots = _hf_hub_cache() / f"models--{repo.replace('/', '--')}/snapshots"
    try:
        return sorted(snapshots.glob(f"*/{filename}"))
    except OSError:
        return []


def _sdk_library_candidates() -> list[Path]:
    roots: list[Path] = []
    configured = _env_path("OCEANSCRIBE_NEMO_SDK_LIB")
    if configured:
        roots.append(configured)
    roots += [
        PRODUCTION_ROOT / "build/nemo-install/lib",
        PRODUCTION_ROOT / "build/nemo-sdk/bin",
    ]
    return [root / "libnemo_speech_asr_c.so.1" for root in roots]


def _executable_info(path: Path) -> dict[str, Any]:
    try:
        result = subprocess.run(
            [str(path), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            shell=False,
        )
        return {"returncode": result.returncode, "output": (result.stdout + result.stderr).strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


@dataclass(frozen=True)
class LocalASRRunner:
    """Runs locally cached Nemotron streaming and offline Parakeet TDT models."""

    worker_path: Path
    model_path: Path
    sdk_library: Path
    parakeet_model_path: Path | None = None
    engine_name: str = "nemo-speech.cpp"
    model_name: str = "Nemotron 3.5 ASR Streaming 0.6B"

    @classmethod
    def discover(cls, parakeet_model_path: Path | None = None) -> LocalASRRunner:
        worker = PRODUCTION_ROOT / "build/asr-worker/oceanscribe-asr-worker"
        model_paths = _model_candidates(
            "nvidia/nemotron-3.5-asr-streaming-0.6b",
            "nemotron-3.5-asr-streaming-0.6b.q8_0.gguf",
            "OCEANSCRIBE_ASR_MODEL",
        )
        if parakeet_model_path is None:
            parakeet_candidates = _model_candidates(
                "nvidia/parakeet-tdt-0.6b-v3",
                "parakeet-tdt-0.6b-v3.q8_0.gguf",
                "OCEANSCRIBE_PARAKEET_MODEL",
            )
            parakeet_model_path = next(
                (path for path in reversed(parakeet_candidates) if path.is_file()), None
            )
        elif not Path(parakeet_model_path).is_file():
            raise ASRUnavailable(f"Parakeet model does not exist: {parakeet_model_path}")
        libraries = _sdk_library_candidates()
        valid_library = next((path for path in libraries if path.is_file()), None)
        valid_model = next((path for path in reversed(model_paths) if path.is_file()), None)
        if not worker.is_file():
            raise ASRUnavailable(f"OceanScribe ASR worker is not built: {worker}")
        if valid_model is None:
            raise ASRUnavailable(
                "OceanScribe Nemotron model is unavailable; expected the existing local "
                "Hugging Face cache or OCEANSCRIBE_ASR_MODEL override."
            )
        if valid_library is None:
            raise ASRUnavailable(
                "OceanScribe's installed NeMo-Speech.cpp C SDK is unavailable; "
                "build OceanScribe locally before using Studio dictation."
            )
        return cls(
            worker.resolve(),
            valid_model.absolute(),
            valid_library.resolve(),
            Path(parakeet_model_path).absolute() if parakeet_model_path else None,
        )

    def _load_api(self) -> ctypes.CDLL:
        try:
            api = ctypes.CDLL(str(self.sdk_library))
        except OSError as exc:
            raise ASRUnavailable(f"Could not load OceanScribe NeMo SDK: {exc}") from exc
        api.nemo_speech_asr_create.argtypes = [
            ctypes.POINTER(_RecognizerConfig),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        api.nemo_speech_asr_create.restype = ctypes.c_int
        api.nemo_speech_asr_destroy.argtypes = [ctypes.c_void_p]
        api.nemo_speech_asr_recognition_options_default.restype = _RecognitionOptions
        api.nemo_speech_asr_recognize_f32.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_RecognitionOptions),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        api.nemo_speech_asr_recognize_f32.restype = ctypes.c_int
        api.nemo_speech_asr_streaming_recognize.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_RecognitionOptions),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        api.nemo_speech_asr_streaming_recognize.restype = ctypes.c_int
        api.nemo_speech_asr_stream_push_f32.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_int32,
        ]
        api.nemo_speech_asr_stream_push_f32.restype = ctypes.c_int
        api.nemo_speech_asr_stream_finish.argtypes = [ctypes.c_void_p]
        api.nemo_speech_asr_stream_finish.restype = ctypes.c_int
        api.nemo_speech_asr_stream_next.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        api.nemo_speech_asr_stream_next.restype = ctypes.c_int
        api.nemo_speech_asr_stream_close.argtypes = [ctypes.c_void_p]
        api.nemo_speech_asr_result_transcript.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        api.nemo_speech_asr_result_transcript.restype = ctypes.c_char_p
        api.nemo_speech_asr_result_alternative_count.argtypes = [ctypes.c_void_p]
        api.nemo_speech_asr_result_alternative_count.restype = ctypes.c_size_t
        api.nemo_speech_asr_result_destroy.argtypes = [ctypes.c_void_p]
        api.nemo_speech_asr_last_error.restype = ctypes.c_char_p
        api.nemo_speech_asr_version.restype = ctypes.c_char_p
        return api

    def status(self) -> dict[str, Any]:
        try:
            api = self._load_api()
            raw_version = api.nemo_speech_asr_version()
            sdk_version = raw_version.decode("utf-8", "replace") if raw_version else "unknown"
            sdk_error = None
        except (ASRUnavailable, AttributeError) as exc:
            sdk_version = None
            sdk_error = str(exc)
        engines = {
            "nemotron": self._model_status(
                self.model_path,
                self.model_name,
                sdk_error,
                "nvidia/nemotron-3.5-asr-streaming-0.6b",
                "openmdw-1.1",
            ),
            "parakeet": self._model_status(
                self.parakeet_model_path,
                "Parakeet TDT 0.6B v3",
                sdk_error,
                "nvidia/parakeet-tdt-0.6b-v3",
                "CC-BY-4.0",
            ),
        }
        worker_version = _executable_info(self.worker_path)
        return {
            "available": all(model["available"] for model in engines.values()),
            "integration_complete": all(model["available"] for model in engines.values()),
            "engine": self.engine_name,
            "backend": "cpu",
            "worker": str(self.worker_path),
            "worker_version": worker_version,
            "worker_sha256": _sha256(self.worker_path),
            "sdk_library": str(self.sdk_library),
            "sdk_version": sdk_version,
            "sdk_sha256": _sha256(self.sdk_library),
            "engines": engines,
            "error": sdk_error,
        }

    def _model_status(
        self,
        path: Path | None,
        name: str,
        sdk_error: str | None,
        repo: str,
        license_id: str,
    ) -> dict[str, Any]:
        if path is None or not path.is_file():
            return {"available": False, "actual_model": name, "error": "Model is not cached."}
        model_sha256 = _sha256(path)
        worker_sha256 = _sha256(self.worker_path)
        sdk_sha256 = _sha256(self.sdk_library)
        config_sha256 = _json_sha256(
            self._pipeline_config("parakeet" if "Parakeet" in name else "nemotron", "auto")
        )
        return {
            "available": sdk_error is None,
            "actual_model": name,
            "model_repo": repo,
            "model_license": license_id,
            "model_path": str(path),
            "model_revision": self._revision(path),
            "model_quantization": self._quantization(path),
            "model_sha256": model_sha256,
            "config_sha256": config_sha256,
            "pipeline_sha256": _json_sha256(
                {
                    "config_sha256": config_sha256,
                    "worker_sha256": worker_sha256,
                    "sdk_sha256": sdk_sha256,
                    "model_sha256": model_sha256,
                }
            ),
            "error": sdk_error,
        }

    @staticmethod
    def _revision(path: Path) -> str:
        match = re.search(r"snapshots/([^/]+)", path.as_posix())
        return match.group(1) if match else path.parent.name

    @staticmethod
    def _quantization(path: Path) -> str:
        match = re.search(r"(?:^|[-_.])(q\d+(?:_\d+)?|f16|f32)(?:[-_.]|$)", path.name)
        return match.group(1) if match else "unknown"

    def _pipeline_config(self, engine: str, language: str) -> dict[str, Any]:
        common: dict[str, Any] = {
            "runtime": self.engine_name,
            "engine": engine,
            "backend": "cpu",
            "sample_rate": 16000,
            "channels": 1,
            "sample_format": "float32",
            "decoder": "model default",
            "automatic_punctuation": True,
            "language_mode": "auto-detect" if engine == "parakeet" else language,
            "ffmpeg": ["-ac", "1", "-ar", "16000", "-f", "f32le"],
            "max_audio_bytes": DEFAULT_MAX_AUDIO_BYTES,
            "max_audio_seconds": DEFAULT_MAX_AUDIO_SECONDS,
        }
        if engine == "nemotron":
            common["streaming"] = {
                "chunk_size_seconds": 0.16,
                "ctc_left_padding_seconds": 1.92,
                "ctc_right_padding_seconds": 1.92,
                "rnnt_right_context": -1,
                "ring_block_samples": 1600,
            }
        else:
            common["streaming"] = None
            common["inference_mode"] = "offline whole-utterance"
        return common

    def transcribe(
        self, audio_path: Path, language: str = "auto", engine: str = "nemotron"
    ) -> dict[str, Any]:
        result = self.transcribe_many(audio_path, language=language, engines=(engine,))
        if not result["variants"]:
            raise ASRUnavailable(result["errors"][0]["error"])
        return result["variants"][0]

    def transcribe_many(
        self,
        audio_path: Path,
        language: str = "auto",
        engines: tuple[str, ...] = ("nemotron", "parakeet"),
    ) -> dict[str, Any]:
        source = Path(audio_path).expanduser().resolve()
        if not source.is_file():
            raise AudioInputError(f"Audio file does not exist: {source}")
        size = source.stat().st_size
        if size <= 0 or size > DEFAULT_MAX_AUDIO_BYTES:
            raise AudioInputError(
                f"Audio file size must be between 1 byte and {DEFAULT_MAX_AUDIO_BYTES} bytes."
            )
        if language not in {"auto", "mixed", "de-DE", "en-US", "de", "en"}:
            raise AudioInputError("Language must be auto, mixed, de-DE, de, en-US, or en.")
        if not engines or any(engine not in {"nemotron", "parakeet"} for engine in engines):
            raise AudioInputError("Engines must contain nemotron and/or parakeet.")
        if len(set(engines)) != len(engines):
            raise AudioInputError("Each engine may be selected only once per audio attempt.")

        audio_meta, pcm = self._decode_audio(source)
        samples = array.array("f")
        samples.frombytes(pcm)
        if os.sys.byteorder != "little":
            samples.byteswap()
        if not samples:
            raise AudioInputError("Audio contains no samples.")

        variants = []
        errors = []
        for engine in engines:
            try:
                variants.append(
                    self._transcribe_samples(engine, samples, source, audio_meta, language)
                )
            except ASRUnavailable as exc:
                errors.append({"engine": engine, "available": False, "error": str(exc)})
        return {"variants": variants, "errors": errors}

    def _transcribe_samples(
        self,
        engine: str,
        samples: array.array,
        source: Path,
        audio_meta: dict[str, Any],
        language: str,
    ) -> dict[str, Any]:
        if engine == "nemotron":
            model_path = self.model_path
            model_name = self.model_name
        else:
            model_path = self.parakeet_model_path
            model_name = "Parakeet TDT 0.6B v3"
            if model_path is None or not model_path.is_file():
                raise ASRUnavailable(
                    "Parakeet model is not available in the local Hugging Face cache."
                )

        api = self._load_api()
        backend = _BackendConfig(ctypes.sizeof(_BackendConfig), -1)
        model_path_bytes = os.fsencode(model_path)
        model = _ModelConfig(ctypes.sizeof(_ModelConfig), model_path_bytes, None)
        streaming = _StreamingConfig(ctypes.sizeof(_StreamingConfig), 0.16, 1.92, 1.92, -1)
        config = _RecognizerConfig(
            ctypes.sizeof(_RecognizerConfig),
            ctypes.pointer(backend),
            ctypes.pointer(model),
            ctypes.pointer(streaming),
            None,
            None,
            None,
            None,
            None,
            None,
        )
        recognizer = ctypes.c_void_p()
        stream = ctypes.c_void_p()
        native_result = ctypes.c_void_p()
        code = api.nemo_speech_asr_create(ctypes.byref(config), ctypes.byref(recognizer))
        if code != 0:
            raise ASRUnavailable(self._last_error(api, "ASR model initialization failed"))
        try:
            options = api.nemo_speech_asr_recognition_options_default()
            locale = {"de": "de-DE", "en": "en-US"}.get(language, language)
            options.language_code = (
                None
                if locale in {"auto", "mixed"} or engine == "parakeet"
                else locale.encode("ascii")
            )
            options.interim_results = engine == "nemotron"
            # Matches OceanScribe's worker and NeMo-Speech.cpp transcribe CLI defaults.
            options.enable_automatic_punctuation = True
            float_samples = (ctypes.c_float * len(samples)).from_buffer(samples)
            if engine == "nemotron":
                code = api.nemo_speech_asr_streaming_recognize(
                    recognizer, ctypes.byref(options), ctypes.byref(stream)
                )
                if code != 0:
                    raise ASRUnavailable(self._last_error(api, "Could not start ASR stream"))
                transcript = self._run_streaming(api, stream, float_samples, len(samples))
            else:
                code = api.nemo_speech_asr_recognize_f32(
                    recognizer,
                    ctypes.byref(options),
                    float_samples,
                    len(samples),
                    16000,
                    ctypes.byref(native_result),
                )
                if code != 0:
                    raise ASRUnavailable(
                        self._last_error(api, "Offline Parakeet transcription failed")
                    )
                text = api.nemo_speech_asr_result_transcript(native_result, 0)
                transcript = text.decode("utf-8") if text else ""
        finally:
            if stream:
                api.nemo_speech_asr_stream_close(stream)
            if native_result:
                api.nemo_speech_asr_result_destroy(native_result)
            api.nemo_speech_asr_destroy(recognizer)

        model_sha256 = _sha256(model_path)
        audio_sha256 = _sha256(source)
        provenance = {
            "engine": engine,
            "runtime": self.engine_name,
            "model": model_name,
            "model_repo": (
                "nvidia/nemotron-3.5-asr-streaming-0.6b"
                if engine == "nemotron"
                else "nvidia/parakeet-tdt-0.6b-v3"
            ),
            "model_license": "openmdw-1.1" if engine == "nemotron" else "CC-BY-4.0",
            "model_revision": self._revision(model_path),
            "model_quantization": self._quantization(model_path),
            "backend": "cpu",
            "decoder": "model default",
            "automatic_punctuation": True,
            "language_requested": language,
            "language_mode": "auto-detect" if engine == "parakeet" else language,
            "model_sha256": model_sha256,
            "worker_sha256": _sha256(self.worker_path),
            "sdk_sha256": _sha256(self.sdk_library),
            "sdk_version": (api.nemo_speech_asr_version() or b"unknown").decode("utf-8", "replace"),
            "preprocessing": "ffmpeg mono 16 kHz float32",
            "inference_mode": "streaming" if engine == "nemotron" else "offline whole-utterance",
            "streaming": (
                {
                    "chunk_size_seconds": 0.16,
                    "ctc_left_padding_seconds": 1.92,
                    "ctc_right_padding_seconds": 1.92,
                    "ring_block_samples": 1600,
                }
                if engine == "nemotron"
                else None
            ),
            "audio_sha256": audio_sha256,
        }
        provenance["config_sha256"] = _json_sha256(self._pipeline_config(engine, language))
        provenance["pipeline_sha256"] = _json_sha256(
            {
                "config_sha256": provenance["config_sha256"],
                "worker_sha256": provenance["worker_sha256"],
                "sdk_sha256": provenance["sdk_sha256"],
                "model_sha256": provenance["model_sha256"],
            }
        )
        return {
            "engine": engine,
            "transcript": transcript,
            "raw_transcript": transcript,
            "provenance": provenance,
            "audio_metadata": {**audio_meta, "sha256": audio_sha256},
        }

    def _run_streaming(
        self,
        api: ctypes.CDLL,
        stream: ctypes.c_void_p,
        samples: Any,
        sample_count: int,
    ) -> str:
        transcript = ""

        def collect_latest() -> None:
            nonlocal transcript
            while True:
                native_result = ctypes.c_void_p()
                status = api.nemo_speech_asr_stream_next(stream, ctypes.byref(native_result))
                if status != 0:
                    raise ASRUnavailable(self._last_error(api, "ASR stream read failed"))
                if not native_result:
                    break
                try:
                    if api.nemo_speech_asr_result_alternative_count(native_result) > 0:
                        text = api.nemo_speech_asr_result_transcript(native_result, 0)
                        if text:
                            transcript = text.decode("utf-8")
                finally:
                    api.nemo_speech_asr_result_destroy(native_result)

        # Match RuntimeController::worker_loop's 1,600-sample (100 ms) ring reads.
        for offset in range(0, sample_count, 1600):
            count = min(1600, sample_count - offset)
            chunk = ctypes.cast(
                ctypes.byref(samples, offset * ctypes.sizeof(ctypes.c_float)),
                ctypes.POINTER(ctypes.c_float),
            )
            code = api.nemo_speech_asr_stream_push_f32(stream, chunk, count, 16000)
            if code != 0:
                raise ASRUnavailable(self._last_error(api, "ASR audio input failed"))
            collect_latest()
        code = api.nemo_speech_asr_stream_finish(stream)
        if code != 0:
            raise ASRUnavailable(self._last_error(api, "ASR stream finalization failed"))
        collect_latest()
        return transcript

    @staticmethod
    def _last_error(api: ctypes.CDLL, fallback: str) -> str:
        message = api.nemo_speech_asr_last_error()
        return message.decode("utf-8", "replace") if message else fallback

    @staticmethod
    def _decode_audio(source: Path) -> tuple[dict[str, Any], bytes]:
        ffprobe = shutil.which("ffprobe")
        ffmpeg = shutil.which("ffmpeg")
        if not ffprobe or not ffmpeg:
            raise ASRUnavailable(
                "Local ffmpeg and ffprobe are required for bounded audio decoding."
            )
        try:
            probe = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration,size:stream=codec_name,sample_rate,channels",
                    "-of",
                    "json",
                    str(source),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
                shell=False,
            )
            data = json.loads(probe.stdout)
            declared = data.get("format", {}).get("duration")
            duration = float(declared) if declared not in (None, "N/A") else None
        except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError) as exc:
            raise AudioInputError(f"Could not inspect uploaded audio: {exc}") from exc
        if duration is not None and (duration <= 0 or duration > DEFAULT_MAX_AUDIO_SECONDS):
            raise AudioInputError(
                "Audio duration must be greater than 0 and no more than "
                f"{DEFAULT_MAX_AUDIO_SECONDS:g} seconds."
            )
        streams = data.get("streams", [])
        first = streams[0] if streams else {}
        try:
            converted = subprocess.run(
                [
                    ffmpeg,
                    "-nostdin",
                    "-v",
                    "error",
                    "-i",
                    str(source),
                    "-t",
                    # Decode one extra second to detect oversize streaming WebM,
                    # whose MediaRecorder header commonly omits its duration.
                    str(DEFAULT_MAX_AUDIO_SECONDS + 1),
                    "-map",
                    "0:a:0",
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-f",
                    "f32le",
                    "pipe:1",
                ],
                check=True,
                capture_output=True,
                timeout=max(30, min(180, int((duration or DEFAULT_MAX_AUDIO_SECONDS) * 2 + 30))),
                shell=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise AudioInputError(f"Could not decode audio locally: {exc}") from exc
        pcm = converted.stdout
        decoded_duration = len(pcm) / (16000 * 4)
        if len(pcm) % 4 or not 0 < decoded_duration <= DEFAULT_MAX_AUDIO_SECONDS:
            raise AudioInputError(
                "Audio muss hörbare Samples enthalten und darf höchstens "
                f"{DEFAULT_MAX_AUDIO_SECONDS:g} Sekunden lang sein."
            )
        return (
            {
                "filename": source.name,
                "bytes": source.stat().st_size,
                "duration_seconds": decoded_duration,
                "container_duration_seconds": duration,
                "source_codec": first.get("codec_name"),
                "source_sample_rate": first.get("sample_rate"),
                "source_channels": first.get("channels"),
                "transcription_sample_rate": 16000,
                "transcription_channels": 1,
            },
            pcm,
        )
