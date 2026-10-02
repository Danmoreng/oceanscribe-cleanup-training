from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from oceanscribe_cleanup.review_studio.asr import (
    PRODUCTION_ROOT,
    AudioInputError,
    LocalASRRunner,
)


def test_production_fixture_audio_is_decoded_to_production_sample_format() -> None:
    fixture = PRODUCTION_ROOT / "third_party/NeMo-Speech.cpp/test_files/asr/wav/test/jfk.wav"
    if not fixture.is_file():
        pytest.skip("NeMo-Speech.cpp public JFK fixture is not present in this checkout")

    metadata, pcm = LocalASRRunner._decode_audio(fixture)

    assert metadata["duration_seconds"] > 0
    assert metadata["transcription_sample_rate"] == 16000
    assert metadata["transcription_channels"] == 1
    assert len(pcm) > 0
    assert len(pcm) % 4 == 0


def test_transcribe_rejects_missing_audio_before_runtime_access(tmp_path: Path) -> None:
    runner = LocalASRRunner(Path("/missing/worker"), Path("/missing/model"), Path("/missing/sdk"))

    with pytest.raises(AudioInputError, match="does not exist"):
        runner.transcribe(tmp_path / "missing.wav")


def test_transcribe_rejects_unsupported_locale_before_runtime_access(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    runner = LocalASRRunner(Path("/missing/worker"), Path("/missing/model"), Path("/missing/sdk"))

    with pytest.raises(AudioInputError, match="Language must be"):
        runner.transcribe(audio, language="fr-FR")


@pytest.mark.parametrize("seconds,oversize", [(0.5, False), (121.5, True)])
def test_streaming_webm_without_duration_decodes_and_rejects_oversize(tmp_path, seconds, oversize):
    if not shutil.which("ffmpeg"):
        pytest.skip("Local ffmpeg required")
    audio = tmp_path / "browser.webm"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000",
            "-t",
            str(seconds),
            "-c:a",
            "libopus",
            "-f",
            "webm",
            "-live",
            "1",
            str(audio),
        ],
        check=True,
        timeout=20,
    )
    if oversize:
        with pytest.raises(AudioInputError, match="höchstens"):
            LocalASRRunner._decode_audio(audio)
    else:
        metadata, pcm = LocalASRRunner._decode_audio(audio)
        assert metadata["container_duration_seconds"] is None
        assert metadata["duration_seconds"] == len(pcm) / 64000
        assert 0.4 < metadata["duration_seconds"] < 0.6


@pytest.mark.skipif(
    os.environ.get("OCEANSCRIBE_RUN_ASR_INTEGRATION") != "1",
    reason="Set OCEANSCRIBE_RUN_ASR_INTEGRATION=1 for real local model inference",
)
def test_real_local_engines_transcribe_same_public_nonprivate_audio() -> None:
    fixture = PRODUCTION_ROOT / "third_party/NeMo-Speech.cpp/test_files/asr/wav/test/jfk.wav"
    runner = LocalASRRunner.discover()

    result = runner.transcribe_many(fixture, language="en-US")

    assert result["errors"] == []
    assert [variant["engine"] for variant in result["variants"]] == ["nemotron", "parakeet"]
    assert all(variant["transcript"].strip() for variant in result["variants"])
    assert len({variant["audio_metadata"]["sha256"] for variant in result["variants"]}) == 1
    assert all(variant["provenance"]["model_sha256"] for variant in result["variants"])
    assert all(variant["provenance"]["pipeline_sha256"] for variant in result["variants"])
    assert result["variants"][0]["provenance"]["inference_mode"] == "streaming"
    assert result["variants"][1]["provenance"]["inference_mode"] == "offline whole-utterance"
