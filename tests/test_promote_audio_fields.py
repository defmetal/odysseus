"""Lift /api/generated-audio URLs from generate_music stdout."""
from src.tool_execution import _promote_audio_fields


def _result(stdout, exit_code=0, **extra):
    out = {"exit_code": exit_code, "stdout": stdout}
    out.update(extra)
    return out


def test_relative_audio_url_promoted():
    r = _result("Generated music for: generic ambient\n/api/generated-audio/abcd1234ef.mp3")
    _promote_audio_fields(r)
    assert r["audio_url"] == "/api/generated-audio/abcd1234ef.mp3"


def test_absolute_audio_url_promoted():
    r = _result("https://odysseus.example.com/api/generated-audio/abcd1234ef.mp3")
    _promote_audio_fields(r)
    assert r["audio_url"].endswith("/api/generated-audio/abcd1234ef.mp3")


def test_existing_audio_url_not_overwritten():
    r = _result("/api/generated-audio/other.mp3", audio_url="/api/generated-audio/kept.mp3")
    _promote_audio_fields(r)
    assert r["audio_url"] == "/api/generated-audio/kept.mp3"


def test_nonzero_exit_not_promoted():
    r = _result("/api/generated-audio/abcd1234ef.mp3", exit_code=1)
    _promote_audio_fields(r)
    assert "audio_url" not in r
