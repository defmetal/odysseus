"""Generated-audio filename resolver — generic hashes only."""
from fastapi import HTTPException

from src.generated_audio import GENERATED_AUDIO_RE, resolve_generated_audio_path


def test_filename_regex_accepts_generic_hashes():
    assert GENERATED_AUDIO_RE.fullmatch("abcd1234ef567890.mp3")
    assert GENERATED_AUDIO_RE.fullmatch("aa" * 8 + ".wav")
    assert not GENERATED_AUDIO_RE.fullmatch("../secret.mp3")
    assert not GENERATED_AUDIO_RE.fullmatch("toei90s.mp3")
    assert not GENERATED_AUDIO_RE.fullmatch("smoon.wav")


def test_resolve_rejects_invalid_names():
    for name in ("../etc/passwd", "smoon.mp3", "studio_toei.wav", ""):
        try:
            resolve_generated_audio_path(name)
            assert False, name
        except HTTPException as exc:
            assert exc.status_code in (400, 404)
