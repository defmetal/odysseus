"""Resolve generated-audio filenames for /api/generated-audio/{filename}."""
import os
import re
from pathlib import Path

from fastapi import HTTPException

from src.constants import GENERATED_AUDIO_DIR


GENERATED_AUDIO_DIR_PATH = Path(GENERATED_AUDIO_DIR)
GENERATED_AUDIO_RE = re.compile(
    r"^[a-f0-9]{8,64}\.(mp3|wav|flac|ogg|m4a|aac)$"
)
GENERATED_AUDIO_HEADERS = {
    "Cache-Control": "public, max-age=31536000, immutable",
    "X-Content-Type-Options": "nosniff",
}


def resolve_generated_audio_path(filename: str) -> Path:
    if not isinstance(filename, str) or not GENERATED_AUDIO_RE.fullmatch(filename):
        raise HTTPException(status_code=400, detail="Invalid filename")
    root = GENERATED_AUDIO_DIR_PATH.resolve()
    path = (GENERATED_AUDIO_DIR_PATH / filename).resolve()
    try:
        if os.path.commonpath([str(root), str(path)]) != str(root):
            raise ValueError
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio not found")
    return path
