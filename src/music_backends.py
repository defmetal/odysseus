"""Pluggable music-generation backends (MiniMax Music 3 first).

Registry: config/music_backends.json. Engine keys:
  - minimax_music3_comfy — local ComfyUI Music 3 graph, no API key
  - minimax_music3_api   — hosted MiniMax Music 3; key from env/settings only

This is Music 3, not MiniMax H3 video.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_REGISTRY_PATH = Path(__file__).resolve().parent.parent / "config" / "music_backends.json"

_FALLBACK_REGISTRY = {
    "default": "minimax_music3_comfy",
    "backends": {
        "minimax_music3_comfy": {
            "name": "MiniMax Music 3 (local ComfyUI)",
            "engine": "minimax_music3_comfy",
            "needs_comfy": True,
            "needs_api_key": False,
            "enabled": True,
        },
        "minimax_music3_api": {
            "name": "MiniMax Music 3 (hosted API)",
            "engine": "minimax_music3_api",
            "needs_comfy": False,
            "needs_api_key": True,
            "api_url": "https://api.minimax.io/v1/music_generation",
            "model": "music-3.0",
            "enabled": True,
        },
    },
}


def load_music_backends(path: Optional[str] = None) -> dict:
    p = Path(path) if path else _REGISTRY_PATH
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return dict(_FALLBACK_REGISTRY)
    if not isinstance(data, dict) or not isinstance(data.get("backends"), dict) or not data.get("backends"):
        return dict(_FALLBACK_REGISTRY)
    return data


def minimax_api_key() -> str:
    """Key from env first, then settings. Never logged. Never read from the registry file."""
    env = (os.environ.get("MINIMAX_API_KEY") or "").strip()
    if env:
        return env
    try:
        from src.settings import get_setting
        return (get_setting("minimax_api_key", "") or "").strip()
    except Exception:
        return ""


def music_gen_enabled() -> bool:
    try:
        from src.settings import get_setting
        return bool(get_setting("music_gen_enabled", True))
    except Exception:
        return True


def _comfy_url() -> str:
    try:
        from src.settings import get_setting
        return (get_setting("comfy_base_url", "") or "").strip()
    except Exception:
        return ""


def backend_available(entry: dict, *, comfy_reachable: Optional[bool] = None) -> bool:
    if not isinstance(entry, dict) or not entry.get("enabled", True):
        return False
    if entry.get("needs_api_key") and not minimax_api_key():
        return False
    if entry.get("needs_comfy"):
        if not _comfy_url():
            return False
        if comfy_reachable is False:
            return False
    return True


def list_backends(*, comfy_reachable: Optional[bool] = None) -> list[dict]:
    registry = load_music_backends()
    out = []
    for key, entry in (registry.get("backends") or {}).items():
        if not isinstance(entry, dict):
            continue
        item = {**entry, "key": key}
        item["available"] = backend_available(entry, comfy_reachable=comfy_reachable)
        out.append(item)
    return out


def default_backend_key(*, comfy_reachable: Optional[bool] = None) -> Optional[str]:
    registry = load_music_backends()
    preferred = registry.get("default")
    available = [b for b in list_backends(comfy_reachable=comfy_reachable) if b.get("available")]
    if not available:
        return None
    for b in available:
        if b.get("key") == preferred:
            return preferred
    return available[0]["key"]


def get_backend(key: Optional[str], *, comfy_reachable: Optional[bool] = None) -> Optional[dict]:
    registry = load_music_backends()
    backends = registry.get("backends") or {}
    if key and key in backends:
        entry = {**backends[key], "key": key}
        entry["available"] = backend_available(backends[key], comfy_reachable=comfy_reachable)
        return entry
    chosen = default_backend_key(comfy_reachable=comfy_reachable)
    if not chosen:
        return None
    entry = {**backends[chosen], "key": chosen}
    entry["available"] = True
    return entry


def music_status(*, comfy_reachable: Optional[bool] = None) -> dict[str, Any]:
    enabled = music_gen_enabled()
    backends = list_backends(comfy_reachable=comfy_reachable)
    available = [b for b in backends if b.get("available")] if enabled else []
    return {
        "ok": bool(available),
        "available": bool(available),
        "enabled": enabled,
        "default": default_backend_key(comfy_reachable=comfy_reachable) if enabled else None,
        "backends": backends,
        "reason": None if available else ("disabled" if not enabled else "no_backend"),
    }
