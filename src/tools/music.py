"""Music-domain agent tool: generate_music.

Posts to /api/music/generate (never /api/comfy with a music kind) and waits
for the job to finish via GET /api/music/job/{id}.
"""
from __future__ import annotations

import json
from typing import Dict, Optional

import httpx

from src.tools._common import _INTERNAL_BASE, _internal_headers, _parse_tool_args


def _parse_generate_music(content: str) -> Dict:
    raw = (content or "").strip()
    if raw.startswith("{"):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if isinstance(parsed, dict) and (parsed.get("prompt") or parsed.get("caption") or parsed.get("lyrics")):
            return parsed
    try:
        args = _parse_tool_args(content)
        if isinstance(args, dict) and (args.get("prompt") or args.get("caption") or args.get("lyrics")):
            return args
    except ValueError:
        pass
    lines = raw.split("\n")
    out = {"prompt": lines[0].strip() if lines else ""}
    for i, key in enumerate(["lyrics", "backend", "seconds"], 1):
        if len(lines) > i and lines[i].strip():
            out[key] = lines[i].strip()
    return out


async def do_generate_music(content: str, session_id: Optional[str] = None, owner: Optional[str] = None) -> Dict:
    """Generate a song via the pluggable /api/music backends (MiniMax Music 3 first)."""
    args = _parse_generate_music(content)
    prompt = str(args.get("prompt") or args.get("caption") or "").strip()
    lyrics = str(args.get("lyrics") or "")
    if not prompt and not lyrics:
        return {"error": "prompt or lyrics is required", "exit_code": 1}

    params = {
        "prompt": prompt,
        "caption": prompt,
        "lyrics": lyrics,
        "backend": args.get("backend") or args.get("engine") or "",
        "randomize_seed": True,
    }
    if args.get("seconds") not in (None, ""):
        try:
            params["seconds"] = float(args["seconds"])
        except (TypeError, ValueError):
            pass
    if args.get("instrumental") in (True, "true", "1", "yes"):
        params["instrumental"] = True

    headers = _internal_headers(owner)
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{_INTERNAL_BASE}/api/music/generate",
                json={"session_id": session_id, "params": params},
                headers=headers,
            )
            data = resp.json() if resp.content else {}
        if resp.status_code >= 400 or not data.get("job_id"):
            detail = data.get("detail") or data.get("error") or f"HTTP {resp.status_code}"
            return {"error": str(detail), "exit_code": 1}
        job_id = data["job_id"]
        result = await _wait_for_job(job_id, headers)
        if result.get("error"):
            return {"error": result["error"], "exit_code": 1}
        audio = (result.get("audio") or [{}])[0]
        url = audio.get("url") or ""
        filename = audio.get("filename") or ""
        lines = [
            f"Generated music for: {prompt or lyrics[:80]}",
            f"model: {result.get('backend') or params.get('backend') or 'music'}",
        ]
        if url:
            lines.append(url)
        out = {
            "output": "\n".join(lines),
            "exit_code": 0,
            "audio_url": url,
            "audio_id": audio.get("id") or "",
            "audio_prompt": prompt,
            "audio_model": result.get("backend") or "music",
        }
        if filename:
            out["audio_filename"] = filename
        return out
    except Exception as e:
        return {"error": str(e), "exit_code": 1}


async def _wait_for_job(job_id: str, headers: dict, *, timeout: float = 600.0) -> Dict:
    """Poll GET /api/music/stream until done/error. The SSE relay is 1s."""
    import asyncio

    deadline = asyncio.get_event_loop().time() + timeout
    async with httpx.AsyncClient(timeout=30.0) as client:
        while asyncio.get_event_loop().time() < deadline:
            snap = await client.get(f"{_INTERNAL_BASE}/api/music/job/{job_id}", headers=headers)
            if snap.status_code == 200:
                body = snap.json()
                status = body.get("status")
                if status == "done":
                    return body
                if status in ("error", "cancelled"):
                    return {"error": body.get("error") or status}
            await asyncio.sleep(1.5)
    return {"error": "music generation timed out"}
