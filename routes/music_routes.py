"""Music generation API -- /api/music/*.

Separate from /api/comfy (do not extend comfy kind). MiniMax Music 3 first:
local Comfy engine or hosted API. Image/Video stay on /api/comfy/*.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.auth_helpers import require_privilege, require_user
from src.comfy_client import ComfyClient, ComfyError, DEFAULT_COMFY_BASE_URL, new_client_id, new_job_id
from src.settings import get_setting
from src.comfy_graphs import build_minimax_music3_graph, introspect_graph
from src.constants import GENERATED_AUDIO_DIR
from src.generated_audio import GENERATED_AUDIO_RE
from src.music_backends import (
    default_backend_key,
    get_backend,
    list_backends,
    load_music_backends,
    minimax_api_key,
    music_gen_enabled,
    music_status,
)
from src.url_safety import check_outbound_url

logger = logging.getLogger(__name__)


def _comfy_base_url() -> str:
    return (get_setting("comfy_base_url", DEFAULT_COMFY_BASE_URL) or DEFAULT_COMFY_BASE_URL).strip().rstrip("/")


_JOBS: dict[str, dict] = {}
_JOB_MAX_AGE_SECONDS = 30 * 60
_ALLOWED_AUDIO_EXTS = {"mp3", "wav", "flac", "ogg", "m4a", "aac"}


class MusicParams(BaseModel):
    prompt: str = ""
    caption: Optional[str] = None
    lyrics: Optional[str] = None
    seconds: Optional[float] = None
    seed: Optional[int] = None
    randomize_seed: bool = True
    backend: Optional[str] = None
    engine: Optional[str] = None
    instrumental: bool = False
    tiled_decode: bool = False
    unet: Optional[str] = None
    clip: Optional[str] = None
    vae: Optional[str] = None


class MusicGenerateRequest(BaseModel):
    session_id: Optional[str] = None
    params: MusicParams = MusicParams()


def _require_user(request: Request) -> str:
    return require_user(request)


def _prune_old_jobs() -> None:
    now = time.time()
    dead = []
    for job_id, job in list(_JOBS.items()):
        finished = job.get("finished_at")
        if job.get("status") in ("done", "error", "cancelled"):
            stamp = finished if finished else now
            if now - stamp > _JOB_MAX_AGE_SECONDS:
                dead.append(job_id)
    for job_id in dead:
        _JOBS.pop(job_id, None)


def _mark_finished(job: dict) -> None:
    if not job.get("finished_at"):
        job["finished_at"] = time.time()


def _new_job(owner: str, session_id: Optional[str], params: dict, backend: str) -> dict:
    return {
        "owner": owner,
        "kind": "music",
        "session_id": session_id,
        "params": params,
        "backend": backend,
        "status": "queued",
        "percent": 0,
        "node": "",
        "error": None,
        "audio": [],
        "started_at": time.time(),
        "finished_at": None,
        "_ready": asyncio.Event(),
        "_chat_written": False,
        "_task": None,
    }


def _spawn_bg(coro):
    task = asyncio.create_task(coro)
    return task


async def _comfy_reachable() -> Optional[bool]:
    url = _comfy_base_url()
    if not url:
        return False
    ok, _reason = check_outbound_url(url)
    if not ok:
        return False
    try:
        await ComfyClient(url).status()
        return True
    except Exception:
        return False


def _save_audio_bytes(data: bytes, ext: str = "mp3") -> str:
    ext = ext.lower().lstrip(".")
    if ext not in _ALLOWED_AUDIO_EXTS:
        ext = "mp3"
    digest = hashlib.sha256(data).hexdigest()[:16]
    filename = f"{digest}.{ext}"
    os.makedirs(GENERATED_AUDIO_DIR, exist_ok=True)
    path = Path(GENERATED_AUDIO_DIR) / filename
    if not path.exists():
        path.write_bytes(data)
    return filename


def _insert_generated_audio(*, filename: str, prompt: str, lyrics: str, model: str,
                            session_id: Optional[str], owner: str, gen_params: dict,
                            file_size: int) -> str:
    from core.database import GeneratedAudio, SessionLocal
    audio_id = uuid.uuid4().hex[:16]
    db = SessionLocal()
    try:
        row = GeneratedAudio(
            id=audio_id,
            filename=filename,
            prompt=prompt,
            lyrics=lyrics or "",
            model=model,
            session_id=session_id,
            owner=owner or None,
            gen_params=json.dumps(gen_params) if gen_params else None,
            file_size=file_size,
        )
        db.add(row)
        db.commit()
    finally:
        db.close()
    return audio_id


def _write_user_turn(session_id: Optional[str], owner: str, prompt: str) -> None:
    if not session_id:
        return
    try:
        from core.session_manager import session_manager
        from src.database import ChatMessage
        sess = session_manager.get_session(session_id)
        if not sess:
            return
        sess.add_message(ChatMessage("user", prompt, metadata={"via": "music"}))
    except Exception:
        logger.exception("music: failed to persist user turn")


def _write_terminal_turn(job: dict) -> None:
    if job.get("_chat_written"):
        return
    session_id = job.get("session_id")
    if not session_id:
        return
    try:
        from core.session_manager import session_manager
        from src.database import ChatMessage
        sess = session_manager.get_session(session_id)
        if not sess:
            return
        audio = (job.get("audio") or [{}])[0]
        status = job.get("status")
        if status == "done" and audio.get("url"):
            content = f"Generated music: {job.get('params', {}).get('prompt') or ''}"
            tool_events = [{
                "tool": "generate_music",
                "audio_url": audio.get("url"),
                "audio_id": audio.get("id"),
                "image_url": audio.get("url"),
                "image_prompt": job.get("params", {}).get("prompt") or "",
                "image_model": job.get("backend") or "music",
            }]
        else:
            content = job.get("error") or status or "music job ended"
            tool_events = [{"tool": "generate_music", "error": content}]
        sess.add_message(ChatMessage("assistant", content, metadata={
            "tool_events": tool_events,
            "round_texts": [content],
            "model": job.get("backend") or "music",
        }))
        job["_chat_written"] = True
    except Exception:
        logger.exception("music: failed to persist terminal turn")


async def _run_comfy_job(job_id: str, graph: dict) -> None:
    job = _JOBS[job_id]
    client_id = new_client_id()
    client = ComfyClient(_comfy_base_url())
    try:
        job["status"] = "running"
        async with __import__("contextlib").aclosing(client.run_and_stream(graph, client_id)) as stream:
            async for event in stream:
                mtype = event.get("type")
                data = event.get("data") or {}
                if mtype == "queued":
                    job["prompt_id"] = data.get("prompt_id")
                    job["_ready"].set()
                elif mtype == "progress_state":
                    nodes = data.get("nodes") or {}
                    running = None
                    for node_id, info in nodes.items():
                        if isinstance(info, dict) and info.get("state") == "running":
                            running = info
                            job["node"] = str(info.get("title") or node_id)
                            break
                    if running:
                        value = float(running.get("value") or 0)
                        mx = float(running.get("max") or 1) or 1
                        job["percent"] = int(100 * value / mx)
                elif mtype == "execution_error":
                    job["status"] = "error"
                    job["error"] = data.get("exception_message") or "ComfyUI execution error"
                    return
                elif mtype == "execution_interrupted":
                    job["status"] = "cancelled"
                    job["error"] = "cancelled"
                    return
        history = await client.history(job.get("prompt_id") or "")
        outputs = (history.get("outputs") or {}) if isinstance(history, dict) else {}
        landed = []
        for _nid, node_out in outputs.items():
            if not isinstance(node_out, dict):
                continue
            for key in ("audio", "images", "gifs"):
                for item in node_out.get(key) or []:
                    if not isinstance(item, dict) or not item.get("filename"):
                        continue
                    filename = item["filename"]
                    ext = filename.rsplit(".", 1)[-1].lower()
                    if ext not in _ALLOWED_AUDIO_EXTS:
                        continue
                    raw = await client.view_bytes(
                        filename,
                        subfolder=item.get("subfolder") or "",
                        folder_type=item.get("type") or "output",
                    )
                    local_name = _save_audio_bytes(raw, ext)
                    audio_id = _insert_generated_audio(
                        filename=local_name,
                        prompt=job.get("params", {}).get("prompt") or "",
                        lyrics=job.get("params", {}).get("lyrics") or "",
                        model=job.get("backend") or "minimax_music3_comfy",
                        session_id=job.get("session_id"),
                        owner=job.get("owner") or "",
                        gen_params={"params": job.get("params"), "kind": "music", "backend": job.get("backend")},
                        file_size=len(raw),
                    )
                    landed.append({
                        "id": audio_id,
                        "filename": local_name,
                        "url": f"/api/generated-audio/{local_name}",
                    })
        if not landed:
            job["status"] = "error"
            job["error"] = "ComfyUI finished but produced no audio"
            return
        job["audio"] = landed
        job["status"] = "done"
        job["percent"] = 100
    except ComfyError as e:
        job["status"] = "error"
        job["error"] = str(e)
    except Exception as e:
        logger.exception("music comfy job failed")
        job["status"] = "error"
        job["error"] = str(e)
    finally:
        _mark_finished(job)
        _write_terminal_turn(job)
        _prune_old_jobs()
        try:
            await client.free()
        except Exception:
            pass
        job["_ready"].set()


def _extract_api_audio(payload: Any) -> tuple[Optional[bytes], str]:
    """Best-effort extract audio bytes from a MiniMax Music 3 API response."""
    if not isinstance(payload, dict):
        return None, "mp3"
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    for key in ("audio", "audio_hex", "hex"):
        raw = data.get(key) if isinstance(data, dict) else None
        if isinstance(raw, str) and len(raw) > 32 and all(c in "0123456789abcdefABCDEF" for c in raw[:32]):
            try:
                return bytes.fromhex(raw), "mp3"
            except ValueError:
                pass
    for key in ("audio_base64", "audio"):
        raw = data.get(key) if isinstance(data, dict) else None
        if isinstance(raw, str) and len(raw) > 64 and not raw.startswith("http"):
            import base64
            try:
                return base64.b64decode(raw), "mp3"
            except Exception:
                pass
    return None, "mp3"


async def _download_url_audio(url: str) -> tuple[bytes, str]:
    ok, reason = check_outbound_url(url)
    if not ok:
        raise HTTPException(400, f"audio URL failed SSRF checks: {reason}")
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.get(url)
        r.raise_for_status()
        ext = "mp3"
        path = url.split("?", 1)[0]
        if "." in path.rsplit("/", 1)[-1]:
            cand = path.rsplit(".", 1)[-1].lower()
            if cand in _ALLOWED_AUDIO_EXTS:
                ext = cand
        return r.content, ext


async def _run_api_job(job_id: str, backend: dict) -> None:
    job = _JOBS[job_id]
    key = minimax_api_key()
    if not key:
        job["status"] = "error"
        job["error"] = "MINIMAX_API_KEY is not set"
        _mark_finished(job)
        _write_terminal_turn(job)
        job["_ready"].set()
        return
    api_url = str(backend.get("api_url") or "https://api.minimax.io/v1/music_generation")
    ok, reason = check_outbound_url(api_url)
    if not ok:
        job["status"] = "error"
        job["error"] = f"music API URL failed SSRF checks: {reason}"
        _mark_finished(job)
        _write_terminal_turn(job)
        job["_ready"].set()
        return
    params = job.get("params") or {}
    payload = {
        "model": backend.get("model") or "music-3.0",
        "prompt": params.get("caption") or params.get("prompt") or "",
        "lyrics": params.get("lyrics") or "",
        "is_instrumental": bool(params.get("instrumental")),
    }
    if params.get("seconds") is not None:
        payload["audio_setting"] = {"sample_rate": 44100, "bitrate": 256000, "format": "mp3"}
    job["status"] = "running"
    job["percent"] = 10
    job["_ready"].set()
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            r = await client.post(
                api_url,
                json=payload,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            )
        try:
            body = r.json()
        except Exception:
            body = {}
        if r.status_code >= 400:
            detail = body.get("base_resp", {}).get("status_msg") if isinstance(body, dict) else None
            raise RuntimeError(detail or f"MiniMax API HTTP {r.status_code}")
        audio_bytes, ext = _extract_api_audio(body)
        if audio_bytes is None:
            data = body.get("data") if isinstance(body, dict) else {}
            url = None
            if isinstance(data, dict):
                url = data.get("audio_url") or data.get("url") or data.get("download_url")
            if isinstance(url, str) and url.startswith("http"):
                audio_bytes, ext = await _download_url_audio(url)
        if not audio_bytes:
            raise RuntimeError("MiniMax API returned no audio bytes")
        local_name = _save_audio_bytes(audio_bytes, ext)
        audio_id = _insert_generated_audio(
            filename=local_name,
            prompt=params.get("prompt") or "",
            lyrics=params.get("lyrics") or "",
            model=backend.get("key") or "minimax_music3_api",
            session_id=job.get("session_id"),
            owner=job.get("owner") or "",
            gen_params={"params": params, "kind": "music", "backend": backend.get("key")},
            file_size=len(audio_bytes),
        )
        job["audio"] = [{
            "id": audio_id,
            "filename": local_name,
            "url": f"/api/generated-audio/{local_name}",
        }]
        job["status"] = "done"
        job["percent"] = 100
    except Exception as e:
        logger.exception("music API job failed")
        job["status"] = "error"
        job["error"] = str(e)
    finally:
        _mark_finished(job)
        _write_terminal_turn(job)
        _prune_old_jobs()
        job["_ready"].set()


def setup_music_routes() -> APIRouter:
    router = APIRouter(tags=["music"])

    @router.get("/api/music/status")
    async def music_status_route(request: Request):
        _require_user(request)
        if not music_gen_enabled():
            return music_status(comfy_reachable=None)
        reachable = await _comfy_reachable()
        return music_status(comfy_reachable=reachable)

    @router.get("/api/music/backends")
    async def music_backends_route(request: Request):
        _require_user(request)
        reachable = await _comfy_reachable()
        return {
            "default": default_backend_key(comfy_reachable=reachable),
            "backends": list_backends(comfy_reachable=reachable),
            "registry": load_music_backends().get("default"),
        }

    @router.get("/api/music/job/{job_id}")
    async def music_job(request: Request, job_id: str):
        owner = _require_user(request)
        job = _JOBS.get(job_id)
        if not job or (job.get("owner") and owner and job["owner"] != owner):
            raise HTTPException(404, "Job not found")
        return {
            "job_id": job_id,
            "status": job.get("status"),
            "percent": job.get("percent", 0),
            "error": job.get("error"),
            "audio": job.get("audio") or [],
            "backend": job.get("backend"),
            "params": job.get("params"),
        }

    @router.post("/api/music/generate")
    async def music_generate(request: Request, body: MusicGenerateRequest):
        require_privilege(request, "can_generate_images")
        owner = _require_user(request)
        try:
            from routes.gpu_routes import _load_helper
            await _load_helper().prepare_for("music")
        except Exception:
            logger.warning("gpu prepare_for(music) skipped", exc_info=True)
        if not music_gen_enabled():
            raise HTTPException(503, "Music generation is disabled")
        reachable = await _comfy_reachable()
        params = body.params.model_dump()
        caption = (params.get("caption") or params.get("prompt") or "").strip()
        lyrics = (params.get("lyrics") or "").strip()
        if not caption and not lyrics:
            raise HTTPException(400, "prompt/caption or lyrics is required")
        params["prompt"] = caption
        params["caption"] = caption
        params["lyrics"] = lyrics
        backend = get_backend(params.get("backend") or params.get("engine"), comfy_reachable=reachable)
        if not backend or not backend.get("available"):
            raise HTTPException(503, "No music backend is available. Connect ComfyUI with Music 3 weights or set MINIMAX_API_KEY.")
        engine = backend.get("engine") or backend.get("key")
        job_id = new_job_id().replace("cj-", "mj-", 1)
        job = _new_job(owner, body.session_id, params, backend.get("key") or engine)
        _JOBS[job_id] = job
        _write_user_turn(body.session_id, owner, caption or lyrics[:200])
        if engine == "minimax_music3_comfy":
            try:
                graph = build_minimax_music3_graph(params)
                resolved = introspect_graph(graph)["params"]
                resolved["randomize_seed"] = False
                resolved["backend"] = backend.get("key")
                job["params"] = {**params, **resolved}
            except Exception as e:
                raise HTTPException(400, f"Could not build the Music 3 graph: {e}")
            job["_task"] = _spawn_bg(_run_comfy_job(job_id, graph))
        elif engine == "minimax_music3_api":
            job["_task"] = _spawn_bg(_run_api_job(job_id, backend))
        else:
            raise HTTPException(400, f"Unknown music engine {engine!r}")
        try:
            await asyncio.wait_for(job["_ready"].wait(), timeout=20)
        except asyncio.TimeoutError:
            pass
        return {"job_id": job_id, "prompt_id": job.get("prompt_id"), "backend": backend.get("key")}

    @router.get("/api/music/stream/{job_id}")
    async def music_stream(request: Request, job_id: str):
        owner = _require_user(request)
        job = _JOBS.get(job_id)
        if not job or (job.get("owner") and owner and job["owner"] != owner):
            raise HTTPException(404, "Job not found")

        async def events():
            last = None
            while True:
                current = (
                    job.get("status"),
                    job.get("percent"),
                    job.get("node"),
                    job.get("error"),
                    len(job.get("audio") or []),
                )
                if current != last:
                    last = current
                    status = job.get("status")
                    if status == "done":
                        yield f"event: done\ndata: {json.dumps({'audio': job.get('audio'), 'params': job.get('params'), 'backend': job.get('backend')})}\n\n"
                        return
                    if status in ("error", "cancelled"):
                        yield f"event: error\ndata: {json.dumps({'message': job.get('error') or status})}\n\n"
                        return
                    elapsed = int(time.time() - (job.get("started_at") or time.time()))
                    yield (
                        "event: progress\n"
                        f"data: {json.dumps({'percent': job.get('percent', 0), 'node': job.get('node') or '', 'elapsed': elapsed})}\n\n"
                    )
                await asyncio.sleep(1)

        return StreamingResponse(events(), media_type="text/event-stream")

    @router.post("/api/music/cancel/{job_id}")
    async def music_cancel(request: Request, job_id: str):
        require_privilege(request, "can_generate_images")
        owner = _require_user(request)
        job = _JOBS.get(job_id)
        if not job or (job.get("owner") and owner and job["owner"] != owner):
            raise HTTPException(404, "Job not found")
        if job.get("status") in ("done", "landing"):
            return {"ok": True, "status": job.get("status")}
        job["status"] = "cancelled"
        job["error"] = "cancelled"
        _mark_finished(job)
        prompt_id = job.get("prompt_id")
        if prompt_id:
            try:
                url = _comfy_base_url()
                if url:
                    await ComfyClient(url).cancel(prompt_id)
            except Exception:
                logger.warning("music cancel: comfy interrupt failed", exc_info=True)
        _write_terminal_turn(job)
        return {"ok": True, "status": "cancelled"}

    return router
