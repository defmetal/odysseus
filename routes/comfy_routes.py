# routes/comfy_routes.py
"""ComfyUI-backed Image/Video generation API -- /api/comfy/*.

New, studio-only file. This is the BACKEND half of the Image/Video tabs
feature described in data/studio/PLAN-IMAGE-VIDEO-TABS.md; static/
(frontend) is a parallel effort coded against the API contract in that
plan/the task that produced this file. Do not change the request/response
shapes here without updating that contract.

Generation is driven from an explicit UI panel, not agent tool-calls (plan
§9 risk 2/7) -- this file has no dependency on src/agent_loop.py,
src/tool_implementations.py, or any of the other studio-registration hot-zone
files.

Job model: POST /generate builds a graph (src/comfy_graphs.py), opens a
ComfyUI WS connection, queues it, and returns as soon as ComfyUI has
acknowledged the prompt (so the caller gets a real `prompt_id` back
synchronously). A background asyncio task keeps consuming the WS stream,
updates an in-memory job record (`_JOBS`, keyed by our own `job_id` --
mirrors ResearchHandler._active_tasks' shape, see routes/research/research_routes.py),
and lands finished images into the Gallery. GET /stream/{job_id} is a
plain polling-SSE relay over that shared record, mirroring
/api/research/stream/{id}'s transport (plan §5).

`_JOBS` is in-memory only (lost on restart, not shared across worker
processes if ever run with >1) -- an accepted limitation matching the
existing ResearchHandler precedent, not something this file tries to solve.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.auth_helpers import get_current_user, require_privilege, _auth_disabled
from src.constants import DATA_DIR, GENERATED_IMAGES_DIR
from src.comfy_client import ComfyClient, ComfyError, DEFAULT_COMFY_BASE_URL, new_client_id, new_job_id
from src.comfy_graphs import (
    GraphConversionError,
    apply_style_trigger,
    build_image_graph,
    build_wan_i2v_graph,
    filter_safetensors,
    introspect_graph,
    resolve_lora_entries,
    ui_to_api,
)
from src.generated_images import GENERATED_IMAGE_RE, resolve_generated_image_path

logger = logging.getLogger(__name__)

# filename -> local landed-copy extension allow-list. Mirrors
# src/generated_images.py's GENERATED_IMAGE_RE allow-list exactly (that
# module's serve regex already allows the video extensions too, per plan §6.4).
_ALLOWED_EXTS = {"png", "jpg", "jpeg", "webp", "gif", "mp4", "mov", "webm", "mkv", "m4v"}

# job_id -> job record (dict). See _new_job() for the shape.
_JOBS: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class LoraParam(BaseModel):
    comfy_name: str
    weight: float = 1.0
    # Video-only disambiguator ("high_noise" | "low_noise"). static/js/
    # genParams.js's video path sends this on both of its two loras[] entries
    # (comfy_name is just the sentinel 'wan_high_noise'/'wan_low_noise'
    # there, not a real filename -- src/comfy_graphs.py's
    # _extract_wan_lora_weights() resolves the real Wan lightning-LoRA
    # filenames itself). MUST be declared here or Pydantic silently drops it
    # as an undeclared extra field (model_dump() below would then hand
    # build_wan_i2v_graph() a loras list with no way to tell high from low
    # noise). None/absent for the image path, where it's unused.
    role: Optional[str] = None
    # "style" | "character", mirrored from data/studio/scripts/loras.json so
    # comfy_generate() can tell whether any STYLE LoRA is in play and decide
    # whether the style trigger applies. Like `role` above, this MUST be
    # declared or Pydantic drops it as an undeclared extra and every render
    # silently becomes general-mode.
    kind: Optional[str] = None


class GenerateParams(BaseModel):
    prompt: str = ""
    negative_prompt: Optional[str] = None
    loras: List[LoraParam] = []
    width: Optional[int] = None
    height: Optional[int] = None
    batch: Optional[int] = None
    seed: Optional[int] = None
    randomize_seed: bool = False
    steps: Optional[int] = None
    cfg: Optional[float] = None
    sampler: Optional[str] = None
    scheduler: Optional[str] = None
    denoise: Optional[float] = None
    shift: Optional[float] = None
    unet: Optional[str] = None
    clip: Optional[str] = None
    vae: Optional[str] = None
    input_image: Optional[str] = None
    frames: Optional[int] = None
    fps: Optional[int] = None
    # Optional explicit style key/alias from styles.json ("smoon", "chflash",
    # "90s anime", ...). When set, the style trigger is applied even with no
    # style LoRA selected. Leave unset for normal behaviour.
    style_hint: Optional[str] = None


class GenerateRequest(BaseModel):
    kind: str = "image"
    workflow: Optional[str] = None
    session_id: Optional[str] = None
    params: GenerateParams = GenerateParams()


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def _client() -> ComfyClient:
    from src.settings import get_setting
    return ComfyClient(get_setting("comfy_base_url", DEFAULT_COMFY_BASE_URL))


def _require_user(request: Request) -> str:
    user = get_current_user(request)
    if not user:
        if _auth_disabled():
            return ""
        raise HTTPException(401, "Not authenticated")
    return user


def _workflows_dir() -> Path:
    return Path(DATA_DIR) / "studio" / "comfy" / "ComfyUI" / "user" / "default" / "workflows"


def _resolve_workflow_file(directory: Path, name: str) -> Optional[Path]:
    """Confine `name` to `directory` (no traversal) and require it to exist.
    Mirrors src/generated_images.py's resolve_generated_image_path /
    routes/research/research_routes.py's _find_research_path pattern."""
    if not name or "/" in name or "\\" in name or ".." in name:
        return None
    candidate = directory / f"{name}.json"
    try:
        root = directory.resolve()
        resolved = candidate.resolve()
        resolved.relative_to(root)
    except (ValueError, OSError):
        return None
    if not resolved.is_file():
        return None
    return resolved


def _classify_workflow(path: Path) -> str:
    """image | video | unknown, by presence of a tell-tale class_type string
    anywhere in the file. Simple substring check rather than a structural
    walk: the video workflow's real nodes live inside `definitions.
    subgraphs[]` (plan §3b), not the top-level `nodes[]`, so a class_type
    can appear at either depth -- a raw text search of the parsed dict's
    round-tripped JSON dodges having to special-case that structure."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "unknown"
    text = json.dumps(raw)
    if '"SaveVideo"' in text or '"WanImageToVideo"' in text:
        return "video"
    if '"SaveImage"' in text:
        return "image"
    return "unknown"


def _load_curated_loras() -> list[dict]:
    """Enabled entries from data/studio/scripts/loras.json (plan §6.3a)."""
    path = Path(DATA_DIR) / "studio" / "scripts" / "loras.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("comfy_options: could not read loras.json: %s", e)
        return []
    out = []
    for key, entry in (raw.get("loras") or {}).items():
        if not isinstance(entry, dict) or not entry.get("enabled", True):
            continue
        out.append({
            "key": key,
            "name": entry.get("name", key),
            "comfy_name": entry.get("comfy_name", ""),
            "kind": entry.get("kind", ""),
            "default_weight": entry.get("default_weight", 1.0),
            "status": entry.get("status", ""),
            "note": entry.get("note", ""),
        })
    return out


# DEFECT 1 -- a short-lived, in-process cache of ComfyUI's full /object_info
# response. Mirrors static/js/genParams.js's OWN ~60s _CACHE_MS on the
# frontend (see that file's _fetchOptions()) -- short enough that a ComfyUI
# restart (which re-reads extra_model_paths.yaml and starts reporting the
# widened LoRA roots, per the DEFECT 1 note in src/comfy_graphs.py) is picked
# up here within about a minute, with NO Odysseus restart required (this is
# a plain in-process dict, not persisted anywhere) and no manual action.
# Also cuts the number of full-registry fetches against ComfyUI when several
# browser tabs/users hit /api/comfy/options or /api/comfy/generate close
# together.
_OBJECT_INFO_CACHE: dict = {"ts": 0.0, "data": None}
_OBJECT_INFO_CACHE_TTL_SECONDS = 60.0


async def _cached_object_info(*, force: bool = False) -> dict:
    now = time.time()
    if not force and _OBJECT_INFO_CACHE["data"] is not None and (now - _OBJECT_INFO_CACHE["ts"]) < _OBJECT_INFO_CACHE_TTL_SECONDS:
        return _OBJECT_INFO_CACHE["data"]
    data = await _client().object_info()
    _OBJECT_INFO_CACHE["ts"] = time.time()
    _OBJECT_INFO_CACHE["data"] = data
    return data


def _combo_options(object_info: dict, class_type: str, input_name: str) -> list[str]:
    """Pull a COMBO (dropdown) input's option list out of a /object_info
    response -- `input.required.<name>` is `[options_list_or_type, {...}]`;
    only the list form is a combo."""
    node = object_info.get(class_type) or {}
    required = ((node.get("input") or {}).get("required") or {})
    spec = required.get(input_name)
    if not isinstance(spec, list) or not spec or not isinstance(spec[0], list):
        return []
    return [str(o) for o in spec[0]]


def _valid_session_id(session_id: Optional[str]) -> Optional[str]:
    """Only pass through a session_id that actually exists -- sessions.id is
    a hard FK on gallery_images (foreign-key enforcement is on globally, per
    core/database.py's migration-section warning), so a stale/bogus id would
    otherwise raise IntegrityError on insert."""
    if not session_id:
        return None
    from core.database import SessionLocal
    from core.database import Session as DbSession
    db = SessionLocal()
    try:
        return session_id if db.query(DbSession).filter(DbSession.id == session_id).first() else None
    finally:
        db.close()


async def _resolve_input_image(input_image: str) -> tuple[str, str, str]:
    """Resolve params['input_image'] into a (name, subfolder, type) triple
    ComfyUI's LoadImage node can actually see (GAP 1 -- upload bridge).

    ComfyUI's LoadImage resolves filenames against ITS OWN input/ directory
    (folder_paths.get_annotated_filepath(), data/studio/comfy/ComfyUI/
    folder_paths.py:343-357) -- it has no idea Odysseus's gallery exists, so
    an Odysseus gallery filename passed straight through always 404s inside
    ComfyUI. Two cases:

      1. `input_image` matches GENERATED_IMAGE_RE -- an existing Odysseus
         render (e.g. picked from the Gallery as an img2img source or a
         video start frame -- plan §3b's Tetsuya-keyframe -> Wan-video
         chain). Read the bytes from data/generated_images/ and push them
         through ComfyUI's own POST /upload/image so LoadImage can resolve
         them -- this is what makes "use an existing render" work at all;
         the frontend never has to know this happens.
      2. Anything else is assumed to already be a name sitting in ComfyUI's
         own input/ directory -- i.e. the frontend already called
         POST /api/comfy/upload (which forwards straight to ComfyUI) and is
         passing the resulting filename back verbatim. Passed through as
         ("input", "") -- no re-upload, no wasted round trip.

    The returned triple is fed straight into build_image_graph() as
    input_image / input_image_subfolder / input_image_type, which applies
    ComfyUI's " [type]" annotation convention itself (src/comfy_graphs.py's
    comfy_image_ref()) -- this function only decides WHAT to reference, not
    how to spell the reference ComfyUI needs.
    """
    if GENERATED_IMAGE_RE.fullmatch(input_image):
        path = resolve_generated_image_path(input_image)  # raises HTTPException if missing/unsafe
        file_bytes = path.read_bytes()
        try:
            result = await _client().upload_image(file_bytes, input_image)
        except Exception as e:
            raise HTTPException(502, f"Could not forward gallery image {input_image!r} to ComfyUI: {e}")
        return (
            result.get("name", input_image),
            result.get("subfolder", ""),
            result.get("type", "input"),
        )
    return (input_image, "", "input")


def _format_error_event(data: dict) -> str:
    msg = data.get("exception_message")
    if not msg:
        node = data.get("node_type") or data.get("node_id")
        msg = f"Execution interrupted at node {node}" if node else "Execution interrupted"
    exc_type = data.get("exception_type")
    if exc_type and str(exc_type) not in str(msg):
        msg = f"{exc_type}: {msg}"
    return str(msg)[:1000]


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------------------
# Job bookkeeping + the background runner
# ---------------------------------------------------------------------------

def _new_job(owner: str, kind: str, workflow: Optional[str], session_id: Optional[str], params: dict, graph: dict) -> dict:
    return {
        "owner": owner,
        "kind": kind,
        "workflow": workflow,
        "session_id": session_id,
        "params": params,
        "status": "queued",       # queued -> running -> landing -> done | error | cancelled
        "prompt_id": None,
        "percent": 0.0,
        "node": None,
        "step": 0,
        "max": 0,
        "images": None,
        "error": None,
        "started_at": time.time(),
        "_node_titles": {nid: n.get("class_type", nid) for nid, n in graph.items()},
        "_ready": asyncio.Event(),  # set once prompt_id is known OR queueing failed
    }


def _apply_progress_state(job: dict, data: dict) -> None:
    nodes = data.get("nodes") or {}
    if not nodes:
        return
    total_value = sum(float(n.get("value") or 0) for n in nodes.values())
    total_max = sum(float(n.get("max") or 0) for n in nodes.values())
    job["step"] = total_value
    job["max"] = total_max
    if total_max:
        job["percent"] = round(min(100.0, (total_value / total_max) * 100), 1)
    # Surface whichever node this update is about as the "current" node.
    last_node_id = next(iter(nodes.keys()), None)
    if last_node_id is not None:
        job["node"] = last_node_id


# ---------------------------------------------------------------------------
# GAP 3 -- WS-drop poll fallback (plan §5 point 6). A dropped WEBSOCKET does
# not mean the RENDER failed -- ComfyUI keeps executing server-side
# independently of any one client's connection to it -- so both ways
# _run_job's WS stream can end without ComfyUI's own completion signal (a
# clean disconnect, or an exception raised mid-stream) get one chance to
# resolve via polling before being reported as a failure. The dangerous half
# (a false "done" on a clean disconnect) is already closed by the
# `completed` flag in _run_job; this closes the annoying half where a
# recoverable blip kills an otherwise-fine render.
# ---------------------------------------------------------------------------

_POLL_FALLBACK_INTERVAL_SECONDS = 2.0
# Generous on purpose: plan §9 risk 1 notes a GPU-swap stall alone can run
# 1-3 minutes before a render even starts. This is how long to wait for a
# result AFTER the WS already dropped, not the render's own time budget.
_POLL_FALLBACK_TIMEOUT_SECONDS = 300.0


async def _poll_until_resolved(
    client: ComfyClient, prompt_id: str, *,
    interval: float = _POLL_FALLBACK_INTERVAL_SECONDS,
    timeout: float = _POLL_FALLBACK_TIMEOUT_SECONDS,
) -> tuple[bool, Optional[str]]:
    """Poll GET /history/{prompt_id} and GET /queue every `interval` seconds.
    ComfyUI's history entry always carries a status regardless of outcome
    (data/studio/comfy/ComfyUI/main.py:372-377 sets
    status_str='success'|'error' unconditionally on BOTH branches), so a
    non-empty /history response is resolved either way, not just on success.

    Returns (ok, error_message):
      (True, None)   -- history shows the prompt finished successfully;
                         caller should proceed to land images as normal.
      (False, <msg>) -- history shows an error status, OR the prompt_id
                         disappeared from both history and the queue (lost),
                         OR `timeout` elapsed with no resolution either way.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        await asyncio.sleep(interval)
        try:
            history = await client.history(prompt_id)
        except Exception as e:
            logger.warning("comfy poll fallback: GET /history/%s failed: %s", prompt_id, e)
            history = {}
        if history:
            status = history.get("status") or {}
            if status.get("status_str") == "error":
                messages = status.get("messages")
                detail = f" ({messages})" if messages else ""
                return False, f"ComfyUI reported an error for prompt {prompt_id}{detail}"
            return True, None

        try:
            queue = await client.queue_state()
        except Exception as e:
            logger.warning("comfy poll fallback: GET /queue failed: %s", e)
            continue  # one bad /queue poll isn't fatal -- keep trying
        still_present = any(
            isinstance(entry, (list, tuple)) and len(entry) > 1 and entry[1] == prompt_id
            for key in ("queue_running", "queue_pending")
            for entry in (queue.get(key) or [])
        )
        if not still_present:
            return False, f"Lost track of prompt {prompt_id} after the connection dropped (not in ComfyUI's history or queue)."

    return False, f"Timed out waiting for prompt {prompt_id} to finish after the connection dropped."


async def _finish_via_poll_or_error(job: dict, job_id: str, client: ComfyClient, stream_exc: Optional[BaseException]) -> None:
    """Shared tail for the two ways _run_job's WS stream can end without
    ComfyUI's own completion signal: a clean disconnect (`async for` just
    stops, stream_exc=None) or a raised exception (an abnormal WS close,
    stream_exc=that exception). Mutates `job` in place; never raises --
    callers rely on job["status"]/job["error"] rather than an exception.
    """
    prompt_id = job.get("prompt_id")
    if not prompt_id:
        # Never even got ComfyUI's "queued" ack -- nothing to poll.
        job["status"] = "error"
        job["error"] = str(stream_exc) if stream_exc else (
            "ComfyUI's progress stream ended before the run finished (connection dropped?)."
        )
        return
    logger.warning(
        "comfy job %s: WS stream for prompt %s ended early (%s) -- falling back to polling /history + /queue",
        job_id, prompt_id, stream_exc or "clean disconnect",
    )
    ok, err = await _poll_until_resolved(client, prompt_id)
    if not ok:
        job["status"] = "error"
        job["error"] = err
        return
    try:
        job["status"] = "landing"
        job["images"] = await _land_images(job)
        job["status"] = "done"
    except Exception as e:
        logger.exception("comfy job %s failed while landing after poll-fallback recovery", job_id)
        job["status"] = "error"
        job["error"] = str(e)


async def _run_job(job_id: str, graph: dict, client_id: str) -> None:
    job = _JOBS.get(job_id)
    if job is None:
        return
    client = _client()
    # Set True only by ComfyUI's own completion signal (an "executing" event
    # with node=None for our prompt_id -- see ComfyClient.run_and_stream).
    # Without this flag, a WS connection that drops cleanly (no exception,
    # the `async for` just stops) partway through a run would fall through
    # to _land_images() and report a false "done" with zero images instead
    # of a clear error -- see _finish_via_poll_or_error for what happens
    # instead (GAP 3 / plan §5 point 6: poll before giving up).
    completed = False
    try:
        async for event in client.run_and_stream(graph, client_id):
            mtype = event.get("type")
            data = event.get("data") or {}
            if mtype == "queued":
                job["prompt_id"] = data.get("prompt_id")
                job["status"] = "running"
                job["_ready"].set()
            elif mtype == "progress_state":
                _apply_progress_state(job, data)
            elif mtype == "executing":
                if data.get("node") is not None:
                    job["node"] = data.get("node")
                else:
                    completed = True
            elif mtype == "executed":
                pass  # per-node completion; progress_state already drives percent
            elif mtype in ("execution_error", "execution_interrupted"):
                job["status"] = "error" if mtype == "execution_error" else "cancelled"
                job["error"] = _format_error_event(data)
                return

        if not completed:
            await _finish_via_poll_or_error(job, job_id, client, None)
            return

        job["status"] = "landing"
        job["images"] = await _land_images(job)
        job["status"] = "done"
    except ComfyError as e:
        job["status"] = "error"
        job["error"] = str(e)
    except Exception as e:
        if job.get("status") == "landing":
            # _land_images() itself failed (DB/disk) -- the render already
            # finished server-side; polling ComfyUI again can't fix a LOCAL
            # write failure, so report it as-is rather than retrying.
            logger.exception("comfy job %s failed while landing", job_id)
            job["status"] = "error"
            job["error"] = str(e)
        else:
            # The WS raised mid-stream (e.g. an abnormal close from a
            # network blip) instead of ending cleanly -- same recoverable-
            # blip fallback as the clean-disconnect branch above.
            await _finish_via_poll_or_error(job, job_id, client, e)
    finally:
        job["_ready"].set()


async def _land_images(job: dict) -> list[dict]:
    """Fetch every SaveImage output for job['prompt_id'] via /view, write it
    into data/generated_images/, and insert a GalleryImage row with the full
    params + workflow name in gen_params (plan §6.4)."""
    prompt_id = job.get("prompt_id")
    if not prompt_id:
        return []
    client = _client()
    history = await client.history(prompt_id)
    outputs = (history or {}).get("outputs") or {}

    from core.database import SessionLocal, GalleryImage

    img_dir = Path(GENERATED_IMAGES_DIR)
    img_dir.mkdir(parents=True, exist_ok=True)

    params = job.get("params") or {}
    session_id = _valid_session_id(job.get("session_id"))
    gen_params_json = json.dumps({
        "params": params,
        "workflow": job.get("workflow"),
        "kind": job.get("kind"),
        "prompt_id": prompt_id,
    })
    width, height = params.get("width"), params.get("height")
    size = f"{width}x{height}" if width and height else None

    landed: list[dict] = []
    for node_output in outputs.values():
        if not isinstance(node_output, dict):
            continue
        for img in (node_output.get("images") or []):
            filename = img.get("filename")
            if not filename:
                continue
            try:
                content = await client.view_bytes(filename, img.get("subfolder", ""), img.get("type", "output"))
            except Exception as e:
                logger.warning("comfy: could not fetch %s via /view: %s", filename, e)
                continue

            ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "png"
            if ext not in _ALLOWED_EXTS:
                ext = "png"
            local_name = f"{uuid.uuid4().hex[:12]}.{ext}"
            (img_dir / local_name).write_bytes(content)

            gallery_id: Optional[str] = None
            db = SessionLocal()
            try:
                gallery_id = str(uuid.uuid4())
                db.add(GalleryImage(
                    id=gallery_id,
                    filename=local_name,
                    prompt=params.get("prompt", ""),
                    model="ComfyUI",
                    size=size,
                    tags=f"comfy,{job.get('kind') or 'image'}",
                    session_id=session_id,
                    owner=job.get("owner") or None,
                    is_active=True,
                    gen_params=gen_params_json,
                    file_size=len(content),
                ))
                db.commit()
            except Exception:
                db.rollback()
                logger.exception("comfy: failed to write gallery row for %s", local_name)
                gallery_id = None
            finally:
                db.close()

            landed.append({
                "filename": local_name,
                "url": f"/api/generated-image/{local_name}",
                "gallery_id": gallery_id,
            })
    return landed


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def setup_comfy_routes() -> APIRouter:
    router = APIRouter(tags=["comfy"])

    @router.get("/api/comfy/status")
    async def comfy_status(request: Request):
        _require_user(request)
        try:
            info = await _client().status()
            return {
                "ok": True,
                "version": info.get("version", ""),
                "queue_pending": info.get("queue_pending", 0),
                "queue_running": info.get("queue_running", 0),
                "error": None,
            }
        except Exception as e:
            return {"ok": False, "version": "", "queue_pending": 0, "queue_running": 0, "error": str(e)}

    @router.get("/api/comfy/options")
    async def comfy_options(request: Request):
        _require_user(request)
        result = {
            "loras": _load_curated_loras(),
            "loras_all": [],
            "unets": [],
            "vaes": [],
            "clips": [],
            "samplers": [],
            "schedulers": [],
        }
        try:
            info = await _cached_object_info()
        except Exception as e:
            # Degrade gracefully -- the curated list above needs no network,
            # and /api/comfy/status is the dedicated reachability probe. We
            # cannot verify availability against a server we can't reach, so
            # pass every curated entry through unchanged (available=True,
            # resolved_name=its own comfy_name) rather than greying
            # everything out: ComfyUI being fully down is a different,
            # already-surfaced problem (see /api/comfy/status), not the
            # "needs a restart to see newly-mapped LoRA roots" case DEFECT 1
            # targets -- and greying out every LoRA on a transient network
            # blip would be a strictly worse regression than today.
            logger.warning("comfy_options: could not reach ComfyUI /object_info: %s", e)
            result["loras"] = [
                {**entry, "available": True, "resolved_name": entry.get("comfy_name")}
                for entry in result["loras"]
            ]
            return result
        # DEFECT 1: resolve each curated entry's comfy_name against ComfyUI's
        # LIVE LoraLoaderModelOnly list (exact -> basename -> unavailable; see
        # src/comfy_graphs.py's resolve_lora_entries() docstring). The "show
        # all detected" fallback list is filtered to real adapter files only
        # -- the live list currently also reports "optimizer.pt", a training
        # checkpoint artifact, never a valid LoRA.
        live_lora_names = filter_safetensors(_combo_options(info, "LoraLoaderModelOnly", "lora_name"))
        result["loras"] = resolve_lora_entries(result["loras"], live_lora_names)
        result["loras_all"] = live_lora_names
        result["unets"] = _combo_options(info, "UNETLoader", "unet_name")
        result["vaes"] = _combo_options(info, "VAELoader", "vae_name")
        result["clips"] = _combo_options(info, "CLIPLoader", "clip_name")
        result["samplers"] = _combo_options(info, "KSampler", "sampler_name")
        result["schedulers"] = _combo_options(info, "KSampler", "scheduler")
        return result

    @router.get("/api/comfy/workflows")
    async def comfy_workflows(request: Request):
        _require_user(request)
        out = []
        d = _workflows_dir()
        if d.is_dir():
            for p in sorted(d.glob("*.json")):
                if p.stem.endswith("_api"):
                    continue  # a sibling Export (API) file, not its own workflow entry
                out.append({"name": p.stem, "filename": p.name, "kind": _classify_workflow(p)})
        return {"workflows": out}

    @router.get("/api/comfy/workflows/{name}/params")
    async def comfy_workflow_params(name: str, request: Request):
        _require_user(request)
        d = _workflows_dir()
        path = _resolve_workflow_file(d, name)
        if path is None:
            return {"ok": False, "params": {}, "unsupported": [], "error": "Workflow not found"}

        api_graph = None
        # Tier 1: a sibling "<name>_api.json" export (plan §4c).
        api_sibling = path.with_name(path.stem + "_api.json")
        if api_sibling.is_file():
            try:
                api_graph = json.loads(api_sibling.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("comfy: sibling API json unreadable for %s: %s", name, e)

        if api_graph is None:
            # Tier 2: convert on the fly via live /object_info.
            try:
                ui_graph = json.loads(path.read_text(encoding="utf-8"))
            except Exception as e:
                return {"ok": False, "params": {}, "unsupported": [], "error": f"Could not read workflow file: {e}"}
            try:
                object_info = await _client().object_info()
                api_graph = ui_to_api(ui_graph, object_info)
            except GraphConversionError as e:
                # Tier 3: the honest escape hatch (plan §4c).
                return {
                    "ok": False, "params": {}, "unsupported": [],
                    "error": (
                        f"Could not auto-convert '{name}': {e} Open this workflow in "
                        f"ComfyUI once and use File -> Export (API) to save a sibling "
                        f"'{path.stem}_api.json' next to it, then try again."
                    ),
                }
            except Exception as e:
                return {"ok": False, "params": {}, "unsupported": [], "error": f"Could not reach ComfyUI: {e}"}

        result = introspect_graph(api_graph)
        return {"ok": True, "params": result["params"], "unsupported": result["unsupported"], "error": None}

    @router.post("/api/comfy/upload")
    async def comfy_upload(request: Request, file: UploadFile = File(...)):
        """GAP 1 -- upload bridge. Forwards a browser-uploaded file straight
        to ComfyUI's own POST /upload/image so a subsequent /generate call's
        input_image can reference the returned filename directly -- no
        Odysseus-side storage of the upload; ComfyUI's own input/ directory
        IS the storage for this path. Same auth guard as /generate (this is
        part of the generate flow, not a general-purpose upload endpoint --
        for that, see routes/upload_routes.py's POST /api/upload).

        Response keys are deliberately "filename"/"subfolder"/"type" (not
        ComfyUI's own "name") to match this codebase's usual upload-response
        shape (see e.g. _land_images()'s landed-image dicts) -- the frontend
        should never need to know ComfyUI spells it "name".
        """
        require_privilege(request, "can_generate_images")
        content = await file.read()
        if not content:
            raise HTTPException(400, "Empty file")
        try:
            result = await _client().upload_image(content, file.filename or "upload.png")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(502, f"ComfyUI upload failed: {e}")
        return {
            "filename": result.get("name"),
            "subfolder": result.get("subfolder", ""),
            "type": result.get("type", "input"),
        }

    @router.post("/api/comfy/generate")
    async def comfy_generate(body: GenerateRequest, request: Request):
        user = require_privilege(request, "can_generate_images")

        if body.kind not in ("image", "video"):
            raise HTTPException(400, f"Unknown kind: {body.kind!r}")

        params = body.params.model_dump()

        if body.kind == "image":
            # Style trigger is applied ONLY when a style LoRA is actually in
            # play. The trigger words ("toei90s style, smoon") are the tokens
            # the style LoRA was trained against -- with no style LoRA loaded
            # they are not neutral, they are noise: "smoon" means nothing to
            # base Z-Image and "toei90s style" drags it toward generic retro
            # anime. Prepending them unconditionally made it impossible to get
            # a clean general-purpose render out of this tab even with every
            # LoRA deselected.
            #
            # So: zero style LoRAs == "general mode", the ComfyUI-side
            # equivalent of the :8100 server's Z-Image-General variant.
            # An explicit style_hint still forces the trigger on, and
            # apply_style_trigger() stays idempotent, so a saved workflow's
            # hand-typed trigger is never doubled up (plan §7).
            _style_loras = [
                l for l in (params.get("loras") or [])
                if (l.get("kind") or "").lower() == "style"
            ]
            if _style_loras or params.get("style_hint"):
                params["prompt"] = apply_style_trigger(
                    params.get("prompt") or "", params.get("style_hint")
                )

            # DEFECT 1 defense-in-depth: re-resolve every requested LoRA's
            # comfy_name against ComfyUI's LIVE list here too, not just at
            # /api/comfy/options -- a stale frontend options cache (up to
            # genParams.js's own _CACHE_MS old) or any other caller of this
            # API could still submit a registry comfy_name ComfyUI's current
            # combo doesn't recognize. Resolving (rather than only rejecting)
            # is what makes generation actually WORK today against a
            # pre-restart ComfyUI even if the client sent the registry's raw
            # nested-path spelling.
            #
            # Deliberately SKIPPED for kind == "video": build_wan_i2v_graph()
            # never treats params["loras"][i]["comfy_name"] as a real
            # filename -- genParams.js sends sentinel values
            # ('wan_high_noise' / 'wan_low_noise') purely so
            # _extract_wan_lora_weights() can tell which weight is which; the
            # ACTUAL Wan lightning-LoRA filenames are always the fixed
            # DEFAULT_WAN_LORA_HIGH/LOW constants, never registry- or
            # request-supplied, so resolving them against the live registry
            # list here would incorrectly reject every video job.
            if params.get("loras"):
                try:
                    info = await _cached_object_info()
                    live_lora_names = filter_safetensors(_combo_options(info, "LoraLoaderModelOnly", "lora_name"))
                except Exception as e:
                    logger.warning("comfy_generate: could not verify LoRA availability against live ComfyUI: %s", e)
                    live_lora_names = None
                if live_lora_names is not None:
                    resolved_loras = resolve_lora_entries(params["loras"], live_lora_names)
                    unavailable = [l.get("comfy_name") for l in resolved_loras if not l.get("available")]
                    if unavailable:
                        raise HTTPException(
                            400,
                            "LoRA(s) not available on the ComfyUI server (it may need a "
                            f"restart to pick up new adapter files): {', '.join(map(str, unavailable))}",
                        )
                    params["loras"] = [{**l, "comfy_name": l["resolved_name"]} for l in resolved_loras]
                # else: ComfyUI unreachable right now -- let the LoRAs through
                # unresolved; the actual failure will surface naturally when
                # _run_job() tries to queue the prompt (same graceful-degrade
                # rationale as /api/comfy/options above).
        # else: kind == "video" -- deliberately NOT triggered. Wan has no
        # knowledge of the studio style/character LoRAs (plan §3b /
        # build_wan_i2v_graph()'s docstring); prepending the trigger would
        # just add tokens Wan's own text encoder has no use for.

        if params.get("input_image"):
            # GAP 1 -- gallery passthrough: an Odysseus gallery filename (or
            # an already-uploaded-to-Comfy filename from POST
            # /api/comfy/upload) becomes whatever LoadImage can actually
            # resolve inside ComfyUI's own input/ directory. Shared by both
            # kinds -- image's img2img source and video's start frame both
            # resolve through the exact same bridge (task step 11: "reuse
            # them; do not duplicate").
            name, subfolder, ftype = await _resolve_input_image(params["input_image"])
            params["input_image"] = name
            params["input_image_subfolder"] = subfolder
            params["input_image_type"] = ftype

        builder = build_image_graph if body.kind == "image" else build_wan_i2v_graph
        try:
            graph = builder(params)
        except Exception as e:
            raise HTTPException(400, f"Could not build the ComfyUI graph: {e}")

        # GAP 2 -- seed round-trip: recover the FINAL concrete params (post
        # style-trigger, post seed-resolution) from the graph we just built,
        # instead of persisting the pre-resolution request body. Without
        # this, a randomize_seed: true request has its seed rolled inside
        # build_image_graph() and then discarded -- it would never reach the
        # client or the gallery row, defeating the entire point of exposing
        # seed. This is the documented round-trip contract from
        # src/comfy_graphs.py's module docstring ("this round-trip is also
        # how routes/comfy_routes.py recovers the FINAL concrete params...
        # for the gen_params column").
        resolved_params = introspect_graph(graph)["params"]
        # The seed above is now a concrete, fixed value already baked into
        # the graph -- resubmitting these resolved params verbatim (e.g. a
        # UI "re-roll with one value changed" action) must NOT re-roll a
        # fresh random seed.
        resolved_params["randomize_seed"] = False

        session_id = body.session_id
        job_id = new_job_id()
        client_id = new_client_id()
        job = _new_job(user, body.kind, body.workflow, session_id, resolved_params, graph)
        _JOBS[job_id] = job

        asyncio.create_task(_run_job(job_id, graph, client_id))

        try:
            await asyncio.wait_for(job["_ready"].wait(), timeout=20)
        except asyncio.TimeoutError:
            logger.warning("comfy_generate: timed out waiting for ComfyUI to accept job %s", job_id)

        if job.get("prompt_id") is None and job.get("error"):
            raise HTTPException(502, job["error"])

        return {"job_id": job_id, "prompt_id": job.get("prompt_id")}

    @router.get("/api/comfy/stream/{job_id}")
    async def comfy_stream(job_id: str, request: Request):
        user = _require_user(request)
        job = _JOBS.get(job_id)
        if job is None or job.get("owner", "") != user:
            raise HTTPException(404, "Job not found")

        async def _generate():
            while True:
                current = _JOBS.get(job_id)
                if current is None:
                    yield _sse("error", {"message": "Job not found"})
                    return
                status = current.get("status")
                if status == "error":
                    yield _sse("error", {"message": current.get("error") or "Unknown error"})
                    return
                if status == "cancelled":
                    yield _sse("error", {"message": current.get("error") or "Cancelled"})
                    return
                if status == "done":
                    # GAP 2 -- seed round-trip: "params" carries the FULLY
                    # RESOLVED params (post style-trigger, post seed
                    # resolution -- see comfy_generate()) so the client can
                    # persist/re-submit the exact seed that produced this
                    # render, not the pre-resolution request it sent.
                    yield _sse("done", {
                        "images": current.get("images") or [],
                        "params": current.get("params") or {},
                    })
                    return
                node = current.get("node")
                titles = current.get("_node_titles") or {}
                yield _sse("progress", {
                    "percent": current.get("percent", 0.0),
                    "node": node,
                    "node_title": titles.get(node, node),
                    "step": current.get("step", 0),
                    "max": current.get("max", 0),
                    "elapsed": round(time.time() - current.get("started_at", time.time()), 1),
                })
                await asyncio.sleep(1.0)

        return StreamingResponse(
            _generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.post("/api/comfy/cancel/{job_id}")
    async def comfy_cancel(job_id: str, request: Request):
        user = require_privilege(request, "can_generate_images")
        job = _JOBS.get(job_id)
        if job is None or job.get("owner", "") != user:
            raise HTTPException(404, "Job not found")
        try:
            await _client().cancel(job.get("prompt_id"))
        except Exception as e:
            logger.warning("comfy cancel: request failed for job %s: %s", job_id, e)
        if job.get("status") not in ("done", "error"):
            job["status"] = "cancelled"
            job["error"] = job.get("error") or "Cancelled by user"
        return {"ok": True}

    return router
