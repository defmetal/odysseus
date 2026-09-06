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
import contextlib
import json
import logging
import time
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.auth_helpers import require_privilege, require_user
from src.constants import DATA_DIR, GENERATED_IMAGES_DIR
from src.comfy_client import ComfyClient, ComfyError, DEFAULT_COMFY_BASE_URL, new_client_id, new_job_id
from src.comfy_graphs import (
    GraphConversionError,
    apply_model_preset,
    apply_style_trigger,
    build_image_graph,
    build_minimax_h3_graph,
    build_wan_i2v_graph,
    filter_safetensors,
    introspect_graph,
    load_model_registry,
    resolve_lora_entries,
    resolve_model_entries,
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
    # DEFECT 7: Optional[...] = None (not []) so an ABSENT `loras` field and
    # a DELIBERATE empty selection (the UI does let a user uncheck every
    # LoRA row) are distinguishable on the wire -- see apply_model_preset()'s
    # own docstring (src/comfy_graphs.py) for the None-vs-[] contract this
    # enables. model_dump()'s "loras" key is therefore None when the
    # frontend omits it, and [] only when it was sent explicitly.
    loras: Optional[List[LoraParam]] = None
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
    # data/studio/scripts/models.json preset key ("studio_toei",
    # "zimage_general", "qwen_image_2512", ...). When set (image kind only),
    # comfy_generate() resolves it and calls apply_model_preset() to fill any
    # unet/clip/clip_type/vae/loras/defaults the caller didn't already
    # override -- "pick a model, type a prompt, hit Generate" (models.json's
    # own _comment). Its `style_trigger` flag also becomes the AUTHORITY for
    # whether the studio trigger is applied, superseding the old any-style-
    # LoRA-selected gate. Unset means "no preset" -- the caller is
    # responsible for unet/clip/vae/loras itself and the style-trigger gate
    # falls back to its pre-existing kind-based behaviour, exactly like every
    # request before this field existed (a direct API caller that never
    # passes `model` keeps working unchanged).
    model: Optional[str] = None
    # CLIPLoader "type" combo ("lumina2" for Z-Image, "qwen_image" for
    # Qwen-Image, "minimax" for MiniMax H3, ...). Normally filled by a model
    # preset; settable directly for a caller that supplies unet/clip/vae
    # itself without a preset key.
    clip_type: Optional[str] = None
    # -- MiniMax H3 only (kind == "video", model == "minimax_h3") -- all
    # Optional/None-default so every OTHER kind/model combination is fully
    # unaffected by their existence (Pydantic simply never sees them set).
    # duration, in SECONDS -- converted to build_minimax_h3_graph()'s raw
    # `length` frame count via src/comfy_graphs.py's _h3_length(). NOT a raw
    # frame count itself (task: expose duration in seconds to the user).
    seconds: Optional[float] = None
    # OPTIONAL end frame -- a SECOND, independent image slot alongside the
    # existing `input_image` (H3's own start/"first" frame). Resolved through
    # the EXACT SAME upload/gallery-passthrough bridge as input_image (see
    # _resolve_input_image() below) -- just a second filename.
    last_frame: Optional[str] = None
    # MiniMaxH3SigmaShift's two floats (12.0/3.0 are that node's own
    # defaults). The node is omitted from the built graph entirely unless a
    # request differs from them -- see build_minimax_h3_graph().
    shift_video: Optional[float] = None
    shift_audio: Optional[float] = None
    # The SECOND VAELoader (the existing `vae` field is the VIDEO vae for
    # H3). Both are REQUIRED by build_minimax_h3_graph() -- there is no
    # "no audio" mode for that graph.
    audio_vae: Optional[str] = None


class GenerateRequest(BaseModel):
    kind: str = "image"
    workflow: Optional[str] = None
    session_id: Optional[str] = None
    params: GenerateParams = GenerateParams()
    # Debug escape hatch for Image-tab CPU prompt rewrite. Default ON for
    # kind == "image"; video/music never rewrite. Query ?skip_rewrite=true
    # is also honored in comfy_generate().
    skip_rewrite: bool = False


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def _client() -> ComfyClient:
    from src.settings import get_setting
    return ComfyClient(get_setting("comfy_base_url", DEFAULT_COMFY_BASE_URL))


def _require_user(request: Request) -> str:
    # DEFECT 10: delegate to the shared require_user() (src/auth_helpers.py)
    # instead of re-implementing a narrower subset of its cases. The old body
    # only handled AUTH_ENABLED=false; require_user() also covers the
    # unconfigured-first-run + loopback case and LOCALHOST_BYPASS=true +
    # loopback -- under LOCALHOST_BYPASS, /generate (which already went
    # through require_privilege -> require_user) used to succeed while every
    # OTHER route here (/stream, /status, /options, /workflows) 401'd through
    # this function instead, and this now also rejects an `ody_` bearer token
    # the same way require_user() does (this file has no scope-aware handling
    # for one).
    return require_user(request)


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


def _read_loras_registry() -> dict:
    """Raw data/studio/scripts/loras.json "loras" dict (key -> entry),
    UNFILTERED by "enabled" -- unlike _load_curated_loras() below (which only
    returns entries meant to populate the default LoRA-picker dropdown), a
    model preset's own loras[].key (models.json) may reference any
    registered LoRA regardless of that curation flag, so key resolution
    (_resolve_preset_lora_keys()) must see the full registry, not the
    curated subset. Both functions share this one file-read rather than
    each doing their own (plan/task: "reuse the existing resolution path --
    don't duplicate it")."""
    path = Path(DATA_DIR) / "studio" / "scripts" / "loras.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("comfy: could not read loras.json: %s", e)
        return {}
    return raw.get("loras") or {}


def _load_curated_loras() -> list[dict]:
    """Enabled entries from data/studio/scripts/loras.json (plan §6.3a)."""
    out = []
    for key, entry in _read_loras_registry().items():
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


def _resolve_preset_lora_keys(keyed_loras: Optional[list]) -> list[dict]:
    """Translate a model preset's OWN loras list (models.json's registry
    form, `[{"key": <loras.json key>, "weight": ...}, ...]` -- see that
    file's _comment) into the wire shape the rest of this module already
    understands (`[{"comfy_name": ..., "weight": ..., "kind": ...}, ...]`),
    by looking each "key" up in loras.json via `_read_loras_registry()`.

    An entry whose key isn't a real (or complete) loras.json entry is
    dropped -- logged, not raised -- rather than reaching build_image_graph()
    with an empty comfy_name, which raises ValueError deep inside the graph
    builder (src/comfy_graphs.py) instead of degrading; a hand-edited
    models.json typo should not 500 the request.

    This function ONLY does the key -> comfy_name translation. The actual
    live-availability check (does ComfyUI currently have this file?) is left
    to happen exactly where it already did for a directly wire-submitted
    loras list -- comfy_generate()'s existing DEFECT-1 resolve_lora_entries()
    block below runs on whatever this function returns, so there remains
    exactly one code path that ever rejects an unavailable LoRA.
    """
    registry = _read_loras_registry()
    out = []
    for item in keyed_loras or []:
        if not isinstance(item, dict):
            continue
        key = item.get("key")
        entry = registry.get(key) if key else None
        if not isinstance(entry, dict) or not entry.get("comfy_name"):
            logger.warning("comfy: model preset referenced unknown/incomplete loras.json key %r -- dropped", key)
            continue
        out.append({
            "comfy_name": entry["comfy_name"],
            "weight": item.get("weight", entry.get("default_weight", 1.0)),
            "kind": entry.get("kind", ""),
        })
    return out


def _load_model_presets() -> tuple[dict, list[dict]]:
    """Returns (registry, flattened_list):
      - registry: the raw load_model_registry() dict (`{"default": ...,
        "models": {key: {...}, ...}}`) -- used at generate time to look up
        ONE preset by key, "enabled" or not (see comfy_generate()).
      - flattened_list: registry["models"] turned into a list of dicts each
        carrying its own "key", ENABLED entries only -- mirrors how
        _load_curated_loras() flattens loras.json -- for /api/comfy/options'
        `models: [...]` response (the dropdown should not offer a disabled
        preset, same "enabled hides it from the default list without
        deleting the record" convention loras.json already documents).
    """
    registry = _cached_model_registry()  # DEFECT 15
    out = []
    for key, entry in (registry.get("models") or {}).items():
        if not isinstance(entry, dict) or not entry.get("enabled", True):
            continue
        out.append({**entry, "key": key})
    return registry, out


def _load_video_model_presets() -> tuple[dict, list[dict]]:
    """Video counterpart of _load_model_presets() -- same (registry,
    flattened_list) shape, reading data/studio/scripts/models.json's
    `video_models`/`default_video_model` keys instead of `models`/`default`
    (added alongside the existing image section; see that file's own
    `_comment_video_models` for the schema).

    load_model_registry() already returns the FULL parsed models.json dict
    (not just the "models" subtree) -- so a valid file's `video_models`/
    `default_video_model` keys are already present on `registry` with ZERO
    changes needed to that function. This only does its OWN flattening step
    (ENABLED entries only, each carrying its own "key"), mirroring
    _load_model_presets()'s exact contract for the image side.
    """
    registry = _cached_model_registry()  # DEFECT 15
    out = []
    for key, entry in (registry.get("video_models") or {}).items():
        if not isinstance(entry, dict) or not entry.get("enabled", True):
            continue
        out.append({**entry, "key": key})
    return registry, out


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


# DEFECT 15: same short-lived-cache pattern as _OBJECT_INFO_CACHE above, for
# data/studio/scripts/models.json. Before this, /api/comfy/options parsed it
# twice per request (_load_model_presets() + _load_video_model_presets(),
# called back to back) and comfy_generate() a third time independently --
# load_model_registry() itself does a synchronous disk read + json.loads
# with no caching of its own.
_MODEL_REGISTRY_CACHE: dict = {"ts": 0.0, "data": None}
_MODEL_REGISTRY_CACHE_TTL_SECONDS = 60.0


def _cached_model_registry(*, force: bool = False) -> dict:
    now = time.time()
    if not force and _MODEL_REGISTRY_CACHE["data"] is not None and (now - _MODEL_REGISTRY_CACHE["ts"]) < _MODEL_REGISTRY_CACHE_TTL_SECONDS:
        return _MODEL_REGISTRY_CACHE["data"]
    data = load_model_registry()
    _MODEL_REGISTRY_CACHE["ts"] = time.time()
    _MODEL_REGISTRY_CACHE["data"] = data
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


# DEFECT 2: _JOBS is otherwise never evicted -- every /generate call adds an
# entry and nothing ever removes it, so it grows without bound over the
# container's uptime. Pruned from _run_job's own `finally` (once per
# completed job, not on a timer) so growth is bounded by job throughput.
# _JOB_MAX_AGE_SECONDS mirrors static/js/genParams.js's own
# INFLIGHT_MAX_AGE_MS (30 min) -- comfortably longer than /stream's SSE loop
# or a /cancel call needs to still find a just-finished job.
_JOB_MAX_AGE_SECONDS = 30 * 60


def _prune_old_jobs(*, now: Optional[float] = None) -> None:
    """Evict terminal jobs from the in-memory _JOBS dict.

    Age is measured from `finished_at`, NOT `started_at`. That distinction is
    the whole bug this docstring exists to prevent recurring:

    _prune_old_jobs() runs in _run_job()'s `finally`, i.e. the instant a job
    reaches a terminal state. Measuring from `started_at` meant any render
    that took longer than _JOB_MAX_AGE_SECONDS was ALREADY "stale" the moment
    it succeeded, so it was deleted in the same breath as being marked done.
    GET /stream's 1 s poller then read _JOBS.get(job_id) -> None and emitted
    `error: Job not found` instead of `done`.

    Observed live 2026-08-07 on a MiniMax H3 video (renders run many minutes,
    easily past the 30 min TTL at 1344x768): the mp4 landed in the Gallery and
    both chat turns were written to the DB, but the UI showed no completion at
    all, because the SSE stream errored instead of delivering `done` -- so the
    frontend's _onJobDone() never ran and never painted the result bubble.

    Measuring from `finished_at` means the retention window is "30 minutes to
    collect your result", independent of how long the render itself took,
    which is what the TTL was always meant to express.
    """
    now = now if now is not None else time.time()
    stale = [
        jid for jid, j in _JOBS.items()
        if j.get("status") in ("done", "error", "cancelled")
        # Fall back to `now` (never `0`) when finished_at is somehow unset, so
        # a missing timestamp keeps the job rather than instantly dropping it.
        and (now - float(j.get("finished_at") or now)) > _JOB_MAX_AGE_SECONDS
    ]
    for jid in stale:
        _JOBS.pop(jid, None)


def _mark_finished(job: dict) -> None:
    """Stamp the terminal-state time exactly once, for _prune_old_jobs()'s
    retention window. Idempotent: a job that reaches `finally` after already
    being cancelled keeps its original finish time."""
    if not job.get("finished_at"):
        job["finished_at"] = time.time()


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
    # DEFECT 6: `nodes` is keyed by node_id; ComfyUI populates it in
    # execution order, but the FIRST key is often just the first node
    # ComfyUI loaded (e.g. UNETLoader), not the one actually running -- so
    # the label used to stick there for the whole run instead of advancing
    # to e.g. "KSampler". Prefer whichever node ComfyUI currently reports as
    # "running"; fall back to the LAST reported node (the most recently
    # started, per that same execution order) rather than the first, so the
    # label still advances even on an update shape this doesn't recognize.
    running_id = next((nid for nid, n in nodes.items() if n.get("state") == "running"), None)
    if running_id is None:
        node_ids = list(nodes.keys())
        running_id = node_ids[-1] if node_ids else None
    if running_id is not None:
        job["node"] = running_id


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
        # DEFECT 17: wrap in contextlib.aclosing so an early `return` from
        # inside the loop (the execution_error/execution_interrupted branch
        # below) still deterministically closes the underlying async
        # generator -- and with it, its open websocket -- instead of leaving
        # both open until the garbage collector eventually finalizes them.
        async with contextlib.aclosing(client.run_and_stream(graph, client_id)) as stream:
            async for event in stream:
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
        # Stamp the finish time BEFORE anything that could prune, so this job's
        # own 30 min retention window starts here rather than at started_at
        # (see _prune_old_jobs' docstring -- a long render used to be pruned
        # the instant it succeeded, killing its own `done` SSE event).
        _mark_finished(job)
        job["_ready"].set()
        # DEFECT 16: plan section 9 risk 1's stated VRAM mitigation ("POST
        # :8188/free after each Comfy job") was never actually wired up
        # anywhere -- best-effort, never allowed to mask the job's real
        # outcome above.
        try:
            await client.free()
        except Exception as e:
            logger.warning("comfy job %s: POST /free failed (non-fatal): %s", job_id, e)
        _write_terminal_turn(job)
        _prune_old_jobs()  # DEFECT 2


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
# Chat-history persistence -- Image/Video tabs previously wrote a GalleryImage
# row (above) but NEVER a ChatMessage, so a session used only from these tabs
# had message_count == 0 forever: SessionManager.load_sessions() only loads
# sessions with message_count > 0 into RAM at boot (core/session_manager.py),
# so the session vanished from GET /sessions (the sidebar) after any restart
# -- exactly the reported bug ("chats that use video or images only don't
# seem to be saved... they just disappear"). Fixed by writing a real turn
# through the SAME session_manager.add_message() path Chat/Agent mode use
# (routes/chat_routes.py), which also handles message_count/last_message_at/
# updated_at (TimestampMixin's onupdate) for free -- see core/session_manager.
# py's _persist_message().
# ---------------------------------------------------------------------------

def _write_user_turn(
    session_id: Optional[str], kind: str, workflow: Optional[str],
    prompt: str, model_key: Optional[str],
) -> None:
    """Persist the Image/Video-tab request as the USER half of a chat turn.

    Called from POST /generate itself (comfy_generate(), before the
    background job/render even starts) so the session's message_count goes
    from 0 to 1 immediately -- "the turn appears immediately, not only after
    a slow render" (task requirement), and the session survives a restart
    even if the render is still in flight or later fails/gets cancelled.

    `prompt` is the ORIGINAL, pre-style-trigger text the user typed (the
    caller passes job["user_prompt"], captured from body.params.prompt before
    apply_style_trigger() mutates the working `params` dict) -- so the visible
    user bubble shows their own words, not "toei90s style, smoon, ...".

    Never raises: a chat-history write failure must not break generation.
    """
    # DEFECT 5: _valid_session_id() hits the DB -- it must be INSIDE the try
    # too, or a DB failure here propagates straight out of comfy_generate()
    # (this runs before the background job task is even created), i.e. a 500
    # plus an orphaned _JOBS entry, instead of the "never raises" contract
    # this function documents.
    try:
        sid = _valid_session_id(session_id)
        if not sid:
            return
        from core.models import ChatMessage, get_session_manager_instance
        session_manager = get_session_manager_instance()
        if session_manager is None:
            return
        sess = session_manager.get_session(sid)

        kind_label = "Video" if kind == "video" else "Image"
        note_bits = [f"{kind_label} tab"]
        if model_key:
            note_bits.append(f"model: {model_key}")
        elif workflow and workflow != "Custom":
            note_bits.append(f"workflow: {workflow}")
        note = " · ".join(note_bits)
        text = (prompt or "").strip()
        content = f"{text}\n\n_(via {note})_" if text else f"_(via {note})_"

        sess.add_message(ChatMessage("user", content, metadata={"source": "comfy", "kind": kind}))
    except Exception:
        logger.exception("comfy: failed to persist user turn for session %s", session_id)


def _write_terminal_turn(job: dict) -> None:
    """Persist a finished/failed/cancelled Comfy job as the ASSISTANT half of
    the turn _write_user_turn() opened.

    Reuses the EXACT SAME rendering convention Chat mode's agent image-gen
    tool call already uses (routes/chat_routes.py's do_generate_image()
    branch + static/js/chatRenderer.js's addMessage()):
    metadata.tool_events = [{round, tool, command, output, exit_code,
    image_url, image_id, image_prompt, image_model, image_size}, ...] -- one
    event per landed image (or one video). image_url doubles as the video URL;
    the frontend tells them apart by extension (chatRenderer.buildVideoBubble
    vs buildImageBubble), same as genParams.js's own live-rendering path
    already does for the transient bubble.

    Called from _run_job()'s `finally` (covers done/error/cancelled-via-WS)
    AND from POST /cancel (covers a user-initiated cancel, which may resolve
    before _run_job's WS stream ever sees `execution_interrupted` -- e.g. a
    still-queued job). Idempotent via job["_chat_written"] so a job reachable
    from both never double-writes the turn. Never raises.
    """
    if job.get("_chat_written"):
        return
    job["_chat_written"] = True
    # DEFECT 5: same fix as _write_user_turn() above -- _valid_session_id()
    # must be inside the try. This runs from _run_job()'s `finally`, so a DB
    # failure here previously escaped as an unretrieved exception on the
    # background task instead of being swallowed per this function's own
    # "Never raises" contract.
    try:
        sid = _valid_session_id(job.get("session_id"))
        if not sid:
            return
        from core.models import ChatMessage, get_session_manager_instance
        from routes.chat_helpers import needs_auto_name, auto_name_session, _spawn_bg
        session_manager = get_session_manager_instance()
        if session_manager is None:
            return
        sess = session_manager.get_session(sid)

        kind = job.get("kind") or "image"
        kind_label = "Video" if kind == "video" else "Image"
        tool_name = "generate_video" if kind == "video" else "generate_image"
        params = job.get("params") or {}
        prompt = (job.get("user_prompt") or params.get("prompt") or "").strip()[:100] or "(no prompt)"
        model_label = params.get("model") or "ComfyUI"
        width, height = params.get("width"), params.get("height")
        size = f"{width}x{height}" if width and height else None
        images = job.get("images") or []
        status = job.get("status")

        tool_events: list[dict] = []
        if status == "done" and images:
            for img in images:
                tool_events.append({
                    "round": 1,
                    "tool": tool_name,
                    "command": prompt,
                    "output": "",
                    "exit_code": 0,
                    "image_url": img.get("url"),
                    "image_id": img.get("gallery_id"),
                    "image_prompt": prompt,
                    "image_model": model_label,
                    "image_size": size,
                })
            if kind == "video":
                content = f"Generated video for: {prompt}"
            else:
                n = len(images)
                content = f"Generated {n} image{'s' if n != 1 else ''} for: {prompt}"
        elif status == "done":
            content = f"{kind_label} generation finished with no output."
            tool_events.append({
                "round": 1, "tool": tool_name, "command": prompt,
                "output": "ComfyUI reported success but produced no images.", "exit_code": 1,
            })
        elif status == "cancelled":
            content = f"{kind_label} generation cancelled."
            tool_events.append({
                "round": 1, "tool": tool_name, "command": prompt,
                "output": job.get("error") or "Cancelled by user.", "exit_code": 1,
            })
        else:  # "error", or any other status reached defensively
            content = f"{kind_label} generation failed: {job.get('error') or 'unknown error'}"
            tool_events.append({
                "round": 1, "tool": tool_name, "command": prompt,
                "output": job.get("error") or "", "exit_code": 1,
            })

        # DEFECT 4: static/js/chatRenderer.js's addMessage() reads
        # metadata.round_texts (falling back to []) to get the actual text
        # to render per round -- tool_events alone renders no bubble text at
        # all on reload (only a collapsed <details> with the reason), per
        # src/agent_loop.py's own convention of always writing round_texts
        # alongside tool_events.
        sess.add_message(ChatMessage("assistant", content, metadata={"tool_events": tool_events, "round_texts": [content], "model": model_label}))

        if needs_auto_name(sess.name):
            _spawn_bg(auto_name_session(session_manager, sess))
    except Exception:
        logger.exception("comfy: failed to persist terminal turn for session %s", job.get("session_id"))


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
        registry, model_list = _load_model_presets()
        video_registry, video_model_list = _load_video_model_presets()
        result = {
            "loras": _load_curated_loras(),
            "loras_all": [],
            "unets": [],
            "vaes": [],
            "clips": [],
            "samplers": [],
            "schedulers": [],
            "models": model_list,
            "default_model": registry.get("default"),
            # Video counterpart (task item 3) -- data/studio/scripts/
            # models.json's new `video_models` section, resolved for
            # availability the SAME way as `models` below (reusing
            # resolve_model_entries() against the same live unets/clips/vaes
            # -- H3 uses UNETLoader/CLIPLoader/VAELoader too, just different
            # filenames/type values, so no separate combo lookup is needed).
            "video_models": video_model_list,
            "default_video_model": video_registry.get("default_video_model"),
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
            # Same rationale for model presets -- e.g. the Qwen-Image preset,
            # whose ~30GB download is in progress per the task this shipped
            # under, must not be greyed out just because ComfyUI happens to
            # be unreachable THIS request; that's orthogonal to whether its
            # files exist. Same for video_models -- the H3 download is its
            # own multi-file, ~42.5GB in-progress case.
            result["models"] = [
                {**entry, "available": True, "missing": []}
                for entry in result["models"]
            ]
            result["video_models"] = [
                {**entry, "available": True, "missing": []}
                for entry in result["video_models"]
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
        # Model presets: same exact -> basename -> unavailable resolution as
        # the LoRA registry, applied to each preset's unet/clip/vae (plan
        # §14.6 / src/comfy_graphs.py's resolve_model_entries() docstring).
        # This is what marks the Qwen preset unavailable today (its files
        # are mid-download -- task constraint) and flips it to available
        # later with zero code change once they land and ComfyUI restarts.
        live_for_models = {"unets": result["unets"], "clips": result["clips"], "vaes": result["vaes"]}
        result["models"] = resolve_model_entries(result["models"], live_for_models)
        # Same function, same live lists -- resolve_model_entries() also
        # checks unet_low/audio_vae when a video preset declares them (Wan's
        # low-noise unet, H3's audio vae). This is what marks the MiniMax H3
        # preset unavailable while its ~42.5GB download is in progress.
        result["video_models"] = resolve_model_entries(result["video_models"], live_for_models)
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
                # DEFECT 15: use the shared ~60s cache (mirrors
                # /api/comfy/options and comfy_generate()'s own use of it)
                # instead of pulling ComfyUI's ENTIRE node registry over the
                # network on every single workflow-params request.
                object_info = await _cached_object_info()
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
        try:
            from routes.gpu_routes import _load_helper
            await _load_helper().prepare_for(body.kind)
        except Exception:
            logger.warning("gpu prepare_for(%s) skipped", body.kind, exc_info=True)

        if body.kind not in ("image", "video"):
            raise HTTPException(400, f"Unknown kind: {body.kind!r}")

        params = body.params.model_dump()
        rewrite_info = {
            "original": (body.params.prompt or "").strip(),
            "rewritten": (body.params.prompt or "").strip(),
            "model": None,
            "used_rewrite": False,
        }
        skip_rewrite = bool(body.skip_rewrite)
        q_skip = str(request.query_params.get("skip_rewrite") or "").strip().lower()
        if q_skip in {"1", "true", "yes", "on"}:
            skip_rewrite = True
        if body.kind == "image" and not skip_rewrite:
            try:
                from src.image_prompt_rewrite import rewrite_image_prompt
                rewrite_info = await rewrite_image_prompt(
                    body.params.prompt or "",
                    img2img=bool(params.get("input_image")),
                )
                if rewrite_info.get("used_rewrite") and rewrite_info.get("rewritten"):
                    params["prompt"] = rewrite_info["rewritten"]
            except Exception:
                logger.exception("comfy: image prompt rewrite failed; using original")
        # Which video graph builder to use -- resolved inside the
        # `elif body.kind == "video"` block below when a `model` key is
        # given; stays at this default (Wan, the pre-H3 behaviour) otherwise,
        # which is exactly the "keep the existing Wan behaviour working
        # unchanged when no model is given" contract the task requires. Never
        # read at all for kind == "image".
        video_engine = "wan22_i2v"

        if body.kind == "image":
            # Model preset (data/studio/scripts/models.json) resolution --
            # MUST run before the style-trigger decision below (a resolved
            # preset's style_trigger becomes the authority) and before
            # build_image_graph() (apply_model_preset() can fill unet/clip/
            # clip_type/vae/loras). Image kind only: models.json's `models`
            # section is scoped to the Image tab -- a video request's `model`
            # field is resolved against the SEPARATE `video_models` section
            # in the elif branch below, never this one.
            preset: Optional[dict] = None
            model_key = params.get("model")
            if model_key:
                model_registry = _cached_model_registry()  # DEFECT 15
                preset = (model_registry.get("models") or {}).get(model_key)
                if preset is None:
                    raise HTTPException(
                        400,
                        f"Unknown model {model_key!r}. Known models: "
                        + ", ".join(sorted((model_registry.get("models") or {}).keys())),
                    )
                try:
                    info = await _cached_object_info()
                except Exception as e:
                    # ComfyUI unreachable right now -- cannot verify, so let
                    # it through unresolved (same graceful-degrade rationale
                    # as the LoRA availability block below, and
                    # /api/comfy/options): the actual failure will surface
                    # naturally when _run_job() tries to queue the prompt.
                    logger.warning(
                        "comfy_generate: could not verify model %r availability against live ComfyUI: %s",
                        model_key, e,
                    )
                else:
                    live_for_model = {
                        "unets": _combo_options(info, "UNETLoader", "unet_name"),
                        "clips": _combo_options(info, "CLIPLoader", "clip_name"),
                        "vaes": _combo_options(info, "VAELoader", "vae_name"),
                    }
                    checked = resolve_model_entries([preset], live_for_model)[0]
                    if not checked["available"]:
                        raise HTTPException(
                            400,
                            f"Model {model_key!r} is not available on the ComfyUI server yet "
                            f"(missing: {', '.join(checked['missing'])}). Its files may still be "
                            f"downloading, or ComfyUI may need a restart to see them.",
                        )

                # A caller-supplied loras list is always respected as-is
                # (including a deliberate empty one -- see
                # apply_model_preset()'s docstring for why "absent" and
                # "deliberately empty" can't be told apart at this layer);
                # only fill from the preset's OWN loras when none was given.
                # DEFECT 7: only a truly ABSENT loras field (None) means
                # "use the preset's own LoRAs" -- an explicit [] is now a
                # deliberate "the user unchecked every row" (GenerateParams.
                # loras is Optional[...] = None, so the two are no longer
                # wire-identical; see that field's own comment).
                used_preset_loras = bool(params.get("loras") is None and preset.get("loras"))
                params = apply_model_preset(params, preset)
                if used_preset_loras:
                    # The preset's loras are KEYED (models.json's
                    # {"key": <loras.json key>, "weight": ...} form -- not
                    # yet a real comfy_name). Translate through loras.json
                    # BEFORE the existing live-availability block below (which
                    # only understands the comfy_name wire shape) sees them.
                    params["loras"] = _resolve_preset_lora_keys(params["loras"])

            # Style trigger. NEW AUTHORITY: when a model preset was resolved
            # above, ITS style_trigger flag decides -- not which LoRAs
            # happen to be attached. This is what makes picking e.g.
            # "Qwen-Image" or "Z-Image (base)" reliably general-mode
            # (style_trigger: false in models.json) and picking "Studio --
            # Toei 90s" reliably triggered (true), regardless of what ends
            # up in params["loras"]. The trigger words ("toei90s style,
            # smoon") are the tokens the style LoRA was trained against --
            # with no style LoRA loaded they are not neutral, they are
            # noise: "smoon" means nothing to base Z-Image and "toei90s
            # style" drags it toward generic retro anime.
            #
            # A direct API caller that never sends `model` (predating this
            # feature, or any script that only knows the old contract) falls
            # back to the ORIGINAL any-style-LoRA-selected gate, unchanged --
            # so it keeps working exactly as before. An explicit style_hint
            # still forces the trigger on either way, and apply_style_trigger()
            # stays idempotent, so a saved workflow's hand-typed trigger is
            # never doubled up (plan §7).
            if preset is not None:
                trigger_on = bool(preset.get("style_trigger")) or bool(params.get("style_hint"))
            else:
                _style_loras = [
                    l for l in (params.get("loras") or [])
                    if (l.get("kind") or "").lower() == "style"
                ]
                trigger_on = bool(_style_loras) or bool(params.get("style_hint"))
            if trigger_on:
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

        elif body.kind == "video":
            # Video model preset (data/studio/scripts/models.json's
            # `video_models` section) -- same "pick a model, the recipe
            # follows" idea as the image block above, but deliberately
            # simpler: NO style trigger (Wan and H3 both have no knowledge of
            # the studio style/character LoRAs -- build_wan_i2v_graph()'s /
            # build_minimax_h3_graph()'s own docstrings) and no loras.json
            # key translation (Wan's LoRA FILENAMES are fixed constants; H3
            # has no LoRA slot at all). "Keep the existing Wan behaviour
            # working unchanged when no model is given" (task) -- an absent
            # `model` key skips this ENTIRE block, exactly like every request
            # before this feature existed (video_engine stays at its
            # "wan22_i2v" default set above, and build_wan_i2v_graph() sees
            # the SAME params dict it always has).
            video_model_key = params.get("model")
            if video_model_key:
                video_registry = _cached_model_registry()  # DEFECT 15
                video_preset = (video_registry.get("video_models") or {}).get(video_model_key)
                if video_preset is None:
                    raise HTTPException(
                        400,
                        f"Unknown video model {video_model_key!r}. Known video models: "
                        + ", ".join(sorted((video_registry.get("video_models") or {}).keys())),
                    )
                try:
                    info = await _cached_object_info()
                except Exception as e:
                    # Same graceful-degrade rationale as the image block
                    # above -- ComfyUI unreachable right now, cannot verify,
                    # so let it through unresolved.
                    logger.warning(
                        "comfy_generate: could not verify video model %r availability against live ComfyUI: %s",
                        video_model_key, e,
                    )
                else:
                    live_for_video_model = {
                        "unets": _combo_options(info, "UNETLoader", "unet_name"),
                        "clips": _combo_options(info, "CLIPLoader", "clip_name"),
                        "vaes": _combo_options(info, "VAELoader", "vae_name"),
                    }
                    checked = resolve_model_entries([video_preset], live_for_video_model)[0]
                    if not checked["available"]:
                        raise HTTPException(
                            400,
                            f"Video model {video_model_key!r} is not available on the ComfyUI server yet "
                            f"(missing: {', '.join(checked['missing'])}). Its files may still be "
                            f"downloading, or ComfyUI may need a restart to see them.",
                        )
                params = apply_model_preset(params, video_preset)
                # AUTHORITATIVE dispatch key for which builder to call below
                # -- NOT the model key itself (a registry key could in
                # principle be renamed without this file changing), and NOT
                # "arch" the way image presets use it (image's arch drives
                # DEFAULTS ONLY, one builder serves both -- but Wan and H3
                # are genuinely different graphs; see models.json's own
                # `_comment_video_models`). Falls back to the Wan default if
                # a hand-edited preset omits "engine" -- the lighter-weight,
                # no-huge-download path, same bias as _FALLBACK_MODEL_REGISTRY.
                video_engine = str(video_preset.get("engine") or "wan22_i2v")

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

        if params.get("last_frame"):
            # MiniMax H3's OPTIONAL end frame -- a second, independent
            # filename resolved through the EXACT SAME upload/gallery-
            # passthrough bridge as input_image above (task step 11: "reuse
            # them; do not duplicate"). _resolve_input_image() doesn't care
            # about the semantic role of the filename it's given.
            name, subfolder, ftype = await _resolve_input_image(params["last_frame"])
            params["last_frame"] = name
            params["last_frame_subfolder"] = subfolder
            params["last_frame_type"] = ftype

        if body.kind == "image":
            builder = build_image_graph
        elif video_engine == "minimax_h3":
            builder = build_minimax_h3_graph
        else:
            builder = build_wan_i2v_graph
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
        # Persist which model preset (if any) made this render, alongside
        # the resolved seed, so a gallery item's gen_params records "which
        # model" the same way it already records the seed that produced it.
        # params["model"] still holds the ORIGINAL requested key regardless
        # of whether a preset was applied above -- apply_model_preset()
        # never touches that key itself, only unet/clip/clip_type/vae/loras/
        # defaults.
        resolved_params["model"] = params.get("model")

        session_id = body.session_id
        job_id = new_job_id()
        client_id = new_client_id()
        job = _new_job(user, body.kind, body.workflow, session_id, resolved_params, graph)
        # Original, pre-style-trigger prompt text -- what the user actually
        # typed -- kept separately from resolved_params["prompt"] (which may
        # have "toei90s style, smoon" prepended by apply_style_trigger()
        # above) so the chat-history turn shows their own words, not the
        # internal trigger tokens. See _write_user_turn()/_write_terminal_turn().
        job["user_prompt"] = (body.params.prompt or "").strip()
        _JOBS[job_id] = job

        # Write the USER half of the chat turn now, before the (possibly
        # slow) render -- see _write_user_turn()'s docstring.
        _write_user_turn(session_id, body.kind, body.workflow, job["user_prompt"], resolved_params.get("model"))

        # DEFECT 3: asyncio only holds a WEAK reference to a task created via
        # a bare create_task() call -- nothing else here was keeping this one
        # alive, so the GC was free to collect it mid-run. _spawn_bg()
        # (routes/chat_helpers.py) holds a strong ref until the task
        # finishes (the same helper _write_terminal_turn() above already
        # uses for auto-naming); stashing the handle on the job record also
        # means it's available for real cancellation later, not just GC
        # safety.
        from routes.chat_helpers import _spawn_bg
        job["_task"] = _spawn_bg(_run_job(job_id, graph, client_id))

        try:
            await asyncio.wait_for(job["_ready"].wait(), timeout=20)
        except asyncio.TimeoutError:
            logger.warning("comfy_generate: timed out waiting for ComfyUI to accept job %s", job_id)

        if job.get("prompt_id") is None and job.get("error"):
            raise HTTPException(502, job["error"])

        return {
            "job_id": job_id,
            "prompt_id": job.get("prompt_id"),
            "original_prompt": rewrite_info.get("original") or (body.params.prompt or "").strip(),
            "rewritten_prompt": (rewrite_info.get("rewritten") or rewrite_info.get("original") or ""),
            "used_rewrite": bool(rewrite_info.get("used_rewrite")),
            "rewrite_model": rewrite_info.get("model"),
        }

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
        # DEFECT 14: "landing" (images are being fetched/written -- the
        # render itself already finished successfully server-side) and
        # "cancelled" (idempotency -- a second /cancel call, or a race with
        # _run_job already having set it) must not be clobbered back to
        # "cancelled"; only a still-queued/running job is actually
        # cancellable here.
        if job.get("status") not in ("done", "error", "landing", "cancelled"):
            job["status"] = "cancelled"
            job["error"] = job.get("error") or "Cancelled by user"
            # Same retention-window stamp as _run_job's finally. _run_job will
            # also reach its finally shortly and call _mark_finished(), which
            # is idempotent, so the earlier (cancel) time is the one kept.
            _mark_finished(job)
            _write_terminal_turn(job)
        return {"ok": True}

    return router
