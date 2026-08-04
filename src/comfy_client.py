# src/comfy_client.py
"""Async HTTP + WebSocket client for one ComfyUI server.

New, studio-only file -- see data/studio/PLAN-IMAGE-VIDEO-TABS.md §6.1. Every
endpoint/message shape below was verified directly against the vendored
source in data/studio/comfy/ComfyUI/ during this session (not just the plan's
summary of it):

    server.py:516      GET  /view
    server.py:397-467  POST /upload/image  -> {name, subfolder, type} (multipart --
                                                see upload_image()'s own docstring for the
                                                full field/response detail; verified this
                                                session directly, not just the plan)
    server.py:686-737  GET  /system_stats
    server.py:800-819  GET  /object_info[/{class}]
    server.py:1059-1070 GET /history/{prompt_id}, GET /queue
    server.py:1072-1144 POST /prompt        -> {prompt_id, number, node_errors}
                                                or {"error": {...}, "node_errors": {...}} (400)
    server.py:1146-1158 POST /queue {"delete": [prompt_id]} / {"clear": true}
    server.py:1160-1190 POST /interrupt {"prompt_id": ...}  (targeted) or {} (global)
    server.py:1192-1201 POST /free {"unload_models": bool, "free_memory": bool}
    server.py:269-327  WS   /ws?clientId=<id>

WS message wire shape is `{"type": <event>, "data": {...}}` for every event
(confirmed both by server.py's websocket_handler and by ComfyUI's own
script_examples/websockets_api_example.py, which this client's queue+listen
ordering mirrors). Event payload shapes verified at:

    comfy_execution/progress.py:160-185  "progress_state" -> {prompt_id, nodes:{...}}
    execution.py:496                      "executing"      -> {node, display_node, prompt_id}
                                           (node is None => this prompt_id is fully done --
                                           the completion signal script_examples/
                                           websockets_api_example.py itself waits on)
    execution.py:436,578                  "executed"       -> {node, display_node, output, prompt_id}
    execution.py:536,712                  "execution_error" -> {prompt_id, node_id, node_type,
                                                                  exception_message, exception_type,
                                                                  traceback, ...}
    execution.py:699                      "execution_interrupted" -> {prompt_id, node_id, executed}

The `client_id` passed to queue_prompt() MUST be the same id used as the WS
`clientId` query param -- ComfyUI keys its per-connection message routing on
it (server.py's `self.server_instance.client_id`), so a mismatch means
silence, not an error. `run_and_stream()` below exists specifically to make
that pairing (and the WS-before-POST ordering that avoids missing the first
progress frames -- see its docstring) hard to get wrong.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import uuid
from typing import Any, AsyncIterator, Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_COMFY_BASE_URL = "http://host.docker.internal:8188"

# Verified reachable from inside the odysseus container on 2026-08-03,
# ComfyUI v0.29.0 (plan §2, Phase 0).


class ComfyError(Exception):
    """Raised for a non-recoverable ComfyUI API failure (HTTP or protocol)."""


def new_client_id() -> str:
    """A fresh id suitable for both the WS `clientId` query param and the
    POST /prompt `client_id` body field. Must be the SAME string for both."""
    return uuid.uuid4().hex


def new_job_id() -> str:
    return f"cj-{uuid.uuid4().hex[:12]}"


class ComfyClient:
    """Thin async client for one ComfyUI server. Stateless aside from
    `base_url` -- safe to construct fresh per request (see
    routes/comfy_routes.py's `_client()`), which also means a settings change
    to `comfy_base_url` takes effect on the next call with no restart."""

    def __init__(self, base_url: Optional[str] = None, *, timeout: float = 60.0):
        self.base_url = (base_url or DEFAULT_COMFY_BASE_URL).rstrip("/")
        self._timeout = timeout

    # -- internal helpers ----------------------------------------------------

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _ws_url(self, client_id: str) -> str:
        ws_base = self.base_url.replace("https://", "wss://").replace("http://", "ws://")
        return f"{ws_base}/ws?clientId={client_id}"

    # -- HTTP API --------------------------------------------------------------

    async def status(self) -> dict:
        """Condensed status for GET /api/comfy/status: version + queue depth
        (server.py's /system_stats + /queue)."""
        async with httpx.AsyncClient(timeout=self._timeout) as c:
            stats_r = await c.get(self._url("/system_stats"))
            stats_r.raise_for_status()
            stats = stats_r.json()
            queue_r = await c.get(self._url("/queue"))
            queue_r.raise_for_status()
            queue = queue_r.json()
        return {
            "version": ((stats.get("system") or {}).get("comfyui_version")) or "",
            "queue_pending": len(queue.get("queue_pending") or []),
            "queue_running": len(queue.get("queue_running") or []),
        }

    async def upload_image(
        self, file_bytes: bytes, filename: str, *,
        subfolder: str = "", folder_type: str = "input", overwrite: bool = False,
    ) -> dict:
        """POST /upload/image (multipart/form-data) -- pushes bytes into
        ComfyUI's OWN input/ (or output/temp) directory so a LoadImage node
        can resolve them by name. This is the fix for GAP 1 (upload bridge):
        params.input_image previously passed an Odysseus filename straight
        into LoadImage, which ComfyUI resolves against ITS OWN input/
        directory -- not Odysseus's data/generated_images/ -- so it always
        failed to find the file.

        Multipart fields and the JSON response were verified directly
        against data/studio/comfy/ComfyUI/server.py's `image_upload()`
        (:397-457, routed via `@routes.post("/upload/image")` at :464-467),
        not guessed:

          Request fields (read via aiohttp's `request.post()` server-side):
            "image"     -- the file itself; ComfyUI reads `image.filename`
                            (the multipart part's filename, not a separate
                            form field) as the base save name, and may
                            rename it on save (see "overwrite" below).
            "type"      -- "input" | "output" | "temp"; server.py's
                            `get_dir_by_type()` treats an absent/None value
                            as "input" (:370-372).
            "subfolder" -- default "" (:410).
            "overwrite" -- the STRING "true" or "1" to overwrite a same-name
                            file; anything else (including omitted) triggers
                            ComfyUI's own de-dupe: if a same-name file
                            already exists it content-hashes the two, reuses
                            the existing name as-is with no write if
                            identical (`image_is_duplicate`), else appends
                            " (1)", " (2)", ... to the stem until a free name
                            is found (:420-432) -- so calling this repeatedly
                            with the same (filename, bytes) is idempotent and
                            always resolves back to the same name.

          Response body (:441): `{"name": <str>, "subfolder": <str>, "type":
          <str>}` -- note the key is "name", NOT "filename"; server.py never
          calls it that.

        Returns the decoded response dict verbatim (name/subfolder/type,
        plus an "asset" key ComfyUI only adds when `--enable-assets` is on --
        unused by every caller in this codebase). Raises ComfyError if the
        body has no "name" (a malformed/unexpected response -- fail loudly
        rather than let a caller silently key off a missing field).
        """
        data = {"type": folder_type, "subfolder": subfolder}
        if overwrite:
            data["overwrite"] = "true"
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        files = {"image": (filename, file_bytes, content_type)}
        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.post(self._url("/upload/image"), data=data, files=files)
        r.raise_for_status()
        try:
            result = r.json()
        except Exception as e:
            raise ComfyError(f"POST /upload/image returned a non-JSON body: {e}")
        if not isinstance(result, dict) or "name" not in result:
            raise ComfyError(f"POST /upload/image returned no 'name': {json.dumps(result)[:500]}")
        return result

    async def queue_prompt(self, graph: dict, client_id: str) -> dict:
        """POST /prompt. Returns the decoded body on success:
        {"prompt_id": str, "number": float, "node_errors": {}}.

        Raises ComfyError on an HTTP error status OR a body-level `"error"`
        key (server.py returns 400 + `{"error": {...}, "node_errors": {...}}`
        for both "no prompt" and prompt-validation failures -- see
        server.py:1088-1144). A non-empty `node_errors` alongside a 200 (i.e.
        the prompt WAS queued) is passed through rather than raised -- the
        server's own accept/reject signal is the status code + presence of
        `error`, not node_errors' emptiness.
        """
        payload = {"prompt": graph, "client_id": client_id}
        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.post(self._url("/prompt"), json=payload)
        try:
            data = r.json()
        except Exception:
            data = {}
        if r.status_code >= 400 or "error" in data:
            err = data.get("error")
            detail = err.get("message") if isinstance(err, dict) else (str(err) if err else None)
            node_errors = data.get("node_errors") or {}
            if node_errors:
                detail = f"{detail or 'Invalid graph'}: {json.dumps(node_errors)[:1000]}"
            raise ComfyError(detail or f"POST /prompt failed with status {r.status_code}")
        if "prompt_id" not in data:
            raise ComfyError(f"POST /prompt returned no prompt_id: {json.dumps(data)[:500]}")
        return data

    async def history(self, prompt_id: str) -> dict:
        """GET /history/{prompt_id}. Returns {} if the prompt hasn't produced
        history yet (still queued/running) -- NOT an error."""
        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.get(self._url(f"/history/{prompt_id}"))
        if r.status_code == 404:
            return {}
        r.raise_for_status()
        data = r.json()
        return data.get(prompt_id, {}) if isinstance(data, dict) else {}

    async def queue_state(self) -> dict:
        """GET /queue -- {"queue_running": [...], "queue_pending": [...]},
        each entry [number, prompt_id, prompt, extra_data, outputs_to_execute].
        Used by the polling fallback (plan §5 point 6) when the WS drops."""
        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.get(self._url("/queue"))
        r.raise_for_status()
        return r.json()

    async def view_bytes(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        """GET /view?filename=&subfolder=&type= -- fetch rendered output bytes
        for landing into the gallery."""
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.get(self._url("/view"), params=params)
        r.raise_for_status()
        return r.content

    async def object_info(self, class_type: Optional[str] = None) -> dict:
        """GET /object_info (all classes) or /object_info/{class_type} (one).
        Always returns a dict keyed by class_type."""
        path = f"/object_info/{class_type}" if class_type else "/object_info"
        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.get(self._url(path))
        r.raise_for_status()
        return r.json()

    async def interrupt(self, prompt_id: Optional[str] = None) -> None:
        """POST /interrupt. With a prompt_id, ComfyUI only interrupts if that
        prompt_id is the one CURRENTLY EXECUTING (a no-op, not an error, if
        it isn't -- server.py:1160-1190); omit prompt_id for a global
        interrupt. Prefer `cancel()` below, which also removes a
        still-queued (not yet started) prompt -- `/interrupt` alone does
        nothing for a job that hasn't started running yet.
        """
        body = {"prompt_id": prompt_id} if prompt_id else {}
        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.post(self._url("/interrupt"), json=body)
        r.raise_for_status()

    async def dequeue(self, prompt_id: str) -> None:
        """POST /queue {"delete": [prompt_id]} -- removes a PENDING (not yet
        started) prompt from the queue. A no-op if it already started or
        already finished (server.py:1146-1158)."""
        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.post(self._url("/queue"), json={"delete": [prompt_id]})
        r.raise_for_status()

    async def cancel(self, prompt_id: Optional[str]) -> None:
        """Best-effort cancel covering both queue states a job can be in:
        still pending (dequeue) or currently executing (targeted interrupt).
        Safe to call even if the job already finished -- both calls are
        no-ops in that case. If prompt_id is falsy (job never got far enough
        to be queued), this is a no-op."""
        if not prompt_id:
            return
        try:
            await self.dequeue(prompt_id)
        except Exception as e:
            logger.warning("comfy cancel: dequeue(%s) failed: %s", prompt_id, e)
        await self.interrupt(prompt_id)

    async def free(self, *, unload_models: bool = True, free_memory: bool = True) -> None:
        """POST /free -- release VRAM (plan §9 risk 1 mitigation: call this
        after a Comfy job so it doesn't fight the :8100 FP8 server / Ollama
        for GPU memory)."""
        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.post(self._url("/free"), json={
                "unload_models": unload_models, "free_memory": free_memory,
            })
        r.raise_for_status()

    # -- WebSocket progress ----------------------------------------------------

    async def stream_progress(self, prompt_id: str, client_id: str) -> AsyncIterator[dict]:
        """Open WS /ws?clientId=<client_id> and yield decoded `{"type", "data"}`
        messages relevant to `prompt_id` (progress_state / executing / executed
        / execution_error / execution_interrupted) until the run finishes or
        errors, then return.

        This assumes the prompt is already queued (or about to be, by another
        concurrent caller) -- it does NOT queue anything itself. Use this to
        re-attach to an in-flight prompt_id (e.g. after a process restart).
        For a fresh job, prefer `run_and_stream()`, which avoids the race
        where the prompt starts executing before this WS is listening.

        Binary frames (latent preview images, `PREVIEW_IMAGE_WITH_METADATA` --
        comfy_execution/progress.py:206-228) are skipped; live-preview
        thumbnails are phase-2 per plan §5 point 5.
        """
        import websockets  # local import: comfy_graphs.py stays stdlib-only;
        # this is the one module that needs a real WS client, only when used.

        uri = self._ws_url(client_id)
        async with websockets.connect(uri, max_size=None) as ws:
            async for raw in ws:
                event = _decode_ws_message(raw)
                if event is None:
                    continue
                mtype = event.get("type")
                data = event.get("data") or {}
                if mtype not in ("progress_state", "executing", "executed", "execution_error", "execution_interrupted"):
                    continue  # e.g. "status", "feature_flags" -- not relevant here
                event_prompt_id = data.get("prompt_id")
                if event_prompt_id is not None and event_prompt_id != prompt_id:
                    continue
                yield event
                if mtype in ("execution_error", "execution_interrupted"):
                    return
                if mtype == "executing" and data.get("node") is None:
                    # ComfyUI's own completion signal for this prompt_id --
                    # mirrors script_examples/websockets_api_example.py.
                    return

    async def run_and_stream(self, graph: dict, client_id: str) -> AsyncIterator[dict]:
        """Connect the WS FIRST, THEN queue the prompt, THEN stream progress --
        in that order, on the same open connection -- so no early progress
        frame can be emitted before we're listening (a real race: ComfyUI
        starts executing, and therefore sending "executing"/"progress_state"
        events, essentially immediately after POST /prompt returns).

        The first item yielded is always `{"type": "queued", "data": <the
        raw POST /prompt response>}` so the caller can capture `prompt_id`
        before any per-step frames arrive. Raises ComfyError immediately
        (without yielding anything) if queueing itself fails.
        """
        import websockets

        uri = self._ws_url(client_id)
        async with websockets.connect(uri, max_size=None) as ws:
            queued = await self.queue_prompt(graph, client_id)
            prompt_id = queued["prompt_id"]
            yield {"type": "queued", "data": queued}

            async for raw in ws:
                event = _decode_ws_message(raw)
                if event is None:
                    continue
                mtype = event.get("type")
                data = event.get("data") or {}
                if mtype not in ("progress_state", "executing", "executed", "execution_error", "execution_interrupted"):
                    continue
                event_prompt_id = data.get("prompt_id")
                if event_prompt_id is not None and event_prompt_id != prompt_id:
                    continue
                yield event
                if mtype in ("execution_error", "execution_interrupted"):
                    return
                if mtype == "executing" and data.get("node") is None:
                    return


def _decode_ws_message(raw: Any) -> Optional[dict]:
    """Decode one WS frame to `{"type", "data"}`, or None for a binary frame
    (skipped -- see stream_progress's docstring) or malformed JSON."""
    if isinstance(raw, (bytes, bytearray)):
        return None
    try:
        message = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return message if isinstance(message, dict) else None
