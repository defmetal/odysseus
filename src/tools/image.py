"""Image-domain tool implementations.

Extracted from tool_implementations.py as part of slice 1 (#4082/#4071).
Holds the edit_image (gallery) tool.
``src.tool_implementations`` re-exports these for backward compatibility.
``_INTERNAL_BASE`` still lives in tool_implementations.py and is pulled back
function-locally here.
"""
import re
from typing import Dict, Optional

from src.tools._common import _parse_tool_args


async def do_edit_image(content: str, owner: Optional[str] = None) -> Dict:
    """Edit a gallery image (upscale, rembg, inpaint, harmonize)."""
    import httpx
    from src.tool_implementations import _INTERNAL_BASE  # shared constant, still lives in the facade
    try:
        args = _parse_tool_args(content)
    except ValueError:
        return {"error": "Invalid JSON arguments", "exit_code": 1}
    image_id = args.get("image_id", "")
    action = args.get("action", "")
    if not image_id or not action:
        return {"error": "image_id and action are required", "exit_code": 1}
    payload = {"image_id": image_id}
    if args.get("prompt"):
        payload["prompt"] = args["prompt"]
    if args.get("scale"):
        payload["scale"] = args["scale"]
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{_INTERNAL_BASE}/api/gallery/{action}", json=payload)
            data = resp.json()
        if data.get("success") or data.get("id"):
            return {"output": f"Image edited ({action}). New image ID: {data.get('id', '?')}", "exit_code": 0}
        return {"error": data.get("error", f"{action} failed"), "exit_code": 1}
    except Exception as e:
        return {"error": str(e), "exit_code": 1}


# --- Studio image-edit tools (restyle / inpaint / controlnet) --------------
# Re-homed here from tool_implementations.py during the 2026-07-09 upstream
# merge (upstream split tool_implementations.py into src/tools/, #4423).
# They shell out to the studio's data/studio/scripts/*.py via the diffusion
# server; self-contained (only typing + asyncio).
async def _run_studio_image_script(script: str, cli_args: list, owner: Optional[str]) -> Dict:
    """Run a studio image script (restyle.py/inpaint.py) deterministically and
    lift the resulting gallery URL out of its stdout. The script handles
    file-finding, the diffusion call, the database, and the owner — the model
    only supplies the region/prompt, so it can't improvise paths or DBs."""
    import asyncio as _aio
    cmd = ["python3", f"/app/data/studio/scripts/{script}", "--image", "latest"] + cli_args
    if owner:
        cmd += ["--owner", owner]
    try:
        proc = await _aio.create_subprocess_exec(
            *cmd, stdout=_aio.subprocess.PIPE, stderr=_aio.subprocess.STDOUT)
        out, _ = await proc.communicate()
        out = (out or b"").decode(errors="replace").strip()
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "error": f"could not run {script}: {e}", "exit_code": 1}
    if proc.returncode != 0:
        # Surface the script's own (clear) error; do NOT let the model improvise.
        msg = out.splitlines()[-1] if out else f"{script} failed"
        return {"stdout": "", "stderr": out, "error": msg, "exit_code": 1}
    # stdout carries the `/api/generated-image/<file>` line; _promote_image_fields lifts it.
    # stderr key is required — agent_loop reads result["stderr"] directly.
    return {"stdout": out, "stderr": "", "output": out, "exit_code": 0}


# Strength keywords -> img2img denoising strength (how far from the input). The
# studio's core use is converting an image TOWARD the style, so restyle.py's
# default already leans strong; these let the agent go fuller or gentler.
_RESTYLE_STRENGTH = {
    "light": 0.45, "subtle": 0.45, "slight": 0.45, "gentle": 0.45,
    "medium": 0.6, "moderate": 0.6,
    "full": 0.8, "strong": 0.8, "complete": 0.8, "heavy": 0.8, "max": 0.85,
}


async def do_restyle_image(content: str, owner: Optional[str] = None) -> Dict:
    """Restyle the user's most recent uploaded image into the trained style (img2img).
    Line 1 = prompt (start with the project's style trigger).
    Line 2 (optional) = strength: a keyword (light / medium / full) OR a 0.0-1.0
    number — higher = a fuller repaint into the style (e.g. photo -> anime)."""
    lines = [l.strip() for l in (content or "").strip().split("\n") if l.strip()]
    if not lines:
        return {"stdout": "", "stderr": "", "error": "A prompt is required (line 1).", "exit_code": 1}
    args = ["--prompt", lines[0]]
    if len(lines) > 1:
        s = lines[1].lower()
        strength = _RESTYLE_STRENGTH.get(s)
        if strength is None:
            try:
                v = float(s)
                strength = v if 0.0 < v <= 1.0 else None
            except ValueError:
                strength = None
        if strength is not None:
            args += ["--strength", str(strength)]
    return await _run_studio_image_script("restyle.py", args, owner)


async def do_inpaint_region(content: str, owner: Optional[str] = None) -> Dict:
    """Fix one region of the user's most recent uploaded image (inpaint).
    Line 1 = region in plain words; line 2 = prompt (start with the style trigger)."""
    lines = [l.strip() for l in (content or "").strip().split("\n") if l.strip()]
    if len(lines) < 2:
        return {"stdout": "", "stderr": "", "error": "Provide region on line 1 and prompt on line 2.", "exit_code": 1}
    region, prompt = lines[0], lines[1]
    return await _run_studio_image_script("inpaint.py", ["--region", region, "--prompt", prompt], owner)


async def do_controlnet(content: str, owner: Optional[str] = None) -> Dict:
    """Turn the user's most recent uploaded sketch/reference into an on-model
    frame whose COMPOSITION follows it (ControlNet). Line 1 = prompt (start with
    the project's style trigger); optional line 2 = 'canny' (default) or 'scribble'."""
    lines = [l.strip() for l in (content or "").strip().split("\n") if l.strip()]
    if not lines:
        return {"stdout": "", "stderr": "", "error": "A prompt is required (line 1).", "exit_code": 1}
    prompt = lines[0]
    control = lines[1].lower() if len(lines) > 1 and lines[1].lower() in ("canny", "scribble") else "canny"
    return await _run_studio_image_script("controlnet.py", ["--prompt", prompt, "--control", control], owner)


# --- fix_faces: full-body face-drift fix (close-up inpaint on the face) ---
# data/studio/TRAINING-GUIDE.md: "Full-body face drift (faces too tiny in wide
# shots for the LoRA to enforce): generate the body, then INPAINT the face
# with the character LoRA on the face region — a workflow fix you'll use
# regardless of LoRA quality." fix_faces is that workflow, exposed as a tool.
#
# Implementation choice: inpaint.py's EXISTING --region/--prompt interface is
# already generic enough to do this with ZERO changes to inpaint.py and NO new
# script file — "face" is just as valid a --region value as "her left hand",
# and inpaint.py's own apply_trigger() call already auto-prepends the style
# trigger, so fix_faces only needs to build the right PROMPT convention
# (character trigger + "close-up of the face" + an optional hint) and hand it
# straight to the existing _run_studio_image_script("inpaint.py", ...) helper
# do_inpaint_region already uses. This keeps the diff to one new do_* function
# instead of new inpaint.py flags or a fix_faces.py wrapper script.
_PLACEHOLDER_LINE_RE = re.compile(r"^<.*>$")


def _clean_tool_lines(content: str) -> list:
    """Split into stripped, non-blank lines; drop lines that are ONLY a
    literal <placeholder> token echoed from a fenced-block template (same
    weak-model-proofing precedent as board.py's _clean_lines /
    _parse_generate_image in tool_execution.py)."""
    lines = [ln.strip() for ln in (content or "").strip().split("\n")]
    return [ln for ln in lines if ln and not _PLACEHOLDER_LINE_RE.fullmatch(ln)]


def _parse_fix_faces(content: str):
    """Parse fix_faces' two optional lines: a character trigger (line 1,
    e.g. 'tetsuya_oc') and an expression/detail hint (line 2). Both are
    optional — an entirely empty body is a valid call (generic close-up
    redraw of the face). Returns (trigger, hint), each '' when not given."""
    lines = _clean_tool_lines(content)
    trigger = lines[0] if len(lines) > 0 else ""
    hint = lines[1] if len(lines) > 1 else ""
    return trigger, hint


def _fix_faces_prompt(trigger: str, hint: str) -> str:
    """Build the inpaint prompt: '<trigger>, close-up of the face, <hint>',
    dropping either optional part when blank. Do NOT prepend a style trigger
    here — inpaint.py's own apply_trigger() call already does that."""
    parts = [p.strip() for p in (trigger, "close-up of the face", hint) if p and p.strip()]
    return ", ".join(parts)


async def do_fix_faces(content: str, owner: Optional[str] = None) -> Dict:
    """Fix face drift on the user's most recently uploaded image: locate the
    face (reusing inpaint.py's qwen3-vl region grounding, region="face") and
    inpaint it with a close-up prompt. Line 1 (optional) = a character
    trigger, e.g. 'tetsuya_oc' — include it so whatever variant is fused on
    the image server (style-only or style+character) gets a chance to lock
    identity. Line 2 (optional) = an expression/detail hint. Both lines are
    optional; an empty body is a valid call.

    v1 scope: inpaint.py's locate_region() returns a SINGLE bounding box, so
    this fixes the largest/most prominent face in the shot only.
    TODO(multi-face): iterating every face needs locate_region to return
    every detected box (today it returns one) and one inpaint pass per box —
    left for a later phase.
    """
    trigger, hint = _parse_fix_faces(content)
    prompt = _fix_faces_prompt(trigger, hint)
    return await _run_studio_image_script("inpaint.py", ["--region", "face", "--prompt", prompt], owner)


# --- reference_edit: one-off chat access to the QIE dataset-factory CLI ----
# Wraps data/studio/scripts/qie_edit.py (colorize/vary/turnaround), which is
# deliberately CLI-only per qie_edit_README.md: "The reference_edit agent tool
# that would expose this to chat is not wired yet — this is a host/docker-exec
# CLI tool only, on purpose, until it proves out." This is that tool.
#
# qie_edit.py itself is NOT modified. It has no --image latest / --owner /
# Gallery integration by design (a human curates its _candidates/ output by
# hand for the training pipeline — "Gallery doesn't see this — it's a plain
# file"), so this wrapper resolves the latest upload itself, invokes the CLI
# exactly as documented, and promotes the ONE freshly-generated candidate into
# the Gallery afterward — mirroring what inpaint.py/controlnet.py do inside
# the script, just done here instead since qie_edit.py intentionally doesn't.
# QIE (Qwen-Image-Edit-2511) is a wholly separate model from the Z-Image house
# style — no style trigger is ever applied to its prompts.
_REFERENCE_EDIT_MODES = ("colorize", "vary", "turnaround")
_REFERENCE_EDIT_MODE_ALIASES = {
    "colorize": "colorize", "colourize": "colorize", "colorise": "colorize",
    "color": "colorize", "colour": "colorize", "colorization": "colorize",
    "colourisation": "colorize",
    "vary": "vary", "variation": "vary", "variant": "vary", "pose": "vary",
    "turnaround": "turnaround", "turn around": "turnaround", "turn-around": "turnaround",
    "sheet": "turnaround", "character sheet": "turnaround", "character-sheet": "turnaround",
    "reference sheet": "turnaround",
}
_REF_LINE_RE = re.compile(r"^ref(?:erence)?\s*:\s*(.*)$", re.I)
_REFERENCE_EDIT_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
# Generous: qie_edit_README.md's own (estimated) budget is ~1-3 min gguf
# swap-in + ~30-90s/image at steps=40 — this comfortably exceeds that,
# following controlnet.py's "heavier swap-in mode" precedent.
_REFERENCE_EDIT_TIMEOUT = 600
_REFERENCE_EDIT_OUT_DIR = "/app/data/studio/dataset/_chat_reference_edits/_candidates"
_REFERENCE_EDIT_UPLOADS_INDEX = "/app/data/uploads/uploads.json"
_REFERENCE_EDIT_CHAR_DIR = "/app/data/studio/dataset/characters"


def _normalize_reference_edit_mode(raw: str) -> str:
    """Canonicalize the mode line to 'colorize'/'vary'/'turnaround', or ''
    when unrecognized (same alias-normalization spirit as board.py's
    _normalize_status, applied to reference_edit's own vocabulary)."""
    text = re.sub(r"\s+", " ", (raw or "").strip().lower()).strip(" .!\"'")
    return _REFERENCE_EDIT_MODE_ALIASES.get(text, "")


def _parse_reference_edit(content: str):
    """Parse reference_edit's body: line 1 = mode (raw, not yet validated —
    the caller normalizes it); an optional `ref: <character>` line (also
    accepts `reference:`); every other non-blank, non-placeholder line is
    treated as (part of) the prompt, joined with spaces so a model that
    splits it across lines still yields one usable string. Returns
    (raw_mode, prompt, ref_name) — '' for whichever part is absent."""
    lines = _clean_tool_lines(content)
    if not lines:
        return "", "", ""
    raw_mode = lines[0]
    prompt_parts = []
    ref_name = ""
    for ln in lines[1:]:
        m = _REF_LINE_RE.match(ln)
        if m:
            val = m.group(1).strip()
            if val and not _PLACEHOLDER_LINE_RE.fullmatch(val):
                ref_name = val
            continue
        prompt_parts.append(ln)
    return raw_mode, " ".join(prompt_parts).strip(), ref_name


def _resolve_latest_upload_path(owner: Optional[str] = None,
                                 uploads_index: str = _REFERENCE_EDIT_UPLOADS_INDEX):
    """(path, owner) of the most recent image upload, optionally restricted
    to `owner`. Duplicates inpaint.py's/controlnet.py's own
    resolve_latest_upload() rather than importing it: those are standalone
    subprocess SCRIPTS executed inside the container, while this helper runs
    in-process in the tool dispatcher for a CLI (qie_edit.py) that, unlike its
    siblings, has no 'latest upload' concept of its own to delegate to."""
    import json
    from pathlib import Path
    idx = Path(uploads_index)
    if not idx.exists():
        return None, None
    try:
        entries = [
            e for e in json.loads(idx.read_text()).values()
            if (e.get("mime", "")).startswith("image/")
            and (not owner or e.get("owner") == owner)
        ]
        if not entries:
            return None, None
        newest = max(entries, key=lambda e: e.get("uploaded_at", ""))
        p = newest.get("path", "")
        if p and Path(p).exists():
            return p, (newest.get("owner") or "admin")
        return None, None
    except Exception:
        return None, None


def _resolve_color_ref(name: str, base_dir: str = _REFERENCE_EDIT_CHAR_DIR):
    """Resolve a `ref: <character>` line to a canonical color-reference image
    under dataset/characters/<name>/color/ (no characters.json registry
    exists yet to consult, so this is a filesystem heuristic — documented
    here since it's a judgment call):
      1. normalize `name` — lowercase, strip, drop a trailing '_oc' (the
         character-LoRA trigger suffix a user may type out of habit);
      2. look directly inside .../<name>/color/ (non-recursive, matching
         qie_edit.py's own directory-glob convention) for an image file
         (.png/.jpg/.jpeg/.webp) whose stem starts with 'canonical';
      3. else fall back to the alphabetically-first remaining image file.
    Files/dirs starting with '_' are excluded (color/'s existing _dups,
    _sheets, _tiles, _candidates convention). Returns a Path, or None if the
    character has no color/ directory or no usable image inside it."""
    from pathlib import Path
    key = (name or "").strip().lower()
    if key.endswith("_oc"):
        key = key[: -len("_oc")]
    if not key:
        return None
    color_dir = Path(base_dir) / key / "color"
    if not color_dir.is_dir():
        return None
    candidates = sorted(
        p for p in color_dir.iterdir()
        if p.is_file() and not p.name.startswith("_")
        and p.suffix.lower() in _REFERENCE_EDIT_IMAGE_EXTS
    )
    if not candidates:
        return None
    for p in candidates:
        if p.stem.lower().startswith("canonical"):
            return p
    return candidates[0]


async def do_reference_edit(content: str, owner: Optional[str] = None) -> Dict:
    """One-off chat edit via the Qwen-Image-Edit-2511 dataset-factory CLI
    (data/studio/scripts/qie_edit.py). Line 1 = mode (colorize/vary/
    turnaround); an optional `ref: <character>` line resolves to that
    character's canonical color reference (see _resolve_color_ref); every
    other line is the prompt. REQUIRED: `ref:` for colorize (its
    --color-ref flag is mandatory), a prompt for vary (its --prompt/
    --prompts-file is mandatory and this tool only ever passes --prompt —
    batching via --prompts-file is CLI-only, out of scope here).

    Always operates on the user's most recently uploaded image, exactly like
    its restyle_image/inpaint_region/controlnet siblings — `ref:` is an
    auxiliary SECOND reference (required for colorize, optional for vary,
    unused for turnaround since qie_edit.py's turnaround takes a single
    --ref), never a substitute for the upload. Handles ONE input image per
    call; qie_edit.py's own directory-batch input mode is CLI-only, not
    exposed here. Always passes --quant gguf (bitsandbytes/nf4 is not
    installed in this container; the gguf weights are already local).
    """
    raw_mode, prompt, ref_name = _parse_reference_edit(content)
    mode = _normalize_reference_edit_mode(raw_mode)
    if not mode:
        return {
            "stdout": "", "stderr": "",
            "error": f"reference_edit needs a mode on line 1: colorize, vary, or turnaround (got {raw_mode!r}).",
            "exit_code": 1,
        }

    upload_path, up_owner = _resolve_latest_upload_path(owner)
    if not upload_path:
        return {"stdout": "", "stderr": "",
                "error": "No uploaded image found. Attach an image and try again.", "exit_code": 1}
    effective_owner = owner or up_owner or "admin"

    ref_path = None
    if ref_name:
        ref_path = _resolve_color_ref(ref_name)
        if ref_path is None:
            return {
                "stdout": "", "stderr": "",
                "error": f"No color reference image found for '{ref_name}' "
                         f"under dataset/characters/{ref_name.strip().lower()}/color/.",
                "exit_code": 1,
            }

    if mode == "colorize" and ref_path is None:
        return {"stdout": "", "stderr": "",
                "error": "colorize needs a `ref: <character>` line naming a color reference.", "exit_code": 1}
    if mode == "vary" and not prompt:
        return {"stdout": "", "stderr": "",
                "error": "vary needs a prompt line describing the change.", "exit_code": 1}

    import asyncio as _aio
    import json as _json
    import shutil as _shutil
    import uuid as _uuid
    from pathlib import Path

    out_dir = Path(_REFERENCE_EDIT_OUT_DIR)
    common_args = ["--quant", "gguf", "--num", "1", "--out-dir", str(out_dir)]
    if mode == "colorize":
        cli = ["colorize", "--input", upload_path, "--color-ref", str(ref_path), *common_args]
        if prompt:
            cli += ["--prompt", prompt]
    elif mode == "vary":
        refs = [upload_path] + ([str(ref_path)] if ref_path else [])
        cli = ["vary", "--ref", *refs, "--prompt", prompt, "--tag", "reference_edit", *common_args]
    else:  # turnaround
        cli = ["turnaround", "--ref", upload_path, "--tag", "reference_edit", *common_args]
        if prompt:
            cli += ["--prompt", prompt]

    manifest_path = out_dir / "_manifest.jsonl"
    before_count = 0
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            before_count = sum(1 for _ in f)

    cmd = ["python3", "/app/data/studio/scripts/qie_edit.py", *cli]
    try:
        proc = await _aio.create_subprocess_exec(
            *cmd, stdout=_aio.subprocess.PIPE, stderr=_aio.subprocess.STDOUT)
        try:
            out_b, _ = await _aio.wait_for(proc.communicate(), timeout=_REFERENCE_EDIT_TIMEOUT)
        except _aio.TimeoutError:
            proc.kill()
            await proc.wait()
            return {
                "stdout": "", "stderr": "",
                "error": f"reference_edit timed out after {_REFERENCE_EDIT_TIMEOUT}s "
                         f"(model swap-in + generation can take minutes) — try again.",
                "exit_code": 1,
            }
        out = (out_b or b"").decode(errors="replace").strip()
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "error": f"could not run qie_edit.py: {e}", "exit_code": 1}

    if proc.returncode != 0:
        msg = out.splitlines()[-1] if out else "qie_edit.py failed"
        return {"stdout": "", "stderr": out, "error": msg, "exit_code": 1}

    # qie_edit.py never touches the Gallery/DB itself (see the module-level
    # comment above); lift its freshly-appended manifest record into the
    # Gallery here so the chat can render it. Only the lines appended DURING
    # this call are considered (line-count diff), so a shared manifest file
    # across calls can't pick up a stale/foreign record.
    try:
        new_lines = []
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as f:
                new_lines = f.readlines()[before_count:]
        if not new_lines:
            return {"stdout": out, "stderr": "",
                    "error": "qie_edit.py finished but wrote no new manifest record.", "exit_code": 1}
        record = _json.loads(new_lines[-1])
        produced = Path(record["output"])
        if not produced.exists():
            return {"stdout": out, "stderr": "",
                    "error": f"qie_edit.py reported {produced} but the file is missing.", "exit_code": 1}

        from src.constants import GENERATED_IMAGES_DIR
        from src.database import SessionLocal, GalleryImage
        gdir = Path(GENERATED_IMAGES_DIR)
        gdir.mkdir(parents=True, exist_ok=True)
        filename = f"{_uuid.uuid4().hex[:12]}.png"
        _shutil.copyfile(produced, gdir / filename)

        tags = f"reference_edit,{mode}" + (f",{ref_name.strip().lower()}" if ref_name else "")
        size = f"{record.get('width', '')}x{record.get('height', '')}".strip("x")
        db = SessionLocal()
        try:
            db.add(GalleryImage(
                id=str(_uuid.uuid4()), filename=filename,
                prompt=record.get("prompt") or prompt, model=f"Qwen-Image-Edit-2511 ({mode})",
                size=size, quality="high", tags=tags, owner=effective_owner, is_active=True,
            ))
            db.commit()
        finally:
            db.close()
    except Exception as e:
        return {"stdout": out, "stderr": "",
                "error": f"qie_edit.py succeeded but saving to the Gallery failed: {e}", "exit_code": 1}

    result_out = f"{out}\nOK: reference_edit ({mode}) -> gallery image {filename}\n    /api/generated-image/{filename}"
    return {"stdout": result_out, "stderr": "", "output": result_out, "exit_code": 0}
