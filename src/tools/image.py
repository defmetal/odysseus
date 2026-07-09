"""Image-domain tool implementations.

Extracted from tool_implementations.py as part of slice 1 (#4082/#4071).
Holds the edit_image (gallery) tool.
``src.tool_implementations`` re-exports these for backward compatibility.
``_INTERNAL_BASE`` still lives in tool_implementations.py and is pulled back
function-locally here.
"""
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


