# src/comfy_graphs.py
"""Pure ComfyUI API-format graph builders + introspection helpers.

No I/O, no FastAPI/httpx/torch/DB imports, no dependency on the rest of the
Odysseus package -- stdlib only (json/pathlib for the one function that reads
the style-trigger registry; everything else is plain dict/list manipulation).
This is deliberate: it makes the module trivially unit-testable without a
running ComfyUI server, a database, or even fastapi installed (see
tests/test_comfy_graphs.py, which runs under plain `python3`).

See data/studio/PLAN-IMAGE-VIDEO-TABS.md for the full design. Section
references below (e.g. "plan §3a") point at that document.

-- Canonical image "params" dict --------------------------------------------
Every function in this module that talks about "params" for an IMAGE graph
uses this shape (a subset matches POST /api/comfy/generate's request body --
see routes/comfy_routes.py):

    {
        "prompt": str,
        "negative_prompt": str,
        "loras": [{"comfy_name": str, "weight": float}, ...],
        "width": int, "height": int, "batch": int,
        "input_image": str | None,       # LoadImage filename -> img2img when set
        "input_image_subfolder": str,    # optional; ComfyUI /upload/image "subfolder"
                                          # (GAP 1 -- see comfy_image_ref()); default ""
        "input_image_type": str,         # optional; "input"|"output"|"temp"; default "input"
        "denoise": float,                # ignored (forced 1.0) for txt2img
        "seed": int | None,
        "randomize_seed": bool,
        "steps": int, "cfg": float,
        "sampler": str, "scheduler": str,
        "shift": float,
        "unet": str, "clip": str, "vae": str,
        "filename_prefix": str,
    }

`build_image_graph(params)` consumes this shape (every key optional, sane
defaults applied). `introspect_graph(build_image_graph(params))["params"]`
returns this same shape back out (mapped from the built graph's live values,
including a resolved concrete seed) -- see tests/test_comfy_graphs.py's
round-trip test. This round-trip is also how routes/comfy_routes.py recovers
the FINAL concrete params (esp. the resolved seed) for the gen_params column
after a graph is built, without duplicating the resolution logic.
------------------------------------------------------------------------------

-- Canonical video "params" dict (build_wan_i2v_graph) ----------------------
Deliberately reuses as many keys/shapes from the image params dict above as
apply (plan §7 / task step 11: "reuse, don't invent") -- `prompt`,
`negative_prompt`, `loras`, `width`, `height`, `batch`, `input_image` (+
`_subfolder`/`_type`), `seed`, `randomize_seed`, `cfg`, `sampler`,
`scheduler`, `shift`, `clip`, `vae`, `filename_prefix` all mean exactly what
they mean for the image graph. What differs:

    {
        ... (all the shared keys above, plus:) ...
        "loras": [{"comfy_name": str, "weight": float, "role": str}, ...],
                                          # exactly 2 conceptual entries: the
                                          # HIGH-noise and LOW-noise Wan
                                          # lightning LoRAs. Only WEIGHT is
                                          # ever caller-controlled -- the
                                          # FILENAMES are always
                                          # DEFAULT_WAN_LORA_HIGH/LOW (plan
                                          # §3b: "Wan does NOT know the
                                          # studio style/character LoRAs...
                                          # Only the two Wan lightning LoRAs
                                          # are relevant"). Entries are
                                          # matched to high/low by "role"
                                          # first (what static/js/genParams.js
                                          # already sends), else a
                                          # "high"/"low" substring match on
                                          # comfy_name, else position
                                          # [0]=high/[1]=low -- see
                                          # _extract_wan_lora_weights().
        "frames": int,                   # WanImageToVideo "length"
        "fps": int | float,              # CreateVideo "fps"
        "seed": int | None,              # STAGE 1 (high-noise) noise_seed
                                          # ONLY -- stage 2 has
                                          # add_noise="disable", so its own
                                          # noise_seed is inert and is
                                          # deliberately never derived from
                                          # this (task step 11).
        "steps": int,                    # TOTAL steps, auto-split across
                                          # both KSamplerAdvanced stages --
                                          # see _wan_step_split().
        "unet": str,                     # HIGH-noise UNETLoader override
                                          # (same key name as the image
                                          # dict's single "unet" -- video's
                                          # high-noise chain is the
                                          # "primary" one).
        "unet_low": str,                 # LOW-noise UNETLoader override.
    }

No "denoise" (KSamplerAdvanced has no such input -- verified against
data/studio/comfy/ComfyUI/nodes.py; forcing denoise is meaningless for a
two-stage advanced sampler) and no "batch"-affecting size preset beyond
`width`/`height`/`frames` (WanImageToVideo's own "batch_size" widget, default
1). `build_wan_i2v_graph(params)` / `introspect_graph(...)["params"]`
round-trip this shape the same way the image pair does -- see
tests/test_comfy_graphs.py's video round-trip test.
------------------------------------------------------------------------------
"""

from __future__ import annotations

import itertools
import json
import os
import random
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Shared node-schema constants (verified against
# data/studio/comfy/ComfyUI/user/default/workflows/Studio - Tetsuya image
# (wide).json and "Studio - Restyle as Tetsuya (img2img).json" -- read in
# full; see plan §3a).
# ---------------------------------------------------------------------------

DEFAULT_UNET = "z_image_bf16.safetensors"
DEFAULT_CLIP = "qwen_3_4b.safetensors"
DEFAULT_CLIP_TYPE = "lumina2"
DEFAULT_CLIP_DEVICE = "default"
DEFAULT_VAE = "ae.safetensors"
DEFAULT_UNET_WEIGHT_DTYPE = "default"
DEFAULT_SHIFT = 3
DEFAULT_STEPS = 30
DEFAULT_CFG = 4.5
DEFAULT_SAMPLER = "res_multistep"
DEFAULT_SCHEDULER = "simple"
DEFAULT_IMG2IMG_DENOISE = 0.6
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_BATCH = 1
DEFAULT_NEGATIVE_PROMPT = "text, letters, lettering, logo, watermark, signature"
DEFAULT_FILENAME_PREFIX = "odysseus"

_SEED_MAX = 2**32 - 1


class GraphConversionError(Exception):
    """Raised by ui_to_api() when a UI-format node can't be confidently
    converted to API format (unknown class_type in /object_info, or a
    widgets_values count that doesn't line up with the declared inputs).

    Callers should catch this and fall back to the tier-3 escape hatch:
    "open this workflow in ComfyUI once and File -> Export (API)" (plan §4c).
    """


def _new_id_gen():
    """Fresh per-call node-id generator. Links use `["<id_str>", <out_idx>]`
    (plan §3a) -- IDs are plain incrementing strings; ComfyUI executes by
    dependency graph, not by id order, so there is no ordering requirement
    beyond uniqueness."""
    counter = itertools.count(1)
    return lambda: str(next(counter))


# ---------------------------------------------------------------------------
# A. build_image_graph
# ---------------------------------------------------------------------------

def comfy_image_ref(name: str, subfolder: str = "", folder_type: str = "input") -> str:
    """Build the LoadImage 'image' widget value ComfyUI actually resolves,
    from an /upload/image response's (name, subfolder, type) -- GAP 1
    (upload bridge).

    Verified against data/studio/comfy/ComfyUI/nodes.py:1738-1739
    (`LoadImage.load_image(self, image)` calls
    `folder_paths.get_annotated_filepath(image)` with no `default_dir`) and
    folder_paths.py:257-270 (`annotated_filepath`) + :343-357
    (`get_annotated_filepath`):

      - A subfolder is NOT a separate field on the node -- it must be
        PREFIXED onto the name (`"<subfolder>/<name>"`), because ComfyUI
        resolves the final path as `os.path.join(base_dir, name)`, and the
        file was actually SAVED at `<upload_dir>/<subfolder>/<name>`
        (server.py:410-412) -- a bare name with a separately-stored
        subfolder would silently 404 inside ComfyUI.
      - `get_annotated_filepath()` falls back to the INPUT directory
        whenever `name` has no trailing " [input]" / " [output]" / " [temp]"
        annotation (LoadImage never passes a `default_dir`) -- so the
        annotation is only actually REQUIRED when `folder_type != "input"`.
        It's therefore only appended in that case, rather than
        unconditionally, matching what ComfyUI's own frontend / a real
        Export (API) would write for the common `type == "input"` case
        (plan §3a: match real exports).

    A bare filename with the default subfolder="" and folder_type="input" --
    the common case for both a fresh /api/comfy/upload and the gallery-
    passthrough path in routes/comfy_routes.py, which both upload with those
    exact defaults -- round-trips unchanged, so this is a no-op passthrough
    for the case every existing caller/test exercises.
    """
    ref = f"{subfolder}/{name}" if subfolder else name
    if folder_type and folder_type != "input":
        ref = f"{ref} [{folder_type}]"
    return ref


def build_image_graph(params: dict) -> dict:
    """Build a ComfyUI API-format graph (flat {id: {class_type, inputs}} dict)
    for the Z-Image image pipeline, from a plain params dict (see module
    docstring for the shape). Handles:

      - txt2img (EmptySD3LatentImage) vs img2img (LoadImage + VAEEncode),
        selected by whether params["input_image"] is set (plan §3a/§3). The
        LoadImage value itself is built via comfy_image_ref() (GAP 1) so an
        upload response's subfolder/type -- not just its bare name -- is
        represented correctly; see that function's docstring for the
        " [type]"-annotation trap this avoids.
      - an ARBITRARY-length LoRA stack: N chained LoraLoaderModelOnly nodes,
        each one's `model` input rewired to the previous node's output, in
        the order given in params["loras"] -- this is the thing a simple
        "inject values into a template" approach can't do, per plan §3.

    Every field has a default (see the DEFAULT_* constants above, all
    verified against the two real workflows) so a bare `{}` still builds a
    valid graph. Raises ValueError for a malformed loras entry (missing
    comfy_name) rather than silently emitting a broken LoraLoaderModelOnly.

    Pure function: no I/O, no network, no randomness *unless* the caller asks
    for a random seed (params["randomize_seed"] truthy, or no seed given) --
    ComfyUI's own "randomize" control (`control_after_generate`) is a
    UI-only, this-run-has-no-effect widget (plan §3a/§7), so nothing else in
    the pipeline will ever roll a seed for you; this is the one place it can
    happen.
    """
    prompt = str(params.get("prompt") or "")
    negative_prompt = params.get("negative_prompt")
    negative_prompt = str(negative_prompt) if negative_prompt not in (None, "") else DEFAULT_NEGATIVE_PROMPT

    loras = params.get("loras") or []
    for i, lora in enumerate(loras):
        if not isinstance(lora, dict) or not lora.get("comfy_name"):
            raise ValueError(f"loras[{i}] is missing a comfy_name: {lora!r}")

    input_image = params.get("input_image") or None
    input_image_subfolder = str(params.get("input_image_subfolder") or "")
    input_image_type = str(params.get("input_image_type") or "input")

    width = int(params.get("width") or DEFAULT_WIDTH)
    height = int(params.get("height") or DEFAULT_HEIGHT)
    batch = int(params.get("batch") or DEFAULT_BATCH)

    if input_image:
        denoise = params.get("denoise")
        denoise = float(denoise) if denoise is not None else DEFAULT_IMG2IMG_DENOISE
    else:
        # txt2img denoise is always 1.0 -- a caller-supplied value is ignored
        # on purpose (plan §1: "Correct — txt2img denoise is always 1.0").
        denoise = 1.0

    seed = params.get("seed")
    if params.get("randomize_seed") or seed is None:
        seed = random.randint(0, _SEED_MAX)
    else:
        seed = int(seed)

    steps = int(params.get("steps") or DEFAULT_STEPS)
    cfg = float(params["cfg"]) if params.get("cfg") is not None else DEFAULT_CFG
    sampler = str(params.get("sampler") or DEFAULT_SAMPLER)
    scheduler = str(params.get("scheduler") or DEFAULT_SCHEDULER)
    shift = float(params["shift"]) if params.get("shift") is not None else DEFAULT_SHIFT

    unet = str(params.get("unet") or DEFAULT_UNET)
    clip = str(params.get("clip") or DEFAULT_CLIP)
    vae = str(params.get("vae") or DEFAULT_VAE)
    filename_prefix = str(params.get("filename_prefix") or DEFAULT_FILENAME_PREFIX)

    nid = _new_id_gen()
    graph: dict[str, Any] = {}

    unet_id = nid()
    graph[unet_id] = {
        "class_type": "UNETLoader",
        "inputs": {"unet_name": unet, "weight_dtype": DEFAULT_UNET_WEIGHT_DTYPE},
    }

    model_link = [unet_id, 0]
    for lora in loras:
        lora_id = nid()
        graph[lora_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": model_link,
                "lora_name": lora["comfy_name"],
                "strength_model": float(lora.get("weight", 1.0)),
            },
        }
        model_link = [lora_id, 0]

    sampling_id = nid()
    graph[sampling_id] = {
        "class_type": "ModelSamplingAuraFlow",
        "inputs": {"model": model_link, "shift": shift},
    }
    model_link = [sampling_id, 0]

    clip_id = nid()
    graph[clip_id] = {
        "class_type": "CLIPLoader",
        "inputs": {"clip_name": clip, "type": DEFAULT_CLIP_TYPE, "device": DEFAULT_CLIP_DEVICE},
    }
    clip_link = [clip_id, 0]

    pos_id = nid()
    graph[pos_id] = {"class_type": "CLIPTextEncode", "inputs": {"clip": clip_link, "text": prompt}}

    neg_id = nid()
    graph[neg_id] = {"class_type": "CLIPTextEncode", "inputs": {"clip": clip_link, "text": negative_prompt}}

    vae_id = nid()
    graph[vae_id] = {"class_type": "VAELoader", "inputs": {"vae_name": vae}}
    vae_link = [vae_id, 0]

    if input_image:
        load_id = nid()
        graph[load_id] = {
            "class_type": "LoadImage",
            "inputs": {"image": comfy_image_ref(input_image, input_image_subfolder, input_image_type)},
        }
        encode_id = nid()
        graph[encode_id] = {
            "class_type": "VAEEncode",
            "inputs": {"pixels": [load_id, 0], "vae": vae_link},
        }
        latent_link = [encode_id, 0]
    else:
        latent_id = nid()
        graph[latent_id] = {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": width, "height": height, "batch_size": batch},
        }
        latent_link = [latent_id, 0]

    sampler_id = nid()
    graph[sampler_id] = {
        "class_type": "KSampler",
        "inputs": {
            "model": model_link,
            "positive": [pos_id, 0],
            "negative": [neg_id, 0],
            "latent_image": latent_link,
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler,
            "scheduler": scheduler,
            "denoise": denoise,
        },
    }

    decode_id = nid()
    graph[decode_id] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": [sampler_id, 0], "vae": vae_link},
    }

    save_id = nid()
    graph[save_id] = {
        "class_type": "SaveImage",
        "inputs": {"images": [decode_id, 0], "filename_prefix": filename_prefix},
    }

    return graph


# ---------------------------------------------------------------------------
# B. build_wan_i2v_graph
# ---------------------------------------------------------------------------
#
# Verified this session directly against the ComfyUI source (not just the
# plan's widgets_values table, and not just "Studio - Ride video (Wan2.2)
# .json"'s subgraph, which was independently flattened by hand into a new
# sibling file -- "Studio - Ride video FLAT (Wan2.2).json", build order step
# 10 -- and cross-checked node-for-node against this builder's output):
#
#   nodes.py:1609-1641      KSamplerAdvanced.INPUT_TYPES() -- inputs in order
#                            model, add_noise, noise_seed, steps, cfg,
#                            sampler_name, scheduler, positive, negative,
#                            latent_image, start_at_step, end_at_step,
#                            return_with_leftover_noise. NO "denoise" input
#                            (unlike KSampler) -- it's a python-level kwarg
#                            default (1.0) inside sample(), never a declared
#                            node input, so it must never appear in a built
#                            graph's "inputs" dict for this class_type.
#   comfy_extras/nodes_model_advanced.py:116-121  ModelSamplingSD3 --
#                            {model, shift}. THE PLAN'S OWN §3b NODE TABLE
#                            OMITS THIS NODE -- the real workflow chains
#                            LoraLoaderModelOnly -> ModelSamplingSD3 (shift
#                            5.0) -> KSamplerAdvanced on BOTH the high- and
#                            low-noise sides (subgraph-internal nodes 109/124
#                            in the original file). Skipping it would silently
#                            run Wan with the image pipeline's flow-matching
#                            defaults instead of Wan's own tuned schedule.
#   comfy_extras/nodes_wan.py:16-61          WanImageToVideo.define_schema()
#                            -- required: positive, negative, vae, width
#                            (default 832), height (default 480), length
#                            (default 81), batch_size (default 1); optional:
#                            clip_vision_output, start_image. width/height/
#                            length/batch_size carry defaults but are NOT
#                            `optional=True`, so (like EmptySD3LatentImage in
#                            build_image_graph()) they're always given as
#                            literal ints, never omitted.
#   comfy_extras/nodes_video.py:75-165        SaveVideo.define_schema() --
#                            {video, filename_prefix (default
#                            "video/ComfyUI"), format (combo, default
#                            "auto"), codec (combo, default "auto")}.
#                            CreateVideo.define_schema() -- required {images,
#                            fps (FLOAT, default 30.0)}; optional {audio,
#                            bit_depth (default 8)} -- both omitted here,
#                            ComfyUI uses their python-level defaults.
#   nodes.py:966-1016        UNETLoader {unet_name, weight_dtype},
#                            CLIPLoader {clip_name, type, device} -- note
#                            type="wan" here, NOT "lumina2" (that's the image
#                            pipeline's CLIP family); "wan" is a real, listed
#                            CLIPLoader "type" combo option.
#   nodes.py:740-752, 754-847   LoraLoaderModelOnly {model, lora_name,
#                            strength_model}, VAELoader {vae_name} -- same
#                            shape build_image_graph() already uses.
#
# Confirms (and corrects) plan §3b's node/param table; see the module
# docstring's "Canonical video params dict" section above for the params
# shape this function consumes, and introspect_graph()'s video-handling
# section below (now verified, not best-effort) for the inverse mapping.

DEFAULT_WAN_UNET_HIGH = "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
DEFAULT_WAN_UNET_LOW = "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
DEFAULT_WAN_UNET_WEIGHT_DTYPE = "default"
DEFAULT_WAN_CLIP = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
DEFAULT_WAN_CLIP_TYPE = "wan"
DEFAULT_WAN_CLIP_DEVICE = "default"
DEFAULT_WAN_VAE = "wan_2.1_vae.safetensors"
DEFAULT_WAN_LORA_HIGH = "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"
DEFAULT_WAN_LORA_LOW = "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"
DEFAULT_WAN_LORA_WEIGHT = 1.0
DEFAULT_WAN_SHIFT = 5.0  # ModelSamplingSD3 -- Wan's own tuned value (verified against the real workflow); distinct from the image pipeline's ModelSamplingAuraFlow shift=3.
DEFAULT_WAN_WIDTH = 720
DEFAULT_WAN_HEIGHT = 720
DEFAULT_WAN_FRAMES = 81
DEFAULT_WAN_BATCH = 1
DEFAULT_WAN_FPS = 16
DEFAULT_WAN_STEPS = 4
DEFAULT_WAN_CFG = 1.0
DEFAULT_WAN_SAMPLER = "euler"
DEFAULT_WAN_SCHEDULER = "simple"
# Reuses the SAME fallback text as the image pipeline (and what
# static/js/genParams.js's VIDEO_BASELINE also defaults to) rather than the
# reference workflow's own (Chinese) negative prompt -- that stronger,
# Wan-community-tuned text is preserved verbatim only in the saved "Ride
# video FLAT" workflow file itself, as an opt-in starting point (plan §4a:
# picking a saved workflow "seeds every field"; every field stays editable).
DEFAULT_WAN_NEGATIVE_PROMPT = DEFAULT_NEGATIVE_PROMPT
DEFAULT_WAN_FILENAME_PREFIX = "video/odysseus"
DEFAULT_WAN_FORMAT = "auto"
DEFAULT_WAN_CODEC = "auto"

# Stage 2 has add_noise="disable", which makes its noise_seed a dead input
# (common_ksampler() only consumes noise_seed when disable_noise is False --
# nodes.py:1634-1641). Kept at the reference workflow's own literal 0 rather
# than exposed as a second seed param (task step 11 is explicit: "do not
# expose it as a second seed").
_WAN_STAGE2_NOISE_SEED = 0


def _wan_step_split(total_steps: int) -> tuple[int, int, int]:
    """Derive (stage1_end, stage2_start, stage2_end) from one "total steps"
    value -- stage 1 always runs [0, mid]; stage 2 always runs [mid, total]
    (stage2_start == stage1_end == mid, so the two stages tile the schedule
    with no gap or overlap, matching the reference workflow's 4 -> [0,2] then
    [2,4] exactly). mid = total // 2 (integer division): an odd total gives
    stage 1 the smaller half (e.g. 5 -> [0,2] then [2,5]) -- there is no
    "more correct" split for an odd total, and every real Wan lightning-LoRA
    workflow uses an even total (4 or 8) in practice. Negative input is
    clamped to 0 rather than producing a negative range.
    """
    total = max(int(total_steps), 0)
    mid = total // 2
    return mid, mid, total


def _extract_wan_lora_weights(loras: Optional[list]) -> tuple[float, float]:
    """Resolve (high_noise_weight, low_noise_weight) out of params["loras"]
    -- reusing the SAME [{comfy_name, weight}, ...] list shape
    build_image_graph() consumes rather than inventing new dedicated fields
    (task step 11 / plan §3b: Wan's LoRA FILENAMES are always
    DEFAULT_WAN_LORA_HIGH/LOW -- only the STRENGTH is ever caller-controlled).

    static/js/genParams.js -- the already-built frontend for this feature --
    sends exactly
    `[{comfy_name:'wan_high_noise', weight, role:'high_noise'},
      {comfy_name:'wan_low_noise', weight, role:'low_noise'}]`.
    Entries are matched, in order of trust:
      1. an explicit "role" field ("high_noise"/"low_noise") -- what
         genParams.js actually sends (LoraParam in routes/comfy_routes.py
         gained a `role` field so this survives Pydantic validation instead
         of being silently dropped as an undeclared extra key);
      2. a "high"/"low" substring match on comfy_name -- covers a caller
         that omits role, AND covers re-submitting introspect_graph()'s own
         output verbatim (its comfy_name values are the REAL filenames
         DEFAULT_WAN_LORA_HIGH/LOW, which themselves contain "high"/"low");
      3. positional order, [0]=high / [1]=low -- the final fallback for a
         caller that supplies neither role nor a recognisable name.
    Missing/malformed entries fall back to DEFAULT_WAN_LORA_WEIGHT rather
    than raising -- unlike build_image_graph()'s arbitrary user-selected LoRA
    stack, a video LoRA entry can never reference an unknown/bad file (the
    filename is never caller-supplied), so there is nothing to validate
    loudly against.
    """
    high = low = None
    for i, lora in enumerate(loras or []):
        if not isinstance(lora, dict):
            continue
        weight = lora.get("weight")
        if weight is None:
            continue
        role = str(lora.get("role") or "").lower()
        name = str(lora.get("comfy_name") or "").lower()
        if high is None and (role == "high_noise" or (not role and ("high" in name or i == 0))):
            high = float(weight)
        elif low is None and (role == "low_noise" or (not role and ("low" in name or i == 1))):
            low = float(weight)
    return (
        high if high is not None else DEFAULT_WAN_LORA_WEIGHT,
        low if low is not None else DEFAULT_WAN_LORA_WEIGHT,
    )


def build_wan_i2v_graph(params: dict) -> dict:
    """Build a ComfyUI API-format graph for the Wan2.2 I2V pipeline (plan
    §3b) from a plain params dict (see module docstring for the shape).

    Two independent model chains -- UNETLoader -> LoraLoaderModelOnly ->
    ModelSamplingSD3 -- feed a two-stage KSamplerAdvanced pair sharing one
    WanImageToVideo latent:

      stage 1 (HIGH noise): add_noise="enable", the only REAL seed,
        start_at_step=0 -> end_at_step=mid, return_with_leftover_noise="enable"
      stage 2 (LOW noise):  add_noise="disable" (noise_seed therefore inert),
        start_at_step=mid -> end_at_step=steps, return_with_leftover_noise="disable"

    where mid = steps // 2 (_wan_step_split()) -- a single params["steps"]
    (the TOTAL, e.g. 4) drives both stages' start/end consistently, exactly
    like static/js/genParams.js's one "total steps" field is documented to
    (plan §7 Video Advanced: "total steps (4, auto-split across both
    samplers)").

    Deliberately does NOT call apply_style_trigger() -- Wan has no knowledge
    of the studio style/character LoRAs (plan §3b), so triggering the prompt
    would just add tokens Wan's text encoder has no use for. Deliberately
    does not offer a studio-LoRA stack either -- only the two fixed Wan
    lightning LoRAs (see _extract_wan_lora_weights()) are ever wired in.

    Start frame handling mirrors build_image_graph()'s img2img path exactly
    -- comfy_image_ref() builds the same LoadImage "image" widget value from
    (input_image, input_image_subfolder, input_image_type); when no
    input_image is given, the LoadImage node and WanImageToVideo's optional
    start_image input are both simply omitted (WanImageToVideo supports a
    text-only/no-start-frame mode per its schema), rather than raising --
    keeping this a pure, always-buildable function like build_image_graph(),
    consistent with the "every field has a default" design (module
    docstring). In production this repo only ever calls it in the I2V case
    (the frontend always supplies a start frame), but the graph builder
    itself doesn't need to be the one to enforce that.

    Pure function: no I/O, no network, no randomness unless the caller asks
    for a random seed (params["randomize_seed"] truthy, or no seed given) --
    same seed-resolution contract as build_image_graph().
    """
    prompt = str(params.get("prompt") or "")
    negative_prompt = params.get("negative_prompt")
    negative_prompt = str(negative_prompt) if negative_prompt not in (None, "") else DEFAULT_WAN_NEGATIVE_PROMPT

    input_image = params.get("input_image") or None
    input_image_subfolder = str(params.get("input_image_subfolder") or "")
    input_image_type = str(params.get("input_image_type") or "input")

    width = int(params.get("width") or DEFAULT_WAN_WIDTH)
    height = int(params.get("height") or DEFAULT_WAN_HEIGHT)
    frames = int(params.get("frames") or DEFAULT_WAN_FRAMES)
    batch = int(params.get("batch") or DEFAULT_WAN_BATCH)
    fps = params.get("fps")
    fps = float(fps) if fps is not None else float(DEFAULT_WAN_FPS)

    seed = params.get("seed")
    if params.get("randomize_seed") or seed is None:
        seed = random.randint(0, _SEED_MAX)
    else:
        seed = int(seed)

    steps = int(params.get("steps") or DEFAULT_WAN_STEPS)
    stage1_end, stage2_start, stage2_end = _wan_step_split(steps)
    cfg = float(params["cfg"]) if params.get("cfg") is not None else DEFAULT_WAN_CFG
    sampler = str(params.get("sampler") or DEFAULT_WAN_SAMPLER)
    scheduler = str(params.get("scheduler") or DEFAULT_WAN_SCHEDULER)
    shift = float(params["shift"]) if params.get("shift") is not None else DEFAULT_WAN_SHIFT

    unet_high = str(params.get("unet") or DEFAULT_WAN_UNET_HIGH)
    unet_low = str(params.get("unet_low") or DEFAULT_WAN_UNET_LOW)
    clip = str(params.get("clip") or DEFAULT_WAN_CLIP)
    vae = str(params.get("vae") or DEFAULT_WAN_VAE)
    filename_prefix = str(params.get("filename_prefix") or DEFAULT_WAN_FILENAME_PREFIX)

    lora_high_weight, lora_low_weight = _extract_wan_lora_weights(params.get("loras"))

    nid = _new_id_gen()
    graph: dict[str, Any] = {}

    # -- HIGH NOISE model chain (built FIRST -- introspect_graph() relies on
    # insertion order to pick this as the "primary" UNETLoader/KSamplerAdvanced). --
    unet_high_id = nid()
    graph[unet_high_id] = {
        "class_type": "UNETLoader",
        "inputs": {"unet_name": unet_high, "weight_dtype": DEFAULT_WAN_UNET_WEIGHT_DTYPE},
    }
    lora_high_id = nid()
    graph[lora_high_id] = {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {"model": [unet_high_id, 0], "lora_name": DEFAULT_WAN_LORA_HIGH, "strength_model": lora_high_weight},
    }
    sampling_high_id = nid()
    graph[sampling_high_id] = {
        "class_type": "ModelSamplingSD3",
        "inputs": {"model": [lora_high_id, 0], "shift": shift},
    }

    # -- LOW NOISE model chain: same shape, independent nodes. --
    unet_low_id = nid()
    graph[unet_low_id] = {
        "class_type": "UNETLoader",
        "inputs": {"unet_name": unet_low, "weight_dtype": DEFAULT_WAN_UNET_WEIGHT_DTYPE},
    }
    lora_low_id = nid()
    graph[lora_low_id] = {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {"model": [unet_low_id, 0], "lora_name": DEFAULT_WAN_LORA_LOW, "strength_model": lora_low_weight},
    }
    sampling_low_id = nid()
    graph[sampling_low_id] = {
        "class_type": "ModelSamplingSD3",
        "inputs": {"model": [lora_low_id, 0], "shift": shift},
    }

    # -- CLIP / text (single CLIPLoader feeds both encodes, like the image graph). --
    clip_id = nid()
    graph[clip_id] = {
        "class_type": "CLIPLoader",
        "inputs": {"clip_name": clip, "type": DEFAULT_WAN_CLIP_TYPE, "device": DEFAULT_WAN_CLIP_DEVICE},
    }
    clip_link = [clip_id, 0]

    pos_id = nid()
    graph[pos_id] = {"class_type": "CLIPTextEncode", "inputs": {"clip": clip_link, "text": prompt}}
    neg_id = nid()
    graph[neg_id] = {"class_type": "CLIPTextEncode", "inputs": {"clip": clip_link, "text": negative_prompt}}

    # -- VAE (shared by WanImageToVideo's encode and the final VAEDecode). --
    vae_id = nid()
    graph[vae_id] = {"class_type": "VAELoader", "inputs": {"vae_name": vae}}
    vae_link = [vae_id, 0]

    # -- WanImageToVideo: width/height/length/batch_size are required-but-
    # literal INT inputs (NOT optional=True, despite carrying schema
    # defaults) -- same pattern as EmptySD3LatentImage in build_image_graph().
    wan_inputs: dict[str, Any] = {
        "positive": [pos_id, 0],
        "negative": [neg_id, 0],
        "vae": vae_link,
        "width": width,
        "height": height,
        "length": frames,
        "batch_size": batch,
    }
    if input_image:
        load_id = nid()
        graph[load_id] = {
            "class_type": "LoadImage",
            "inputs": {"image": comfy_image_ref(input_image, input_image_subfolder, input_image_type)},
        }
        wan_inputs["start_image"] = [load_id, 0]

    wan_id = nid()
    graph[wan_id] = {"class_type": "WanImageToVideo", "inputs": wan_inputs}
    # WanImageToVideo outputs, in RETURN_TYPES/schema order: positive(0),
    # negative(1), latent(2) -- verified against comfy_extras/nodes_wan.py.
    wan_positive, wan_negative, wan_latent = [wan_id, 0], [wan_id, 1], [wan_id, 2]

    # -- Stage 1 (HIGH noise): add_noise enable, the only REAL seed, 0 -> stage1_end. --
    stage1_id = nid()
    graph[stage1_id] = {
        "class_type": "KSamplerAdvanced",
        "inputs": {
            "model": [sampling_high_id, 0],
            "positive": wan_positive,
            "negative": wan_negative,
            "latent_image": wan_latent,
            "add_noise": "enable",
            "noise_seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler,
            "scheduler": scheduler,
            "start_at_step": 0,
            "end_at_step": stage1_end,
            "return_with_leftover_noise": "enable",
        },
    }

    # -- Stage 2 (LOW noise): add_noise disable (noise_seed inert), stage2_start -> steps. --
    stage2_id = nid()
    graph[stage2_id] = {
        "class_type": "KSamplerAdvanced",
        "inputs": {
            "model": [sampling_low_id, 0],
            "positive": wan_positive,
            "negative": wan_negative,
            "latent_image": [stage1_id, 0],
            "add_noise": "disable",
            "noise_seed": _WAN_STAGE2_NOISE_SEED,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler,
            "scheduler": scheduler,
            "start_at_step": stage2_start,
            "end_at_step": stage2_end,
            "return_with_leftover_noise": "disable",
        },
    }

    decode_id = nid()
    graph[decode_id] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": [stage2_id, 0], "vae": vae_link},
    }

    video_id = nid()
    graph[video_id] = {
        "class_type": "CreateVideo",
        "inputs": {"images": [decode_id, 0], "fps": fps},
    }

    save_id = nid()
    graph[save_id] = {
        "class_type": "SaveVideo",
        "inputs": {
            "video": [video_id, 0],
            "filename_prefix": filename_prefix,
            "format": DEFAULT_WAN_FORMAT,
            "codec": DEFAULT_WAN_CODEC,
        },
    }

    return graph


# ---------------------------------------------------------------------------
# C. introspect_graph
# ---------------------------------------------------------------------------

# Node types this module understands well enough to map to panel controls
# (plan §4b). Anything else encountered in a graph is reported in
# `unsupported` rather than silently dropped.
#
# KSamplerAdvanced / WanImageToVideo / CreateVideo / SaveVideo / ModelSamplingSD3
# field-level extraction is now VERIFIED (task step 11), not best-effort --
# checked directly against data/studio/comfy/ComfyUI/nodes.py's
# KSamplerAdvanced.INPUT_TYPES(), comfy_extras/nodes_wan.py's
# WanImageToVideo.define_schema(), comfy_extras/nodes_video.py's
# CreateVideo/SaveVideo.define_schema(), and
# comfy_extras/nodes_model_advanced.py's ModelSamplingSD3.INPUT_TYPES() --
# see build_wan_i2v_graph()'s header comment for the full citation. All of
# plan §3b's inferred field names (width/height/length/batch_size on
# WanImageToVideo, fps on CreateVideo, filename_prefix on SaveVideo,
# noise_seed on KSamplerAdvanced) turned out correct; ModelSamplingSD3 itself
# was missing from the plan's node table entirely (now added below).
_KNOWN_CLASS_TYPES = {
    "UNETLoader", "LoraLoaderModelOnly", "ModelSamplingAuraFlow", "CLIPLoader",
    "CLIPTextEncode", "EmptySD3LatentImage", "LoadImage", "VAEEncode",
    "KSampler", "VAELoader", "VAEDecode", "SaveImage",
    # Video (verified -- see note above):
    "KSamplerAdvanced", "WanImageToVideo", "CreateVideo", "SaveVideo", "ModelSamplingSD3",
}

_SAMPLER_CLASS_TYPES = ("KSampler", "KSamplerAdvanced")


def _resolve_conditioning_text(nodes: dict, ref: Any, input_name: str, *, _depth: int = 0) -> str:
    """Resolve the prompt TEXT feeding a sampler's positive/negative
    CONDITIONING input.

    For the image graph this is a single hop: KSampler's positive/negative
    links straight to a CLIPTextEncode node, which has "text" right there in
    its own inputs. The video graph is NOT a single hop -- KSamplerAdvanced's
    positive/negative link to WanImageToVideo's OUTPUTS, and WanImageToVideo
    is a pass-through conditioning node: it takes positive/negative CLIP
    conditioning IN as its own inputs and re-emits ITS OWN positive/negative
    conditioning OUT (verified against comfy_extras/nodes_wan.py's
    WanImageToVideo.execute(), which wraps -- via
    node_helpers.conditioning_set_values() -- rather than replaces the
    incoming conditioning). So resolving a video graph's prompt text means
    following the referenced node's OWN same-named input one more hop, not
    assuming the first node reached already has a "text" field.

    This generalises rather than special-cases WanImageToVideo: any node
    encountered that doesn't itself have a "text" input is assumed to be a
    pass-through, and its own `input_name` input is followed recursively
    (bounded by `_depth` against a malformed/cyclic graph) until a real
    CLIPTextEncode-shaped node is found or the trail runs out (-> ""). A
    direct CLIPTextEncode link (the image graph's shape) resolves on the
    FIRST call with zero extra hops, so this is a strict generalisation of
    the previous one-hop-only behaviour, not a video-only special case.
    """
    if _depth > 4 or not (isinstance(ref, list) and ref and ref[0] in nodes):
        return ""
    node_inputs = nodes[ref[0]].get("inputs", {})
    if "text" in node_inputs:
        return node_inputs.get("text", "")
    return _resolve_conditioning_text(nodes, node_inputs.get(input_name), input_name, _depth=_depth + 1)


def introspect_graph(api_graph: dict) -> dict:
    """Walk an API-format graph and return `{"params": {...}, "unsupported":
    [node_id, ...]}` -- the panel schema + current values (plan §4b), using
    the same field names as the canonical params dict documented at the top
    of this module, so `introspect_graph(build_image_graph(p))["params"]`
    round-trips `p` (see tests/test_comfy_graphs.py).

    Positive vs negative CLIPTextEncode is disambiguated by which one is
    wired to the sampler's `positive` / `negative` input (not by node order
    or id) -- required because a hand-authored or exported graph can number
    nodes in any order.

    The LoRA chain is recovered by walking the `model`-typed link chain
    forward from the UNETLoader (producer -> the node whose `model` input
    references it) rather than by sorting node ids, so LoRA order is correct
    even when node ids aren't sequential/chain-ordered.

    Any node whose class_type isn't in `_KNOWN_CLASS_TYPES` is listed in
    `unsupported` (plan §4b: "left untouched, shown read-only in an 'other
    nodes' list") rather than raising -- an unrecognised node must never
    break introspection of the rest of the graph.
    """
    nodes: dict = api_graph or {}
    by_type: dict[str, list[str]] = {}
    for node_id, node in nodes.items():
        by_type.setdefault(node.get("class_type", ""), []).append(node_id)

    unsupported = [
        node_id for node_id, node in nodes.items()
        if node.get("class_type", "") not in _KNOWN_CLASS_TYPES
    ]

    params: dict[str, Any] = {}

    # --- sampler (KSampler or KSamplerAdvanced) ---
    sampler_id = None
    sampler_node = None
    for ct in _SAMPLER_CLASS_TYPES:
        ids = by_type.get(ct) or []
        if ids:
            sampler_id, sampler_node = ids[0], nodes[ids[0]]
            break

    positive_text = negative_text = ""
    if sampler_node is not None:
        s_in = sampler_node.get("inputs", {})
        pos_ref = s_in.get("positive")
        neg_ref = s_in.get("negative")
        positive_text = _resolve_conditioning_text(nodes, pos_ref, "positive")
        negative_text = _resolve_conditioning_text(nodes, neg_ref, "negative")

        params["steps"] = s_in.get("steps")
        params["cfg"] = s_in.get("cfg")
        params["sampler"] = s_in.get("sampler_name")
        params["scheduler"] = s_in.get("scheduler")
        params["denoise"] = s_in.get("denoise")
        # KSampler uses "seed"; KSamplerAdvanced uses "noise_seed" (verified
        # against nodes.py's KSamplerAdvanced.INPUT_TYPES() -- see the
        # module-level note above). For a video graph this resolves to
        # STAGE 1's noise_seed (the real one) because sampler_id above is
        # always by_type["KSamplerAdvanced"][0], and build_wan_i2v_graph()
        # always creates stage 1 before stage 2 -- stage 2's inert noise_seed
        # is never read here.
        params["seed"] = s_in.get("seed", s_in.get("noise_seed"))

    params["prompt"] = positive_text
    params["negative_prompt"] = negative_text

    # --- LoRA chain(s): walk the `model` link forward from EVERY UNETLoader ---
    # (not just the first). The image graph only ever has one UNETLoader, so
    # for it this is behaviourally identical to walking unet_ids[0] alone --
    # but the video graph (task step 11) has TWO independent chains (high-
    # noise / low-noise), and looping over all of them is what makes both
    # LoRAs -- and the low-noise UNETLoader's own name -- recoverable instead
    # of the second chain being silently dropped.
    unet_ids = by_type.get("UNETLoader") or []
    loras: list[dict] = []
    if unet_ids:
        params["unet"] = nodes[unet_ids[0]].get("inputs", {}).get("unet_name")
        if len(unet_ids) > 1:
            # Video's low-noise UNETLoader -- surfaced under its own key so
            # the single-UNET image-graph contract ("unet" = the only one)
            # is unchanged; build_wan_i2v_graph() reads this same key back
            # as its low-noise override (module docstring's video params
            # shape).
            params["unet_low"] = nodes[unet_ids[1]].get("inputs", {}).get("unet_name")

        model_consumer: dict[str, str] = {}
        for node_id, node in nodes.items():
            m = node.get("inputs", {}).get("model")
            if isinstance(m, list) and len(m) == 2:
                model_consumer[m[0]] = node_id

        for unet_id in unet_ids:
            cur = unet_id
            visited: set = set()
            while cur in model_consumer and model_consumer[cur] not in visited:
                nxt_id = model_consumer[cur]
                visited.add(nxt_id)
                nxt = nodes[nxt_id]
                if nxt.get("class_type") == "LoraLoaderModelOnly":
                    ni = nxt.get("inputs", {})
                    loras.append({"comfy_name": ni.get("lora_name"), "weight": ni.get("strength_model")})
                    cur = nxt_id
                else:
                    if nxt.get("class_type") in ("ModelSamplingAuraFlow", "ModelSamplingSD3"):
                        # setdefault: the FIRST chain's shift wins (mirrors
                        # "unet" above always being the first/primary chain);
                        # Wan's two chains always carry the same shift value
                        # in practice, so this is a non-issue there.
                        params.setdefault("shift", nxt.get("inputs", {}).get("shift"))
                    break
    params["loras"] = loras

    # --- CLIP / VAE loaders ---
    clip_ids = by_type.get("CLIPLoader") or []
    if clip_ids:
        params["clip"] = nodes[clip_ids[0]].get("inputs", {}).get("clip_name")

    vae_ids = by_type.get("VAELoader") or []
    if vae_ids:
        params["vae"] = nodes[vae_ids[0]].get("inputs", {}).get("vae_name")

    # --- latent source: txt2img vs img2img vs video ---
    if by_type.get("EmptySD3LatentImage"):
        lat = nodes[by_type["EmptySD3LatentImage"][0]].get("inputs", {})
        params["width"] = lat.get("width")
        params["height"] = lat.get("height")
        params["batch"] = lat.get("batch_size")
        params["input_image"] = None
    elif by_type.get("LoadImage"):
        li = nodes[by_type["LoadImage"][0]].get("inputs", {})
        params["input_image"] = li.get("image")

    # WanImageToVideo carries its OWN width/height/length/batch_size --
    # deliberately a separate `if`, not an `elif` off the block above. A
    # video graph ALSO has a LoadImage node (the start frame), which would
    # otherwise win the elif chain and silently prevent width/height/frames/
    # batch from ever being read (this was a real bug: for any video graph,
    # the WanImageToVideo branch was unreachable dead code). Verified against
    # comfy_extras/nodes_wan.py -- see build_wan_i2v_graph()'s header note.
    if by_type.get("WanImageToVideo"):
        wi = nodes[by_type["WanImageToVideo"][0]].get("inputs", {})
        params["width"] = wi.get("width")
        params["height"] = wi.get("height")
        params["frames"] = wi.get("length")
        params["batch"] = wi.get("batch_size")

    # --- output node ---
    save_ids = by_type.get("SaveImage") or []
    if save_ids:
        params["filename_prefix"] = nodes[save_ids[0]].get("inputs", {}).get("filename_prefix")
    elif by_type.get("SaveVideo"):
        # Verified against comfy_extras/nodes_video.py's SaveVideo schema.
        params["filename_prefix"] = nodes[by_type["SaveVideo"][0]].get("inputs", {}).get("filename_prefix")

    if by_type.get("CreateVideo"):
        # Verified against comfy_extras/nodes_video.py's CreateVideo schema
        # (single "fps" widget).
        params["fps"] = nodes[by_type["CreateVideo"][0]].get("inputs", {}).get("fps")

    return {"params": params, "unsupported": unsupported}


# ---------------------------------------------------------------------------
# D. ui_to_api
# ---------------------------------------------------------------------------

# ComfyUI's frontend renders an extra "control_after_generate" combo widget
# immediately after certain seed-like INT widgets. It is injected purely
# client-side -- it has NO entry in the node's declared /object_info inputs
# at all, so it can't be located by name; it can only be found positionally,
# immediately after the widget value it rides alongside. Verified for
# KSampler at plan §3a: widgets order
# [seed, control_after_generate, steps, cfg, sampler, scheduler, denoise].
# KSamplerAdvanced's "noise_seed" companion is the same mechanism -- verified
# directly against nodes.py's KSamplerAdvanced.INPUT_TYPES(), whose
# "noise_seed" entry carries the same {"control_after_generate": True} flag
# seen on KSampler's "seed".
_CONTROL_AFTER_GENERATE_COMPANION = {
    "KSampler": "seed",
    "KSamplerAdvanced": "noise_seed",
}


def ui_to_api(ui_graph: dict, object_info: dict) -> dict:
    """Convert a ComfyUI UI-format graph (`{"nodes": [...], "links": [...]}`)
    to API format (`{"<id>": {"class_type", "inputs"}}`), using the ordered
    `input.required` (+ `input.optional`) dict per class_type from ComfyUI's
    `GET /object_info` (plan §4c tier 2) to map `widgets_values` positions to
    input names -- this is what the ComfyUI frontend itself does.

    `object_info` must be the FULL /object_info response (a dict keyed by
    every class_type present in the graph), not a single-class lookup.

    Deliberately skips the two known UI-only widget traps (plan §3a/§4c):
      - `control_after_generate` on KSampler/KSamplerAdvanced (positional,
        see `_CONTROL_AFTER_GENERATE_COMPANION` above -- it has no declared
        input name to skip by name).
      - LoadImage's second widgets_values entry (the upload-mode marker,
        e.g. "image") -- also undeclared in /object_info.

    Raises GraphConversionError -- rather than silently mis-mapping -- for
    any node whose class_type is missing from `object_info`, whose linked
    input references an unknown link id, or whose widgets_values count
    doesn't line up with its declared (non-linked) inputs once the two known
    traps above are accounted for. Per plan §4c, tier 2 is a best-effort
    convenience; when it can't be confident, the honest move is to fail
    loudly so the caller can fall back to "export API format manually",
    not to guess.
    """
    ui_nodes = ui_graph.get("nodes") or []
    ui_links = ui_graph.get("links") or []

    # link_id -> (origin_node_id_str, origin_output_slot)
    link_index: dict[Any, tuple[str, int]] = {}
    for link in ui_links:
        if not isinstance(link, (list, tuple)) or len(link) < 5:
            continue
        link_id, origin_id, origin_slot = link[0], link[1], link[2]
        link_index[link_id] = (str(origin_id), origin_slot)

    api_graph: dict[str, Any] = {}
    for node in ui_nodes:
        node_id = str(node.get("id"))
        class_type = node.get("type", "")
        info = object_info.get(class_type)
        if not info:
            raise GraphConversionError(
                f"Node {node_id} ('{class_type}') is not in ComfyUI's /object_info "
                f"-- cannot determine its input schema."
            )
        node_input_spec = info.get("input") or {}
        ordered_names = list((node_input_spec.get("required") or {}).keys()) + \
            list((node_input_spec.get("optional") or {}).keys())

        # Which declared input names does this node instance satisfy via a link?
        resolved_links: dict[str, list] = {}
        for inp in (node.get("inputs") or []):
            name = inp.get("name")
            link_id = inp.get("link")
            if name is None or link_id is None:
                continue
            if link_id not in link_index:
                raise GraphConversionError(
                    f"Node {node_id} ('{class_type}') input '{name}' references "
                    f"unknown link id {link_id!r}."
                )
            origin_id, origin_slot = link_index[link_id]
            resolved_links[name] = [origin_id, origin_slot]

        widget_names = [n for n in ordered_names if n not in resolved_links]
        raw_values = list(node.get("widgets_values") or [])
        companion_after = _CONTROL_AFTER_GENERATE_COMPANION.get(class_type)

        inputs: dict[str, Any] = dict(resolved_links)
        vi = 0
        for name in widget_names:
            if vi >= len(raw_values):
                raise GraphConversionError(
                    f"Node {node_id} ('{class_type}'): ran out of widgets_values "
                    f"before filling required input '{name}' (have {len(raw_values)} "
                    f"value(s) for {widget_names})."
                )
            inputs[name] = raw_values[vi]
            vi += 1
            if companion_after and name == companion_after:
                vi += 1  # discard the control_after_generate value

        leftover = len(raw_values) - vi
        if leftover:
            # LoadImage's trailing upload-mode marker is the one known,
            # harmless leftover (plan §3a). Anything else is suspicious.
            if class_type == "LoadImage" and leftover == 1:
                pass
            else:
                raise GraphConversionError(
                    f"Node {node_id} ('{class_type}'): {leftover} unexplained "
                    f"leftover widgets_values entr{'y' if leftover == 1 else 'ies'} "
                    f"after mapping {widget_names}."
                )

        api_graph[node_id] = {"class_type": class_type, "inputs": inputs}

    return api_graph


# ---------------------------------------------------------------------------
# E. apply_style_trigger -- the one function here that touches the filesystem
# ---------------------------------------------------------------------------

# data/studio/scripts/styles.json lives two directories up from this file
# (src/comfy_graphs.py -> src/ -> repo root -> data/studio/scripts/). Resolved
# relative to this file (not via src.constants.DATA_DIR) so this module stays
# free of any repo-internal import -- stdlib only, per the module docstring.
_STYLES_REGISTRY_PATH = Path(__file__).resolve().parent.parent / "data" / "studio" / "scripts" / "styles.json"

# Used only if styles.json is missing (e.g. a fresh clone -- data/ is
# gitignored per CLAUDE.md) so this function degrades instead of crashing.
_FALLBACK_STYLE_REGISTRY = {
    "default": "smoon",
    "styles": {
        "smoon": {"trigger": "toei90s style, smoon", "label": "default style", "aliases": []},
    },
}


def _load_style_registry(registry_path: Optional[str] = None) -> dict:
    path = Path(registry_path) if registry_path else _STYLES_REGISTRY_PATH
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _FALLBACK_STYLE_REGISTRY


def _resolve_style_key(text: str, registry: dict) -> str:
    t = (text or "").lower()
    best, best_len = None, 0
    for key, s in registry["styles"].items():
        for alias in [key] + list(s.get("aliases", [])):
            if alias in t and len(alias) > best_len:
                best, best_len = key, len(alias)
    return best or registry["default"]


def apply_style_trigger(prompt: str, style_hint: Optional[str] = None, *, registry_path: Optional[str] = None) -> str:
    """Port of data/studio/scripts/styles.py's `apply_trigger()` for
    ComfyUI-built graphs.

    Critical: unlike the :8100 FP8 server (which auto-prepends the trigger
    via `_styled()`), **ComfyUI does not** -- its saved workflows have the
    trigger typed into the prompt text by hand (plan §7). A Custom-mode
    prompt built here that skips this silently renders off-style.

    Idempotent: if `prompt` already starts with a known trigger (as every
    saved workflow's prompt already does), it is returned unchanged -- so it
    is always safe to call on any prompt regardless of origin, not just
    "Custom" mode ones.

    This is the one function in this module that touches the filesystem (it
    reads the styles.json registry); everything else here is a pure,
    no-I/O transform. Falls back to a minimal built-in registry if
    styles.json can't be read, rather than raising.
    """
    registry = _load_style_registry(registry_path)
    p = (prompt or "").strip()
    low = p.lower()
    for s in registry["styles"].values():
        if low.startswith(s["trigger"].lower()):
            return p  # already triggered
    hint = (style_hint or "").strip()
    key = _resolve_style_key(hint, registry) if hint else _resolve_style_key(p, registry)
    trigger = registry["styles"][key]["trigger"]
    return f"{trigger}, {p}"


# ---------------------------------------------------------------------------
# F. filter_safetensors / resolve_lora_entries -- DEFECT 1 fix: registry
# comfy_name vs ComfyUI's LIVE LoraLoaderModelOnly list
# ---------------------------------------------------------------------------
#
# data/studio/scripts/loras.json's `comfy_name` values were authored in the
# NESTED form ("toei90s_zbase_v4_1/toei90s_zbase_v4_1.safetensors") because
# data/studio/comfy/extra_model_paths.yaml was widened to add a recursive
# "training" root -- but ComfyUI only re-reads that YAML at its OWN startup,
# and a live query against a running v0.29.0 server (GET /object_info/
# LoraLoaderModelOnly -> input.required.lora_name[0]) shows it currently
# reports 30 entries, ALL BARE FILENAMES ("tetsuya_char_v5_000006000
# .safetensors"), zero nested-path entries -- so every curated registry entry
# would 404 inside ComfyUI until that restart happens, and nothing in this
# codebase's tooling can trigger that restart. The functions below make LoRA
# selection work correctly BOTH before and after the eventual restart, with
# no manual intervention and no redeploy -- see routes/comfy_routes.py's
# /api/comfy/options and /api/comfy/generate for where these are called.

def filter_safetensors(names: Optional[list]) -> list[str]:
    """Keep only real adapter files. ComfyUI's live LoraLoaderModelOnly list
    also includes junk like "optimizer.pt" (a training-run checkpoint
    artifact that happens to sit in a mapped models dir) -- never a valid
    LoRA to load, so it must never reach a "show all detected" fallback
    dropdown or the resolution match set below."""
    return [str(n) for n in (names or []) if str(n).lower().endswith(".safetensors")]


def resolve_lora_entries(entries: Optional[list], live_names: Optional[list]) -> list[dict]:
    """Add `available` (bool) and `resolved_name` (str | None) to each dict in
    `entries` (anything with a "comfy_name" key -- both
    data/studio/scripts/loras.json's curated registry rows AND a raw
    POST /api/comfy/generate request's params["loras"] list share this shape),
    resolved against `live_names` (ComfyUI's LIVE, CURRENT LoraLoaderModelOnly
    `lora_name` combo options -- see module note above).

    Three-way resolution, in order, per entry:
      1. EXACT match in `live_names` -> resolved_name = the entry's own
         comfy_name unchanged, available = True. This is the steady-state
         case once ComfyUI is eventually restarted with the widened
         extra_model_paths.yaml and starts reporting nested paths itself.
      2. Else, a BASENAME match (os.path.basename of the entry's comfy_name
         against os.path.basename of each live name) -> resolved_name = the
         LIVE entry's OWN exact string (never the registry's) -- this is what
         makes a nested-path registry entry
         ("toei90s_zbase_v4_1/toei90s_zbase_v4_1.safetensors") resolve TODAY
         against a pre-restart ComfyUI that only knows the bare filename
         ("toei90s_zbase_v4_1.safetensors"), and it is exactly what a
         LoraLoaderModelOnly node's `lora_name` must be spelled as for
         ComfyUI to accept it (a combo value ComfyUI itself didn't declare is
         a validation failure at POST /prompt time, not a warning). If more
         than one live name shares that basename (a namespace collision
         across two mapped roots), the FIRST one encountered in `live_names`
         wins -- deterministic, not an error, since the whole point of the
         multi-root mapping is that a file can legitimately be reachable
         under more than one name.
      3. Else: available = False, resolved_name = None -- there is no string
         this entry could be submitted as that ComfyUI would currently
         accept; the caller (the /api/comfy/options route, and genParams.js
         after it) must refuse to offer or submit it rather than let it fail
         deep inside a queued ComfyUI job.

    Pure function, no I/O -- returns NEW dicts (shallow copy + 2 added keys);
    never mutates `entries`. Every other key on each input entry (name, key,
    kind, weight, role, ...) is preserved verbatim, so this works identically
    whether it's called on loras.json's registry rows or on a generate
    request's wire-format LoRA list.
    """
    live_list = [str(n) for n in (live_names or [])]
    live_set = set(live_list)
    live_by_basename: dict[str, str] = {}
    for live in live_list:
        base = os.path.basename(live)
        live_by_basename.setdefault(base, live)

    out = []
    for entry in entries or []:
        entry_dict = entry if isinstance(entry, dict) else {}
        comfy_name = str(entry_dict.get("comfy_name") or "")
        if comfy_name and comfy_name in live_set:
            resolved, available = comfy_name, True
        else:
            match = live_by_basename.get(os.path.basename(comfy_name)) if comfy_name else None
            if match:
                resolved, available = match, True
            else:
                resolved, available = None, False
        out.append({**entry_dict, "available": available, "resolved_name": resolved})
    return out
