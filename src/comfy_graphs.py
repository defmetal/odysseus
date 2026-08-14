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
        "unet": str, "clip": str, "clip_type": str, "vae": str,
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

-- Canonical MiniMax H3 video params dict (build_minimax_h3_graph) ----------
A THIRD, deliberately SMALLER video shape -- H3 is structurally not a
KSampler(Advanced) pipeline at all (see build_minimax_h3_graph()'s own header
comment), so this reuses only the keys that genuinely still apply:

    {
        "prompt": str,                   # goes straight into
                                          # MiniMaxH3ImageToVideo as a STRING
                                          # -- there is no CLIPTextEncode in
                                          # this graph at all, so unlike the
                                          # other two shapes there is no
                                          # "negative_prompt" key here (never
                                          # read -- H3's BasicGuider is
                                          # unguided/CFG-free, so there is no
                                          # "cfg" key either).
        "input_image": str | None,       # FIRST frame -- same key/shape as
                                          # the other two builders' own start
                                          # frame (comfy_image_ref() via
                                          # input_image_subfolder/_type), so
                                          # routes/comfy_routes.py's existing
                                          # upload/gallery-passthrough bridge
                                          # needs no H3-specific change.
        "last_frame": str | None,        # OPTIONAL end frame -- a SECOND,
                                          # independent image slot (same
                                          # comfy_image_ref() shape via
                                          # "last_frame_subfolder"/"_type").
        "seconds": float,                # duration -- converted to
                                          # MiniMaxH3ImageToVideo's `length`
                                          # (raw frame count) via _h3_length().
        "width": int, "height": int,
        "fps": int | float,              # CreateVideo "fps" (24, not Wan's 16)
        "seed": int | None, "randomize_seed": bool,
        "steps": int,                    # BasicScheduler "steps"
        "sampler": str,                  # KSamplerSelect "sampler_name"
        "scheduler": str,                # BasicScheduler "scheduler"
        "shift_video": float, "shift_audio": float,
                                          # MiniMaxH3SigmaShift's two floats.
                                          # That node is OMITTED from the
                                          # graph entirely unless at least one
                                          # of these differs from its own
                                          # default (12.0 / 3.0) -- see
                                          # build_minimax_h3_graph().
        "unet": str, "clip": str,        # same keys the other two builders
        "vae": str,                      # use (vae = the VIDEO vae here).
        "audio_vae": str,                # the SECOND, audio VAELoader --
                                          # H3's headline feature is
                                          # synchronized audio generated in
                                          # the SAME pass; there is no
                                          # "no audio" mode for this graph,
                                          # so unlike "vae" this has no
                                          # meaningful use anywhere else.
        "filename_prefix": str,
    }

`build_minimax_h3_graph(params)` / `introspect_graph(...)["params"]`
round-trip this shape too (introspect_graph() gained a dedicated branch for
H3's SamplerCustomAdvanced chain -- see that function's own comments) -- see
tests/test_comfy_graphs.py's H3 round-trip test.
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
    # None == "caller said nothing" -> fall back to the studio anti-text
    # negative. "" == "caller explicitly wants NO negative prompt" -> honour it.
    # These are deliberately NOT the same. Collapsing them (the original
    # behaviour) forced the anti-text negative onto every render, including the
    # general-purpose presets in models.json whose defaults set it to "".
    # That is actively harmful for Qwen-Image, whose headline strength is
    # rendering readable text -- suppressing "text, letters, lettering" is the
    # exact opposite of what you'd want from it.
    negative_prompt = params.get("negative_prompt")
    negative_prompt = DEFAULT_NEGATIVE_PROMPT if negative_prompt is None else str(negative_prompt)

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
    # Model-preset registry key (data/studio/scripts/models.json): "lumina2"
    # for Z-Image, "qwen_image" for Qwen-Image. NOT a cosmetic default --
    # getting this wrong silently produces garbage (models.json's own
    # _comment), so it must be a real per-request field, not hardcoded.
    clip_type = str(params.get("clip_type") or DEFAULT_CLIP_TYPE)
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
        "inputs": {"clip_name": clip, "type": clip_type, "device": DEFAULT_CLIP_DEVICE},
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
# C. build_minimax_h3_graph -- MiniMax H3 (omni-modal video + synchronized
#    audio). Every node shape/default below comes from a LIVE /object_info
#    dump against a running ComfyUI 0.30.1 plus ComfyUI's own bundled
#    template "video_minimax_h3_i2v.json" ("Image to Video (MiniMax H3)"
#    subgraph, read in full) -- not re-derived or guessed here. See
#    data/studio/PLAN-IMAGE-VIDEO-TABS.md sections 14.11/14.13 for the model files
#    chosen and why (fl2va pruned-INT8 + nvfp4 text encoder, ~42.5GB); the
#    exact graph wiring below is from that separate live-introspection pass,
#    which supersedes any node list in the plan doc itself.
#
# Structurally NOT a KSampler/KSamplerAdvanced pipeline like the two builders
# above -- H3 uses ComfyUI's "custom sampler" node family (BasicGuider /
# RandomNoise / KSamplerSelect / BasicScheduler / SamplerCustomAdvanced), and
# the prompt goes straight into MiniMaxH3ImageToVideo as a STRING. Two
# load-bearing differences from every other builder in this module, both
# deliberate, verified against the template, not omissions:
#   - NO negative prompt, NO cfg anywhere -- BasicGuider is unguided/CFG-free.
#     params["negative_prompt"] / params["cfg"] are never read by this
#     function, and neither key (nor a "negative" CONDITIONING input) ever
#     appears in the built graph.
#   - TWO VAEs, both REQUIRED (video + audio) -- H3 generates synchronized
#     audio in the SAME pass; dropping the audio VAE silently loses the
#     audio track, which is H3's headline feature over Wan.
# ---------------------------------------------------------------------------

DEFAULT_H3_UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
DEFAULT_H3_UNET_WEIGHT_DTYPE = "default"
DEFAULT_H3_CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
DEFAULT_H3_CLIP_TYPE = "minimax"  # NOT "lumina2"/"qwen_image"/"wan" -- a distinct, verified live CLIPLoader "type" combo value
DEFAULT_H3_CLIP_DEVICE = "default"
DEFAULT_H3_VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
DEFAULT_H3_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
# The official 0.4MP 16:9 preview baseline, and the ONLY configuration
# verified end-to-end on this machine (2026-08-06: 864x480 / 124 frames /
# 20 steps produced MiniMax_H3_00001_.mp4 with both video and audio tracks).
# 1344x768 is H3's native canvas and is offered in models.json, but it is a
# "scale up after a clean run" target, not a safe default.
DEFAULT_H3_WIDTH = 864
DEFAULT_H3_HEIGHT = 480
DEFAULT_H3_SECONDS = 5.0  # -> _h3_length() = 124, the trained-range minimum
DEFAULT_H3_FPS = 24.0
DEFAULT_H3_STEPS = 20
DEFAULT_H3_SAMPLER = "res_multistep"
DEFAULT_H3_SCHEDULER = "simple"
DEFAULT_H3_SHIFT_VIDEO = 12.0
DEFAULT_H3_SHIFT_AUDIO = 3.0
DEFAULT_H3_FILENAME_PREFIX = "video/MiniMax_H3"
DEFAULT_H3_FORMAT = "auto"
DEFAULT_H3_CODEC = "auto"


# MiniMax H3's trained frame range starts around 124 frames (ComfyUI's own
# guidance is "5-15 requested seconds" @ 24fps). Below this the model does not
# produce a short clip -- it fails in VAEDecodeAudio with a device-mismatch
# error that looks nothing like "your duration is too short". Proven
# 2026-08-06: length=56 fails, length=124 succeeds on the identical graph.
_H3_MIN_FRAMES = 124


def _h3_length(seconds: float) -> int:
    """seconds -> MiniMax H3's `length` (frame count) input, satisfying the
    mod-17 constraint ComfyUI's own bundled template computes
    ("video_minimax_h3_i2v.json", "Image to Video (MiniMax H3)" subgraph) --
    NOT something invented here. Frontend/model-preset callers work in
    DURATION (seconds), not a raw frame count, so this is the one place that
    conversion happens -- mirroring how _wan_step_split() above is the one
    place a single "total steps" value gets split across Wan's two sampler
    stages.

        L = max(5, round(seconds * 24))     # requested frames @ 24fps, floored at 5
        length = L + ((5 - (L % 17)) % 17)  # round UP to the next L % 17 == 5

    Verified against the template's own worked example: seconds=2 -> L=48 ->
    48 % 17 == 14 -> (5 - 14) % 17 == 8 -> length=56. The resulting `length`
    always satisfies `length % 17 == 5` for ANY non-negative `L` -- a plain
    modular-arithmetic identity of the formula itself, not something that
    depends on round()'s tie-breaking rule.

    TRAINED-RANGE FLOOR (added 2026-08-06, proven empirically):
    H3's documented trained range is ~124-362 frames, and ComfyUI's guidance
    is "5-15 requested seconds". Asking for LESS does not merely give a short
    clip -- it fails, deep in the graph, with a misleading error. Every run at
    seconds=2 (length=56) died in VAEDecodeAudio with
    "Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor)
    should be the same", which reads like a device-placement bug and sent us
    chasing --gpu-only for hours (PLAN-IMAGE-VIDEO-TABS.md 14.14/14.17).
    The identical graph at seconds=5 (length=124) succeeds, audio track and
    all. So: clamp UP to the trained minimum rather than letting a caller
    request a length the model cannot produce. Clamping is right here instead
    of raising -- a slightly longer clip is a far better outcome than an
    opaque failure 6 minutes into a render.
    """
    L = max(_H3_MIN_FRAMES, round(seconds * 24))
    return L + ((5 - (L % 17)) % 17)


def build_minimax_h3_graph(params: dict) -> dict:
    """Build a ComfyUI API-format graph for MiniMax H3 image-to-video (task
    spec's "THE AUTHORITATIVE GRAPH", dumped from ComfyUI's own bundled
    template) from a plain params dict (see the module docstring's
    "Canonical MiniMax H3 video params dict" section for the shape). Reuses
    as many keys/shapes from the image/Wan params dicts as apply -- `prompt`,
    `input_image` (+ `_subfolder`/`_type`, the FIRST frame -- same key
    build_image_graph() and build_wan_i2v_graph() already use for their own
    single start-frame slot, so routes/comfy_routes.py's existing upload/
    gallery-passthrough bridge (_resolve_input_image()) needs no H3-specific
    change), `seed`, `randomize_seed`, `width`, `height`, `fps`,
    `filename_prefix`, `unet`, `clip`, `vae` mean exactly what they mean
    elsewhere. What's new:

      "last_frame" (+ "_subfolder"/"_type")
                                        -- OPTIONAL end frame, same
                                           comfy_image_ref() shape as
                                           input_image but a SECOND,
                                           independent LoadImage node (H3's
                                           MiniMaxH3ImageToVideo takes both
                                           first_frame and last_frame as
                                           separate optional IMAGE inputs --
                                           neither is required, so a
                                           text-only call with neither is
                                           still valid, same "every field has
                                           a default" contract as the other
                                           two builders).
      "seconds"                        -- duration; converted to the
                                           `length` frame count via
                                           _h3_length() (task spec: expose
                                           DURATION IN SECONDS, not a raw
                                           frame count).
      "audio_vae"                      -- the SECOND VAELoader (the video
                                           vae is the existing "vae" key).
                                           Both are REQUIRED -- there is no
                                           "no audio" mode for this graph.
      "shift_video" / "shift_audio"    -- MiniMaxH3SigmaShift's two floats.
                                           The node is OMITTED entirely
                                           (UNETLoader feeds BasicGuider and
                                           BasicScheduler directly) UNLESS at
                                           least one differs from its own
                                           default (12.0 / 3.0) -- verified:
                                           "MiniMaxH3SigmaShift is NOT in the
                                           [authoritative] template."

    Deliberately does NOT read params["negative_prompt"] or params["cfg"] --
    this pipeline structurally has neither (see the section header comment
    above). Deliberately does NOT call apply_style_trigger() and offers no
    LoRA stack at all -- like Wan, H3 has no knowledge of the studio style/
    character LoRAs, and unlike Wan it has no LoRA slot whatsoever (no
    LoraLoaderModelOnly anywhere in the authoritative template).

    Pure function: no I/O, no network, no randomness unless the caller asks
    for a random seed (params["randomize_seed"] truthy, or no seed given) --
    same seed-resolution contract as the other two builders.
    """
    prompt = str(params.get("prompt") or "")

    input_image = params.get("input_image") or None
    input_image_subfolder = str(params.get("input_image_subfolder") or "")
    input_image_type = str(params.get("input_image_type") or "input")

    last_frame = params.get("last_frame") or None
    last_frame_subfolder = str(params.get("last_frame_subfolder") or "")
    last_frame_type = str(params.get("last_frame_type") or "input")

    width = int(params.get("width") or DEFAULT_H3_WIDTH)
    height = int(params.get("height") or DEFAULT_H3_HEIGHT)
    seconds = params.get("seconds")
    seconds = float(seconds) if seconds is not None else DEFAULT_H3_SECONDS
    length = _h3_length(seconds)

    fps = params.get("fps")
    fps = float(fps) if fps is not None else DEFAULT_H3_FPS

    seed = params.get("seed")
    if params.get("randomize_seed") or seed is None:
        seed = random.randint(0, _SEED_MAX)
    else:
        seed = int(seed)

    steps = int(params.get("steps") or DEFAULT_H3_STEPS)
    sampler = str(params.get("sampler") or DEFAULT_H3_SAMPLER)
    scheduler = str(params.get("scheduler") or DEFAULT_H3_SCHEDULER)

    # MiniMaxH3SigmaShift is inserted ONLY when the caller's value differs
    # from the node's own default -- "UNETLoader feeds BasicGuider and
    # BasicScheduler directly" in the authoritative (no-override) template.
    shift_video_in = params.get("shift_video")
    shift_audio_in = params.get("shift_audio")
    needs_sigma_shift = (
        (shift_video_in is not None and float(shift_video_in) != DEFAULT_H3_SHIFT_VIDEO)
        or (shift_audio_in is not None and float(shift_audio_in) != DEFAULT_H3_SHIFT_AUDIO)
    )
    shift_video = float(shift_video_in) if shift_video_in is not None else DEFAULT_H3_SHIFT_VIDEO
    shift_audio = float(shift_audio_in) if shift_audio_in is not None else DEFAULT_H3_SHIFT_AUDIO

    unet = str(params.get("unet") or DEFAULT_H3_UNET)
    clip = str(params.get("clip") or DEFAULT_H3_CLIP)
    video_vae = str(params.get("vae") or DEFAULT_H3_VIDEO_VAE)
    audio_vae = str(params.get("audio_vae") or DEFAULT_H3_AUDIO_VAE)
    filename_prefix = str(params.get("filename_prefix") or DEFAULT_H3_FILENAME_PREFIX)

    nid = _new_id_gen()
    graph: dict[str, Any] = {}

    unet_id = nid()
    graph[unet_id] = {
        "class_type": "UNETLoader",
        "inputs": {"unet_name": unet, "weight_dtype": DEFAULT_H3_UNET_WEIGHT_DTYPE},
    }
    model_link = [unet_id, 0]

    if needs_sigma_shift:
        shift_id = nid()
        graph[shift_id] = {
            "class_type": "MiniMaxH3SigmaShift",
            "inputs": {"model": model_link, "shift_video": shift_video, "shift_audio": shift_audio},
        }
        model_link = [shift_id, 0]

    clip_id = nid()
    graph[clip_id] = {
        "class_type": "CLIPLoader",
        "inputs": {"clip_name": clip, "type": DEFAULT_H3_CLIP_TYPE, "device": DEFAULT_H3_CLIP_DEVICE},
    }
    clip_link = [clip_id, 0]

    # Video VAE MUST be created before the audio one -- introspect_graph()
    # relies on VAELoader insertion order to tell them apart (vae_ids[0] ==
    # video, vae_ids[1] == audio), the SAME convention build_wan_i2v_graph()
    # already established for its two UNETLoaders (unet_ids[0]==high/primary,
    # [1]==low).
    video_vae_id = nid()
    graph[video_vae_id] = {"class_type": "VAELoader", "inputs": {"vae_name": video_vae}}
    video_vae_link = [video_vae_id, 0]

    audio_vae_id = nid()
    graph[audio_vae_id] = {"class_type": "VAELoader", "inputs": {"vae_name": audio_vae}}
    audio_vae_link = [audio_vae_id, 0]

    h3_inputs: dict[str, Any] = {
        "clip": clip_link,
        "vae": video_vae_link,
        "prompt": prompt,
        "width": width,
        "height": height,
        "length": length,
    }
    if input_image:
        first_load_id = nid()
        graph[first_load_id] = {
            "class_type": "LoadImage",
            "inputs": {"image": comfy_image_ref(input_image, input_image_subfolder, input_image_type)},
        }
        h3_inputs["first_frame"] = [first_load_id, 0]
    if last_frame:
        last_load_id = nid()
        graph[last_load_id] = {
            "class_type": "LoadImage",
            "inputs": {"image": comfy_image_ref(last_frame, last_frame_subfolder, last_frame_type)},
        }
        h3_inputs["last_frame"] = [last_load_id, 0]

    h3_id = nid()
    graph[h3_id] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": h3_inputs}
    # (CONDITIONING, LATENT) -- verified output order (task spec).
    h3_conditioning, h3_latent = [h3_id, 0], [h3_id, 1]

    guider_id = nid()
    graph[guider_id] = {
        "class_type": "BasicGuider",
        "inputs": {"model": model_link, "conditioning": h3_conditioning},
    }

    noise_id = nid()
    graph[noise_id] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}

    sampler_sel_id = nid()
    graph[sampler_sel_id] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": sampler}}

    scheduler_id = nid()
    graph[scheduler_id] = {
        "class_type": "BasicScheduler",
        # denoise is always 1.0 -- the authoritative template's own literal
        # value, not something exposed as a param (H3 is guided by
        # conditioning, not a partial-noise img2img-style denoise).
        "inputs": {"model": model_link, "scheduler": scheduler, "steps": steps, "denoise": 1.0},
    }

    adv_id = nid()
    graph[adv_id] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": [noise_id, 0],
            "guider": [guider_id, 0],
            "sampler": [sampler_sel_id, 0],
            "sigmas": [scheduler_id, 0],
            "latent_image": h3_latent,
        },
    }

    decode_id = nid()
    graph[decode_id] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": [adv_id, 0], "vae": video_vae_link},
    }
    decode_audio_id = nid()
    graph[decode_audio_id] = {
        "class_type": "VAEDecodeAudio",
        "inputs": {"samples": [adv_id, 0], "vae": audio_vae_link},
    }

    video_id = nid()
    graph[video_id] = {
        "class_type": "CreateVideo",
        "inputs": {"images": [decode_id, 0], "audio": [decode_audio_id, 0], "fps": fps},
    }

    save_id = nid()
    graph[save_id] = {
        "class_type": "SaveVideo",
        "inputs": {
            "video": [video_id, 0],
            "filename_prefix": filename_prefix,
            "format": DEFAULT_H3_FORMAT,
            "codec": DEFAULT_H3_CODEC,
        },
    }

    return graph


# ---------------------------------------------------------------------------
# D. introspect_graph
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
    # MiniMax H3 (build_minimax_h3_graph()) -- its own "custom sampler" node
    # family, entirely distinct from KSampler(Advanced).
    "MiniMaxH3ImageToVideo", "BasicGuider", "RandomNoise", "KSamplerSelect",
    "BasicScheduler", "SamplerCustomAdvanced", "VAEDecodeAudio", "MiniMaxH3SigmaShift",
    # MiniMax Music 3 (build_minimax_music3_graph()) -- audio, not H3 video.
    "MiniMaxMusic3TextEncode", "EmptyMiniMaxMusic3LatentAudio",
    "SaveAudioMP3", "SaveAudio", "SaveAudioAdvanced",
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
    # DEFECT 19: H3 has no negative-prompt concept at all (module docstring;
    # BasicGuider is unguided/CFG-free) -- set True in the H3 branch below so
    # the final params["negative_prompt"] assignment can be skipped entirely
    # for it, rather than always emitting "" (which contradicts this
    # module's own docstring and looks like a real, empty-but-present
    # negative prompt to a caller).
    h3_no_negative = False
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

    elif by_type.get("SamplerCustomAdvanced"):
        h3_no_negative = True
        # MiniMax H3's CFG-free custom-sampler chain (build_minimax_h3_graph())
        # -- no single node carries steps/cfg/seed/prompt together the way
        # KSampler(Advanced) does; each lives on a DISTINCT upstream node,
        # reached by following SamplerCustomAdvanced's own typed inputs
        # (noise/sampler/sigmas/guider). Deliberately sets NO "cfg" and NO
        # "denoise" key at all here (left absent, not None/0) -- H3
        # structurally has neither (BasicGuider is unguided/CFG-free; see
        # build_minimax_h3_graph()'s header comment).
        adv_in = nodes[by_type["SamplerCustomAdvanced"][0]].get("inputs", {})

        noise_ref = adv_in.get("noise")
        if isinstance(noise_ref, list) and noise_ref and noise_ref[0] in nodes:
            params["seed"] = nodes[noise_ref[0]].get("inputs", {}).get("noise_seed")

        sampler_ref = adv_in.get("sampler")
        if isinstance(sampler_ref, list) and sampler_ref and sampler_ref[0] in nodes:
            params["sampler"] = nodes[sampler_ref[0]].get("inputs", {}).get("sampler_name")

        sigmas_ref = adv_in.get("sigmas")
        if isinstance(sigmas_ref, list) and sigmas_ref and sigmas_ref[0] in nodes:
            sched_in = nodes[sigmas_ref[0]].get("inputs", {})
            params["scheduler"] = sched_in.get("scheduler")
            params["steps"] = sched_in.get("steps")

        # Positive prompt: BasicGuider.conditioning -> MiniMaxH3ImageToVideo's
        # OWN "prompt" STRING input -- NOT a CLIPTextEncode chase (H3 has no
        # CLIPTextEncode at all), so _resolve_conditioning_text() (which only
        # knows how to find a "text" field) does not apply here.
        guider_ref = adv_in.get("guider")
        if isinstance(guider_ref, list) and guider_ref and guider_ref[0] in nodes:
            cond_ref = nodes[guider_ref[0]].get("inputs", {}).get("conditioning")
            if isinstance(cond_ref, list) and cond_ref and cond_ref[0] in nodes:
                h3_node = nodes[cond_ref[0]]
                if h3_node.get("class_type") == "MiniMaxH3ImageToVideo":
                    positive_text = str(h3_node.get("inputs", {}).get("prompt") or "")

    params["prompt"] = positive_text
    if not h3_no_negative:
        params["negative_prompt"] = negative_text

    encode_ids = by_type.get("MiniMaxMusic3TextEncode") or []
    if encode_ids:
        enc_in = nodes[encode_ids[0]].get("inputs", {})
        params["caption"] = enc_in.get("caption") or params.get("prompt") or ""
        params["lyrics"] = enc_in.get("lyrics") or ""
        params["max_duration"] = enc_in.get("max_duration")
        params["seconds"] = enc_in.get("max_duration")
        params["cfg_scale"] = enc_in.get("cfg_scale")
        params["top_k"] = enc_in.get("top_k")
        if enc_in.get("seed") is not None:
            params["seed"] = enc_in.get("seed")
        if not params.get("prompt"):
            params["prompt"] = params["caption"]

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
        clip_inputs = nodes[clip_ids[0]].get("inputs", {})
        params["clip"] = clip_inputs.get("clip_name")
        # "type" is the CLIPLoader combo picking the text-encoder family
        # ("lumina2" for Z-Image, "qwen_image" for Qwen-Image, "wan" for the
        # video pipeline). Round-tripping it matters for the same reason
        # build_image_graph() takes it as a real param now, not a hardcoded
        # constant: getting it wrong silently produces garbage, not an error.
        params["clip_type"] = clip_inputs.get("type")

    vae_ids = by_type.get("VAELoader") or []
    if vae_ids:
        params["vae"] = nodes[vae_ids[0]].get("inputs", {}).get("vae_name")
        if len(vae_ids) > 1:
            # MiniMax H3's second VAELoader (build_minimax_h3_graph()) is the
            # AUDIO vae, inserted right after the video one -- same "surfaced
            # under its own key so the single-VAE contract for every other
            # graph is unchanged" pattern as unet/unet_low above.
            params["audio_vae"] = nodes[vae_ids[1]].get("inputs", {}).get("vae_name")

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

    # MiniMax H3's own width/height/length + optional first/last frame --
    # SAME "separate `if`, not `elif`" reasoning as WanImageToVideo above (a
    # LoadImage for first_frame/last_frame must not shadow this via the
    # earlier EmptySD3LatentImage/LoadImage elif chain).
    if by_type.get("MiniMaxH3ImageToVideo"):
        hi = nodes[by_type["MiniMaxH3ImageToVideo"][0]].get("inputs", {})
        params["width"] = hi.get("width")
        params["height"] = hi.get("height")
        length = hi.get("length")
        params["frames"] = length
        # DEFECT 19: recover `seconds` (the duration a re-roll needs) from
        # `length` -- _h3_length() rounds UP to satisfy H3's own mod-17
        # constraint, so this is not a lossless inverse of "what the caller
        # originally typed"; it's the actual, exact duration the graph will
        # render (length / fps), which is the more honest value to hand back
        # anyway. Reads the CreateVideo node's own fps directly (rather than
        # relying on params["fps"], set later below) so this doesn't depend
        # on this block's position relative to that one.
        create_video_ids = by_type.get("CreateVideo") or []
        fps_for_seconds = (
            nodes[create_video_ids[0]].get("inputs", {}).get("fps") if create_video_ids else None
        )
        fps_for_seconds = float(fps_for_seconds) if fps_for_seconds else DEFAULT_H3_FPS
        if isinstance(length, (int, float)):
            params["seconds"] = length / fps_for_seconds
        # DEFECT 11: with only last_frame set, the earlier EmptySD3LatentImage
        # /LoadImage `elif` chain above (there being no EmptySD3LatentImage in
        # an H3 graph) falls to `elif by_type.get("LoadImage")`, which grabs
        # LoadImage[0] -- the ONLY LoadImage node when first_frame is absent,
        # i.e. the END frame -- as `input_image`. Explicitly null it back out
        # here when first_frame is genuinely absent, instead of leaving that
        # wrong value in place.
        first_ref = hi.get("first_frame")
        if isinstance(first_ref, list) and first_ref and first_ref[0] in nodes:
            params["input_image"] = nodes[first_ref[0]].get("inputs", {}).get("image")
        else:
            params["input_image"] = None
        last_ref = hi.get("last_frame")
        if isinstance(last_ref, list) and last_ref and last_ref[0] in nodes:
            params["last_frame"] = nodes[last_ref[0]].get("inputs", {}).get("image")

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
# E. ui_to_api
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
        required_spec = node_input_spec.get("required") or {}
        optional_spec = node_input_spec.get("optional") or {}
        ordered_names = list(required_spec.keys()) + list(optional_spec.keys())
        combined_spec = {**required_spec, **optional_spec}

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

        # DEFECT 12: NOT every declared, non-linked input is a real widget --
        # an IMAGE/CONDITIONING/CLIP_VISION_OUTPUT/... typed OPTIONAL input
        # left disconnected (e.g. WanImageToVideo's clip_vision_output) has
        # NO widgets_values entry at all (ComfyUI's own frontend never writes
        # one for it), so treating it as positional over-counted the values
        # needed and raised "ran out of widgets_values" on an otherwise
        # perfectly normal export. Only a combo (spec type is a list) or a
        # primitive widget type actually gets a widgets_values slot.
        def _is_widget_input(name: str) -> bool:
            spec = combined_spec.get(name)
            if not isinstance(spec, (list, tuple)) or not spec:
                return False
            type_or_options = spec[0]
            if isinstance(type_or_options, list):
                return True  # combo dropdown
            return type_or_options in ("INT", "FLOAT", "STRING", "BOOLEAN")

        widget_names = [n for n in ordered_names if n not in resolved_links and _is_widget_input(n)]
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
# F. apply_style_trigger -- the one function here that touches the filesystem
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
# G. filter_safetensors / resolve_lora_entries -- DEFECT 1 fix: registry
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


# ---------------------------------------------------------------------------
# H. load_model_registry / resolve_model_entries / apply_model_preset --
# data/studio/scripts/models.json: "which model" as ONE user-facing choice
# ---------------------------------------------------------------------------
#
# models.json collapses unet+clip+clip_type+vae+loras+defaults into a single
# named preset (its own _comment has the full schema/rationale) so the Image
# tab becomes "pick a model, type a prompt, hit Generate" instead of eight
# separate knobs. This section mirrors the styles.json (_load_style_registry)
# and loras.json (resolve_lora_entries) patterns already in this module:
# pure, stdlib-only, degrade-don't-raise on a missing/malformed file.

_MODELS_REGISTRY_PATH = Path(__file__).resolve().parent.parent / "data" / "studio" / "scripts" / "models.json"

# Used only if models.json is missing/malformed (e.g. a fresh clone -- data/
# is gitignored per CLAUDE.md) so callers degrade instead of crashing. A
# single safe, always-available preset: plain Z-Image, no LoRAs, no style
# trigger -- the same shape as models.json's own "zimage_general" entry, but
# built from this module's existing DEFAULT_* constants rather than
# duplicating literals that could drift out of sync with build_image_graph().
_FALLBACK_MODEL_REGISTRY = {
    "default": "zimage_general",
    "models": {
        "zimage_general": {
            "name": "Z-Image (base)",
            "group": "General",
            "description": "Plain Z-Image with no LoRAs and no style trigger.",
            "arch": "z_image",
            "unet": DEFAULT_UNET,
            "clip": DEFAULT_CLIP,
            "clip_type": DEFAULT_CLIP_TYPE,
            "vae": DEFAULT_VAE,
            "loras": [],
            "style_trigger": False,
            "characters_allowed": False,
            "enabled": True,
            "sizes": [f"{DEFAULT_WIDTH}x{DEFAULT_HEIGHT}"],
            "defaults": {
                "steps": DEFAULT_STEPS, "cfg": DEFAULT_CFG, "sampler": DEFAULT_SAMPLER,
                "scheduler": DEFAULT_SCHEDULER, "shift": DEFAULT_SHIFT,
                "size": f"{DEFAULT_WIDTH}x{DEFAULT_HEIGHT}", "negative_prompt": "",
            },
        },
    },
    # Video counterpart of the image fallback above -- same rationale: a
    # single safe, always-available preset (Wan, not H3 -- H3's ~42.5GB of
    # weights are far less likely to exist on a fresh clone/dev machine than
    # Wan's, mirroring why "zimage_general" rather than "studio_toei" (needs
    # a LoRA file) was chosen as the IMAGE fallback), built from this
    # module's existing DEFAULT_WAN_* constants so it can't drift out of sync
    # with build_wan_i2v_graph()'s own defaults.
    "default_video_model": "wan22_i2v",
    "video_models": {
        "wan22_i2v": {
            "name": "Wan 2.2 (I2V)",
            "group": "Video",
            "description": "Image-to-video, no audio.",
            "engine": "wan22_i2v",
            "unet": DEFAULT_WAN_UNET_HIGH,
            "unet_low": DEFAULT_WAN_UNET_LOW,
            "clip": DEFAULT_WAN_CLIP,
            "clip_type": DEFAULT_WAN_CLIP_TYPE,
            "vae": DEFAULT_WAN_VAE,
            "has_audio": False,
            "enabled": True,
            "sizes": [f"{DEFAULT_WAN_WIDTH}x{DEFAULT_WAN_HEIGHT}"],
            "defaults": {
                "steps": DEFAULT_WAN_STEPS, "cfg": DEFAULT_WAN_CFG, "sampler": DEFAULT_WAN_SAMPLER,
                "scheduler": DEFAULT_WAN_SCHEDULER, "shift": DEFAULT_WAN_SHIFT, "fps": DEFAULT_WAN_FPS,
                "size": f"{DEFAULT_WAN_WIDTH}x{DEFAULT_WAN_HEIGHT}", "negative_prompt": DEFAULT_WAN_NEGATIVE_PROMPT,
            },
        },
    },
}


def load_model_registry(path: Optional[str] = None) -> dict:
    """Read data/studio/scripts/models.json -- the model-preset registry (see
    that file's own _comment for the full per-entry schema). Mirrors
    _load_style_registry()'s degradation contract exactly: a missing,
    unreadable, or structurally malformed file (no dict, no non-empty
    "models" dict) returns `_FALLBACK_MODEL_REGISTRY` instead of raising, so
    every caller (routes/comfy_routes.py's /api/comfy/options and /generate)
    always has at least one usable preset with no try/except of its own.
    """
    p = Path(path) if path else _MODELS_REGISTRY_PATH
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return _FALLBACK_MODEL_REGISTRY
    if not isinstance(data, dict) or not isinstance(data.get("models"), dict) or not data.get("models"):
        return _FALLBACK_MODEL_REGISTRY
    return data


def resolve_model_entries(entries: Optional[list], live: Optional[dict]) -> list[dict]:
    """Add `available` (bool) and `missing` (list[str]) to each dict in
    `entries` -- data/studio/scripts/models.json's preset rows, flattened to
    a list the same way routes/comfy_routes.py's `_load_curated_loras()`
    already flattens loras.json's dict-of-dicts into a list of entries each
    carrying its own "key" (mirror that shape here: one entry per preset,
    "key" = the models.json key, plus its unet/clip/vae/... fields verbatim).

    `live` is `{"unets": [...], "clips": [...], "vaes": [...]}` -- exactly
    ComfyUI's live UNETLoader.unet_name / CLIPLoader.clip_name /
    VAELoader.vae_name combo lists (routes/comfy_routes.py already has
    `_combo_options()` for pulling each of these out of a cached
    /object_info response). LoRAs are deliberately NOT checked here -- a
    preset's `loras` entries are `{"key": ..., "weight": ...}` referencing
    loras.json, not a live comfy_name directly, and resolving THOSE is
    apply_model_preset()'s + /api/comfy/generate's job (reusing
    resolve_lora_entries()), not this function's.

    For each of "unet"/"clip"/"vae" independently: the SAME exact ->
    basename -> unresolved fallback resolve_lora_entries() already uses
    (plan §14.6 -- ComfyUI reports bare filenames, registries/presets may
    hold nested paths). A preset is `available` only when every field it
    actually declares resolves; `missing` lists which of "unet"/"clip"/"vae"
    did not (e.g. the Qwen preset, whose ~30GB download is in progress per
    the task constraints, comes back `available: False, missing: ["unet",
    "clip", "vae"]` today and flips to `available: True, missing: []` once
    the files land AND ComfyUI is restarted to see them -- no code change
    either side of that event). A preset field left blank/unset is not
    treated as "missing" -- there is nothing to check.

    ALSO checks "unet_low" (Wan's second, low-noise UNETLoader -- build_
    wan_i2v_graph()) against the SAME "unets" live list, and "audio_vae"
    (MiniMax H3's second VAELoader -- build_minimax_h3_graph()) against the
    SAME "vaes" live list -- data/studio/scripts/models.json's `video_models`
    section (task item 3: "resolved for availability the same way image
    models are -- reuse resolve_model_entries"). An image preset never
    declares either field, so this is purely additive for it (the "field
    left blank/unset is not missing" rule above already covers "not
    declared at all").

    Pure function, no I/O -- returns NEW dicts (shallow copy + 2 added keys);
    never mutates `entries`. `live=None` (or missing individual keys) means
    nothing resolves for that slot, e.g. every preset with a non-empty field
    comes back unavailable -- callers that can't currently reach ComfyUI
    should NOT call this function at all and should instead degrade the same
    way /api/comfy/options' existing loras handling does (pass every entry
    through as available, since we cannot verify availability against a
    server we can't reach -- ComfyUI being down is a different, already-
    surfaced problem, see /api/comfy/status).
    """
    live = live or {}

    def _resolves(name: str, live_names: Optional[list]) -> bool:
        if not name:
            return True  # nothing declared for this slot -- nothing to check
        live_list = [str(n) for n in (live_names or [])]
        if name in live_list:
            return True
        base = os.path.basename(name)
        return any(os.path.basename(live_name) == base for live_name in live_list)

    out = []
    for entry in entries or []:
        entry_dict = entry if isinstance(entry, dict) else {}
        missing = [
            field for field, live_key in (
                ("unet", "unets"), ("clip", "clips"), ("vae", "vaes"),
                ("unet_low", "unets"), ("audio_vae", "vaes"),
            )
            if not _resolves(str(entry_dict.get(field) or ""), live.get(live_key))
        ]
        out.append({**entry_dict, "available": not missing, "missing": missing})
    return out


# Preset `defaults.*` keys apply_model_preset() copies straight into params
# under the SAME name when the caller hasn't already set that key. "size" is
# handled separately below (it expands to two params keys, width/height).
#
# The last five (fps/frames/seconds/shift_video/shift_audio) are VIDEO-only
# additions (task: reuse apply_model_preset() for data/studio/scripts/
# models.json's new `video_models` section too, rather than writing a second
# fill-function) -- purely additive: an IMAGE preset's `defaults` never
# declares any of them, so `field in defaults` is False and nothing changes
# for the image path. "cfg"/"shift"/"negative_prompt" are Wan-shaped (Wan
# reuses the exact same keys build_image_graph() already does); "seconds"/
# "shift_video"/"shift_audio" are H3-only; "fps" is shared by both video
# engines (image has no such concept at all).
_PRESET_DEFAULT_KEYS = (
    "steps", "cfg", "sampler", "scheduler", "shift", "negative_prompt",
    "fps", "frames", "seconds", "shift_video", "shift_audio",
)


def _parse_preset_size(size: Any) -> Optional[tuple[int, int]]:
    """"1216x672" -> (1216, 672). None for anything that isn't exactly
    WIDTHxHEIGHT of two ints -- defensive against a hand-edited models.json
    typo (this module never raises on a malformed registry, per its other
    degrade-not-crash functions)."""
    if not isinstance(size, str) or "x" not in size:
        return None
    w_str, _, h_str = size.partition("x")
    try:
        return int(w_str), int(h_str)
    except ValueError:
        return None


def apply_model_preset(params: Optional[dict], preset: Optional[dict]) -> dict:
    """Return a NEW params dict with `preset`'s unet/clip/clip_type/vae/loras
    and defaults filled in ONLY where `params` did not already specify a
    value -- models.json's own _comment: "Advanced still exposes every
    individual field, and any field a user overrides wins over the preset's
    default." Never mutates `params` or `preset`.

    Originally written for the IMAGE `models` registry section only; reused
    as-is (no video-specific fork) for the `video_models` section added
    alongside it -- `unet_low`/`audio_vae` (direct-copy fields) and
    `fps`/`frames`/`seconds`/`shift_video`/`shift_audio` (_PRESET_DEFAULT_KEYS)
    are video-only additions that are simply absent from every image preset's
    own data, so this is a strict superset of the original behaviour, not a
    behaviour change for the image path.

    "Caller did not already specify a value" uses the same falsy-still-gets-
    a-default contract build_image_graph() already applies throughout (e.g.
    `params.get("steps") or DEFAULT_STEPS`) -- a key that's absent, None, or
    "" all count as unset -- so this composes with that function rather than
    inventing a second, stricter definition.

    Fields filled from the preset, in order:
      - unet / clip / clip_type / vae -- direct copy from the preset's own
        top-level keys, when present on the preset and unset on `params`.
      - loras -- the preset's OWN loras list (models.json's registry form,
        `[{"key": ..., "weight": ...}, ...]` -- still KEYED, not yet
        resolved to a comfy_name; that resolution is the caller's job,
        exactly like loras.json's own registry rows are resolved elsewhere
        via resolve_lora_entries() -- see routes/comfy_routes.py's
        /api/comfy/generate). Filled only when `params["loras"]` is None
        (DEFECT 7, 2026-08) -- routes/comfy_routes.py's GenerateParams.loras
        is `Optional[List[LoraParam]] = None` specifically so an absent
        field (None) and a deliberate empty selection (`[]`) are
        distinguishable on the wire. An explicit `[]` now means "the caller
        deliberately chose zero LoRAs" (the UI does let a user uncheck every
        row) and is left alone, never refilled from the preset.
      - defaults.steps / .cfg / .sampler / .scheduler / .shift /
        .negative_prompt -- direct copy when that params key is unset.
      - defaults.size -- parsed "WIDTHxHEIGHT" (_parse_preset_size()) and
        expanded into `width`/`height` INDEPENDENTLY (each filled only when
        THAT specific key is unset), so a caller who set width but not
        height still gets the preset's height, not a silently mismatched
        pair, and vice versa.

    `preset=None`/`{}` returns an unmodified shallow copy of `params` (or
    `{}` if `params` is also falsy) -- always safe to call even before a
    model key has been resolved to a preset dict.
    """
    out = dict(params or {})
    preset = preset or {}

    def _unset(key: str) -> bool:
        # DEFECT 1: "" must NOT count as unset for negative_prompt -- the
        # frontend ALWAYS sends `model` (so this preset path is the default,
        # not an edge case), and clearing the Negative box in Advanced sends
        # negative_prompt="" on purpose (build_image_graph() already treats
        # "" as "caller explicitly wants no negative prompt", distinct from
        # None -- see its own docstring). Treating "" as unset here silently
        # refilled the studio anti-text negative right back in, defeating
        # that distinction entirely for every model-preset request.
        if key == "negative_prompt":
            return key not in out or out[key] is None
        return out.get(key) in (None, "")

    # "unet_low" (Wan's second, low-noise UNETLoader -- build_wan_i2v_graph())
    # and "audio_vae" (MiniMax H3's second VAELoader -- build_minimax_h3_graph())
    # are video-only additions to this same direct-copy loop, for the same
    # "purely additive" reason as _PRESET_DEFAULT_KEYS above: an image preset
    # never declares either key.
    for field in ("unet", "clip", "clip_type", "vae", "unet_low", "audio_vae"):
        if field in preset and _unset(field):
            out[field] = preset[field]

    # DEFECT 7: None (absent/never-set) means "the caller didn't specify
    # loras" and should be filled from the preset; [] is a DELIBERATE "the
    # user unchecked every LoRA row" and must be left as-is. The old
    # `not out.get("loras")` check was true for BOTH None and [], so
    # deselecting every LoRA silently re-added the preset's own.
    if out.get("loras") is None and preset.get("loras"):
        out["loras"] = [dict(lora) for lora in preset["loras"] if isinstance(lora, dict)]

    defaults = preset.get("defaults") or {}
    for field in _PRESET_DEFAULT_KEYS:
        if field in defaults and _unset(field):
            out[field] = defaults[field]

    parsed_size = _parse_preset_size(defaults.get("size")) if defaults.get("size") else None
    if parsed_size:
        width, height = parsed_size
        if _unset("width"):
            out["width"] = width
        if _unset("height"):
            out["height"] = height

    return out


# ---------------------------------------------------------------------------
# MiniMax Music 3 -- appended from origin/cursor/generic-generate-modes-dc98
# so studio image/video graphs stay intact. Audio, not H3 video.
# ---------------------------------------------------------------------------
DEFAULT_MUSIC3_UNET = "minimax_music3_dit_fp16.safetensors"
DEFAULT_MUSIC3_CLIP = "minimax_music3_text_encoder_pruned_int8_convrot.safetensors"
DEFAULT_MUSIC3_CLIP_TYPE = "stable_diffusion"
DEFAULT_MUSIC3_VAE = "minimax_music3_dav.safetensors"
DEFAULT_MUSIC3_SECONDS = 60.0
DEFAULT_MUSIC3_STEPS = 20
DEFAULT_MUSIC3_CFG = 1.0
DEFAULT_MUSIC3_SAMPLER = "euler"
DEFAULT_MUSIC3_SCHEDULER = "simple"
DEFAULT_MUSIC3_FILENAME_PREFIX = "audio/odysseus_music3"
DEFAULT_MUSIC3_CFG_SCALE = 1.5
DEFAULT_MUSIC3_TOP_K = 50


def build_minimax_music3_graph(params: dict) -> dict:
    """API-format graph for local MiniMax Music 3 in ComfyUI.

    Params (all optional):
      caption / prompt, lyrics, seconds / max_duration, seed, randomize_seed,
      steps, cfg, sampler, scheduler, cfg_scale, top_k, unet, clip, clip_type,
      vae, filename_prefix, tiled_decode.
    """
    caption = str(params.get("caption") or params.get("prompt") or "")
    lyrics = str(params.get("lyrics") or "")
    seconds = params.get("seconds")
    if seconds is None:
        seconds = params.get("max_duration")
    seconds = float(seconds) if seconds is not None else DEFAULT_MUSIC3_SECONDS
    seconds = max(0.04, min(seconds, 300.0))

    seed = params.get("seed")
    if params.get("randomize_seed") or seed is None:
        seed = random.randint(0, _SEED_MAX)
    else:
        seed = int(seed)

    steps = int(params.get("steps") or DEFAULT_MUSIC3_STEPS)
    cfg = float(params.get("cfg") if params.get("cfg") is not None else DEFAULT_MUSIC3_CFG)
    sampler = str(params.get("sampler") or DEFAULT_MUSIC3_SAMPLER)
    scheduler = str(params.get("scheduler") or DEFAULT_MUSIC3_SCHEDULER)
    cfg_scale = float(params.get("cfg_scale") if params.get("cfg_scale") is not None else DEFAULT_MUSIC3_CFG_SCALE)
    top_k = int(params.get("top_k") or DEFAULT_MUSIC3_TOP_K)
    unet = str(params.get("unet") or DEFAULT_MUSIC3_UNET)
    clip = str(params.get("clip") or DEFAULT_MUSIC3_CLIP)
    clip_type = str(params.get("clip_type") or DEFAULT_MUSIC3_CLIP_TYPE)
    vae = str(params.get("vae") or DEFAULT_MUSIC3_VAE)
    filename_prefix = str(params.get("filename_prefix") or DEFAULT_MUSIC3_FILENAME_PREFIX)

    nid = _new_id_gen()
    graph: dict[str, Any] = {}

    unet_id = nid()
    graph[unet_id] = {
        "class_type": "UNETLoader",
        "inputs": {"unet_name": unet, "weight_dtype": "default"},
    }
    clip_id = nid()
    graph[clip_id] = {
        "class_type": "CLIPLoader",
        "inputs": {"clip_name": clip, "type": clip_type, "device": "default"},
    }
    vae_id = nid()
    graph[vae_id] = {
        "class_type": "VAELoader",
        "inputs": {"vae_name": vae},
    }
    encode_id = nid()
    graph[encode_id] = {
        "class_type": "MiniMaxMusic3TextEncode",
        "inputs": {
            "clip": [clip_id, 0],
            "caption": caption,
            "lyrics": lyrics,
            "seed": seed,
            "max_duration": seconds,
            "cfg_scale": cfg_scale,
            "top_k": top_k,
        },
    }
    latent_id = nid()
    graph[latent_id] = {
        "class_type": "EmptyMiniMaxMusic3LatentAudio",
        "inputs": {"seconds": seconds, "batch_size": 1},
    }
    sampler_id = nid()
    graph[sampler_id] = {
        "class_type": "KSampler",
        "inputs": {
            "model": [unet_id, 0],
            "positive": [encode_id, 0],
            "negative": [encode_id, 0],
            "latent_image": [latent_id, 0],
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler,
            "scheduler": scheduler,
            "denoise": 1.0,
        },
    }
    decode_id = nid()
    decode_inputs: dict[str, Any] = {"samples": [sampler_id, 0], "vae": [vae_id, 0]}
    if params.get("tiled_decode"):
        decode_inputs["tiled"] = True
    graph[decode_id] = {"class_type": "VAEDecodeAudio", "inputs": decode_inputs}
    save_id = nid()
    graph[save_id] = {
        "class_type": "SaveAudioMP3",
        "inputs": {"audio": [decode_id, 0], "filename_prefix": filename_prefix, "quality": "V0"},
    }
    return graph
