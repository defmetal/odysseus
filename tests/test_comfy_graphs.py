"""Unit tests for src/comfy_graphs.py -- pure ComfyUI graph builders.

Deliberately dependency-free (stdlib only, matching the module under test) so
this file runs under plain `python3 tests/test_comfy_graphs.py` with no
fastapi/torch/pytest install, AND is still pytest-discoverable (plain
`def test_*()` + `assert`) when the real suite runs it.
"""
import json
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.comfy_graphs import (  # noqa: E402
    GraphConversionError,
    apply_style_trigger,
    build_image_graph,
    build_wan_i2v_graph,
    comfy_image_ref,
    filter_safetensors,
    introspect_graph,
    resolve_lora_entries,
    ui_to_api,
    _wan_step_split,
)


class Skip(Exception):
    """Raised by a test to mean 'skipped', not 'failed' -- used for the two
    tests below that depend on real repo files under data/, which per
    CLAUDE.md are gitignored and may not exist in a fresh clone."""


# ---------------------------------------------------------------------------
# build_image_graph: node-set shape (txt2img vs img2img)
# ---------------------------------------------------------------------------

def _class_types(graph: dict) -> list[str]:
    return [n["class_type"] for n in graph.values()]


def test_txt2img_node_set():
    graph = build_image_graph({"prompt": "a girl on a beach", "loras": []})
    types = _class_types(graph)
    for expected in ("UNETLoader", "ModelSamplingAuraFlow", "CLIPLoader",
                      "EmptySD3LatentImage", "KSampler", "VAELoader", "VAEDecode", "SaveImage"):
        assert expected in types, f"missing {expected} in txt2img graph: {types}"
    assert types.count("CLIPTextEncode") == 2
    assert "LoadImage" not in types
    assert "VAEEncode" not in types
    assert "LoraLoaderModelOnly" not in types  # no loras requested


def test_img2img_node_set():
    graph = build_image_graph({"prompt": "restyle this", "input_image": "upload_abc.png"})
    types = _class_types(graph)
    assert "LoadImage" in types
    assert "VAEEncode" in types
    assert "EmptySD3LatentImage" not in types
    load_node = next(n for n in graph.values() if n["class_type"] == "LoadImage")
    assert load_node["inputs"] == {"image": "upload_abc.png"}, (
        "LoadImage inputs must be exactly {'image': <filename>} -- no second "
        "widget value (the upload-mode marker is UI-only, plan §3a)"
    )



# ---------------------------------------------------------------------------
# GAP 1 -- upload bridge: comfy_image_ref() + its wiring into build_image_graph
# ---------------------------------------------------------------------------

def test_comfy_image_ref_plain_name_default_type_needs_no_annotation():
    assert comfy_image_ref("photo.png") == "photo.png"
    assert comfy_image_ref("photo.png", "", "input") == "photo.png"


def test_comfy_image_ref_prefixes_subfolder():
    assert comfy_image_ref("photo.png", "session42") == "session42/photo.png"


def test_comfy_image_ref_annotates_non_input_type():
    assert comfy_image_ref("render.png", "", "output") == "render.png [output]"
    assert comfy_image_ref("render.png", "", "temp") == "render.png [temp]"


def test_comfy_image_ref_combines_subfolder_and_type():
    assert comfy_image_ref("render.png", "batch1", "output") == "batch1/render.png [output]"


def test_build_image_graph_applies_input_image_annotation():
    # This is the exact wiring routes/comfy_routes.py's gallery-passthrough
    # relies on: a non-default subfolder/type coming back from ComfyUI's own
    # /upload/image must show up correctly annotated on the LoadImage node,
    # not as a bare (and therefore wrong/unresolvable) filename.
    graph = build_image_graph({
        "prompt": "x",
        "input_image": "render.png",
        "input_image_subfolder": "batch1",
        "input_image_type": "output",
    })
    load_node = next(n for n in graph.values() if n["class_type"] == "LoadImage")
    assert load_node["inputs"] == {"image": "batch1/render.png [output]"}


def test_txt2img_denoise_forced_to_one_regardless_of_input():
    # denoise is meaningless for txt2img; a caller-supplied value must be
    # ignored, not silently accepted (plan §1).
    graph = build_image_graph({"prompt": "x", "denoise": 0.3})
    ksampler = next(n for n in graph.values() if n["class_type"] == "KSampler")
    assert ksampler["inputs"]["denoise"] == 1.0


def test_img2img_default_denoise_is_point_six():
    graph = build_image_graph({"prompt": "x", "input_image": "a.png"})
    ksampler = next(n for n in graph.values() if n["class_type"] == "KSampler")
    assert ksampler["inputs"]["denoise"] == 0.6


def test_img2img_explicit_denoise_respected():
    graph = build_image_graph({"prompt": "x", "input_image": "a.png", "denoise": 0.42})
    ksampler = next(n for n in graph.values() if n["class_type"] == "KSampler")
    assert ksampler["inputs"]["denoise"] == 0.42


# ---------------------------------------------------------------------------
# N-LoRA chaining
# ---------------------------------------------------------------------------

def test_n_lora_chaining_rewires_links():
    loras = [
        {"comfy_name": "toei90s_v4_1/toei90s_zbase_v4_1.safetensors", "weight": 0.75},
        {"comfy_name": "tetsuya_char_v5/tetsuya_char_v5_000006000.safetensors", "weight": 1.0},
        {"comfy_name": "extra_lora.safetensors", "weight": 0.3},
    ]
    graph = build_image_graph({"prompt": "x", "loras": loras})

    unet_id = next(nid for nid, n in graph.items() if n["class_type"] == "UNETLoader")
    lora_nodes = {nid: n for nid, n in graph.items() if n["class_type"] == "LoraLoaderModelOnly"}
    assert len(lora_nodes) == 3

    # Walk the chain starting at the UNETLoader and confirm it visits all 3
    # LoRAs in INPUT order, each one correctly rewired to the previous node's
    # output -- this is the thing a value-injector can't do (plan §3).
    model_consumer = {}
    for nid, n in graph.items():
        m = n["inputs"].get("model")
        if isinstance(m, list):
            model_consumer[m[0]] = nid

    cur = unet_id
    seen_in_order = []
    for _ in range(10):
        nxt = model_consumer.get(cur)
        if nxt is None or graph[nxt]["class_type"] != "LoraLoaderModelOnly":
            break
        seen_in_order.append(nxt)
        cur = nxt

    assert len(seen_in_order) == 3
    for nid, expected in zip(seen_in_order, loras):
        assert graph[nid]["inputs"]["lora_name"] == expected["comfy_name"]
        assert graph[nid]["inputs"]["strength_model"] == expected["weight"]
        assert graph[nid]["inputs"]["model"][1] == 0  # output slot index

    # The node after the last LoRA must be ModelSamplingAuraFlow, and after
    # THAT, the KSampler -- proving the tail of the chain is wired back into
    # the main graph, not left dangling.
    after_last_lora = model_consumer[seen_in_order[-1]]
    assert graph[after_last_lora]["class_type"] == "ModelSamplingAuraFlow"
    ksampler_id = next(nid for nid, n in graph.items() if n["class_type"] == "KSampler")
    assert graph[ksampler_id]["inputs"]["model"] == [after_last_lora, 0]


def test_zero_loras_wires_unet_straight_to_sampling():
    graph = build_image_graph({"prompt": "x", "loras": []})
    unet_id = next(nid for nid, n in graph.items() if n["class_type"] == "UNETLoader")
    sampling = next(n for n in graph.values() if n["class_type"] == "ModelSamplingAuraFlow")
    assert sampling["inputs"]["model"] == [unet_id, 0]


def test_lora_missing_comfy_name_raises():
    try:
        build_image_graph({"prompt": "x", "loras": [{"weight": 1.0}]})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a lora entry missing comfy_name")


# ---------------------------------------------------------------------------
# control_after_generate must never appear in built (API-format) output
# ---------------------------------------------------------------------------

def test_control_after_generate_absent_from_ksampler_output():
    graph = build_image_graph({"prompt": "x", "seed": 42, "randomize_seed": False})
    ksampler = next(n for n in graph.values() if n["class_type"] == "KSampler")
    assert "control_after_generate" not in ksampler["inputs"]
    assert set(ksampler["inputs"].keys()) == {
        "model", "positive", "negative", "latent_image",
        "seed", "steps", "cfg", "sampler_name", "scheduler", "denoise",
    }


# ---------------------------------------------------------------------------
# seed resolution
# ---------------------------------------------------------------------------

def test_fixed_seed_is_used_verbatim():
    graph = build_image_graph({"prompt": "x", "seed": 123456, "randomize_seed": False})
    ksampler = next(n for n in graph.values() if n["class_type"] == "KSampler")
    assert ksampler["inputs"]["seed"] == 123456


def test_randomize_seed_or_missing_seed_produces_an_int_in_range():
    for params in ({"prompt": "x", "randomize_seed": True, "seed": 999},
                    {"prompt": "x"}):
        graph = build_image_graph(params)
        seed = next(n for n in graph.values() if n["class_type"] == "KSampler")["inputs"]["seed"]
        assert isinstance(seed, int)
        assert 0 <= seed <= 2**32 - 1


# ---------------------------------------------------------------------------
# GAP 2 -- seed round-trip: reproducibility (same fixed seed -> identical
# graph) and "the resolved seed must come back, not be discarded"
# ---------------------------------------------------------------------------

def test_identical_seed_and_randomize_false_produces_byte_identical_graphs():
    # The reproducibility promise seed exists for: same params in (fixed
    # seed, randomize_seed=False) => same graph out, every time -- nothing
    # in build_image_graph may vary run to run for this input.
    params = {
        "prompt": "toei90s style, smoon, a hero on a cliff",
        "negative_prompt": "bad hands",
        "loras": [{"comfy_name": "style.safetensors", "weight": 0.8}],
        "width": 1216, "height": 672, "batch": 1,
        "seed": 424242, "randomize_seed": False,
        "steps": 30, "cfg": 4.5, "sampler": "res_multistep", "scheduler": "simple",
    }
    graph_a = build_image_graph(dict(params))
    graph_b = build_image_graph(dict(params))
    assert graph_a == graph_b
    assert json.dumps(graph_a, sort_keys=True) == json.dumps(graph_b, sort_keys=True)


def test_randomize_seed_true_resolved_seed_is_retrievable_via_introspection():
    # This is the exact round-trip routes/comfy_routes.py's /generate now
    # relies on to persist the RESOLVED params (module docstring's "this
    # round-trip is also how routes/comfy_routes.py recovers the FINAL
    # concrete params... for the gen_params column"). The seed rolled inside
    # build_image_graph() must be recoverable from the graph afterward, not
    # silently discarded -- and it must be the SAME value that ended up in
    # the KSampler node, not a freshly re-rolled independent one.
    graph = build_image_graph({"prompt": "x", "randomize_seed": True})
    ksampler_seed = next(n for n in graph.values() if n["class_type"] == "KSampler")["inputs"]["seed"]
    resolved_seed = introspect_graph(graph)["params"]["seed"]
    assert isinstance(resolved_seed, int)
    assert 0 <= resolved_seed <= 2**32 - 1
    assert resolved_seed == ksampler_seed  # recovered, not lost, not re-rolled


# ---------------------------------------------------------------------------
# introspect(build(params)) round-trips
# ---------------------------------------------------------------------------

def test_round_trip_txt2img():
    params = {
        "prompt": "toei90s style, smoon, a hero on a cliff at sunset",
        "negative_prompt": "bad hands, extra fingers",
        "loras": [
            {"comfy_name": "toei90s_v4_1/toei90s_zbase_v4_1.safetensors", "weight": 0.8},
            {"comfy_name": "tetsuya_char_v5/tetsuya_char_v5_000006000.safetensors", "weight": 1.0},
        ],
        "width": 1216, "height": 672, "batch": 2,
        "seed": 777, "randomize_seed": False,
        "steps": 28, "cfg": 5.0, "sampler": "euler", "scheduler": "normal",
        "shift": 4, "unet": "custom_unet.safetensors", "clip": "custom_clip.safetensors",
        "vae": "custom_vae.safetensors", "filename_prefix": "roundtrip_test",
    }
    graph = build_image_graph(params)
    out = introspect_graph(graph)["params"]

    assert out["prompt"] == params["prompt"]
    assert out["negative_prompt"] == params["negative_prompt"]
    assert out["loras"] == params["loras"]
    assert out["width"] == 1216
    assert out["height"] == 672
    assert out["batch"] == 2
    assert out["input_image"] is None
    assert out["seed"] == 777
    assert out["steps"] == 28
    assert out["cfg"] == 5.0
    assert out["sampler"] == "euler"
    assert out["scheduler"] == "normal"
    assert out["denoise"] == 1.0
    assert out["shift"] == 4
    assert out["unet"] == "custom_unet.safetensors"
    assert out["clip"] == "custom_clip.safetensors"
    assert out["vae"] == "custom_vae.safetensors"
    assert out["filename_prefix"] == "roundtrip_test"


def test_round_trip_img2img():
    params = {
        "prompt": "restyle this photo", "negative_prompt": "blurry",
        "loras": [{"comfy_name": "style.safetensors", "weight": 0.75}],
        "input_image": "uploaded_source.png", "denoise": 0.55,
        "seed": 55, "randomize_seed": False,
        "steps": 20, "cfg": 4.0, "sampler": "res_multistep", "scheduler": "simple",
    }
    graph = build_image_graph(params)
    out = introspect_graph(graph)["params"]

    assert out["input_image"] == "uploaded_source.png"
    assert out["denoise"] == 0.55
    assert out["loras"] == params["loras"]
    assert out["seed"] == 55
    assert "width" not in out  # img2img: no EmptySD3LatentImage node to read size from


def test_introspect_reports_unsupported_nodes():
    graph = build_image_graph({"prompt": "x"})
    graph["999"] = {"class_type": "SomeBrandNewCustomNode", "inputs": {"foo": "bar"}}
    result = introspect_graph(graph)
    assert "999" in result["unsupported"]
    # A genuinely unrecognised node must not break introspection of the rest.
    assert result["params"]["prompt"] == "x"


def test_positive_negative_disambiguated_by_wiring_not_by_id_order():
    # Hand-built graph where the NEGATIVE text node has the LOWER id, to
    # prove introspect_graph follows the KSampler's positive/negative links
    # rather than assuming insertion/id order.
    graph = {
        "1": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["9", 0], "text": "NEGATIVE TEXT"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["9", 0], "text": "POSITIVE TEXT"}},
        "9": {"class_type": "CLIPLoader", "inputs": {"clip_name": "c.safetensors", "type": "lumina2", "device": "default"}},
        "5": {"class_type": "KSampler", "inputs": {
            "model": ["9", 0], "positive": ["2", 0], "negative": ["1", 0],
            "latent_image": ["9", 0], "seed": 1, "steps": 1, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
        }},
    }
    out = introspect_graph(graph)["params"]
    assert out["prompt"] == "POSITIVE TEXT"
    assert out["negative_prompt"] == "NEGATIVE TEXT"


# ---------------------------------------------------------------------------
# build_wan_i2v_graph: node-set shape + flatness (task step 11)
# ---------------------------------------------------------------------------

def _looks_like_uuid(s: str) -> bool:
    import re
    return bool(re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", s))


def test_wan_graph_has_no_subgraph_or_definitions_key():
    # The ORIGINAL "Ride video (Wan2.2).json" is a subgraph -- everything
    # tunable lives under a top-level "definitions" key (plan §3b), which is
    # exactly what made it undrivable via POST /prompt (API format has no
    # such concept -- see execution.py's flat node["class_type"] lookups).
    # The graph built here must be completely flat: no "definitions" key
    # anywhere, and no node's class_type is a subgraph reference (a bare
    # UUID string, as node 130 was in the original).
    graph = build_wan_i2v_graph({"prompt": "x", "input_image": "start.png"})
    assert "definitions" not in graph
    for node in graph.values():
        assert "class_type" in node
        assert not _looks_like_uuid(node["class_type"])


def test_wan_graph_node_set():
    graph = build_wan_i2v_graph({"prompt": "x", "input_image": "start.png"})
    types = _class_types(graph)
    for expected in ("LoadImage", "UNETLoader", "LoraLoaderModelOnly", "ModelSamplingSD3",
                      "CLIPLoader", "CLIPTextEncode", "VAELoader", "WanImageToVideo",
                      "KSamplerAdvanced", "VAEDecode", "CreateVideo", "SaveVideo"):
        assert expected in types, f"missing {expected} in wan graph: {types}"
    assert types.count("UNETLoader") == 2, "expected 2 UNETLoaders (high + low noise)"
    assert types.count("LoraLoaderModelOnly") == 2, "expected 2 LoRA loaders (high + low noise)"
    assert types.count("ModelSamplingSD3") == 2, (
        "expected 2 ModelSamplingSD3 nodes (high + low noise chains) -- this node "
        "is missing from the plan's own §3b table but is required in the real workflow"
    )
    assert types.count("KSamplerAdvanced") == 2, "expected 2 KSamplerAdvanced stages"
    assert types.count("CLIPTextEncode") == 2
    assert types.count("CLIPLoader") == 1, "one CLIPLoader shared by both text encodes"
    assert types.count("VAELoader") == 1, "one VAELoader shared by WanImageToVideo + VAEDecode"
    assert "EmptySD3LatentImage" not in types
    assert "VAEEncode" not in types  # video latent comes from WanImageToVideo itself


def test_wan_graph_without_input_image_omits_load_image():
    # Pure, always-buildable function like build_image_graph() -- no start
    # frame must not raise.
    graph = build_wan_i2v_graph({"prompt": "x"})
    types = _class_types(graph)
    assert "LoadImage" not in types
    wan = next(n for n in graph.values() if n["class_type"] == "WanImageToVideo")
    assert "start_image" not in wan["inputs"]


def test_wan_graph_ksampler_advanced_has_no_denoise_input():
    # Unlike KSampler, KSamplerAdvanced has NO declared "denoise" input
    # (verified against nodes.py's KSamplerAdvanced.INPUT_TYPES() -- it's a
    # python-level kwarg default inside sample(), never a node input).
    graph = build_wan_i2v_graph({"prompt": "x"})
    for n in graph.values():
        if n["class_type"] == "KSamplerAdvanced":
            assert "denoise" not in n["inputs"]


# ---------------------------------------------------------------------------
# build_wan_i2v_graph: two-stage step split (the central ask of task step 11)
# ---------------------------------------------------------------------------

def test_wan_step_split_helper_several_values():
    assert _wan_step_split(4) == (2, 2, 4)   # the reference workflow's own value
    assert _wan_step_split(8) == (4, 4, 8)
    assert _wan_step_split(2) == (1, 1, 2)
    assert _wan_step_split(1) == (0, 0, 1)
    assert _wan_step_split(5) == (2, 2, 5)   # odd total: stage 1 gets the smaller half
    assert _wan_step_split(0) == (0, 0, 0)
    assert _wan_step_split(-3) == (0, 0, 0)  # clamped, never a negative range


def _wan_stages(graph: dict) -> tuple:
    """Return (stage1, stage2) KSamplerAdvanced node dicts in the order
    build_wan_i2v_graph() creates them (stage 1/high-noise first) -- this is
    also the order introspect_graph() relies on to pick the "primary" one."""
    stages = [n for n in graph.values() if n["class_type"] == "KSamplerAdvanced"]
    assert len(stages) == 2
    return stages[0], stages[1]


def test_wan_two_stage_split_for_several_step_counts():
    for steps, (mid, start2, end2) in (
        (4, (2, 2, 4)), (8, (4, 4, 8)), (2, (1, 1, 2)), (6, (3, 3, 6)),
    ):
        graph = build_wan_i2v_graph({"prompt": "x", "steps": steps, "input_image": "s.png"})
        stage1, stage2 = _wan_stages(graph)
        assert stage1["inputs"]["start_at_step"] == 0
        assert stage1["inputs"]["end_at_step"] == mid
        assert stage2["inputs"]["start_at_step"] == start2
        assert stage2["inputs"]["end_at_step"] == end2
        # Both stages carry the SAME total "steps" -- it's the conceptual
        # full schedule the start/end pair slices into, not a per-stage
        # count (verified against nodes.py's common_ksampler()).
        assert stage1["inputs"]["steps"] == steps
        assert stage2["inputs"]["steps"] == steps


def test_wan_stage_flags_are_fixed_not_configurable():
    graph = build_wan_i2v_graph({"prompt": "x"})
    stage1, stage2 = _wan_stages(graph)
    assert stage1["inputs"]["add_noise"] == "enable"
    assert stage1["inputs"]["return_with_leftover_noise"] == "enable"
    assert stage2["inputs"]["add_noise"] == "disable"
    assert stage2["inputs"]["return_with_leftover_noise"] == "disable"


def test_wan_only_stage_one_seed_is_real():
    # task step 11: "Only node A's seed is real ... do not expose it as a
    # second seed." Stage 2's noise_seed must be present (KSamplerAdvanced
    # requires it) but must never vary with params["seed"].
    graph_a = build_wan_i2v_graph({"prompt": "x", "seed": 111, "randomize_seed": False})
    graph_b = build_wan_i2v_graph({"prompt": "x", "seed": 999, "randomize_seed": False})
    stage1_a, stage2_a = _wan_stages(graph_a)
    stage1_b, stage2_b = _wan_stages(graph_b)
    assert stage1_a["inputs"]["noise_seed"] == 111
    assert stage1_b["inputs"]["noise_seed"] == 999
    # Stage 2's noise_seed is a fixed, inert constant -- identical regardless
    # of params["seed"].
    assert stage2_a["inputs"]["noise_seed"] == stage2_b["inputs"]["noise_seed"]


def test_wan_randomize_seed_or_missing_seed_produces_an_int_in_range():
    for params in ({"prompt": "x", "randomize_seed": True, "seed": 5},
                    {"prompt": "x"}):
        graph = build_wan_i2v_graph(params)
        stage1, _ = _wan_stages(graph)
        seed = stage1["inputs"]["noise_seed"]
        assert isinstance(seed, int)
        assert 0 <= seed <= 2**32 - 1


def test_wan_identical_seed_and_randomize_false_produces_byte_identical_graphs():
    params = {
        "prompt": "a hero rides down a coastal road", "negative_prompt": "blurry",
        "input_image": "start.png", "width": 720, "height": 720, "frames": 81,
        "fps": 16, "seed": 42424, "randomize_seed": False, "steps": 4, "cfg": 1,
        "sampler": "euler", "scheduler": "simple",
        "loras": [{"comfy_name": "wan_high_noise", "weight": 0.9, "role": "high_noise"},
                  {"comfy_name": "wan_low_noise", "weight": 0.8, "role": "low_noise"}],
    }
    graph_a = build_wan_i2v_graph(dict(params))
    graph_b = build_wan_i2v_graph(dict(params))
    assert json.dumps(graph_a, sort_keys=True) == json.dumps(graph_b, sort_keys=True)


# ---------------------------------------------------------------------------
# build_wan_i2v_graph: frame count / fps / start-frame wiring land on the
# right nodes
# ---------------------------------------------------------------------------

def test_wan_frames_fps_size_land_on_the_right_nodes():
    graph = build_wan_i2v_graph({"prompt": "x", "width": 512, "height": 384, "frames": 49, "fps": 24})
    wan = next(n for n in graph.values() if n["class_type"] == "WanImageToVideo")
    assert wan["inputs"]["width"] == 512
    assert wan["inputs"]["height"] == 384
    assert wan["inputs"]["length"] == 49  # "frames" param -> WanImageToVideo's "length" input
    create_video = next(n for n in graph.values() if n["class_type"] == "CreateVideo")
    assert create_video["inputs"]["fps"] == 24


def test_wan_start_frame_uses_comfy_image_ref_like_the_image_graph():
    graph = build_wan_i2v_graph({
        "prompt": "x", "input_image": "render.png",
        "input_image_subfolder": "batch1", "input_image_type": "output",
    })
    load_node = next(n for n in graph.values() if n["class_type"] == "LoadImage")
    assert load_node["inputs"] == {"image": "batch1/render.png [output]"}
    load_id = next(nid for nid, n in graph.items() if n["class_type"] == "LoadImage")
    wan = next(n for n in graph.values() if n["class_type"] == "WanImageToVideo")
    assert wan["inputs"]["start_image"] == [load_id, 0]


# ---------------------------------------------------------------------------
# build_wan_i2v_graph: LoRA weight extraction (_extract_wan_lora_weights) --
# filenames are always fixed; only weight is ever caller-controlled
# ---------------------------------------------------------------------------

def test_wan_lora_weights_matched_by_role():
    graph = build_wan_i2v_graph({"prompt": "x", "loras": [
        {"comfy_name": "wan_high_noise", "weight": 0.5, "role": "high_noise"},
        {"comfy_name": "wan_low_noise", "weight": 0.25, "role": "low_noise"},
    ]})
    loras = [n for n in graph.values() if n["class_type"] == "LoraLoaderModelOnly"]
    by_name = {n["inputs"]["lora_name"]: n["inputs"]["strength_model"] for n in loras}
    assert by_name["wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"] == 0.5
    assert by_name["wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"] == 0.25


def test_wan_lora_weights_default_when_loras_omitted():
    graph = build_wan_i2v_graph({"prompt": "x"})
    loras = [n for n in graph.values() if n["class_type"] == "LoraLoaderModelOnly"]
    assert len(loras) == 2
    assert all(n["inputs"]["strength_model"] == 1.0 for n in loras)


def test_wan_lora_weights_fall_back_to_positional_order():
    # No "role" key, and comfy_name doesn't contain "high"/"low" either --
    # must still resolve via position: [0]=high, [1]=low.
    graph = build_wan_i2v_graph({"prompt": "x", "loras": [
        {"comfy_name": "aaa", "weight": 0.6},
        {"comfy_name": "bbb", "weight": 0.4},
    ]})
    loras = [n for n in graph.values() if n["class_type"] == "LoraLoaderModelOnly"]
    by_name = {n["inputs"]["lora_name"]: n["inputs"]["strength_model"] for n in loras}
    assert by_name["wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"] == 0.6
    assert by_name["wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"] == 0.4


# ---------------------------------------------------------------------------
# build_wan_i2v_graph: ModelSamplingSD3 shift -- the node missing from the
# plan's own §3b table
# ---------------------------------------------------------------------------

def test_wan_model_sampling_sd3_shift_defaults_and_overrides():
    default_graph = build_wan_i2v_graph({"prompt": "x"})
    shifts = sorted(n["inputs"]["shift"] for n in default_graph.values() if n["class_type"] == "ModelSamplingSD3")
    assert shifts == [5.0, 5.0]

    override_graph = build_wan_i2v_graph({"prompt": "x", "shift": 7})
    shifts = sorted(n["inputs"]["shift"] for n in override_graph.values() if n["class_type"] == "ModelSamplingSD3")
    assert shifts == [7.0, 7.0]


# ---------------------------------------------------------------------------
# introspect_graph(build_wan_i2v_graph(params)) round-trip
# ---------------------------------------------------------------------------

def test_wan_graph_introspection_reports_no_unsupported_nodes():
    # ModelSamplingSD3 was missing from _KNOWN_CLASS_TYPES before this task;
    # a well-formed video graph must now report zero unsupported nodes.
    graph = build_wan_i2v_graph({"prompt": "x", "input_image": "s.png"})
    result = introspect_graph(graph)
    assert result["unsupported"] == []


def test_round_trip_video():
    params = {
        "prompt": "a hero rides a motorcycle down a coastal road",
        "negative_prompt": "blurry, low quality",
        "input_image": "start_frame.png",
        "width": 720, "height": 720, "frames": 81, "fps": 16,
        "seed": 313131, "randomize_seed": False,
        "steps": 4, "cfg": 1.0, "sampler": "euler", "scheduler": "simple",
        "loras": [
            {"comfy_name": "wan_high_noise", "weight": 0.9, "role": "high_noise"},
            {"comfy_name": "wan_low_noise", "weight": 0.7, "role": "low_noise"},
        ],
    }
    graph = build_wan_i2v_graph(params)
    out = introspect_graph(graph)["params"]

    assert out["prompt"] == params["prompt"]
    assert out["negative_prompt"] == params["negative_prompt"]
    # This is the elif->if bug fix: BEFORE it, the presence of a LoadImage
    # node (the start frame) silently shadowed the WanImageToVideo branch,
    # so width/height/frames were never read for ANY video graph with a
    # start frame set -- i.e. every real one. All of the following would
    # have been missing/None under the old code.
    assert out["input_image"] == "start_frame.png"
    assert out["width"] == 720
    assert out["height"] == 720
    assert out["frames"] == 81
    assert out["fps"] == 16
    assert out["seed"] == 313131  # stage 1's real seed, recovered
    assert out["steps"] == 4      # TOTAL steps, not a per-stage count
    assert out["cfg"] == 1.0
    assert out["sampler"] == "euler"
    assert out["scheduler"] == "simple"
    assert out["shift"] == 5.0
    assert out["unet"] == "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
    assert out["unet_low"] == "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
    assert out["loras"] == [
        {"comfy_name": "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors", "weight": 0.9},
        {"comfy_name": "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors", "weight": 0.7},
    ]
    # Stage 2's inert noise_seed must never surface under any key.
    assert "noise_seed" not in out
    assert "seed2" not in out and "seed_2" not in out and "stage2_seed" not in out


def test_round_trip_video_defaults_from_bare_dict():
    # A bare {} must still build a valid, introspectable graph -- same
    # "every field has a default" contract as the image pipeline.
    graph = build_wan_i2v_graph({})
    out = introspect_graph(graph)["params"]
    assert out["width"] == 720
    assert out["height"] == 720
    assert out["frames"] == 81
    assert out["fps"] == 16
    assert out["steps"] == 4
    assert out["cfg"] == 1.0


# ---------------------------------------------------------------------------
# ui_to_api: the two documented UI-only-widget traps + failure modes
# ---------------------------------------------------------------------------

def _stub_object_info(class_type: str, required: dict) -> dict:
    return {class_type: {"input": {"required": required}}}


def test_ui_to_api_skips_control_after_generate_positionally():
    # Declared order deliberately matches plan §3a's widgets order once
    # control_after_generate (UI-only, undeclared) is accounted for:
    # [seed, <control_after_generate, injected>, steps, cfg, sampler_name,
    #  scheduler, positive*, negative*, latent_image*, denoise]  (* = linked)
    object_info = _stub_object_info("KSampler", {
        "model": ("MODEL", {}),
        "seed": ("INT", {}),
        "steps": ("INT", {}),
        "cfg": ("FLOAT", {}),
        "sampler_name": (["res_multistep", "euler"], {}),
        "scheduler": (["simple", "normal"], {}),
        "positive": ("CONDITIONING", {}),
        "negative": ("CONDITIONING", {}),
        "latent_image": ("LATENT", {}),
        "denoise": ("FLOAT", {}),
    })
    ui_graph = {
        "links": [
            [1, 10, 0, 20, 0, "MODEL"],
            [2, 11, 0, 20, 1, "CONDITIONING"],
            [3, 12, 0, 20, 2, "CONDITIONING"],
            [4, 13, 0, 20, 3, "LATENT"],
        ],
        "nodes": [
            {
                "id": 20, "type": "KSampler",
                "inputs": [
                    {"name": "model", "link": 1},
                    {"name": "positive", "link": 2},
                    {"name": "negative", "link": 3},
                    {"name": "latent_image", "link": 4},
                ],
                "widgets_values": [42, "randomize", 30, 4.5, "res_multistep", "simple", 1.0],
            },
        ],
    }
    api_graph = ui_to_api(ui_graph, object_info)
    node = api_graph["20"]
    assert node["class_type"] == "KSampler"
    assert "control_after_generate" not in node["inputs"]
    assert node["inputs"]["seed"] == 42
    assert node["inputs"]["steps"] == 30
    assert node["inputs"]["cfg"] == 4.5
    assert node["inputs"]["sampler_name"] == "res_multistep"
    assert node["inputs"]["scheduler"] == "simple"
    assert node["inputs"]["denoise"] == 1.0
    assert node["inputs"]["model"] == ["10", 0]
    assert node["inputs"]["positive"] == ["11", 0]
    assert node["inputs"]["negative"] == ["12", 0]
    assert node["inputs"]["latent_image"] == ["13", 0]


def test_ui_to_api_skips_load_image_upload_widget():
    object_info = _stub_object_info("LoadImage", {"image": (["example.png"], {"image_upload": True})})
    ui_graph = {
        "links": [],
        "nodes": [
            {"id": 1, "type": "LoadImage", "inputs": [], "widgets_values": ["photo.png", "image"]},
        ],
    }
    api_graph = ui_to_api(ui_graph, object_info)
    assert api_graph["1"]["inputs"] == {"image": "photo.png"}


def test_ui_to_api_raises_on_unknown_class_type():
    ui_graph = {"links": [], "nodes": [{"id": 1, "type": "TotallyMadeUpNode", "inputs": [], "widgets_values": []}]}
    try:
        ui_to_api(ui_graph, {})
    except GraphConversionError:
        pass
    else:
        raise AssertionError("expected GraphConversionError for a class_type missing from object_info")


def test_ui_to_api_raises_on_widget_count_mismatch():
    object_info = _stub_object_info("VAELoader", {"vae_name": (["ae.safetensors"], {})})
    # Missing the required vae_name value entirely.
    ui_graph = {"links": [], "nodes": [{"id": 1, "type": "VAELoader", "inputs": [], "widgets_values": []}]}
    try:
        ui_to_api(ui_graph, object_info)
    except GraphConversionError:
        pass
    else:
        raise AssertionError("expected GraphConversionError for a widgets_values/required-input mismatch")


# ---------------------------------------------------------------------------
# apply_style_trigger
# ---------------------------------------------------------------------------

def _write_temp_registry() -> str:
    registry = {
        "default": "smoon",
        "styles": {
            "smoon": {"trigger": "toei90s style, smoon", "label": "default", "aliases": ["sailor moon", "my style"]},
            "chflash": {"trigger": "toei90s style, chflash", "label": "honey", "aliases": ["cutie honey", "honey flash"]},
        },
    }
    fd, path = tempfile.mkstemp(suffix=".json")
    with open(fd, "w", encoding="utf-8") as f:
        json.dump(registry, f)
    return path


def test_apply_style_trigger_prepends_default():
    path = _write_temp_registry()
    out = apply_style_trigger("a hero on a cliff", registry_path=path)
    assert out == "toei90s style, smoon, a hero on a cliff"


def test_apply_style_trigger_idempotent():
    path = _write_temp_registry()
    already = "toei90s style, smoon, a hero on a cliff"
    assert apply_style_trigger(already, registry_path=path) == already


def test_apply_style_trigger_resolves_alias_from_prompt_text():
    path = _write_temp_registry()
    out = apply_style_trigger("cutie honey flying through the city", registry_path=path)
    assert out.startswith("toei90s style, chflash, ")


def test_apply_style_trigger_hint_takes_precedence_over_prompt_text():
    path = _write_temp_registry()
    # Prompt text alone would resolve to smoon (no alias match), but an
    # explicit style_hint of "honey flash" must win.
    out = apply_style_trigger("a girl runs", style_hint="honey flash", registry_path=path)
    assert out.startswith("toei90s style, chflash, ")


def test_apply_style_trigger_missing_registry_falls_back_instead_of_raising():
    out = apply_style_trigger("anything", registry_path="/nonexistent/path/styles.json")
    assert out.startswith("toei90s style, smoon, ")  # built-in fallback registry


def test_apply_style_trigger_against_real_repo_registry():
    real_path = _REPO_ROOT / "data" / "studio" / "scripts" / "styles.json"
    if not real_path.is_file():
        raise Skip("data/studio/scripts/styles.json not present (gitignored; expected on a fresh clone)")
    out = apply_style_trigger("a hero on a cliff", registry_path=str(real_path))
    assert out.startswith("toei90s style, smoon, ")
    # Idempotency against the REAL registry's actual trigger text.
    assert apply_style_trigger(out, registry_path=str(real_path)) == out


# ---------------------------------------------------------------------------
# DEFECT 1 -- filter_safetensors / resolve_lora_entries: registry comfy_name
# vs ComfyUI's LIVE LoraLoaderModelOnly list. Fake live lists only -- no
# network, no real registry file required (stdlib-only per the module
# docstring).
# ---------------------------------------------------------------------------

def test_filter_safetensors_drops_junk():
    # "optimizer.pt" is the real junk entry observed on the live server (task
    # DEFECT 1) -- a training-run artifact that happens to sit in a mapped
    # models dir, never a valid LoRA to load.
    live = ["tetsuya_char_v5_000006000.safetensors", "optimizer.pt", "toei90s_zbase_v4_1.safetensors", "README.md"]
    assert filter_safetensors(live) == [
        "tetsuya_char_v5_000006000.safetensors", "toei90s_zbase_v4_1.safetensors",
    ]


def test_filter_safetensors_case_insensitive_and_empty_input():
    assert filter_safetensors(["FOO.SAFETENSORS", "bar.safetensors"]) == ["FOO.SAFETENSORS", "bar.safetensors"]
    assert filter_safetensors(None) == []
    assert filter_safetensors([]) == []


def test_resolve_lora_entries_exact_match():
    # The steady-state case: ComfyUI has (already, or post-restart) been
    # widened to report the nested form itself, so the registry's own
    # comfy_name is already a live name verbatim.
    live = ["toei90s_zbase_v4_1/toei90s_zbase_v4_1.safetensors", "some_other.safetensors"]
    entries = [{"key": "toei90s_v4_1", "comfy_name": "toei90s_zbase_v4_1/toei90s_zbase_v4_1.safetensors"}]
    out = resolve_lora_entries(entries, live)
    assert out[0]["available"] is True
    assert out[0]["resolved_name"] == "toei90s_zbase_v4_1/toei90s_zbase_v4_1.safetensors"
    assert out[0]["key"] == "toei90s_v4_1"  # other keys preserved verbatim


def test_resolve_lora_entries_basename_match_uses_the_live_spelling():
    # The CURRENT real-world case per DEFECT 1: a pre-restart ComfyUI only
    # knows the bare filename; the registry's nested-form comfy_name must
    # resolve to that live bare string, NOT to its own nested spelling (which
    # ComfyUI would reject as an unknown combo value).
    live = ["tetsuya_char_v5_000006000.safetensors", "toei90s_zbase_v4_1.safetensors"]
    entries = [
        {"key": "tetsuya_v5", "comfy_name": "tetsuya_char_v5/tetsuya_char_v5_000006000.safetensors"},
        {"key": "toei90s_v4_1", "comfy_name": "toei90s_zbase_v4_1/toei90s_zbase_v4_1.safetensors"},
    ]
    out = resolve_lora_entries(entries, live)
    assert out[0]["available"] is True
    assert out[0]["resolved_name"] == "tetsuya_char_v5_000006000.safetensors"
    assert out[1]["available"] is True
    assert out[1]["resolved_name"] == "toei90s_zbase_v4_1.safetensors"


def test_resolve_lora_entries_unavailable_when_no_match_at_all():
    live = ["some_unrelated_file.safetensors"]
    entries = [{"key": "tetsuya_v1", "comfy_name": "tetsuya_char_v1/tetsuya_char_v1.safetensors"}]
    out = resolve_lora_entries(entries, live)
    assert out[0]["available"] is False
    assert out[0]["resolved_name"] is None


def test_resolve_lora_entries_first_basename_collision_wins_deterministically():
    live = ["shared_name.safetensors", "other.safetensors"]
    # Two DIFFERENT live roots could plausibly both contain a file with this
    # basename; only one literal string is in `live` here, so both registry
    # entries (bare vs nested spelling) must resolve to that same live entry.
    entries = [
        {"key": "a", "comfy_name": "root_one/shared_name.safetensors"},
        {"key": "b", "comfy_name": "shared_name.safetensors"},
    ]
    out = resolve_lora_entries(entries, live)
    assert out[0]["resolved_name"] == "shared_name.safetensors"
    assert out[1]["resolved_name"] == "shared_name.safetensors"
    assert out[1]["available"] is True


def test_resolve_lora_entries_empty_inputs():
    assert resolve_lora_entries([], ["a.safetensors"]) == []
    assert resolve_lora_entries(None, None) == []


def test_resolve_lora_entries_preserves_weight_and_role_for_generate_payload_shape():
    # Same function is reused on POST /api/comfy/generate's wire-format
    # params["loras"] list (comfy_name/weight/role), not just the curated
    # registry rows -- confirm the non-comfy_name keys round-trip untouched.
    live = ["real_file.safetensors"]
    entries = [{"comfy_name": "some_dir/real_file.safetensors", "weight": 0.8, "role": None}]
    out = resolve_lora_entries(entries, live)
    assert out[0]["weight"] == 0.8
    assert out[0]["role"] is None
    assert out[0]["resolved_name"] == "real_file.safetensors"
    assert out[0]["available"] is True


def test_resolve_lora_entries_missing_comfy_name_is_unavailable_not_a_crash():
    out = resolve_lora_entries([{"key": "broken"}], ["a.safetensors"])
    assert out[0]["available"] is False
    assert out[0]["resolved_name"] is None


def test_resolve_lora_entries_non_dict_entry_is_unavailable_not_a_crash():
    # Defensive: a malformed (non-dict) entry must not raise -- treated as an
    # entry with no comfy_name at all (unavailable), same as the dict-with-
    # missing-key case above.
    out = resolve_lora_entries([None, "not-a-dict", 42], ["a.safetensors"])
    assert len(out) == 3
    assert all(o["available"] is False and o["resolved_name"] is None for o in out)


# ---------------------------------------------------------------------------
# Bare-python3 runner (no pytest required) -- also pytest-discoverable above.
# ---------------------------------------------------------------------------

def _run_all():
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    passed = failed = skipped = 0
    for name, fn in tests:
        try:
            fn()
        except Skip as e:
            print(f"SKIP  {name}: {e}")
            skipped += 1
        except Exception as e:
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
            failed += 1
        else:
            print(f"PASS  {name}")
            passed += 1
    total = passed + failed + skipped
    print(f"\n{passed}/{total} passed, {failed} failed, {skipped} skipped")
    return failed == 0


if __name__ == "__main__":
    ok = _run_all()
    sys.exit(0 if ok else 1)
