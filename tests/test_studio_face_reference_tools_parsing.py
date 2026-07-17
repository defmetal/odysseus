"""Parsing + dispatch-reachability tests for the studio face-drift-fix and
QIE dataset-factory chat tools (fix_faces / reference_edit).

Covers the same layers as tests/test_board_tool_parsing.py, per the
documented trap ("ALWAYS test the agent path via parse_tool_blocks(text) —
calling do_X() directly bypasses block-parsing and hides TOOL_TAGS-class
bugs", data/studio/ODYSSEUS.md):

1. Fenced-block RECOGNITION via ``parse_tool_blocks`` — pins that both tags
   are in TOOL_TAGS and the block-parse regex picks them up (a tag missing
   from TOOL_TAGS parses as plain text and silently no-ops — the exact bug
   that historically bit restyle_image/inpaint_region).
2. The pure line-based arg parsers themselves (``_parse_fix_faces``,
   ``_fix_faces_prompt``, ``_parse_reference_edit``,
   ``_normalize_reference_edit_mode``, ``_resolve_color_ref``,
   ``_resolve_latest_upload_path``), including weak-model-proofing (blank
   lines / literal ``<placeholder>`` tokens — the same precedent as
   ``_parse_generate_image`` in src/tool_execution.py and board.py's
   ``_clean_lines``).
3. Dispatch REACHABILITY: execute_tool_block routes a parsed block to the
   correct do_* function. do_fix_faces/do_reference_edit are never called
   directly here — they subprocess studio scripts and touch the Gallery DB,
   neither of which exists in this sandbox — they're monkeypatched with a
   stub so only the WIRING (TOOL_TAGS -> parse -> dispatch elif -> facade
   import) is under test.
"""
import json

from src.agent_tools import TOOL_TAGS
from src.tool_parsing import parse_tool_blocks
from src.tool_execution import execute_tool_block
from src.tools.image import (
    _clean_tool_lines,
    _parse_fix_faces,
    _fix_faces_prompt,
    _REFERENCE_EDIT_MODES,
    _normalize_reference_edit_mode,
    _parse_reference_edit,
    _resolve_latest_upload_path,
    _resolve_color_ref,
)


# ---------------------------------------------------------------------------
# Fenced-block recognition (parse_tool_blocks / TOOL_TAGS)
# ---------------------------------------------------------------------------

def test_new_tags_are_in_tool_tags_allowlist():
    """A tag missing from TOOL_TAGS parses as plain text (silent no-op) — the
    exact bug that historically bit restyle_image/inpaint_region."""
    for tag in ("fix_faces", "reference_edit"):
        assert tag in TOOL_TAGS, f"{tag} missing from TOOL_TAGS — block would parse as text"


def test_fix_faces_fence_parses_with_both_lines():
    blocks = parse_tool_blocks("```fix_faces\ntetsuya_oc\nsmiling\n```")
    assert [(b.tool_type, b.content) for b in blocks] == [
        ("fix_faces", "tetsuya_oc\nsmiling")
    ]


def test_fix_faces_fence_parses_with_empty_body():
    """Both lines are optional — an entirely empty body is a common,
    complete invocation ('fix her face', no character/hint) and must still
    dispatch, not silently vanish (the same class of bug task_list's
    empty-fence fix addressed for the board tools)."""
    blocks = parse_tool_blocks("```fix_faces\n```")
    assert [(b.tool_type, b.content) for b in blocks] == [("fix_faces", "")]


def test_reference_edit_fence_parses():
    blocks = parse_tool_blocks("```reference_edit\ncolorize\nref: tetsuya\n```")
    assert [(b.tool_type, b.content) for b in blocks] == [
        ("reference_edit", "colorize\nref: tetsuya")
    ]


def test_reference_edit_empty_fence_is_dropped_not_dispatched():
    """Unlike fix_faces, reference_edit REQUIRES a mode — an empty body has
    no valid interpretation, so (matching task_add/task_move/task_update's
    precedent, which also require an arg) it is NOT special-cased into an
    empty-content dispatch; parse_tool_blocks drops it entirely."""
    blocks = parse_tool_blocks("```reference_edit\n```")
    assert blocks == []


# ---------------------------------------------------------------------------
# _clean_tool_lines — blank-line + placeholder-line stripping
# ---------------------------------------------------------------------------

def test_clean_tool_lines_drops_blank_lines():
    assert _clean_tool_lines("tetsuya_oc\n\n\nsmiling\n") == ["tetsuya_oc", "smiling"]


def test_clean_tool_lines_drops_whole_line_placeholders():
    assert _clean_tool_lines("<character trigger>\nsmiling") == ["smiling"]


def test_clean_tool_lines_empty_content():
    assert _clean_tool_lines("") == []
    assert _clean_tool_lines(None) == []


# ---------------------------------------------------------------------------
# fix_faces parsing
# ---------------------------------------------------------------------------

def test_parse_fix_faces_both_lines():
    trigger, hint = _parse_fix_faces("tetsuya_oc\nsmiling, blush")
    assert trigger == "tetsuya_oc"
    assert hint == "smiling, blush"


def test_parse_fix_faces_empty_body():
    trigger, hint = _parse_fix_faces("")
    assert trigger == ""
    assert hint == ""


def test_parse_fix_faces_trigger_only():
    trigger, hint = _parse_fix_faces("tetsuya_oc")
    assert trigger == "tetsuya_oc"
    assert hint == ""


def test_parse_fix_faces_tolerates_placeholder_lines():
    trigger, hint = _parse_fix_faces(
        "<character trigger: optional, e.g. tetsuya_oc>\n<expression/detail hint: optional>"
    )
    assert trigger == ""
    assert hint == ""


def test_parse_fix_faces_tolerates_blank_lines():
    trigger, hint = _parse_fix_faces("\n\ntetsuya_oc\n\n")
    assert trigger == "tetsuya_oc"
    assert hint == ""


def test_fix_faces_prompt_both_parts():
    assert _fix_faces_prompt("tetsuya_oc", "smiling") == "tetsuya_oc, close-up of the face, smiling"


def test_fix_faces_prompt_neither_part():
    assert _fix_faces_prompt("", "") == "close-up of the face"


def test_fix_faces_prompt_trigger_only():
    assert _fix_faces_prompt("tetsuya_oc", "") == "tetsuya_oc, close-up of the face"


def test_fix_faces_prompt_hint_only():
    assert _fix_faces_prompt("", "surprised") == "close-up of the face, surprised"


# ---------------------------------------------------------------------------
# reference_edit mode normalization
# ---------------------------------------------------------------------------

def test_normalize_reference_edit_mode_canonical_values():
    for m in _REFERENCE_EDIT_MODES:
        assert _normalize_reference_edit_mode(m) == m


def test_normalize_reference_edit_mode_case_and_space_insensitive():
    assert _normalize_reference_edit_mode("  COLORIZE  ") == "colorize"
    assert _normalize_reference_edit_mode("Turnaround") == "turnaround"


def test_normalize_reference_edit_mode_aliases():
    assert _normalize_reference_edit_mode("colourize") == "colorize"
    assert _normalize_reference_edit_mode("colour") == "colorize"
    assert _normalize_reference_edit_mode("character sheet") == "turnaround"
    assert _normalize_reference_edit_mode("turn-around") == "turnaround"
    assert _normalize_reference_edit_mode("variation") == "vary"


def test_normalize_reference_edit_mode_unknown_returns_empty():
    assert _normalize_reference_edit_mode("upscale") == ""
    assert _normalize_reference_edit_mode("") == ""
    assert _normalize_reference_edit_mode(None) == ""


# ---------------------------------------------------------------------------
# _parse_reference_edit — weak-model-proofed line parsing
# ---------------------------------------------------------------------------

def test_parse_reference_edit_mode_only():
    mode, prompt, ref = _parse_reference_edit("colorize")
    assert mode == "colorize"
    assert prompt == ""
    assert ref == ""


def test_parse_reference_edit_mode_prompt_and_ref():
    mode, prompt, ref = _parse_reference_edit("vary\nchange the pose to sitting\nref: tetsuya")
    assert mode == "vary"
    assert prompt == "change the pose to sitting"
    assert ref == "tetsuya"


def test_parse_reference_edit_ref_accepts_reference_prefix():
    mode, prompt, ref = _parse_reference_edit("colorize\nreference: Tetsuya")
    assert mode == "colorize"
    assert ref == "Tetsuya"


def test_parse_reference_edit_ref_before_prompt():
    """ref: may appear before the prompt line — order shouldn't matter."""
    mode, prompt, ref = _parse_reference_edit("turnaround\nref: tetsuya\na dynamic action pose")
    assert mode == "turnaround"
    assert ref == "tetsuya"
    assert prompt == "a dynamic action pose"


def test_parse_reference_edit_tolerates_blank_lines_and_placeholders():
    mode, prompt, ref = _parse_reference_edit("\n\ncolorize\n\nref: <character name, optional>\n\n")
    assert mode == "colorize"
    assert ref == ""
    assert prompt == ""


def test_parse_reference_edit_empty_content():
    mode, prompt, ref = _parse_reference_edit("")
    assert mode == ""
    assert prompt == ""
    assert ref == ""


def test_parse_reference_edit_multiline_prompt_joined():
    mode, prompt, ref = _parse_reference_edit("vary\nchange the pose\nkeep the face and hair unchanged")
    assert mode == "vary"
    assert prompt == "change the pose keep the face and hair unchanged"


# ---------------------------------------------------------------------------
# _resolve_color_ref — canonical/fallback heuristic (filesystem-backed)
# ---------------------------------------------------------------------------

def test_resolve_color_ref_prefers_canonical_prefixed_file(tmp_path):
    color_dir = tmp_path / "tetsuya" / "color"
    color_dir.mkdir(parents=True)
    (color_dir / "grok-abc123.jpg").write_bytes(b"x")
    (color_dir / "canonical_ref.png").write_bytes(b"x")
    result = _resolve_color_ref("tetsuya", base_dir=str(tmp_path))
    assert result.name == "canonical_ref.png"


def test_resolve_color_ref_falls_back_to_alphabetically_first(tmp_path):
    color_dir = tmp_path / "tetsuya" / "color"
    color_dir.mkdir(parents=True)
    (color_dir / "kJMpX.jpg").write_bytes(b"x")
    (color_dir / "grok-abc123.jpg").write_bytes(b"x")
    result = _resolve_color_ref("tetsuya", base_dir=str(tmp_path))
    assert result.name == "grok-abc123.jpg"


def test_resolve_color_ref_excludes_underscore_prefixed_entries(tmp_path):
    color_dir = tmp_path / "tetsuya" / "color"
    color_dir.mkdir(parents=True)
    (color_dir / "_tiles_contact.jpg").write_bytes(b"x")
    (color_dir / "_dups").mkdir()
    (color_dir / "only_real.jpg").write_bytes(b"x")
    result = _resolve_color_ref("tetsuya", base_dir=str(tmp_path))
    assert result.name == "only_real.jpg"


def test_resolve_color_ref_normalizes_trigger_suffix(tmp_path):
    """'tetsuya_oc' (the character-LoRA trigger form) resolves the same as
    the bare character name 'tetsuya'."""
    color_dir = tmp_path / "tetsuya" / "color"
    color_dir.mkdir(parents=True)
    (color_dir / "canonical.jpg").write_bytes(b"x")
    result = _resolve_color_ref("Tetsuya_OC", base_dir=str(tmp_path))
    assert result.name == "canonical.jpg"


def test_resolve_color_ref_missing_character_dir_returns_none(tmp_path):
    assert _resolve_color_ref("nonexistent_character", base_dir=str(tmp_path)) is None


def test_resolve_color_ref_no_color_dir_returns_none(tmp_path):
    (tmp_path / "marzipan").mkdir()
    assert _resolve_color_ref("marzipan", base_dir=str(tmp_path)) is None


def test_resolve_color_ref_empty_color_dir_returns_none(tmp_path):
    (tmp_path / "tetsuya" / "color").mkdir(parents=True)
    assert _resolve_color_ref("tetsuya", base_dir=str(tmp_path)) is None


def test_resolve_color_ref_ignores_non_image_files(tmp_path):
    color_dir = tmp_path / "tetsuya" / "color"
    color_dir.mkdir(parents=True)
    (color_dir / "color_key.txt").write_bytes(b"x")
    assert _resolve_color_ref("tetsuya", base_dir=str(tmp_path)) is None


# ---------------------------------------------------------------------------
# _resolve_latest_upload_path (filesystem-backed)
# ---------------------------------------------------------------------------

def test_resolve_latest_upload_path_picks_newest_image(tmp_path):
    img_old = tmp_path / "old.png"
    img_new = tmp_path / "new.png"
    img_old.write_bytes(b"x")
    img_new.write_bytes(b"x")
    idx = tmp_path / "uploads.json"
    idx.write_text(json.dumps({
        "admin:1": {"path": str(img_old), "mime": "image/png", "owner": "admin", "uploaded_at": "2026-01-01T00:00:00"},
        "admin:2": {"path": str(img_new), "mime": "image/png", "owner": "admin", "uploaded_at": "2026-06-01T00:00:00"},
    }))
    path, owner = _resolve_latest_upload_path(uploads_index=str(idx))
    assert path == str(img_new)
    assert owner == "admin"


def test_resolve_latest_upload_path_filters_by_owner(tmp_path):
    img_mine = tmp_path / "mine.png"
    img_theirs = tmp_path / "theirs.png"
    img_mine.write_bytes(b"x")
    img_theirs.write_bytes(b"x")
    idx = tmp_path / "uploads.json"
    idx.write_text(json.dumps({
        "wife:1": {"path": str(img_theirs), "mime": "image/png", "owner": "wife", "uploaded_at": "2026-06-02T00:00:00"},
        "admin:1": {"path": str(img_mine), "mime": "image/png", "owner": "admin", "uploaded_at": "2026-01-01T00:00:00"},
    }))
    path, owner = _resolve_latest_upload_path(owner="admin", uploads_index=str(idx))
    assert path == str(img_mine)
    assert owner == "admin"


def test_resolve_latest_upload_path_ignores_non_image_mime(tmp_path):
    doc = tmp_path / "doc.pdf"
    doc.write_bytes(b"x")
    idx = tmp_path / "uploads.json"
    idx.write_text(json.dumps({
        "admin:1": {"path": str(doc), "mime": "application/pdf", "owner": "admin", "uploaded_at": "2026-06-01T00:00:00"},
    }))
    path, owner = _resolve_latest_upload_path(uploads_index=str(idx))
    assert path is None
    assert owner is None


def test_resolve_latest_upload_path_missing_index_returns_none(tmp_path):
    idx = tmp_path / "does_not_exist.json"
    path, owner = _resolve_latest_upload_path(uploads_index=str(idx))
    assert path is None
    assert owner is None


# ---------------------------------------------------------------------------
# Dispatch reachability (execute_tool_block routes to the right do_* — the
# real implementations are stubbed out since they subprocess studio scripts
# / touch the Gallery DB, neither of which exists in this sandbox)
# ---------------------------------------------------------------------------

async def test_fix_faces_dispatch_reachability(monkeypatch):
    import src.tool_implementations as ti
    calls = {}

    async def _fake(content, owner=None):
        calls["content"] = content
        calls["owner"] = owner
        return {"stdout": "ok", "stderr": "", "exit_code": 0}

    monkeypatch.setattr(ti, "do_fix_faces", _fake)
    blocks = parse_tool_blocks("```fix_faces\ntetsuya_oc\nsmiling\n```")
    desc, result = await execute_tool_block(blocks[0], owner="austin")
    assert desc == "fix_faces"
    assert result == {"stdout": "ok", "stderr": "", "exit_code": 0}
    assert calls == {"content": "tetsuya_oc\nsmiling", "owner": "austin"}


async def test_reference_edit_dispatch_reachability(monkeypatch):
    import src.tool_implementations as ti
    calls = {}

    async def _fake(content, owner=None):
        calls["content"] = content
        calls["owner"] = owner
        return {"stdout": "ok", "stderr": "", "exit_code": 0}

    monkeypatch.setattr(ti, "do_reference_edit", _fake)
    blocks = parse_tool_blocks("```reference_edit\ncolorize\nref: tetsuya\n```")
    desc, result = await execute_tool_block(blocks[0], owner="austin")
    assert desc == "reference_edit"
    assert result == {"stdout": "ok", "stderr": "", "exit_code": 0}
    assert calls == {"content": "colorize\nref: tetsuya", "owner": "austin"}
