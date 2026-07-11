"""Parsing tests for the studio production-board tools (task_add/task_move/
task_list/task_update — PM board MVP, Phase 5 of
data/studio/CHARACTER-PIPELINE-DECISION.md).

Covers two layers, per the documented trap ("ALWAYS test the agent path via
parse_tool_blocks(text) — calling do_X() directly bypasses block-parsing and
hides TOOL_TAGS-class bugs", data/studio/ODYSSEUS.md):

1. Fenced-block RECOGNITION via ``parse_tool_blocks`` — pins that the four
   tags are in TOOL_TAGS and the block-parse regex picks them up (a tag
   missing from TOOL_TAGS parses as plain text and silently no-ops).
2. The line-based arg parsers themselves (``_parse_task_add``,
   ``_parse_ident_and_fields``, ``_normalize_status``, ``_clean_lines``),
   including the weak-model-proofing behavior (blank lines and literal
   ``<placeholder>`` tokens are tolerated — the same precedent as
   ``_parse_generate_image`` in src/tool_execution.py).
"""

from src.agent_tools import TOOL_TAGS
from src.tool_parsing import parse_tool_blocks
from src.tools.board import (
    _clean_lines,
    _normalize_status,
    _parse_ident_and_fields,
    _parse_task_add,
)


# ---------------------------------------------------------------------------
# Fenced-block recognition (parse_tool_blocks / TOOL_TAGS)
# ---------------------------------------------------------------------------

def test_board_tags_are_in_tool_tags_allowlist():
    """A tag missing from TOOL_TAGS parses as plain text (silent no-op) — the
    exact bug that bit restyle_image/inpaint_region historically."""
    for tag in ("task_add", "task_move", "task_list", "task_update"):
        assert tag in TOOL_TAGS, f"{tag} missing from TOOL_TAGS — block would parse as text"


def test_task_add_fence_parses():
    blocks = parse_tool_blocks("```task_add\nFix Tetsuya's line art\nstatus: doing\n```")
    assert [(b.tool_type, b.content) for b in blocks] == [
        ("task_add", "Fix Tetsuya's line art\nstatus: doing")
    ]


def test_task_move_fence_parses():
    blocks = parse_tool_blocks("```task_move\nFix Tetsuya's line art\ndone\n```")
    assert [(b.tool_type, b.content) for b in blocks] == [
        ("task_move", "Fix Tetsuya's line art\ndone")
    ]


def test_task_list_fence_parses_with_empty_body():
    blocks = parse_tool_blocks("```task_list\n```")
    assert [(b.tool_type, b.content) for b in blocks] == [("task_list", "")]


def test_task_update_fence_parses():
    blocks = parse_tool_blocks("```task_update\nabc123\ntags: tetsuya, color\n```")
    assert [(b.tool_type, b.content) for b in blocks] == [
        ("task_update", "abc123\ntags: tetsuya, color")
    ]


# ---------------------------------------------------------------------------
# _clean_lines — blank-line + placeholder-line stripping
# ---------------------------------------------------------------------------

def test_clean_lines_drops_blank_lines():
    assert _clean_lines("Fix the hand\n\n\nstatus: doing\n") == ["Fix the hand", "status: doing"]


def test_clean_lines_drops_whole_line_placeholders():
    assert _clean_lines("<title>\nstatus: doing") == ["status: doing"]


def test_clean_lines_empty_content():
    assert _clean_lines("") == []
    assert _clean_lines(None) == []


# ---------------------------------------------------------------------------
# _normalize_status
# ---------------------------------------------------------------------------

def test_normalize_status_canonical_values():
    assert _normalize_status("todo") == "todo"
    assert _normalize_status("doing") == "doing"
    assert _normalize_status("done") == "done"


def test_normalize_status_is_case_and_space_insensitive():
    assert _normalize_status("  DONE  ") == "done"
    assert _normalize_status("In Progress") == "doing"


def test_normalize_status_aliases():
    assert _normalize_status("to-do") == "todo"
    assert _normalize_status("in-progress") == "doing"
    assert _normalize_status("wip") == "doing"
    assert _normalize_status("completed") == "done"
    assert _normalize_status("finished") == "done"


def test_normalize_status_unknown_returns_none():
    assert _normalize_status("archived") is None
    assert _normalize_status("blocked") is None


def test_normalize_status_placeholder_and_empty_return_none():
    assert _normalize_status("<todo|doing|done>") is None
    assert _normalize_status("") is None
    assert _normalize_status(None) is None


# ---------------------------------------------------------------------------
# _parse_task_add — weak-model-proofed line parsing
# ---------------------------------------------------------------------------

def test_parse_task_add_title_only():
    title, fields = _parse_task_add("Colorize Tetsuya's 28 line arts")
    assert title == "Colorize Tetsuya's 28 line arts"
    assert fields == {}


def test_parse_task_add_all_fields():
    title, fields = _parse_task_add(
        "Colorize Tetsuya's line arts\nstatus: doing\nassignee: austin\ntags: tetsuya, color"
    )
    assert title == "Colorize Tetsuya's line arts"
    assert fields == {"status": "doing", "assignee": "austin", "tags": "tetsuya, color"}


def test_parse_task_add_tolerates_blank_lines():
    title, fields = _parse_task_add("\n\nLock color key for Marzipan\n\nstatus: todo\n\n")
    assert title == "Lock color key for Marzipan"
    assert fields == {"status": "todo"}


def test_parse_task_add_tolerates_literal_placeholder_lines():
    """A weak model echoing the fenced template verbatim: the title line
    itself is a placeholder (dropped), leaving no usable title."""
    title, fields = _parse_task_add("<title>\nstatus: <todo|doing|done>\nassignee: <name, optional>")
    assert title == ""
    assert fields == {}


def test_parse_task_add_tolerates_placeholder_values_on_real_title():
    """Real title, but the optional fields are still template placeholders —
    those fields must be dropped (not stored as literal '<...>' values)."""
    title, fields = _parse_task_add(
        "Fix Tetsuya's line art\nstatus: <todo|doing|done, optional — default todo>\nassignee: <name, optional>\ntags: <comma-separated tags, optional>"
    )
    assert title == "Fix Tetsuya's line art"
    assert fields == {}


def test_parse_task_add_explicit_title_key():
    title, fields = _parse_task_add("title: Fix Tetsuya's line art\nstatus: doing")
    assert title == "Fix Tetsuya's line art"
    assert fields == {"status": "doing"}


def test_parse_task_add_field_order_does_not_matter():
    title, fields = _parse_task_add("status: doing\nFix Tetsuya's line art\nassignee: austin")
    assert title == "Fix Tetsuya's line art"
    assert fields == {"status": "doing", "assignee": "austin"}


def test_parse_task_add_empty_content_yields_empty_title():
    title, fields = _parse_task_add("")
    assert title == ""
    assert fields == {}


# ---------------------------------------------------------------------------
# _parse_ident_and_fields — task_update's "<ident>\nfield: value..." shape
# ---------------------------------------------------------------------------

def test_parse_ident_and_fields_basic():
    ident, fields = _parse_ident_and_fields("abc123\ntitle: New title\nassignee: wife")
    assert ident == "abc123"
    assert fields == {"title": "New title", "assignee": "wife"}


def test_parse_ident_and_fields_tolerates_placeholders_and_blanks():
    ident, fields = _parse_ident_and_fields("\nFix Tetsuya's line art\n\ntags: <comma-separated tags, optional>\n")
    assert ident == "Fix Tetsuya's line art"
    assert fields == {}


def test_parse_ident_and_fields_no_fields():
    ident, fields = _parse_ident_and_fields("abc123")
    assert ident == "abc123"
    assert fields == {}
