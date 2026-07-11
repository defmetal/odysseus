"""Board-domain tool implementations (studio production-board MVP, Phase 5 of
data/studio/CHARACTER-PIPELINE-DECISION.md).

Chat-driven task tracking for the animation-studio production board:
task_add / task_move / task_list / task_update. This is a NEW module added
directly under src/tools/ (the post-merge tool-implementation package split,
#4423) — there is no legacy tool_implementations.py-era code to extract this
from; it follows the same domain-module + facade-re-export shape every other
src/tools/*.py file uses (see notes.py / calendar.py for the sibling pattern).

The board is SHARED across both accounts (not owner-scoped, unlike
Note/GalleryImage) — `created_by` records who added a task for display only
and is NEVER used to filter visibility. Both users must see the same board.

Tool-call shape is line-based fenced blocks (matching restyle_image /
inpaint_region / controlnet), not the JSON-action shape used by manage_notes /
manage_calendar — task_add/task_move/task_list/task_update are separate tool
names rather than one manage_* tool with an action line, per the requested
tool surface. Parsing tolerates blank lines and literal ``<placeholder>``
tokens a weak model may echo from the fenced-block template — the same
weak-model-proofing precedent as ``_parse_generate_image`` in
src/tool_execution.py.
"""
import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Status normalization
# ---------------------------------------------------------------------------

_STATUSES = ("todo", "doing", "done")

# Common synonyms a model (or a human typing loosely) might use, normalized
# to the three canonical statuses the studio_tasks.status column stores.
_STATUS_ALIASES = {
    "todo": "todo", "to do": "todo", "to-do": "todo", "backlog": "todo",
    "not started": "todo", "not-started": "todo", "new": "todo", "open": "todo",
    "doing": "doing", "in progress": "doing", "in-progress": "doing",
    "inprogress": "doing", "wip": "doing", "started": "doing", "active": "doing",
    "working": "doing", "in flight": "doing", "in-flight": "doing",
    "done": "done", "complete": "done", "completed": "done",
    "finished": "done", "closed": "done", "shipped": "done",
}


def _normalize_status(raw: Optional[str]) -> Optional[str]:
    """Canonicalize a status string to 'todo'/'doing'/'done'.

    Returns None when *raw* is empty, a literal ``<placeholder>`` token, or
    doesn't match any known status/alias — callers turn that into a clear
    "unknown status" error rather than guessing.
    """
    if not raw:
        return None
    text = re.sub(r"\s+", " ", raw.strip().lower())
    text = text.strip(" .!\"'")
    if not text or _PLACEHOLDER_RE.fullmatch(text):
        return None
    return _STATUS_ALIASES.get(text)


# ---------------------------------------------------------------------------
# Weak-model-proofed line parsing
# ---------------------------------------------------------------------------
# Same defensive shape as _parse_generate_image (src/tool_execution.py): a
# weak model frequently echoes the fenced-block TEMPLATE verbatim, including
# blank filler lines and literal "<placeholder>" tokens. Strip those first so
# an echoed template degrades to "just the title/id", not a literal garbage
# value.

_PLACEHOLDER_RE = re.compile(r"^<.*>$")
_KV_RE = re.compile(r"^(title|status|assignee|tags)\s*:\s*(.*)$", re.I)


def _clean_lines(content: str) -> List[str]:
    """Split into stripped, non-blank lines; drop lines that are ONLY a
    literal <placeholder> token."""
    lines = [ln.strip() for ln in (content or "").strip().split("\n")]
    return [ln for ln in lines if ln and not _PLACEHOLDER_RE.fullmatch(ln)]


def _parse_task_add(content: str):
    """Parse task_add's body: title + optional status:/assignee:/tags: lines.

    The title is assembled from every line that ISN'T a recognized
    status/assignee/tags/title key:value pair (in whatever order they
    appear), so a stray title line before or after the optional fields still
    works. A key:value line whose value is itself a bracket placeholder
    (e.g. ``status: <todo|doing|done>``, echoed straight from the template)
    is treated as not-provided rather than a literal value.
    """
    lines = _clean_lines(content)
    title_parts: List[str] = []
    fields: Dict[str, str] = {}
    for ln in lines:
        m = _KV_RE.match(ln)
        if m:
            key, val = m.group(1).lower(), m.group(2).strip()
            if val and not _PLACEHOLDER_RE.fullmatch(val):
                if key == "title":
                    title_parts.append(val)
                else:
                    fields[key] = val
            continue
        title_parts.append(ln)
    return " ".join(title_parts).strip(), fields


def _parse_ident_and_fields(content: str):
    """Parse an ``<ident>`` line followed by optional key:value lines.

    Used by task_update. Line 1 (first non-kv line) identifies the task
    (id or fuzzy title); any of title:/assignee:/tags: may follow.
    """
    lines = _clean_lines(content)
    ident = ""
    fields: Dict[str, str] = {}
    for ln in lines:
        m = _KV_RE.match(ln)
        if m:
            key, val = m.group(1).lower(), m.group(2).strip()
            if val and not _PLACEHOLDER_RE.fullmatch(val):
                fields[key] = val
            continue
        if not ident:
            ident = ln
    return ident, fields


# ---------------------------------------------------------------------------
# Task resolution (id or fuzzy title match) — shared by task_move/task_update
# ---------------------------------------------------------------------------

_ID_SHAPE_RE = re.compile(r"^[0-9a-f-]{4,40}$", re.I)


def _find_tasks_by_ident(db, ident: str) -> List:
    """Resolve an identifier to StudioTask row(s) on the shared board.

    Order: id / id-prefix match first (board-wide — the board has no owner
    scope to narrow by), else an exact case-insensitive title match, else a
    case-insensitive substring match (either direction, so both a shortened
    reference and a slightly padded one work). Returns a list: empty (not
    found), length 1 (unique — safe to act on), or length > 1 (ambiguous —
    caller must return the candidates, never guess).
    """
    from core.database import StudioTask
    ident = (ident or "").strip()
    if not ident:
        return []
    if _ID_SHAPE_RE.match(ident):
        matches = db.query(StudioTask).filter(StudioTask.id.startswith(ident)).all()
        if matches:
            return matches
    needle = ident.lower()
    all_tasks = db.query(StudioTask).all()
    exact = [t for t in all_tasks if (t.title or "").strip().lower() == needle]
    if exact:
        return exact
    return [
        t for t in all_tasks
        if needle in (t.title or "").lower() or (t.title or "").lower() in needle
    ]


def _ambiguous_result(ident: str, matches: List) -> Dict:
    """Standard "don't guess" response when >1 task matches an identifier."""
    candidates = [{"id": t.id, "title": t.title, "status": t.status} for t in matches]
    listing = "; ".join(f"[{t.id[:8]}] {t.title} ({t.status})" for t in matches)
    return {
        "error": f"Multiple tasks match '{ident}': {listing}. Use the exact id from task_list to pick one.",
        "candidates": candidates,
        "exit_code": 1,
    }


def _fmt_task_line(t) -> str:
    bits = []
    if t.assignee:
        bits.append(f"assignee: {t.assignee}")
    if t.tags:
        bits.append(f"tags: {t.tags}")
    suffix = f" — {', '.join(bits)}" if bits else ""
    return f"- [{t.id[:8]}] {t.title}{suffix}"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

async def do_task_add(content: str, owner: Optional[str] = None) -> Dict:
    """Add a task to the shared studio production board.

    Line 1 (or any non-kv line) = title. Optional status:/assignee:/tags:
    lines in any order. `owner` is recorded as provenance (created_by) only —
    it never restricts who can see or move the task later.
    """
    import uuid as _uuid
    from core.database import SessionLocal, StudioTask

    title, fields = _parse_task_add(content)
    if not title:
        return {"error": "A task title is required (line 1).", "exit_code": 1}

    status = "todo"
    if "status" in fields:
        norm = _normalize_status(fields["status"])
        if not norm:
            return {"error": f"Unknown status '{fields['status']}'. Use todo, doing, or done.", "exit_code": 1}
        status = norm

    assignee = fields.get("assignee") or None
    tags = fields.get("tags") or ""

    db = SessionLocal()
    try:
        task = StudioTask(
            id=str(_uuid.uuid4()),
            title=title,
            status=status,
            assignee=assignee,
            tags=tags,
            created_by=owner,
        )
        db.add(task)
        db.commit()
        return {
            "response": f"Added to board: \"{title}\" [{status}] (id: {task.id[:8]})",
            "task_id": task.id,
            "exit_code": 0,
        }
    except Exception as e:
        logger.error(f"task_add error: {e}")
        db.rollback()
        return {"error": str(e), "exit_code": 1}
    finally:
        db.close()


async def do_task_move(content: str, owner: Optional[str] = None) -> Dict:
    """Move a task to a new status. Line 1 = task id or fuzzy title; line 2 =
    the new status (todo/doing/done, or a common alias). Ambiguous title
    matches return the candidate list instead of guessing which task."""
    from core.database import SessionLocal

    lines = _clean_lines(content)
    if len(lines) < 2:
        return {"error": "Provide the task (id or title) on line 1 and the new status on line 2.", "exit_code": 1}
    ident, status_line = lines[0], lines[1]

    m = _KV_RE.match(status_line)
    raw_status = m.group(2) if (m and m.group(1).lower() == "status") else status_line
    new_status = _normalize_status(raw_status)
    if not new_status:
        return {"error": f"Unknown status '{raw_status}'. Use todo, doing, or done.", "exit_code": 1}

    db = SessionLocal()
    try:
        matches = _find_tasks_by_ident(db, ident)
        if not matches:
            return {"error": f"No task found matching '{ident}'.", "exit_code": 1}
        if len(matches) > 1:
            return _ambiguous_result(ident, matches)
        task = matches[0]
        old_status = task.status
        if old_status == new_status:
            return {
                "response": f"\"{task.title}\" (id: {task.id[:8]}) is already {new_status}.",
                "task_id": task.id,
                "exit_code": 0,
            }
        task.status = new_status
        db.commit()
        return {
            "response": f"Moved \"{task.title}\" (id: {task.id[:8]}): {old_status} -> {new_status}",
            "task_id": task.id,
            "exit_code": 0,
        }
    except Exception as e:
        logger.error(f"task_move error: {e}")
        db.rollback()
        return {"error": str(e), "exit_code": 1}
    finally:
        db.close()


async def do_task_list(content: str, owner: Optional[str] = None) -> Dict:
    """Show the shared studio production board, grouped todo/doing/done.

    Empty body = everything, grouped. A single status word on line 1 filters
    to that column only. Never filtered by owner — the board is shared."""
    from core.database import SessionLocal, StudioTask

    lines = _clean_lines(content)
    status_filter = None
    if lines:
        first = lines[0]
        m = _KV_RE.match(first)
        raw = m.group(2) if (m and m.group(1).lower() == "status") else first
        status_filter = _normalize_status(raw)
        if status_filter is None and raw.strip():
            return {"error": f"Unknown status filter '{raw}'. Use todo, doing, or done.", "exit_code": 1}

    db = SessionLocal()
    try:
        q = db.query(StudioTask)
        if status_filter:
            q = q.filter(StudioTask.status == status_filter)
        tasks = q.order_by(StudioTask.status, StudioTask.created_at.asc()).all()
        if not tasks:
            msg = f"No tasks in {status_filter}." if status_filter else "The production board is empty."
            return {"response": msg, "exit_code": 0}

        if status_filter:
            body = "\n".join(_fmt_task_line(t) for t in tasks)
            out = f"**{status_filter.capitalize()} ({len(tasks)})**\n{body}"
        else:
            groups: Dict[str, List] = {"todo": [], "doing": [], "done": []}
            for t in tasks:
                groups.setdefault(t.status, []).append(t)
            sections = []
            for key in ("todo", "doing", "done"):
                rows = groups.get(key) or []
                header = f"**{key.capitalize()} ({len(rows)})**"
                body = "\n".join(_fmt_task_line(t) for t in rows) if rows else "  (none)"
                sections.append(f"{header}\n{body}")
            out = "\n\n".join(sections)
        return {"response": out, "exit_code": 0}
    except Exception as e:
        logger.error(f"task_list error: {e}")
        return {"error": str(e), "exit_code": 1}
    finally:
        db.close()


async def do_task_update(content: str, owner: Optional[str] = None) -> Dict:
    """Retitle / reassign / re-tag a task. Status changes go through
    task_move, not this tool. Line 1 = task id or fuzzy title (same
    resolution as task_move); following lines = title:/assignee:/tags:."""
    from core.database import SessionLocal

    ident, fields = _parse_ident_and_fields(content)
    if not ident:
        return {"error": "Provide the task (id or title) on line 1.", "exit_code": 1}
    if "status" in fields:
        return {"error": "task_update does not change status — use task_move for that.", "exit_code": 1}
    if not any(k in fields for k in ("title", "assignee", "tags")):
        return {
            "error": "Nothing to update — provide title:, assignee:, and/or tags: on the lines after the task.",
            "exit_code": 1,
        }

    db = SessionLocal()
    try:
        matches = _find_tasks_by_ident(db, ident)
        if not matches:
            return {"error": f"No task found matching '{ident}'.", "exit_code": 1}
        if len(matches) > 1:
            return _ambiguous_result(ident, matches)
        task = matches[0]
        changed = []
        if "title" in fields:
            task.title = fields["title"]
            changed.append("title")
        if "assignee" in fields:
            task.assignee = fields["assignee"]
            changed.append("assignee")
        if "tags" in fields:
            task.tags = fields["tags"]
            changed.append("tags")
        db.commit()
        return {
            "response": f"Updated \"{task.title}\" (id: {task.id[:8]}): {', '.join(changed)}",
            "task_id": task.id,
            "exit_code": 0,
        }
    except Exception as e:
        logger.error(f"task_update error: {e}")
        db.rollback()
        return {"error": str(e), "exit_code": 1}
    finally:
        db.close()
