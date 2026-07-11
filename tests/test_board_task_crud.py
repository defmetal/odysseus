"""DB-backed tests for the studio production-board tools (task_add/task_move/
task_list/task_update — PM board MVP, Phase 5 of
data/studio/CHARACTER-PIPELINE-DECISION.md).

Mirrors tests/test_calendar_batch_events.py's pattern: a file-backed temp
sqlite DB built from core.database's real metadata (so the new StudioTask
table is created exactly the way init_db() creates it via
Base.metadata.create_all), with SessionLocal monkeypatched onto the bound
core.database module for the duration of each test.

Covers: add / list (grouped + filtered) / move (incl. fuzzy-match ambiguity)
/ update (incl. the status-belongs-to-task_move guard), and the SHARED
(non-owner-scoped) visibility requirement — both accounts must see and be
able to move the same tasks.
"""

import sys
import uuid

import pytest

from tests.helpers.import_state import clear_fake_database_modules
from tests.helpers.sqlite_db import make_temp_sqlite

clear_fake_database_modules()

import core.database as cdb
from core.database import StudioTask

_TS, _ENGINE, _TMPDB = make_temp_sqlite(cdb.Base.metadata)


@pytest.fixture(autouse=True)
def _bind_temp_db(monkeypatch):
    monkeypatch.setitem(sys.modules, "core.database", cdb)
    parent = sys.modules.get("core")
    if parent is not None:
        monkeypatch.setattr(parent, "database", cdb, raising=False)
    monkeypatch.setattr(cdb, "SessionLocal", _TS)
    yield
    # Keep the shared temp DB clean between tests — each test owns its own
    # rows via unique titles, but a clean slate keeps assertions simple and
    # order-independent (no test relies on another's leftover rows).
    db = _TS()
    try:
        db.query(StudioTask).delete()
        db.commit()
    finally:
        db.close()


def _title(prefix: str) -> str:
    return f"{prefix} {uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# task_add
# ---------------------------------------------------------------------------

async def test_task_add_defaults_to_todo():
    from src.tool_implementations import do_task_add

    title = _title("Colorize Tetsuya line art")
    res = await do_task_add(title, owner="austin")
    assert res.get("exit_code") == 0, res
    assert "todo" in res.get("response", "")

    db = _TS()
    task = db.query(StudioTask).filter(StudioTask.title == title).first()
    assert task is not None
    assert task.status == "todo"
    assert task.created_by == "austin"
    assert task.assignee is None
    assert task.tags == ""
    db.close()


async def test_task_add_with_all_fields():
    from src.tool_implementations import do_task_add

    title = _title("Lock color key for Marzipan")
    content = f"{title}\nstatus: doing\nassignee: wife\ntags: marzipan, color"
    res = await do_task_add(content, owner="usakochiba")
    assert res.get("exit_code") == 0, res

    db = _TS()
    task = db.query(StudioTask).filter(StudioTask.title == title).first()
    assert task.status == "doing"
    assert task.assignee == "wife"
    assert task.tags == "marzipan, color"
    assert task.created_by == "usakochiba"
    db.close()


async def test_task_add_requires_a_title():
    from src.tool_implementations import do_task_add

    res = await do_task_add("", owner="austin")
    assert res.get("exit_code") == 1
    assert "title" in res.get("error", "").lower()


async def test_task_add_rejects_unknown_status():
    from src.tool_implementations import do_task_add

    res = await do_task_add(f"{_title('Bad status task')}\nstatus: archived", owner="austin")
    assert res.get("exit_code") == 1
    assert "status" in res.get("error", "").lower()


# ---------------------------------------------------------------------------
# task_list
# ---------------------------------------------------------------------------

async def test_task_list_groups_by_status():
    from src.tool_implementations import do_task_add, do_task_list

    t1, t2, t3 = _title("A todo task"), _title("A doing task"), _title("A done task")
    await do_task_add(t1, owner="austin")
    await do_task_add(f"{t2}\nstatus: doing", owner="austin")
    await do_task_add(f"{t3}\nstatus: done", owner="austin")

    res = await do_task_list("", owner="austin")
    assert res.get("exit_code") == 0, res
    out = res["response"]
    assert "**Todo" in out and "**Doing" in out and "**Done" in out
    assert t1 in out and t2 in out and t3 in out


async def test_task_list_status_filter():
    from src.tool_implementations import do_task_add, do_task_list

    doing_title = _title("Only this one is doing")
    await do_task_add(_title("Not this one"), owner="austin")
    await do_task_add(f"{doing_title}\nstatus: doing", owner="austin")

    res = await do_task_list("doing", owner="austin")
    assert res.get("exit_code") == 0, res
    assert doing_title in res["response"]
    assert "Not this one" not in res["response"]


async def test_task_list_empty_board():
    from src.tool_implementations import do_task_list

    res = await do_task_list("", owner="austin")
    assert res.get("exit_code") == 0
    assert "empty" in res.get("response", "").lower()


async def test_task_list_rejects_unknown_status_filter():
    from src.tool_implementations import do_task_list

    res = await do_task_list("archived", owner="austin")
    assert res.get("exit_code") == 1


# ---------------------------------------------------------------------------
# task_move
# ---------------------------------------------------------------------------

async def test_task_move_transitions_status_by_title():
    from src.tool_implementations import do_task_add, do_task_move

    title = _title("Fix Tetsuya's left hand")
    await do_task_add(title, owner="austin")

    res = await do_task_move(f"{title}\ndoing", owner="austin")
    assert res.get("exit_code") == 0, res
    assert "doing" in res["response"]

    db = _TS()
    task = db.query(StudioTask).filter(StudioTask.title == title).first()
    assert task.status == "doing"
    db.close()


async def test_task_move_by_id_prefix():
    from src.tool_implementations import do_task_add, do_task_move

    title = _title("Verify QIE-2511 pipeline class")
    add_res = await do_task_add(title, owner="austin")
    task_id = add_res["task_id"]

    res = await do_task_move(f"{task_id[:8]}\ndone", owner="austin")
    assert res.get("exit_code") == 0, res

    db = _TS()
    task = db.query(StudioTask).filter(StudioTask.id == task_id).first()
    assert task.status == "done"
    db.close()


async def test_task_move_ambiguous_title_returns_candidates_not_a_guess():
    from src.tool_implementations import do_task_add, do_task_move

    shared = _title("Fix hand")
    await do_task_add(f"{shared} left", owner="austin")
    await do_task_add(f"{shared} right", owner="austin")

    res = await do_task_move(f"{shared}\ndone", owner="austin")
    assert res.get("exit_code") == 1
    assert len(res.get("candidates") or []) == 2
    # Neither task was silently moved.
    db = _TS()
    statuses = {t.status for t in db.query(StudioTask).filter(StudioTask.title.like(f"{shared}%")).all()}
    assert statuses == {"todo"}
    db.close()


async def test_task_move_unknown_task_errors():
    from src.tool_implementations import do_task_move

    res = await do_task_move("nonexistent-task-xyz\ndone", owner="austin")
    assert res.get("exit_code") == 1
    assert "no task found" in res.get("error", "").lower()


async def test_task_move_rejects_unknown_status():
    from src.tool_implementations import do_task_add, do_task_move

    title = _title("Some task")
    await do_task_add(title, owner="austin")
    res = await do_task_move(f"{title}\narchived", owner="austin")
    assert res.get("exit_code") == 1
    assert "status" in res.get("error", "").lower()


# ---------------------------------------------------------------------------
# task_update
# ---------------------------------------------------------------------------

async def test_task_update_retitle_and_reassign():
    from src.tool_implementations import do_task_add, do_task_update

    title = _title("Original title")
    add_res = await do_task_add(title, owner="austin")
    task_id = add_res["task_id"]

    new_title = _title("Updated title")
    res = await do_task_update(f"{task_id}\ntitle: {new_title}\nassignee: wife", owner="austin")
    assert res.get("exit_code") == 0, res

    db = _TS()
    task = db.query(StudioTask).filter(StudioTask.id == task_id).first()
    assert task.title == new_title
    assert task.assignee == "wife"
    db.close()


async def test_task_update_rejects_status_field():
    from src.tool_implementations import do_task_add, do_task_update

    title = _title("Do not change my status here")
    await do_task_add(title, owner="austin")
    res = await do_task_update(f"{title}\nstatus: done", owner="austin")
    assert res.get("exit_code") == 1
    assert "task_move" in res.get("error", "")


async def test_task_update_requires_a_field():
    from src.tool_implementations import do_task_add, do_task_update

    title = _title("Nothing to change")
    await do_task_add(title, owner="austin")
    res = await do_task_update(title, owner="austin")
    assert res.get("exit_code") == 1


# ---------------------------------------------------------------------------
# Shared visibility (NOT owner-scoped) — the core PM-board requirement
# ---------------------------------------------------------------------------

async def test_board_is_shared_across_owners():
    """A task added by one account must be visible to, and movable by, the
    other account — the board has no owner filter anywhere."""
    from src.tool_implementations import do_task_add, do_task_list, do_task_move

    title = _title("Shared task across accounts")
    add_res = await do_task_add(title, owner="austin")
    assert add_res["exit_code"] == 0

    # The wife's account (different owner) sees it in task_list...
    listing = await do_task_list("", owner="usakochiba")
    assert title in listing["response"]

    # ...and can move it.
    move_res = await do_task_move(f"{title}\ndoing", owner="usakochiba")
    assert move_res.get("exit_code") == 0, move_res

    db = _TS()
    task = db.query(StudioTask).filter(StudioTask.title == title).first()
    assert task.status == "doing"
    assert task.created_by == "austin"  # provenance preserved, not overwritten
    db.close()
