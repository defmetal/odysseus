"""Domain-classifier wiring for the studio production board (task_add/
task_move/task_list/task_update — PM board MVP, Phase 5 of
data/studio/CHARACTER-PIPELINE-DECISION.md).

Mirrors tests/test_tool_rag_contacts_domain.py: the classifier is
deterministic string matching (no embeddings / no DB), so it can be
exercised directly. Without a matching domain, a message like "what's on the
board" would classify low_signal and never be offered task_list/task_add/etc.
— see _classify_agent_request's `low_signal = not continuation and not
domains` in src/agent_loop.py.
"""

from src.agent_loop import (
    _DOMAIN_RULES,
    _DOMAIN_TOOL_MAP,
    _classify_agent_request,
    _domain_rules_for_tools,
)
from src.agent_tools import TOOL_TAGS
from src.agent_loop import TOOL_SECTIONS


def _classify(text):
    return _classify_agent_request([{"role": "user", "content": text}], text)


def test_board_phrasings_get_board_domain():
    """The exact phrasings called out in the task spec must classify to the
    board domain and must not be treated as low-signal."""
    prompts = [
        "add fix the hand to the board",
        "what's on the board",
        "move fix the hand to done",
        "what's in doing",
        "show me the production board",
        "put colorize tetsuya on the board",
        "mark fix the hand as done",
    ]
    for p in prompts:
        intent = _classify(p)
        assert "board" in intent["domains"], f"expected board domain for: {p!r}"
        assert intent["low_signal"] is False, f"must not be low_signal: {p!r}"


def test_board_domain_seeds_the_four_task_tools():
    """The domain must seed the actual board tools so they're offered even
    when semantic retrieval misses (mirrors _DOMAIN_TOOL_MAP for contacts)."""
    assert _DOMAIN_TOOL_MAP["board"] == {"task_add", "task_move", "task_list", "task_update"}


def test_board_domain_has_a_rule_pack():
    """Every domain in _DOMAIN_TOOL_MAP needs a matching _DOMAIN_RULES entry,
    else _domain_rules_for_tools raises KeyError when the tools are selected."""
    assert "board" in _DOMAIN_RULES
    rules = _domain_rules_for_tools({"task_add"})
    assert any("Production board rules" in r for r in rules)


def test_board_tools_have_tool_sections_entries():
    """Every board tool needs a TOOL_SECTIONS entry, or the model never sees
    a usage description/fenced-block template for it."""
    for tool in ("task_add", "task_move", "task_list", "task_update"):
        assert tool in TOOL_SECTIONS, f"{tool} missing a TOOL_SECTIONS description"
        assert f"```{tool}" in TOOL_SECTIONS[tool]


def test_board_tools_are_registered_in_tool_tags():
    for tool in ("task_add", "task_move", "task_list", "task_update"):
        assert tool in TOOL_TAGS


def test_non_board_requests_do_not_match_board_domain():
    """Guard against over-triggering on ordinary uses of 'board'/'done'."""
    assert "board" not in _classify("the board of directors approved the budget")["domains"]
    assert "board" not in _classify("what is the capital of France")["domains"]
    assert "board" not in _classify("reply to the latest email in my inbox")["domains"]
    assert "board" not in _classify("generate an image of a sunset")["domains"]
    assert "board" not in _classify("I'm done with dinner")["domains"]
    assert "board" not in _classify("check the whiteboard in the office")["domains"]


def test_board_domain_distinct_from_scheduled_tasks_and_notes():
    """Recurring/scheduled-job phrasing and personal-reminder phrasing stay
    on notes_calendar_tasks, not board — the two are different tools
    (manage_tasks / manage_notes) for different concepts."""
    intent = _classify("remind me to buy milk tomorrow")
    assert "notes_calendar_tasks" in intent["domains"]
    assert "board" not in intent["domains"]

    intent2 = _classify("do this every day automatically")
    assert "notes_calendar_tasks" in intent2["domains"]
    assert "board" not in intent2["domains"]
