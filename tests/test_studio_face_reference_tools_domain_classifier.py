"""Domain-classifier + TOOL_SECTIONS wiring for the studio face-drift-fix and
QIE dataset-factory chat tools (fix_faces / reference_edit).

Mirrors tests/test_board_domain_classifier.py and
tests/test_tool_rag_contacts_domain.py: the classifier is deterministic
string matching (no embeddings / no DB), so it can be exercised directly.
Both tools live in the EXISTING "images" domain (alongside restyle_image/
inpaint_region/controlnet) rather than a new domain — see
_DOMAIN_TOOL_MAP["images"] in src/agent_loop.py.
"""

from src.agent_loop import (
    _DOMAIN_RULES,
    _DOMAIN_TOOL_MAP,
    _classify_agent_request,
    _domain_rules_for_tools,
    TOOL_SECTIONS,
)
from src.agent_tools import TOOL_TAGS


def _classify(text):
    return _classify_agent_request([{"role": "user", "content": text}], text)


def test_fix_faces_phrasing_gets_images_domain_and_edit_only():
    prompts = [
        "fix her face",
        "fix his face",
        "fix the faces in this photo",
        "can you fix her face please",
    ]
    for p in prompts:
        intent = _classify(p)
        assert "images" in intent["domains"], f"expected images domain for: {p!r}"
        assert intent["low_signal"] is False, f"must not be low_signal: {p!r}"
        assert intent["edit_only"] is True, f"expected edit_only for: {p!r}"


def test_colorize_phrasing_gets_images_domain():
    prompts = [
        "colorize this line art",
        "can you colorize this for me",
        "colourise this sketch",
    ]
    for p in prompts:
        intent = _classify(p)
        assert "images" in intent["domains"], f"expected images domain for: {p!r}"
        assert intent["low_signal"] is False


def test_turnaround_and_character_sheet_phrasing_gets_images_domain():
    prompts = [
        "make a character sheet from this",
        "build a character turnaround from this reference",
        "give me a turnaround sheet for her",
    ]
    for p in prompts:
        intent = _classify(p)
        assert "images" in intent["domains"], f"expected images domain for: {p!r}"


def test_bare_turnaround_word_alone_does_not_trigger_images_domain():
    """'turnaround' alone is common unrelated business phrasing ('what's the
    turnaround time on my order') — must NOT false-trigger the images domain
    (it only counts paired with 'character'/'sheet')."""
    intent = _classify("what's the turnaround time on my order")
    assert "images" not in intent["domains"]
    intent2 = _classify("we need a quick turnaround on this deal")
    assert "images" not in intent2["domains"]


def test_images_domain_seeds_fix_faces_and_reference_edit():
    """The images domain must seed the new tools alongside the existing
    restyle_image/inpaint_region/controlnet siblings so they're offered even
    when semantic retrieval misses."""
    assert _DOMAIN_TOOL_MAP["images"] == {
        "generate_image", "edit_image", "restyle_image", "inpaint_region",
        "controlnet", "fix_faces", "reference_edit",
    }


def test_images_domain_has_a_rule_pack():
    assert "images" in _DOMAIN_RULES
    rules = _domain_rules_for_tools({"fix_faces"})
    assert any("Image rules" in r for r in rules)
    rules2 = _domain_rules_for_tools({"reference_edit"})
    assert any("Image rules" in r for r in rules2)


def test_fix_faces_and_reference_edit_have_tool_sections_entries():
    for tool in ("fix_faces", "reference_edit"):
        assert tool in TOOL_SECTIONS, f"{tool} missing a TOOL_SECTIONS description"
        assert f"```{tool}" in TOOL_SECTIONS[tool]


def test_fix_faces_and_reference_edit_are_registered_in_tool_tags():
    for tool in ("fix_faces", "reference_edit"):
        assert tool in TOOL_TAGS


def test_non_image_requests_do_not_match_images_domain_via_new_vocabulary():
    """Guard against over-triggering on ordinary uses of the new words."""
    assert "images" not in _classify("what is the capital of France")["domains"]
    assert "images" not in _classify("reply to the latest email in my inbox")["domains"]
    assert "images" not in _classify("what's on the production board")["domains"]
    assert "images" not in _classify("the turnaround time was too slow")["domains"]


def test_attachment_with_fix_face_wording_is_edit_only_not_generate():
    """An attached image + 'fix the face' must classify edit_only so
    generate_image gets dropped and fix_faces/restyle_image/inpaint_region
    are the reachable choices — mirrors the existing restyle/inpaint
    edit_only coverage the sibling tools rely on (agent_loop.py's
    _latest_user_has_image / edit_only handling)."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            {"type": "text", "text": "fix her face"},
        ],
    }]
    intent = _classify_agent_request(messages, "fix her face")
    assert "images" in intent["domains"]
    assert intent["edit_only"] is True
