"""Chat-bar Think toggle: UI effort → OpenAI-compat wire values.

Qwen 3.8 / NVFP4 / local vLLM Qwen reject `reasoning_effort=high` (HTTP 400)
and want `xhigh`. OpenAI / Mistral keep `high`. Think-off must disable
thinking and omit a bogus effort value.
"""

import os

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from src.llm_core import (
    apply_thinking_to_payload,
    map_reasoning_effort,
    normalize_ui_reasoning_effort,
    parse_thinking_enabled,
    thinking_pref,
    uses_xhigh_reasoning_effort,
    _retry_without_reasoning_controls,
)


QWEN38 = "Qwen3.8-27B-AEON-NVFP4"
QWEN38_URL = "http://127.0.0.1:8500/v1/chat/completions"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"


def test_normalize_ui_effort_aliases():
    assert normalize_ui_reasoning_effort("low") == "low"
    assert normalize_ui_reasoning_effort("medium") == "medium"
    assert normalize_ui_reasoning_effort("high") == "high"
    assert normalize_ui_reasoning_effort("med") == "medium"
    assert normalize_ui_reasoning_effort("xhigh") == "high"
    assert normalize_ui_reasoning_effort(None) == "medium"
    assert normalize_ui_reasoning_effort("nope") == "medium"


def test_parse_thinking_enabled():
    assert parse_thinking_enabled(None) is None
    assert parse_thinking_enabled("") is None
    assert parse_thinking_enabled(True) is True
    assert parse_thinking_enabled(False) is False
    assert parse_thinking_enabled("true") is True
    assert parse_thinking_enabled("false") is False
    assert parse_thinking_enabled("off") is False


def test_qwen38_high_maps_to_xhigh():
    assert map_reasoning_effort("high", QWEN38, QWEN38_URL) == "xhigh"
    assert map_reasoning_effort("high", "qwen3.8:27b") == "xhigh"
    assert map_reasoning_effort("high", "Qwen 3.8 27B") == "xhigh"


def test_nvfp4_high_maps_to_xhigh():
    assert uses_xhigh_reasoning_effort("some-model-nvfp4")
    assert map_reasoning_effort("high", "aeon-nvfp4") == "xhigh"


def test_local_vllm_qwen_high_maps_to_xhigh():
    assert map_reasoning_effort(
        "high", "qwen3-27b", "http://127.0.0.1:8500/v1/chat/completions"
    ) == "xhigh"


def test_openai_high_stays_high():
    assert map_reasoning_effort("high", "gpt-4o", OPENAI_URL) == "high"


def test_mistral_high_stays_high():
    assert map_reasoning_effort("high", "mistral-medium", MISTRAL_URL) == "high"


def test_medium_and_low_are_passthrough():
    assert map_reasoning_effort("medium", QWEN38, QWEN38_URL) == "medium"
    assert map_reasoning_effort("low", "gpt-4o", OPENAI_URL) == "low"
    assert map_reasoning_effort("medium", "mistral-small", MISTRAL_URL) == "medium"


def test_think_on_medium_sets_effort_and_enable_thinking():
    payload = {}
    with thinking_pref(True, "medium"):
        apply_thinking_to_payload(payload, QWEN38, QWEN38_URL)
    assert payload["reasoning_effort"] == "medium"
    assert payload["chat_template_kwargs"]["enable_thinking"] is True


def test_think_on_high_qwen38_sends_xhigh():
    payload = {}
    with thinking_pref(True, "high"):
        apply_thinking_to_payload(payload, QWEN38, QWEN38_URL)
    assert payload["reasoning_effort"] == "xhigh"
    assert payload["chat_template_kwargs"]["enable_thinking"] is True


def test_think_on_high_openai_sends_high():
    payload = {}
    with thinking_pref(True, "high"):
        apply_thinking_to_payload(payload, "gpt-4o", OPENAI_URL)
    assert payload["reasoning_effort"] == "high"
    assert payload["chat_template_kwargs"]["enable_thinking"] is True


def test_think_on_high_mistral_sends_high():
    payload = {}
    with thinking_pref(True, "high"):
        apply_thinking_to_payload(payload, "mistral-medium", MISTRAL_URL)
    assert payload["reasoning_effort"] == "high"
    assert payload["chat_template_kwargs"]["enable_thinking"] is True


def test_think_off_disables_thinking_and_omits_effort():
    payload = {"reasoning_effort": "high"}
    with thinking_pref(False, "high"):
        apply_thinking_to_payload(payload, QWEN38, QWEN38_URL)
    assert "reasoning_effort" not in payload
    assert payload["chat_template_kwargs"]["enable_thinking"] is False


def test_no_pref_leaves_payload_alone():
    payload = {"model": "gpt-4o"}
    apply_thinking_to_payload(payload, "gpt-4o", OPENAI_URL)
    assert "reasoning_effort" not in payload
    assert "chat_template_kwargs" not in payload


def _capture_stream_payload(monkeypatch, url, model):
    import asyncio
    import json

    from src import llm_core

    class _FakeResp:
        status_code = 200

        async def aiter_lines(self):
            yield json.dumps({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]})
            yield "data: [DONE]"

        async def aread(self):
            return b""

    class _FakeStreamCtx:
        def __init__(self, captured):
            self._captured = captured

        async def __aenter__(self):
            return _FakeResp()

        async def __aexit__(self, *a):
            return False

    class _FakeClient:
        def __init__(self):
            self.captured_payload = {}

        def stream(self, method, url, **kw):
            self.captured_payload = kw.get("json") or {}
            return _FakeStreamCtx(self.captured_payload)

    client = _FakeClient()
    monkeypatch.setattr(llm_core, "_get_http_client", lambda: client)
    monkeypatch.setattr(llm_core, "_is_host_dead", lambda u: False)
    monkeypatch.setattr(llm_core, "note_model_activity", lambda *a, **k: None)
    monkeypatch.setattr(llm_core, "_clear_host_dead", lambda *a, **k: None)
    monkeypatch.setattr(llm_core, "get_context_length", lambda u, m: 32768)

    async def run():
        return [c async for c in llm_core.stream_llm(
            url, model, [{"role": "user", "content": "hi"}],
        )]

    asyncio.run(run())
    return client.captured_payload


def test_stream_payload_medium_qwen38(monkeypatch):
    with thinking_pref(True, "medium"):
        payload = _capture_stream_payload(monkeypatch, QWEN38_URL, QWEN38)
    assert payload["reasoning_effort"] == "medium"
    assert payload["chat_template_kwargs"]["enable_thinking"] is True


def test_stream_payload_high_qwen38_is_xhigh(monkeypatch):
    with thinking_pref(True, "high"):
        payload = _capture_stream_payload(monkeypatch, QWEN38_URL, QWEN38)
    assert payload["reasoning_effort"] == "xhigh"
    assert payload["chat_template_kwargs"]["enable_thinking"] is True


def test_unknown_effort_400_strips_fields_for_retry():
    payload = {
        "reasoning_effort": "high",
        "chat_template_kwargs": {"enable_thinking": True},
    }
    assert _retry_without_reasoning_controls(payload, 502, "bad gateway") is False
    assert payload["reasoning_effort"] == "high"
    assert _retry_without_reasoning_controls(
        payload, 400, "unknown variant: expected one of low, medium, xhigh"
    ) is True
    assert "reasoning_effort" not in payload
    assert "chat_template_kwargs" not in payload


def test_stream_payload_think_off_qwen38(monkeypatch):
    with thinking_pref(False, "high"):
        payload = _capture_stream_payload(monkeypatch, QWEN38_URL, QWEN38)
    assert "reasoning_effort" not in payload
    assert payload["chat_template_kwargs"]["enable_thinking"] is False
