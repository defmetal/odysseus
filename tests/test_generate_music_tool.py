"""generate_music posts to /api/music, never /api/comfy kind=music."""
import ast
from pathlib import Path

import pytest

from src.tools.music import _parse_generate_music, do_generate_music


def test_parse_plain_prompt():
    parsed = _parse_generate_music("lofi rain, soft piano")
    assert parsed["prompt"] == "lofi rain, soft piano"


def test_parse_json_caption_lyrics():
    parsed = _parse_generate_music('{"prompt": "jazz trio", "lyrics": "[Verse]\\nhello", "seconds": 30}')
    assert parsed["prompt"] == "jazz trio"
    assert "[Verse]" in parsed["lyrics"]
    assert parsed["seconds"] == 30


def test_parse_multiline_fields():
    parsed = _parse_generate_music("caption here\n[Chorus]\nla\nminimax_music3_comfy\n45")
    assert parsed["prompt"] == "caption here"
    assert parsed["lyrics"] == "[Chorus]"
    assert parsed["backend"] == "minimax_music3_comfy"
    assert parsed["seconds"] == "45"


@pytest.mark.asyncio
async def test_generate_music_requires_prompt():
    result = await do_generate_music("")
    assert result.get("exit_code") == 1
    assert "required" in result.get("error", "").lower()


@pytest.mark.asyncio
async def test_generate_music_posts_music_api_not_comfy(monkeypatch):
    calls = []

    class _Resp:
        status_code = 200
        content = b"{}"

        def json(self):
            return {"job_id": "mj-generic-1"}

    class _JobResp:
        status_code = 200

        def json(self):
            return {
                "status": "done",
                "backend": "minimax_music3_comfy",
                "audio": [{"url": "/api/generated-audio/abcd1234.mp3", "id": "a1", "filename": "abcd1234.mp3"}],
            }

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, json=None, headers=None):
            calls.append(("POST", url, json))
            return _Resp()

        async def get(self, url, headers=None):
            calls.append(("GET", url, None))
            return _JobResp()

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    result = await do_generate_music("generic ambient pad")
    assert result["exit_code"] == 0
    assert result["audio_url"] == "/api/generated-audio/abcd1234.mp3"
    assert any(url.endswith("/api/music/generate") for _m, url, _b in calls if _m == "POST")
    assert any("/api/music/job/" in url for _m, url, _b in calls if _m == "GET")
    assert not any("/api/comfy/" in url for _m, url, _b in calls)
    posted = next(body for method, url, body in calls if method == "POST")
    assert posted["params"]["prompt"] == "generic ambient pad"
    assert "kind" not in posted


def test_tool_wiring_covers_six_layers():
    root = Path(__file__).resolve().parents[1]
    tags = (root / "src" / "agent_tools" / "__init__.py").read_text(encoding="utf-8")
    assert '"generate_music"' in tags
    sections = (root / "src" / "agent_loop.py").read_text(encoding="utf-8")
    assert '"generate_music":' in sections
    assert '"media":' in sections and "generate_music" in sections
    execution = (root / "src" / "tool_execution.py").read_text(encoding="utf-8")
    assert "do_generate_music" in execution
    assert 'tool == "generate_music"' in execution
    index = (root / "src" / "tool_index.py").read_text(encoding="utf-8")
    assert '"generate_music"' in index
    security = (root / "src" / "tool_security.py").read_text(encoding="utf-8")
    assert '"generate_music"' in security
    music_src = ast.parse((root / "src" / "tools" / "music.py").read_text(encoding="utf-8"))
    names = {n.name for n in music_src.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert "do_generate_music" in names
    facades = (root / "src" / "tools" / "__init__.py").read_text(encoding="utf-8")
    assert "do_generate_music" in facades
