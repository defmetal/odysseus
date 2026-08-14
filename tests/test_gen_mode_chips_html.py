"""Composer chips exist and stay hidden until a backend is available."""
from pathlib import Path


def test_index_has_hidden_image_video_music_chips():
    html = Path("static/index.html").read_text(encoding="utf-8")
    for chip_id in ("mode-image-btn", "mode-video-btn", "mode-music-btn"):
        assert f'id="{chip_id}"' in html
        # Absence-clean: chips ship hidden; JS unhides after /status.
        start = html.find(f'id="{chip_id}"')
        snippet = html[max(0, start - 80): start + 80]
        assert "hidden" in snippet
    assert 'id="mode-agent-btn"' in html
    assert 'id="mode-chat-btn"' in html
    assert 'id="gen-params-wrap"' in html
    assert 'id="set-comfyBaseUrl"' in html
    assert 'id="set-musicEnabledToggle"' in html
    assert "smoon" not in html
    assert "toei90s" not in html
    assert "studio_toei" not in html


def test_app_js_hides_chips_from_status():
    src = Path("static/app.js").read_text(encoding="utf-8")
    assert "/api/comfy/status" in src
    assert "/api/music/status" in src
    assert "mode-image-btn" in src
    assert "mode-music-btn" in src
    assert "imageBtn.hidden" in src
    assert "musicBtn.hidden" in src
