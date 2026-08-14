"""Image/Video stay hidden when Comfy is unset, SSRF-blocked, or unreachable."""
from fastapi import HTTPException

from routes.comfy_routes import configured_comfy_base_url, validate_comfy_base_url


def test_configured_url_empty_by_default(monkeypatch):
    monkeypatch.setattr("src.settings.get_setting", lambda key, default="": default if key == "comfy_base_url" else default)
    assert configured_comfy_base_url() == ""


def test_validate_unset_raises_503(monkeypatch):
    try:
        validate_comfy_base_url("")
        assert False, "expected HTTPException"
    except HTTPException as exc:
        assert exc.status_code == 503
        assert "not configured" in str(exc.detail).lower()


def test_validate_ssrf_metadata_blocked():
    try:
        validate_comfy_base_url("http://169.254.169.254/latest/meta-data/")
        assert False, "expected HTTPException"
    except HTTPException as exc:
        assert exc.status_code == 400
        assert "ssrf" in str(exc.detail).lower() or "link-local" in str(exc.detail).lower()


def test_validate_loopback_allowed():
    assert validate_comfy_base_url("http://127.0.0.1:8188") == "http://127.0.0.1:8188"


def test_status_payload_reasons_are_documented():
    from pathlib import Path
    src = Path("routes/comfy_routes.py").read_text(encoding="utf-8")
    assert '"reason": "unset"' in src
    assert '"reason": "ssrf"' in src
    assert '"reason": "unreachable"' in src
    assert "available" in src
