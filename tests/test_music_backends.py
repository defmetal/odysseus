"""Pluggable MiniMax Music 3 backends — generic fixture names, no secrets."""
import json

from src.music_backends import (
    backend_available,
    default_backend_key,
    get_backend,
    list_backends,
    load_music_backends,
    music_status,
)


def test_registry_has_comfy_and_api_engines():
    registry = load_music_backends()
    backends = registry["backends"]
    assert backends["minimax_music3_comfy"]["engine"] == "minimax_music3_comfy"
    assert backends["minimax_music3_api"]["engine"] == "minimax_music3_api"
    assert backends["minimax_music3_comfy"]["needs_api_key"] is False
    assert backends["minimax_music3_api"]["needs_api_key"] is True
    dumped = json.dumps(registry)
    assert "sk-" not in dumped
    assert "Bearer" not in dumped


def test_comfy_backend_hidden_without_url(monkeypatch):
    monkeypatch.setattr("src.music_backends._comfy_url", lambda: "")
    monkeypatch.setattr("src.music_backends.minimax_api_key", lambda: "")
    entry = load_music_backends()["backends"]["minimax_music3_comfy"]
    assert backend_available(entry, comfy_reachable=None) is False
    assert default_backend_key(comfy_reachable=None) is None


def test_comfy_backend_hidden_when_unreachable(monkeypatch):
    monkeypatch.setattr("src.music_backends._comfy_url", lambda: "http://127.0.0.1:8188")
    monkeypatch.setattr("src.music_backends.minimax_api_key", lambda: "")
    entry = load_music_backends()["backends"]["minimax_music3_comfy"]
    assert backend_available(entry, comfy_reachable=False) is False
    assert backend_available(entry, comfy_reachable=True) is True


def test_api_backend_hidden_without_key(monkeypatch):
    monkeypatch.setattr("src.music_backends.minimax_api_key", lambda: "")
    entry = load_music_backends()["backends"]["minimax_music3_api"]
    assert backend_available(entry) is False


def test_api_backend_available_with_key(monkeypatch):
    monkeypatch.setattr("src.music_backends.minimax_api_key", lambda: "test-key")
    monkeypatch.setattr("src.music_backends._comfy_url", lambda: "")
    entry = load_music_backends()["backends"]["minimax_music3_api"]
    assert backend_available(entry) is True
    assert default_backend_key() == "minimax_music3_api"


def test_music_status_disabled(monkeypatch):
    monkeypatch.setattr("src.music_backends.music_gen_enabled", lambda: False)
    monkeypatch.setattr("src.music_backends.minimax_api_key", lambda: "test-key")
    status = music_status()
    assert status["available"] is False
    assert status["reason"] == "disabled"


def test_music_status_no_backend(monkeypatch):
    monkeypatch.setattr("src.music_backends.music_gen_enabled", lambda: True)
    monkeypatch.setattr("src.music_backends.minimax_api_key", lambda: "")
    monkeypatch.setattr("src.music_backends._comfy_url", lambda: "")
    status = music_status(comfy_reachable=False)
    assert status["available"] is False
    assert status["reason"] == "no_backend"


def test_list_backends_marks_availability(monkeypatch):
    monkeypatch.setattr("src.music_backends.minimax_api_key", lambda: "test-key")
    monkeypatch.setattr("src.music_backends._comfy_url", lambda: "http://127.0.0.1:8188")
    items = list_backends(comfy_reachable=True)
    by_key = {b["key"]: b for b in items}
    assert by_key["minimax_music3_comfy"]["available"] is True
    assert by_key["minimax_music3_api"]["available"] is True
    chosen = get_backend("minimax_music3_comfy", comfy_reachable=True)
    assert chosen["engine"] == "minimax_music3_comfy"
