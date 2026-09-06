"""Studio-local GPU time-share API.

GET  /api/gpu/status   — who owns the 5090, and whether a switch is needed
POST /api/gpu/prepare  — park the other occupant and wait until ready

Tab clicks must never call these. Frontend calls them only on generate/send.
"""
from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel

from src.auth_helpers import require_user

logger = logging.getLogger(__name__)

_HELPER_CANDIDATES = (
    Path("/app/data/studio/scripts/gpu_timeshare.py"),
    Path(__file__).resolve().parents[1] / "data" / "studio" / "scripts" / "gpu_timeshare.py",
)
_helper = None


def _load_helper():
    global _helper
    if _helper is not None:
        return _helper
    for path in _HELPER_CANDIDATES:
        if path.is_file():
            spec = importlib.util.spec_from_file_location("gpu_timeshare", path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                _helper = mod
                return mod
    raise RuntimeError("gpu_timeshare.py not found")


class PrepareRequest(BaseModel):
    purpose: str
    endpoint_url: Optional[str] = None
    model: Optional[str] = None


def setup_gpu_routes() -> APIRouter:
    router = APIRouter(tags=["gpu-timeshare"])

    @router.get("/api/gpu/status")
    async def gpu_status(
        request: Request,
        purpose: str = "",
        endpoint_url: str = "",
        model: str = "",
    ):
        require_user(request)
        helper = _load_helper()
        return helper.status_payload(purpose, endpoint_url or None, model or None)

    @router.post("/api/gpu/prepare")
    async def gpu_prepare(request: Request, body: PrepareRequest):
        require_user(request)
        helper = _load_helper()
        return await helper.prepare_for(body.purpose, body.endpoint_url, body.model)

    return router
