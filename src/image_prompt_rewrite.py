# src/image_prompt_rewrite.py
"""CPU-only Image-tab prompt rewriter (Ollama 3B).

Turns Grok-style character-name hints into local-safe diffusion prompts
before ComfyUI sees them. NEVER loads a model onto the GPU: num_gpu=0
and the CPU-pinned tag only. On timeout/failure, returns the original.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

REWRITE_MODEL = "qwen2.5-3b-cpu:latest"
REWRITE_TIMEOUT_SECONDS = 8.0
# Names that must never reach CLIP. Not an allow-list of hints.
BANNED = (
    "sailor moon",
    "sailor pluto",
    "pluto",
    "usagi",
    "luna",
    "tuxedo mask",
    "sailor fuku",
    "smoon",
    "mamoru",
)

SYSTEM_PROMPT = (
    "You rewrite casual image-tab text into a proper local diffusion prompt.\n"
    "The user will be vague. Your job is to make a clean CLIP prompt that keeps their intent.\n"
    "Keep every fact they stated: gender, hair, outfit, props, style, era.\n"
    "If they said woman / girl / she, write adult woman. If they said man / guy / he / masculine, write adult man.\n"
    "If they did not state a gender, write this person. Do not default to woman or man.\n"
    "If they said \"this dress\" or \"this\", write wearing this exact outfit.\n"
    "If they named a held prop (sword, fan), keep that prop. Do not invent a new one.\n"
    "Add style words they asked for (1990s Toei cel, hard ink, flat color). Do not add Sailor Moon.\n"
    "Do not copy copyrighted character names. Expand those only into body and era words.\n"
    "Output ONLY the rewritten prompt. No quotes, no preamble."
)

_FEWSHOT_USER = "put this dress on a toei 90s style anime woman in black hair"
_FEWSHOT_ASSISTANT = (
    "adult woman, black hair, 1990s Toei cel, hard ink, flat color, "
    "wearing this exact dress, hands visible"
)

_PREAMBLE_RE = re.compile(
    r"^\s*(here(?:'s| is)|sure[,.]?|rewritten(?: prompt)?\s*:|prompt\s*:)\s*",
    re.I,
)


def _ollama_base_url() -> str:
    raw = (
        os.getenv("OLLAMA_BASE_URL")
        or os.getenv("OLLAMA_URL")
        or ""
    ).strip()
    if raw:
        parsed = urlparse(raw if "://" in raw else "http://" + raw)
        host = parsed.hostname or "host.docker.internal"
        port = parsed.port or 11434
        scheme = parsed.scheme or "http"
        return f"{scheme}://{host}:{port}"
    if os.path.exists("/.dockerenv"):
        return "http://host.docker.internal:11434"
    return "http://127.0.0.1:11434"


def _contains_banned(text: str) -> bool:
    low = (text or "").lower()
    return any(b in low for b in BANNED)


def _looks_garbage(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 12:
        return True
    if t.count("\n") > 6:
        return True
    if t[:1] in {"{", "["} and ("}" in t or "]" in t):
        return True
    return False


def _clean_rewrite(text: str) -> str:
    t = (text or "").strip().strip('"').strip("'").strip()
    t = _PREAMBLE_RE.sub("", t).strip().strip('"').strip("'").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t)
        t = t.strip()
    t = " ".join(t.split())
    return t


def _fallback(original: str, model: Optional[str] = None) -> dict:
    return {
        "original": original,
        "rewritten": original,
        "model": model,
        "used_rewrite": False,
    }


async def _ollama_chat(messages: list[dict], *, model: str, timeout: float) -> str:
    url = _ollama_base_url().rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "keep_alive": "10m",
        "options": {
            "num_gpu": 0,
            "temperature": 0.2,
            "num_predict": 160,
        },
        "messages": messages,
    }
    timeout_cfg = httpx.Timeout(connect=2.0, read=timeout, write=4.0, pool=2.0)
    async with httpx.AsyncClient(timeout=timeout_cfg) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
    return ((data.get("message") or {}).get("content") or "")


async def rewrite_image_prompt(prompt: str, *, img2img: bool = False) -> dict:
    """Rewrite an Image-tab prompt on the host CPU 3B. Never raises."""
    original = (prompt or "").strip()
    if not original:
        return _fallback(original)

    user_text = original
    if img2img:
        user_text = (
            original
            + "\n(img2img: keep this exact outfit and any held props; hands visible. "
            "Keep their stated gender, hair, and props. Do not invent a gender, weapon, or garment.)"
        )

    base_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    fewshot_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _FEWSHOT_USER},
        {"role": "assistant", "content": _FEWSHOT_ASSISTANT},
        {"role": "user", "content": user_text},
    ]

    model = REWRITE_MODEL
    try:
        raw = await asyncio.wait_for(
            _ollama_chat(base_messages, model=model, timeout=REWRITE_TIMEOUT_SECONDS),
            timeout=REWRITE_TIMEOUT_SECONDS,
        )
        cleaned = _clean_rewrite(raw)
        if _contains_banned(cleaned) or _looks_garbage(cleaned):
            logger.info("image rewrite: first pass banned/garbage; retrying few-shot")
            raw = await asyncio.wait_for(
                _ollama_chat(fewshot_messages, model=model, timeout=REWRITE_TIMEOUT_SECONDS),
                timeout=REWRITE_TIMEOUT_SECONDS,
            )
            cleaned = _clean_rewrite(raw)
        if _contains_banned(cleaned) or _looks_garbage(cleaned) or cleaned == original:
            logger.info("image rewrite: rejected output; falling back to original")
            return _fallback(original, model)
        return {
            "original": original,
            "rewritten": cleaned,
            "model": model,
            "used_rewrite": True,
        }
    except Exception as e:
        logger.warning("image rewrite failed (%s); using original prompt", e)
        return _fallback(original, model)


def rewrite_image_prompt_sync(prompt: str, *, img2img: bool = False) -> dict:
    """Sync wrapper for smoke / unit-style calls."""
    return asyncio.run(rewrite_image_prompt(prompt, img2img=img2img))
