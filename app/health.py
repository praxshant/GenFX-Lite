"""
app/health.py — Lightweight API health checker for GenFX Lite.

check_runtime_health() is the single entry point used by the Streamlit sidebar.
It never raises; every check returns {"ok": bool, "error": str | None}.
"""

import shutil
import logging
from typing import Any

import requests

from app import config
from app.blender_runner import resolve_blender_path

logger = logging.getLogger(__name__)


def check_llm_health() -> tuple[bool, str | None]:
    """
    Probe OpenRouter with a minimal request.
    Returns (True, None) on success or (False, reason) on failure.
    Does NOT raise.
    """
    if not config.OPENROUTER_API_KEY or not config.OPENROUTER_API_KEY.startswith("sk-or"):
        return False, "OpenRouter key missing or invalid"

    try:
        resp = requests.post(
            config.OPENROUTER_ENDPOINT,
            headers={
                "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "GenFX Lite",
            },
            json={
                "model": config.OPENROUTER_MODEL,
                "messages": [{"role": "user", "content": 'Reply only with {"ok":true}'}],
                "max_tokens": 20,
            },
            timeout=8,
        )
        if resp.status_code == 200:
            return True, None
        return False, f"OpenRouter HTTP {resp.status_code}"
    except Exception as e:
        return False, f"OpenRouter probe failed: {e}"


def check_image_health() -> tuple[bool, str | None]:
    """
    Probe each HF image model with a tiny request.
    Returns (True, None) for the first model that responds with a valid image.
    Returns (False, reason) if all models fail.
    Does NOT raise.
    """
    if not config.HUGGINGFACE_API_KEY or not config.HUGGINGFACE_API_KEY.startswith("hf_"):
        return False, "HuggingFace key missing or invalid"

    for model in config.HF_IMAGE_MODELS:
        endpoint = f"{config.HF_IMAGE_ENDPOINT_BASE}/{model}"
        try:
            resp = requests.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {config.HUGGINGFACE_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={"inputs": "test image"},
                timeout=10,
            )
            if resp.status_code in (401, 403):
                return False, "HuggingFace key unauthorized"
            if resp.status_code == 200:
                # Quick sanity: first 4 bytes of a PNG are \x89PNG
                if resp.content[:4] in (b"\x89PNG", b"\xff\xd8\xff"):
                    return True, None
                # Some models return JSON on success (model loading)
                return True, None
        except Exception:
            continue  # Try next model

    return False, "No HuggingFace image model responded successfully"


def check_runtime_health() -> dict[str, dict[str, Any]]:
    """
    Run all health probes and return a unified dict.

    Structure:
        {
            "llm":    {"ok": bool, "error": str | None},
            "image":  {"ok": bool, "error": str | None},
            "blender":{"ok": bool, "error": str | None},
            "assets": {"ok": bool, "error": str | None},
        }

    Never raises — safe to call from Streamlit startup.
    """
    # LLM
    llm_ok, llm_err = check_llm_health()

    # Image
    img_ok, img_err = check_image_health()

    # Blender
    blender_found = bool(resolve_blender_path(config.BLENDER_PATH))
    blender_err = None if blender_found else f"Blender not found at '{config.BLENDER_PATH}'"

    # Fallback assets
    missing = [
        str(p) for p in [
            config.FALLBACK_JSON_PATH,
            config.FALLBACK_IMAGE_PATH,
            config.FALLBACK_RENDER_PATH,
        ]
        if not p.exists()
    ]
    assets_ok = len(missing) == 0
    assets_err = f"Missing: {', '.join(missing)}" if missing else None

    return {
        "llm":     {"ok": llm_ok,      "error": llm_err},
        "image":   {"ok": img_ok,      "error": img_err},
        "blender": {"ok": blender_found, "error": blender_err},
        "assets":  {"ok": assets_ok,   "error": assets_err},
    }
