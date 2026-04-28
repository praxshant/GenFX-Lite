"""
Central configuration module for GenFX Lite.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
# Project root is two levels up from this file (app/config.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

ASSETS_DIR = PROJECT_ROOT / "assets"
RUNS_DIR = PROJECT_ROOT / "runs"

FALLBACK_JSON_PATH = ASSETS_DIR / "fallback_scene.json"
FALLBACK_IMAGE_PATH = ASSETS_DIR / "fallback_image.png"
FALLBACK_RENDER_PATH = ASSETS_DIR / "fallback_render.png"

# --- API Keys ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")

# --- Models ---
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct:free")
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
HF_LLM_ENDPOINT = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
HF_IMAGE_ENDPOINT_BASE = "https://api-inference.huggingface.co/models"
HF_IMAGE_MODELS = [
    os.getenv("HF_IMAGE_MODEL", "stabilityai/stable-diffusion-3.5-medium"),
    "black-forest-labs/FLUX.1-schnell",
    "ByteDance/Hyper-SD",
]

# --- External Tools ---
BLENDER_PATH = os.getenv("BLENDER_PATH", "blender")

# --- Retries & Timeouts ---
LLM_RETRY_COUNT = 1
IMAGE_RETRY_COUNT = 3
API_TIMEOUT_SECONDS = 60
BLENDER_TIMEOUT_SECONDS = 120
