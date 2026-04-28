"""
Module 2: Image Generator
Converts a Scene JSON dict into a PNG image using HuggingFace InferenceClient.
Tries multiple models in order before returning the fallback asset.
"""

import logging
from pathlib import Path
from typing import Any

from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

from app import config

logger = logging.getLogger(__name__)


def build_image_prompt(scene_json: dict[str, Any], user_prompt: str | None = None) -> str:
    """
    Build a descriptive text prompt for image generation.

    If user_prompt is provided, it is used directly (with cinematic suffixes added).
    This ensures relevance even when the LLM parser fell back to a generic scene JSON.
    Otherwise, the prompt is derived from the structured scene_json fields.
    """
    if user_prompt:
        prompt = (
            user_prompt.strip()
            + ", cinematic, photorealistic, 8k, film still, VFX production quality"
        )
        logger.debug("Using user prompt for image: %s", prompt)
        return prompt

    env = scene_json.get("environment", {}).get("type", "scene")
    tod = scene_json.get("environment", {}).get("time_of_day", "")
    light = scene_json.get("lighting", {}).get("preset", "")
    shot = scene_json.get("camera", {}).get("shot_type", "")
    angle = scene_json.get("camera", {}).get("angle", "")
    effects_list = scene_json.get("effects", [])
    fx = ", ".join(
        e.get("type", "") for e in effects_list if isinstance(e, dict) and e.get("type")
    )

    parts = [
        f"cinematic {env} scene",
        tod.replace("_", " ") if tod else "",
        light.replace("_", " ") if light else "",
        f"{shot} shot" if shot else "",
        f"{angle} angle" if angle else "",
        fx if fx else "",
        "photorealistic",
        "8k",
        "film still",
        "VFX production quality",
    ]
    prompt = ", ".join(p for p in parts if p)
    logger.debug("Built image prompt from scene JSON: %s", prompt)
    return prompt


def generate_image(
    scene_json: dict[str, Any],
    output_path: str | Path,
    user_prompt: str | None = None,
) -> tuple[str, str | None]:
    """
    Generate a PNG image using HuggingFace InferenceClient.

    user_prompt: the original text entered by the user. When provided, it is
    used as the image prompt even if the LLM parser fell back, so the visual
    output always matches what the user asked for.

    Returns:
        tuple[str, str | None]: (path_to_image, error_string_or_None)
    """
    output_target = Path(output_path)
    fallback_path = str(config.FALLBACK_IMAGE_PATH)

    if not config.HUGGINGFACE_API_KEY or not config.HUGGINGFACE_API_KEY.startswith("hf_"):
        logger.warning("No valid HUGGINGFACE_API_KEY found. Using fallback image.")
        return fallback_path, "HUGGINGFACE_API_KEY missing or invalid"

    client = InferenceClient(token=config.HUGGINGFACE_API_KEY)
    prompt = build_image_prompt(scene_json, user_prompt=user_prompt)
    last_error = ""

    for model in config.HF_IMAGE_MODELS:
        logger.info("Trying HF model via InferenceClient: %s", model)
        try:
            # Returns a PIL Image directly — no raw HTTP handling needed
            pil_image = client.text_to_image(prompt, model=model)

            # Force RGB + explicit PNG format.
            # Some HF models (e.g. FLUX) return JPEG-encoded data internally;
            # saving without format="PNG" can produce a corrupt .png that
            # Blender refuses to load.
            pil_image = pil_image.convert("RGB")
            output_target = output_target.with_suffix(".png")
            output_target.parent.mkdir(parents=True, exist_ok=True)
            pil_image.save(str(output_target), format="PNG")

            # Verify the file actually landed on disk and is not empty
            if not output_target.exists() or output_target.stat().st_size < 50:
                last_error = f"[{model}] Image file missing or corrupt after save"
                logger.warning(last_error)
                continue

            logger.info("Image saved to %s (model: %s, %dx%d)",
                        output_target, model, pil_image.width, pil_image.height)
            return str(output_target), None

        except HfHubHTTPError as e:
            status = e.response.status_code if hasattr(e, "response") else "?"
            last_error = f"[{model}] HTTP {status}: {str(e)[:120]}"
            logger.warning(last_error)
            # 404 → model unavailable, skip immediately
            # 429 / 503 → also skip; InferenceClient doesn't expose retry control
            continue

        except Exception as e:
            last_error = f"[{model}] Error: {str(e)[:120]}"
            logger.warning(last_error)
            continue

    logger.warning("All HF image models failed. Using fallback asset.")
    return fallback_path, f"All HuggingFace image models failed: {last_error}"
