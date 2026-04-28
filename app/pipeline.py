"""
Module 5: Pipeline Orchestrator
Runs the full GenFX Lite pipeline: LLM → Image Gen → Blender Render.
Each stage creates output in a uniquely generated run folder preventing overlaps.
"""

import json
import logging
import uuid
from typing import Any

from app import config
from app.blender_runner import run_blender
from app.image_gen import generate_image
from app.llm_parser import parse_prompt

logger = logging.getLogger(__name__)


def run_pipeline(user_prompt: str) -> dict[str, Any]:
    """
    Execute the full GenFX Lite pipeline for the given user prompt.
    Returns a unified dict mapping status and paths.
    """
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    run_dir = config.RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "prompt": user_prompt,
        "scene_json": None,
        "image_path": None,
        "render_path": None,
        "status": {
            "llm": "pending",
            "image": "pending",
            "render": "pending",
        },
        "diagnostics": {
            "llm": None,
            "image": None,
            "render": None,
        },
        "run_id": run_id
    }

    # ── Stage 1: LLM Parsing ────────────────────────────────────────────────
    logger.info("Pipeline Stage 1: LLM Parsing")
    try:
        parser_result = parse_prompt(user_prompt)
        result["scene_json"] = parser_result.scene_json
        result["status"]["llm"] = parser_result.status
        result["diagnostics"]["llm"] = parser_result.error_message
        
        # Save JSON to run dir
        json_out_path = run_dir / "scene.json"
        with open(json_out_path, "w", encoding="utf-8") as f:
            json.dump(parser_result.scene_json, f, indent=2)
            
    except Exception as e:
        logger.error("Unexpected pipeline exception in Stage 1: %s", e)
        # In case of absolute critical failure escaping parser_prompt
        from app.llm_parser import load_fallback_json
        result["scene_json"] = load_fallback_json()
        result["status"]["llm"] = "fallback"
        result["diagnostics"]["llm"] = f"Critical Pipeline DB Exception: {e}"

    # ── Stage 2: Image Generation ────────────────────────────────────────────
    logger.info("Pipeline Stage 2: Image Generation")
    try:
        expected_img_path = run_dir / "image.png"
        image_path, img_error = generate_image(
            result["scene_json"],
            expected_img_path,
            user_prompt=user_prompt,
        )
        
        result["image_path"] = image_path
        if img_error:
            result["status"]["image"] = "fallback"
            result["diagnostics"]["image"] = img_error
        else:
            result["status"]["image"] = "ok"
    except Exception as e:
        logger.error("Unexpected pipeline exception in Stage 2: %s", e)
        result["image_path"] = str(config.FALLBACK_IMAGE_PATH)
        result["status"]["image"] = "fallback"
        result["diagnostics"]["image"] = f"Unhandled Error: {e}"

    # ── Stage 3: Blender Render ──────────────────────────────────────────────
    logger.info("Pipeline Stage 3: Blender Render")
    try:
        expected_render_path = run_dir / "render.png"
        log_path = run_dir / "blender.log"
        
        render_path, render_error = run_blender(
            image_path=result["image_path"],
            output_path=expected_render_path,
            log_path=log_path
        )
        
        result["render_path"] = render_path
        if render_error:
            result["status"]["render"] = "fallback"
            result["diagnostics"]["render"] = render_error
        else:
            result["status"]["render"] = "ok"
    except Exception as e:
        logger.error("Unexpected pipeline exception in Stage 3: %s", e)
        result["render_path"] = str(config.FALLBACK_RENDER_PATH)
        result["status"]["render"] = "fallback"
        result["diagnostics"]["render"] = f"Unhandled Error: {e}"

    logger.info(
        "Pipeline complete. Status: LLM=%s | Image=%s | Render=%s",
        result["status"]["llm"],
        result["status"]["image"],
        result["status"]["render"],
    )
    return result
