"""
Module 3: Blender Runner
Invokes headless Blender as a subprocess to composite and render the final PNG.
Verifies paths gracefully and captures output to a log file.
"""

import logging
import shutil
import subprocess
from pathlib import Path

import sys
from PIL import Image

from app import config

logger = logging.getLogger(__name__)

BLENDER_SCRIPT = config.PROJECT_ROOT / "blender" / "render.py"


def resolve_blender_path(configured_path: str) -> str | None:
    """Robustly resolve the Blender executable path."""
    # 1. Try as a direct absolute path first
    p = Path(configured_path)
    if p.is_absolute() and p.is_file():
        return str(p)

    # 2. Try PATH lookup (for commands like 'blender')
    which_path = shutil.which(configured_path)
    if which_path:
        return which_path

    # 3. Try common install locations as a last resort
    if sys.platform == "win32":
        common_paths = [
            r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 4.3\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 4.1\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 3.6\blender.exe",
        ]
        for cp in common_paths:
            if Path(cp).is_file():
                return cp
    elif sys.platform == "darwin":
        mac_path = "/Applications/Blender.app/Contents/MacOS/Blender"
        if Path(mac_path).is_file():
            return mac_path

    return None


def run_blender(image_path: str, output_path: str | Path, log_path: str | Path) -> tuple[str, str | None]:
    """
    Run headless Blender to render the final output PNG from a source image.
    Returns:
        tuple[str, str | None]: (Path to render, Error string if fallback was used)
    """
    abs_image_path = Path(image_path).resolve()
    abs_output_path = Path(output_path).resolve()
    abs_log_path = Path(log_path).resolve()
    fallback_path = str(config.FALLBACK_RENDER_PATH)

    # ── Pre-flight: image must exist ──────────────────────────────────────────
    if not abs_image_path.exists():
        msg = f"Blender input image not found at {abs_image_path}"
        logger.warning(msg)
        return fallback_path, msg

    # ── Pre-flight: validate image is a real, loadable PNG ────────────────────
    try:
        with Image.open(abs_image_path) as img:
            img.verify()  # raises if corrupt
        logger.debug("Image pre-check passed: %s", abs_image_path)
    except Exception as e:
        msg = f"Image failed pre-validation (corrupt or wrong format): {e}"
        logger.warning(msg)
        return fallback_path, msg

    # ── Pre-flight: Blender executable ────────────────────────────────────────
    blender_exe = resolve_blender_path(config.BLENDER_PATH)
    if not blender_exe:
        msg = f"Blender executable not found at '{config.BLENDER_PATH}' and not in PATH. Install Blender or check BLENDER_PATH."
        logger.warning(msg)
        return fallback_path, msg

    # Use POSIX-style forward-slash paths for Blender compatibility on Windows.
    # Blender's Python (bpy.data.images.load) handles POSIX paths reliably
    # across all platforms; backslashes can cause issues in some versions.
    cmd = [
        blender_exe,
        "--background",
        "--python", str(BLENDER_SCRIPT.resolve()),
        "--",
        "--image", abs_image_path.as_posix(),
        "--output", abs_output_path.as_posix(),
    ]

    logger.info("Running Blender: %s", " ".join(cmd))

    # ── Run with one automatic retry ──────────────────────────────────────────
    max_attempts = 2
    last_error = ""

    for attempt in range(1, max_attempts + 1):
        try:
            abs_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(abs_log_path, "w", encoding="utf-8") as lf:
                result = subprocess.run(
                    cmd,
                    timeout=config.BLENDER_TIMEOUT_SECONDS,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

            # Trust the output file, not the exit code.
            # Blender headless mode often returns exit code 1 even after a
            # successful render — this is a known quirk.
            if abs_output_path.exists() and abs_output_path.stat().st_size > 0:
                logger.info("Blender render complete (attempt %d): %s", attempt, abs_output_path)
                return str(abs_output_path), None

            last_error = f"Blender exited (code {result.returncode}) but output file missing or empty (attempt {attempt})"
            logger.warning(last_error)

        except subprocess.TimeoutExpired:
            last_error = f"Blender render timed out after {config.BLENDER_TIMEOUT_SECONDS}s (attempt {attempt})"
            logger.warning(last_error)
        except Exception as e:
            last_error = f"Unexpected error running Blender: {type(e).__name__} - {str(e)} (attempt {attempt})"
            logger.warning(last_error)

    # Both attempts failed
    msg = f"Blender failed after {max_attempts} attempts. Last error: {last_error}"
    logger.warning(msg)
    return fallback_path, msg
