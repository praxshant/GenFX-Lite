"""
Module 1: LLM Parser
Converts a natural language scene description into a validated Scene JSON dict.
Uses OpenAI API (primary) or HuggingFace LLM (fallback) for inference.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any

from app import config

logger = logging.getLogger(__name__)

REQUIRED_KEYS = [
    "scene_id",
    "environment",
    "lighting",
    "camera",
    "effects",
    "render_settings",
    "asset_refs",
]

SCENE_SCHEMA_EXAMPLE = """{
  "scene_id": "sc_001",
  "environment": {
    "type": "desert",
    "time_of_day": "golden_hour"
  },
  "lighting": {
    "preset": "cinematic_sunset",
    "intensity": 0.85
  },
  "camera": {
    "shot_type": "wide",
    "angle": "low"
  },
  "effects": [
    { "type": "dust", "density": "medium" }
  ],
  "render_settings": {
    "resolution": "1920x1080",
    "samples": 64
  },
  "asset_refs": []
}"""

SYSTEM_PROMPT = f"""You are a VFX pipeline data formatter for a production rendering system.
Your ONLY job is to convert any scene description into a valid JSON object.

STRICT RULES:
1. Return ONLY valid JSON. No explanation. No markdown. No ```json blocks.
2. Do not include any text before or after the JSON object.
3. Output must be directly parseable by json.loads() with zero pre-processing.
4. Match this schema EXACTLY — do not add or remove any top-level keys.
5. If output is not valid JSON, regenerate internally before returning.

Required schema:
{SCENE_SCHEMA_EXAMPLE}

Fill in values based on the user's scene description.
Keep asset_refs as an empty list — always.
Use snake_case for all string values."""

STRICT_JSON_SUFFIX = (
    "\n\nRespond ONLY with a valid JSON object. "
    "No explanation, no markdown, no backticks. "
    "Start your response with { and end with }."
)


class ValidationError(Exception):
    """Raised when a parsed dict does not match the required scene schema."""


@dataclass
class ParserResult:
    scene_json: dict[str, Any]
    status: str  # "ok" or "fallback"
    error_message: str | None = None
    provider_used: str | None = None


def validate_schema(data: dict) -> bool:
    """
    Validate that all required top-level keys are present in data and 
    nested types match expectations.
    """
    for key in REQUIRED_KEYS:
        if key not in data:
            raise ValidationError(f"Missing required key: '{key}'")
            
    if not isinstance(data.get("environment"), dict):
        raise ValidationError("'environment' must be a dict.")
    if not isinstance(data.get("lighting"), dict):
        raise ValidationError("'lighting' must be a dict.")
    if "intensity" in data["lighting"]:
        val = data["lighting"]["intensity"]
        if not isinstance(val, (int, float)):
            raise ValidationError("'lighting.intensity' must be numeric.")
    if not isinstance(data.get("camera"), dict):
        raise ValidationError("'camera' must be a dict.")
    if not isinstance(data.get("render_settings"), dict):
        raise ValidationError("'render_settings' must be a dict.")
        
    if not isinstance(data.get("effects"), list):
        raise ValidationError("'effects' must be a list.")
    if not isinstance(data.get("asset_refs"), list):
        raise ValidationError("'asset_refs' must be a list.")
        
    return True


def load_fallback_json() -> dict:
    """Load and return the pre-baked fallback scene JSON from disk."""
    try:
        with open(config.FALLBACK_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        # Failsafe if the fallback file itself is missing
        return json.loads(SCENE_SCHEMA_EXAMPLE)


def _parse_with_openrouter(user_prompt: str) -> str:
    """Call OpenRouter chat completion API and return raw response text."""
    import requests

    if not config.OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is not set.")

    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "GenFX Lite",
    }

    payload = {
        "model": config.OPENROUTER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT + STRICT_JSON_SUFFIX,
            },
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": 700,
    }

    resp = requests.post(
        config.OPENROUTER_ENDPOINT,
        headers=headers,
        json=payload,
        timeout=config.API_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()

    data = resp.json()
    return data["choices"][0]["message"]["content"] or ""


def _parse_with_openai(user_prompt: str) -> str:
    """Call OpenAI chat completion and return the raw response text."""
    import openai  # type: ignore

    if not config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set.")

    client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=500,
        timeout=config.API_TIMEOUT_SECONDS,
    )
    return response.choices[0].message.content or ""


def _parse_with_huggingface(user_prompt: str) -> str:
    """Call HuggingFace Inference API LLM and return the raw response text."""
    import requests  # type: ignore

    if not config.HUGGINGFACE_API_KEY:
        raise ValueError("HUGGINGFACE_API_KEY is not set.")

    full_prompt = (
        f"<s>[INST] {SYSTEM_PROMPT}\\n\\nUser request: {user_prompt} [/INST]"
    )
    headers = {
        "Authorization": f"Bearer {config.HUGGINGFACE_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": full_prompt,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.2,
            "return_full_text": False,
        },
    }
    resp = requests.post(
        config.HF_LLM_ENDPOINT, 
        headers=headers, 
        json=payload, 
        timeout=config.API_TIMEOUT_SECONDS
    )
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list) and len(data) > 0:
        return data[0].get("generated_text", "")
    return ""


def _extract_json(raw: str) -> dict:
    """
    Try to extract and parse a JSON object from raw LLM text output.
    Strips markdown code fences if present.
    """
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise json.JSONDecodeError("No JSON object found", text, 0)
    return json.loads(text[start:end])


def _attempt_parsing(user_prompt: str) -> tuple[dict[str, Any], str]:
    """Execute API call and JSON extraction."""
    
    if config.OPENROUTER_API_KEY and config.OPENROUTER_API_KEY.startswith("sk-or"):
        provider = "openrouter"
        raw_response = _parse_with_openrouter(user_prompt)
    elif config.OPENAI_API_KEY and config.OPENAI_API_KEY.startswith("sk-"):
        provider = "openai"
        raw_response = _parse_with_openai(user_prompt)
    elif config.HUGGINGFACE_API_KEY and config.HUGGINGFACE_API_KEY.startswith("hf_"):
        provider = "huggingface"
        raw_response = _parse_with_huggingface(user_prompt)
    else:
        raise ValueError(
            "No valid API key found. Set OPENROUTER_API_KEY, OPENAI_API_KEY, or HUGGINGFACE_API_KEY."
        )

    parsed = _extract_json(raw_response)
    validate_schema(parsed)
    return parsed, provider


def parse_prompt(user_prompt: str) -> ParserResult:
    """
    Parse a natural language scene description into a validated Scene JSON dict.
    Returns a ParserResult containing status and metadata.
    Attempts parsing with internal retries before using fallback.
    """
    last_error = ""
    
    for attempt in range(config.LLM_RETRY_COUNT + 1):
        try:
            parsed, provider = _attempt_parsing(user_prompt)
            # Ensure asset_refs is an empty list as required
            parsed["asset_refs"] = []
            logger.info("LLM parsing successful. scene_id=%s", parsed.get("scene_id"))
            return ParserResult(
                scene_json=parsed,
                status="ok",
                provider_used=provider
            )
        except json.JSONDecodeError as e:
            last_error = f"Invalid JSON format: {str(e)}"
            logger.warning("Attempt %d - %s", attempt + 1, last_error)
        except ValidationError as e:
            last_error = f"Schema validation failed: {str(e)}"
            logger.warning("Attempt %d - %s", attempt + 1, last_error)
        except Exception as e:
            last_error = f"API call failed: {type(e).__name__} - {str(e)}"
            logger.warning("Attempt %d - %s", attempt + 1, last_error)

    # All attempts exhausted
    if not last_error:
        last_error = "Unknown error occurred during parsing."
        
    logger.warning("LLM parsing exhausted. Using fallback JSON.")
    fallback = load_fallback_json()
    return ParserResult(
        scene_json=fallback,
        status="fallback",
        error_message=last_error,
        provider_used=None
    )
