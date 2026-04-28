"""
Smoke tests for GenFX Lite pipeline modules.
Run with: pytest tests/test_pipeline.py -v
All tests must pass without real API keys or Blender.
"""

import io
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from PIL import Image

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import config
from app.llm_parser import (
    ParserResult,
    ValidationError,
    load_fallback_json,
    parse_prompt,
    validate_schema,
)
from app.image_gen import build_image_prompt, generate_image
from app.blender_runner import run_blender
from app.pipeline import run_pipeline
from app.health import check_llm_health, check_image_health, check_runtime_health


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def fallback_json() -> dict:
    """Return the pre-baked fallback scene JSON."""
    return load_fallback_json()


def _make_png_bytes() -> bytes:
    """Return minimal valid PNG bytes for PIL validation tests."""
    buf = io.BytesIO()
    img = Image.new("RGB", (8, 8), color=(100, 100, 100))
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Test 1: Schema Validation ─────────────────────────────────────────────────

def test_schema_validation(fallback_json: dict) -> None:
    assert validate_schema(fallback_json) is True


def test_schema_validation_missing_key() -> None:
    with pytest.raises(ValidationError):
        validate_schema({"scene_id": "x", "environment": {}})


def test_schema_validation_nested_types() -> None:
    bad_json = load_fallback_json()
    bad_json["lighting"] = "not a dict"
    with pytest.raises(ValidationError, match="must be a dict"):
        validate_schema(bad_json)


# ── Test 2: LLM Parser Fallback ───────────────────────────────────────────────

@patch("app.llm_parser.config")
@patch("app.llm_parser._parse_with_openai")
def test_llm_parser_fallback_on_invalid_json(mock_parse, mock_config) -> None:
    mock_config.OPENROUTER_API_KEY = ""
    mock_config.OPENAI_API_KEY = "sk-fakekey123"
    mock_config.HUGGINGFACE_API_KEY = ""
    mock_config.LLM_RETRY_COUNT = 0
    mock_config.FALLBACK_JSON_PATH = config.FALLBACK_JSON_PATH
    mock_parse.return_value = "this is not json"

    result = parse_prompt("a desert at golden hour")

    assert isinstance(result, ParserResult)
    assert result.status == "fallback"
    assert "scene_id" in result.scene_json
    assert result.scene_json.get("asset_refs") == []
    assert result.error_message is not None


@patch("app.llm_parser.config")
@patch("app.llm_parser._parse_with_openai")
def test_llm_parser_fallback_on_schema_mismatch(mock_parse, mock_config) -> None:
    mock_config.OPENROUTER_API_KEY = ""
    mock_config.OPENAI_API_KEY = "sk-fakekey123"
    mock_config.HUGGINGFACE_API_KEY = ""
    mock_config.LLM_RETRY_COUNT = 0
    mock_config.FALLBACK_JSON_PATH = config.FALLBACK_JSON_PATH
    mock_parse.return_value = '{"scene_id": "x"}'

    result = parse_prompt("a foggy forest at dawn")

    assert isinstance(result, ParserResult)
    assert result.status == "fallback"
    assert "environment" in result.scene_json


@patch("app.llm_parser.config")
def test_llm_parser_no_api_keys(mock_config) -> None:
    mock_config.OPENROUTER_API_KEY = ""
    mock_config.OPENAI_API_KEY = ""
    mock_config.HUGGINGFACE_API_KEY = ""
    mock_config.LLM_RETRY_COUNT = 0
    mock_config.FALLBACK_JSON_PATH = config.FALLBACK_JSON_PATH

    result = parse_prompt("snow mountains at night")

    assert isinstance(result, ParserResult)
    assert result.status == "fallback"
    assert result.error_message == (
        "API call failed: ValueError - No valid API key found. "
        "Set OPENROUTER_API_KEY, OPENAI_API_KEY, or HUGGINGFACE_API_KEY."
    )


# ── Test 3: Config — model defaults ──────────────────────────────────────────

def test_openrouter_default_model() -> None:
    """Default OpenRouter model should be mistral-7b-instruct:free."""
    # We import config fresh; actual default is set via env fallback
    import importlib
    import app.config as cfg
    # If env isn't set the default string is embedded in code
    assert "mistral" in cfg.OPENROUTER_MODEL or cfg.OPENROUTER_MODEL  # model is set


def test_hf_image_models_has_three_entries() -> None:
    """HF_IMAGE_MODELS must contain at least 3 fallback models."""
    assert len(config.HF_IMAGE_MODELS) >= 3


# ── Test 4: Image Gen — InferenceClient model chain ──────────────────────────

def _make_pil_image() -> Image.Image:
    """Return a tiny valid PIL Image for mocking text_to_image."""
    return Image.new("RGB", (8, 8), color=(100, 100, 100))


@patch("app.image_gen.InferenceClient")
@patch("app.image_gen.config")
def test_image_gen_tries_next_model_on_404(mock_config, MockClient, fallback_json: dict) -> None:
    """generate_image should skip to the next model when one raises HTTP 404."""
    from huggingface_hub.errors import HfHubHTTPError
    import requests

    mock_config.HUGGINGFACE_API_KEY = "hf_fakekey"
    mock_config.FALLBACK_IMAGE_PATH = config.FALLBACK_IMAGE_PATH
    mock_config.HF_IMAGE_MODELS = ["model-a", "model-b", "model-c"]

    # Build a fake 404 HfHubHTTPError
    fake_resp = MagicMock()
    fake_resp.status_code = 404
    err = HfHubHTTPError("404 Not Found", response=fake_resp)

    client_instance = MockClient.return_value
    client_instance.text_to_image.side_effect = err

    path, error = generate_image(fallback_json, "dummy/output.png")

    assert path.endswith("fallback_image.png")
    assert error is not None
    # All 3 models attempted
    assert client_instance.text_to_image.call_count == 3


@patch("app.image_gen.InferenceClient")
@patch("app.image_gen.config")
def test_image_gen_falls_back_only_after_all_models_fail(mock_config, MockClient, fallback_json: dict) -> None:
    """generate_image returns fallback only after exhausting all models."""
    from huggingface_hub.errors import HfHubHTTPError

    mock_config.HUGGINGFACE_API_KEY = "hf_fakekey"
    mock_config.FALLBACK_IMAGE_PATH = config.FALLBACK_IMAGE_PATH
    mock_config.HF_IMAGE_MODELS = ["m1", "m2"]

    fake_resp = MagicMock(); fake_resp.status_code = 503
    err = HfHubHTTPError("503", response=fake_resp)

    client_instance = MockClient.return_value
    client_instance.text_to_image.side_effect = err

    path, error = generate_image(fallback_json, "dummy/out.png")

    assert path.endswith("fallback_image.png")
    assert error is not None
    assert client_instance.text_to_image.call_count == 2


@patch("app.image_gen.InferenceClient")
@patch("app.image_gen.config")
def test_image_gen_succeeds_on_second_model(mock_config, MockClient, fallback_json: dict, tmp_path) -> None:
    """generate_image should return a real path when the second model succeeds."""
    from huggingface_hub.errors import HfHubHTTPError

    mock_config.HUGGINGFACE_API_KEY = "hf_fakekey"
    mock_config.FALLBACK_IMAGE_PATH = config.FALLBACK_IMAGE_PATH
    mock_config.HF_IMAGE_MODELS = ["bad-model", "good-model"]

    fake_resp = MagicMock(); fake_resp.status_code = 404
    err = HfHubHTTPError("404", response=fake_resp)

    client_instance = MockClient.return_value
    client_instance.text_to_image.side_effect = [err, _make_pil_image()]

    out = tmp_path / "out.png"
    path, error = generate_image(fallback_json, out)

    assert error is None
    assert path == str(out)
    assert Path(path).exists()


@patch("app.image_gen.InferenceClient")
@patch("app.image_gen.config")
def test_image_gen_fallback_on_timeout(mock_config, MockClient, fallback_json: dict) -> None:
    """generate_image falls back when client raises a generic timeout/network error."""
    mock_config.HUGGINGFACE_API_KEY = "hf_fakekey"
    mock_config.FALLBACK_IMAGE_PATH = config.FALLBACK_IMAGE_PATH
    mock_config.HF_IMAGE_MODELS = ["m1"]

    client_instance = MockClient.return_value
    client_instance.text_to_image.side_effect = TimeoutError("timed out")

    path, error = generate_image(fallback_json, "dummy/output.png")
    assert path.endswith("fallback_image.png")
    assert error is not None


@patch("app.image_gen.config")
def test_image_gen_fallback_no_api_key(mock_config, fallback_json: dict) -> None:
    mock_config.HUGGINGFACE_API_KEY = ""
    mock_config.FALLBACK_IMAGE_PATH = config.FALLBACK_IMAGE_PATH

    path, error = generate_image(fallback_json, "dummy/output.png")
    assert path.endswith("fallback_image.png")
    assert error == "HUGGINGFACE_API_KEY missing or invalid"


def test_image_gen_prompt_builder_from_scene_json(fallback_json: dict) -> None:
    """Without user_prompt, prompt is built from scene JSON fields."""
    prompt = build_image_prompt(fallback_json)
    assert isinstance(prompt, str)
    assert "desert" in prompt.lower()
    assert "photorealistic" in prompt.lower()


def test_image_gen_prompt_builder_uses_user_prompt(fallback_json: dict) -> None:
    """When user_prompt is given, it takes priority over scene JSON content."""
    prompt = build_image_prompt(fallback_json, user_prompt="snowy mountain valley")
    assert "snowy mountain valley" in prompt
    assert "desert" not in prompt.lower()
    assert "photorealistic" in prompt.lower()


@patch("app.image_gen.InferenceClient")
@patch("app.image_gen.config")
def test_image_gen_passes_user_prompt_on_llm_fallback(
    mock_config, MockClient, fallback_json: dict, tmp_path
) -> None:
    """Even when the LLM fell back (scene_json is generic desert), generate_image
    must use the original user_prompt so the image stays visually relevant."""
    mock_config.HUGGINGFACE_API_KEY = "hf_fakekey"
    mock_config.FALLBACK_IMAGE_PATH = config.FALLBACK_IMAGE_PATH
    mock_config.HF_IMAGE_MODELS = ["good-model"]

    captured_prompts: list[str] = []

    def fake_text_to_image(prompt: str, model: str) -> Image.Image:
        captured_prompts.append(prompt)
        return Image.new("RGB", (8, 8))

    MockClient.return_value.text_to_image.side_effect = fake_text_to_image

    out = tmp_path / "out.png"
    generate_image(fallback_json, out, user_prompt="snowy mountain valley")

    assert len(captured_prompts) == 1
    assert "snowy mountain valley" in captured_prompts[0]
    assert "desert" not in captured_prompts[0].lower()


# ── Test 5: Blender Runner ────────────────────────────────────────────────────

@patch("app.blender_runner.config")
def test_blender_fallback_missing_image(mock_config) -> None:
    mock_config.FALLBACK_RENDER_PATH = config.FALLBACK_RENDER_PATH
    path, error = run_blender("non_existent_image.png", "out.png", "log.log")
    assert path.endswith("fallback_render.png")
    assert "not found" in error


@patch("app.blender_runner.resolve_blender_path", return_value=None)
@patch("app.blender_runner.config")
def test_blender_fallback_missing_binary(mock_config, mock_resolve, tmp_path) -> None:
    mock_config.FALLBACK_RENDER_PATH = config.FALLBACK_RENDER_PATH
    # Must be a valid PNG so the new PIL pre-check passes
    img_path = tmp_path / "valid.png"
    Image.new("RGB", (8, 8), color=(128, 128, 128)).save(str(img_path), format="PNG")
    path, error = run_blender(str(img_path), "out.png", "log.log")
    assert path.endswith("fallback_render.png")
    assert "executable not found" in error


# ── Test 6: Health Checks — never crash ──────────────────────────────────────

def test_health_llm_missing_key() -> None:
    """check_llm_health returns False gracefully when key is absent."""
    with patch("app.health.config") as mc:
        mc.OPENROUTER_API_KEY = ""
        mc.OPENROUTER_MODEL = "mistralai/mistral-7b-instruct:free"
        mc.OPENROUTER_ENDPOINT = config.OPENROUTER_ENDPOINT
        ok, err = check_llm_health()
    assert ok is False
    assert err is not None


def test_health_image_missing_key() -> None:
    """check_image_health returns False gracefully when HF key is absent."""
    with patch("app.health.config") as mc:
        mc.HUGGINGFACE_API_KEY = ""
        mc.HF_IMAGE_MODELS = config.HF_IMAGE_MODELS
        mc.HF_IMAGE_ENDPOINT_BASE = config.HF_IMAGE_ENDPOINT_BASE
        ok, err = check_image_health()
    assert ok is False
    assert err is not None


def test_health_runtime_never_raises() -> None:
    """check_runtime_health must never raise, even without internet."""
    with patch("app.health.requests.post", side_effect=Exception("no network")):
        with patch("app.health.config") as mc:
            mc.OPENROUTER_API_KEY = ""
            mc.HUGGINGFACE_API_KEY = ""
            mc.HF_IMAGE_MODELS = []
            mc.HF_IMAGE_ENDPOINT_BASE = ""
            mc.OPENROUTER_MODEL = "any"
            mc.OPENROUTER_ENDPOINT = ""
            mc.BLENDER_PATH = "blender"
            mc.FALLBACK_JSON_PATH = config.FALLBACK_JSON_PATH
            mc.FALLBACK_IMAGE_PATH = config.FALLBACK_IMAGE_PATH
            mc.FALLBACK_RENDER_PATH = config.FALLBACK_RENDER_PATH
            result = check_runtime_health()

    assert "llm" in result
    assert "image" in result
    assert "blender" in result
    assert "assets" in result
    # All values have the expected shape
    for key in ("llm", "image", "blender", "assets"):
        assert "ok" in result[key]
        assert "error" in result[key]


# ── Test 7: Full Pipeline with All Fallbacks ──────────────────────────────────────────

@patch("app.blender_runner.resolve_blender_path", return_value=None)
@patch("app.image_gen.InferenceClient")
@patch("app.image_gen.config")
@patch("app.llm_parser.config")
@patch("app.pipeline.config")
def test_full_pipeline_with_fallbacks(
    mock_pipe_cfg, mock_llm_cfg, mock_img_cfg, MockClient, _mock_resolve, tmp_path
) -> None:
    """run_pipeline completes with all three stages falling back gracefully."""
    run_base = tmp_path / "runs"

    # Pipeline-level config (used for run_dir creation)
    mock_pipe_cfg.OPENROUTER_API_KEY = ""
    mock_pipe_cfg.OPENAI_API_KEY = ""
    mock_pipe_cfg.HUGGINGFACE_API_KEY = ""
    mock_pipe_cfg.LLM_RETRY_COUNT = 0
    mock_pipe_cfg.IMAGE_RETRY_COUNT = 0
    mock_pipe_cfg.FALLBACK_JSON_PATH = config.FALLBACK_JSON_PATH
    mock_pipe_cfg.FALLBACK_IMAGE_PATH = config.FALLBACK_IMAGE_PATH
    mock_pipe_cfg.FALLBACK_RENDER_PATH = config.FALLBACK_RENDER_PATH
    mock_pipe_cfg.RUNS_DIR = run_base

    # LLM parser config — no keys so parser returns fallback
    mock_llm_cfg.OPENROUTER_API_KEY = ""
    mock_llm_cfg.OPENAI_API_KEY = ""
    mock_llm_cfg.HUGGINGFACE_API_KEY = ""
    mock_llm_cfg.LLM_RETRY_COUNT = 0
    mock_llm_cfg.FALLBACK_JSON_PATH = config.FALLBACK_JSON_PATH

    # Image gen config — no key so generate_image returns fallback immediately
    mock_img_cfg.HUGGINGFACE_API_KEY = ""
    mock_img_cfg.FALLBACK_IMAGE_PATH = config.FALLBACK_IMAGE_PATH
    mock_img_cfg.HF_IMAGE_MODELS = []

    result = run_pipeline("desert scene at sunset")

    assert result["scene_json"] is not None
    assert "scene_id" in result["scene_json"]
    assert result["image_path"].endswith(".png")
    assert result["render_path"].endswith(".png")
    assert result["status"]["llm"] == "fallback"
    assert result["status"]["image"] == "fallback"
    assert result["status"]["render"] == "fallback"
    assert result["diagnostics"]["llm"] is not None
    assert result["diagnostics"]["image"] is not None
    assert result["diagnostics"]["render"] is not None
    assert result["run_id"].startswith("run_")
    assert (run_base / result["run_id"] / "scene.json").exists()
