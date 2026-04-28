# GenFX Lite

GenFX Lite is a production-style prototype demonstrating how Large Language Models and diffusion-based image generation integrate into a VFX pipeline. Inspired by studio pipeline architectures at facilities like Industrial Light & Magic, it chains together an LLM scene parser, a Stable Diffusion XL image generator, and a headless Blender compositor — all orchestrated through a clean Streamlit interface. 

Every stage degrades gracefully to pre-baked fallback assets, so the pipeline is always demonstrable regardless of API availability.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        GenFX Lite Pipeline                   │
└─────────────────────────────────────────────────────────────┘

  User Prompt (natural language)
          │
          ▼
  ┌───────────────┐
  │  LLM Parser   │  GPT-4o-mini / HF Mixtral
  │  llm_parser.py│  temperature=0.2, max_tokens=500
  └───────┬───────┘
          │ Structured Scene JSON
          ▼
  ┌───────────────┐
  │  Image Gen    │  HuggingFace Inference API
  │  image_gen.py │  SD 3.5 / FLUX / Hyper-SD
  └───────┬───────┘
          │ PNG (1920×1080)
          ▼
  ┌───────────────┐
  │  Blender      │  Headless Cycles renderer
  │  render.py    │  samples=32, image plane composite
  └───────┬───────┘
          │ Final Render PNG
          ▼
  ┌───────────────┐
  │  Streamlit UI │  Dark theme, DM Serif + DM Mono
  │  streamlit_app│  3-column output + pipeline status
  └───────────────┘
```

Fallback System (every stage):
- **API failure** → pre-baked fallback asset
- **JSON invalid** → `fallback_scene.json`
- **Image timeout** → `fallback_image.png`
- **Blender absent** → `fallback_render.png`

---

## File Structure

```
genfx-lite/
│
├── app/
│   ├── config.py           # Master Config (Paths, Keys, Retries)
│   ├── llm_parser.py       # Module 1: Prompt → Scene JSON
│   ├── image_gen.py        # Module 2: Scene JSON → Image
│   ├── blender_runner.py   # Module 3: Subprocess call to Blender
│   └── pipeline.py         # Orchestrator: runs all modules in order
│
├── blender/
│   └── render.py           # Blender Python script (headless Cycles)
│
├── assets/
│   ├── fallback_scene.json # Pre-baked fallback JSON
│   ├── fallback_image.png  # Pre-baked fallback image (1024×576)
│   └── fallback_render.png # Pre-baked fallback render (1024×576)
│
├── runs/                   # [Generated] Per-run pipeline outputs
│
├── ui/
│   └── streamlit_app.py    # Full Streamlit UI
│
├── tests/
│   └── test_pipeline.py    # Smoke tests for each module
│
├── create_fallback_assets.py  # One-time asset generator
├── .env.example            # API key template
├── requirements.txt        # All pip dependencies
└── README.md
```

---

## Setup

### Requirements

- Python 3.10+
- Blender 3.6+ / 4.x / 5.x (must be in PATH or set `BLENDER_PATH` in `.env`)
- OpenAI API key **or** HuggingFace API key

### Installation

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd genfx-lite

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Generate fallback placeholder assets (run once)
python create_fallback_assets.py
```

### Environment

```bash
# 4. Copy the environment template
cp .env.example .env   # Windows: copy .env.example .env

# 5. Fill in your API keys (at minimum, set OPENROUTER_API_KEY + HUGGINGFACE_API_KEY)
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=mistralai/mistral-7b-instruct:free   # recommended free model

HUGGINGFACE_API_KEY=hf_...

# 6. If Blender is not in your PATH, set BLENDER_PATH:
#    Windows example:
BLENDER_PATH=C:\Program Files\Blender Foundation\Blender 5.1\blender.exe
```

### Starting the Pipeline

```bash
# 7. Run smoke tests (22 tests should pass — no API keys required)
pytest tests/test_pipeline.py -v

# 8. Launch the Streamlit UI
streamlit run ui/streamlit_app.py
```

---

## Troubleshooting & Diagnostics

### Understanding FALLBACK badges

A **FALLBACK** badge in the Streamlit UI does **not** mean the pipeline crashed. It means one stage degraded gracefully to a pre-baked asset. The pipeline always completes and always produces output. Open the **Diagnostics** expander to see the exact reason for each fallback.

> [!NOTE]
> **Image relevance on LLM fallback**: When LLM parsing fails, the pipeline still uses the original user prompt for image generation — not the fallback desert scene. So if you type *"snowy mountains at sunset"*, the generated image will depict snowy mountains even if the JSON stage fell back. Visual output always reflects what you asked for.


### Common issues

- **OpenRouter returns no JSON** — The model may respond with prose instead of JSON. The recommended model is `mistralai/mistral-7b-instruct:free`. Set it in `.env`:
  ```
  OPENROUTER_MODEL=mistralai/mistral-7b-instruct:free
  ```

- **HuggingFace image 404 / timeout** — The image generator now tries three models in order before falling back:
  1. `stabilityai/stable-diffusion-3.5-medium`
  2. `black-forest-labs/FLUX.1-schnell`
  3. `ByteDance/Hyper-SD`
  
  A 429 or 503 triggers an exponential-backoff retry within the same model. A 404 immediately skips to the next model. Fallback asset is used only after all three models fail.

- **Blender Not Found** — Verify `BLENDER_PATH` in `.env` matches your install. Test with:
  ```
  & "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe" --version
  ```

- **Sidebar shows live API status** — The System Checks panel probes OpenRouter and HuggingFace at startup (cached 5 minutes). **FALLBACK** there means the API is unreachable, not that the key is wrong.

---

## Storage & Cleanup

> [!WARNING]
> Every time the pipeline is executed, a new unique subfolder is generated in `runs/` (e.g., `runs/run_a1b2c3d4/`). These folders contain your `scene.json`, `image.png`, `render.png`, and a `blender.log`. Over time this directory will grow. You can safely delete any subfolders inside `runs/` to free up space.

---

## Pipeline Integration Notes

### Assumptions Made in This Prototype

- **Blender compositing**: Note that the Blender stage composites a single static image plane. It does *not* generate a full 3D layout matrix or physical mesh structures.
- **GPU not required** — HuggingFace Inference API handles GPU workloads server-side; Blender uses CPU Cycles.
- **No real asset database** — `asset_refs` is always enforced as empty.

### Limitations

- **No ControlNet conditioning** — generated images are prompt-only, not conditioned on storyboards or concept art.
- **No multi-user web support** — While runs are isolated linearly into `runs/`, concurrent browser usage using exact Streamlit references could yield race condition constraints in heavily threaded spaces.
