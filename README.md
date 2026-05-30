# CodeGen API — Development Journey & Client Tools

> **The complete development history and client-side automation tools behind [my-codegen-api2](https://github.com/spearb0lt/my-codegen-api2)** — an AI-powered competitive programming solver built for **Meta Hacker Cup 2025 (AI Track)**.

---

## What This Repository Contains

This repository documents the **full evolution** of the CodeGen API from initial concept to production-ready autonomous solver, along with **reusable client tools** for interacting with the deployed API. It serves as both a development archive and a practical toolkit.

| Folder | Purpose |
|--------|---------|
| `Initial-Prototype/` | The earliest iterations — standalone scripts and the first FastAPI server |
| `Final-Prototype/` | Iterative refinement of the API (v1 → v2 → v3 → v4 → final), including multimodal support |
| `CP_GEN/` | Client-side automation tools for calling the deployed API and testing solutions |

The **production API** lives at: [github.com/spearb0lt/my-codegen-api2](https://github.com/spearb0lt/my-codegen-api2)

---

## The Problem It Solves

Meta Hacker Cup 2025 introduced an **AI Track** where participants could use AI systems to solve the same algorithmic problems that top human programmers struggle with. This system was purpose-built as an **end-to-end autonomous solver**:

1. Accept a problem package (statement + sample I/O + optional diagrams)
2. Generate an optimal Python solution using Google's Gemini LLM (`gemini-2.5-pro`)
3. Validate the solution against sample test cases
4. Iteratively regenerate on failure (up to 4 attempts)
5. Run against real competition test inputs with LLM-based debugging

### Competition Results

Using `gemini-2.5-pro` with this system:

| Problem | Round | Points | Estimated Difficulty |
|---------|-------|--------|---------------------|
| Designing Paths (C) | Round 2 | 23 pts | ~2200–2400 (graph BFS with constrained edge traversal on tram routes) |
| Treehouse Telegram (D) | Round 3 | 24 pts | ~2300–2500 (tree distances + GCD-based pair enumeration using number theory) |

These are **upper-medium to hard** competitive programming problems involving graph algorithms, number theory, and careful complexity analysis.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLIENT (you / CP_GEN tools)                  │
│   Upload .zip (statement + sample_in + sample_out + images)         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ POST /generate
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FastAPI Server (Render)                          │
│                                                                     │
│  1. Unpack ZIP → extract statement, sample I/O, images              │
│  2. Process images:                                                 │
│     a. Local .png/.jpg → PIL.Image                                  │
│     b. URI .txt files → fetch URL → PIL.Image                       │
│     c. If PIL fails → URIs included as text in prompt               │
│  3. Build prompt with problem + sample I/O + image mapping          │
│  4. Call Gemini 2.5 Pro (thinking enabled)                          │
│  5. Extract Python code from markdown response                      │
│  6. Run code against sample input                                   │
│  7. Compare output to expected                                      │
│     ├─ MATCH → save & return solution                               │
│     └─ MISMATCH → feed error back to LLM → retry (up to 4x)       │
│  8. Return solution + metadata                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## CP_GEN — Client-Side Automation Tools

The `CP_GEN/` folder contains **reusable client scripts** for interacting with the deployed API. These scripts automate the full workflow: generating solutions, downloading code, running against real test inputs, and triggering server-side regeneration on failure.

> **Note:** Initially, image-based and non-image problems required **separate pipelines** — hence the `For_Image/` and `For_Non_Image/` split in CP_GEN. Later, the production API ([my-codegen-api2](https://github.com/spearb0lt/my-codegen-api2)) was built as a **single multipurpose API** that seamlessly handles both cases. If images are present in the ZIP, the multimodal pipeline activates automatically; if not, it proceeds with text-only prompting. The CP_GEN folders still reflect the original split for historical context and because the client-side usage patterns differ slightly (image problems need URI files in the ZIP).

### For Problems Without Images (`CP_GEN/For_Non_Image/`)

Use when the problem statement is text-only (no diagrams/figures).

**Quick Start:**
```python
import requests, json
from pathlib import Path

# Generate a solution
url = "https://my-codegen-api2.onrender.com/generate"
files = {"file": ("problem.zip", open("MyQ.zip", "rb"), "application/zip")}
r = requests.post(url, files=files, timeout=600)
j = r.json()
Path("gen_response.json").write_text(json.dumps(j, indent=2))
Path("coding_solution_gen.py").write_text(j["solution"])
```

### For Problems With Images (`CP_GEN/For_Image/`)

Use when the problem contains `{{PHOTO_ID:X}}` placeholders or embedded diagrams. The ZIP should include `*_uri.txt` files containing image URLs or direct image files (`.png`, `.jpg`).

**Key scripts:**
- `1runner.py` — Full automation: generates solution via API, downloads it, runs against test input, posts failures to `/test2` for LLM regeneration
- `1runner2.py` — Variant that only contacts server on error (runs locally first, posts to `/test2` only if the run fails)

**Usage:**
```bash
python 1runner.py --server https://my-codegen-api2.onrender.com --solution-file a.py --test-file test_input.txt
python 1runner2.py --server https://my-codegen-api2.onrender.com --solution-id <ID> --test-file test_input.txt
```

### HyperTest — Waterfall Prototyping (`Final-Prototype/HyperTest (Waterfall-Prototyping)/`)

A local testing harness that runs the generated solution against test inputs **on your machine** and iteratively refines it using the Gemini API directly (without the server). Useful for rapid local iteration before deploying.

---

## Development Evolution

This repository captures the iterative build process — from a single Python script to a production API deployed on Render.

### Phase 1: Initial Prototype (`Initial-Prototype/`)

#### Agentic Standalone Scripts (`Agentic_way/`)
- `gem1_60.py` — The very first iteration: a standalone Python script that takes a ZIP file, unpacks it, calls `gemini-2.5-pro`, runs the generated code, and iterates on failure. Everything hardcoded (paths, API key). Used `google-generativeai` (older SDK).
- `gem1_61.py` — A refinement focused on the **test+fix loop**: takes an already-generated solution and iteratively fixes it against test input using LLM feedback.

#### First Complete API (`1st complete_api_v1/`)
- `api_server.py` — The first FastAPI server. Basic `/generate` endpoint that accepts a ZIP and returns a validated solution. Used `google-generativeai` SDK. No `/test` endpoint yet — testing was client-side only.
- `caller_v2_mix.py` — Client script demonstrating all interaction patterns: generate → download → test (multiple modes: by solution_id, by direct code upload, by solution_file + problem_zip).

#### Image URI Discovery (`final_code_where_image_uri_used.py`)
- The first server version that handled `{{PHOTO_ID:X}}` placeholders and image URI text files. Parsed Meta Hacker Cup's Facebook CDN image URLs from the problem ZIPs and embedded them in the LLM prompt.

### Phase 2: Final Prototype (`Final-Prototype/`)

The API evolved through several named versions, each adding capabilities:

| Version | File | Key Addition |
|---------|------|-------------|
| Two-Step v1 | `api_server_two_step.py` | Added `/test` endpoint — run saved solution on new test input with LLM regeneration on failure |
| Two-Step v2 | `api_server_two_step_v2.py` | Added `/download` endpoint, file upload for test input, artifact persistence |
| Two-Step v3 | `api_server_two_step_v3.py` | Accept direct `solution` text in `/test` (not just solution_id). Unicode-escape decoding for LLM output. Raw LLM text saved. |
| Two-Step v4 | `api_server_two_step_v4.py` | Final merge of v3+v4 features. Increased execution timeout to 60s. Allowed download file whitelist. |
| With Test Input | `api_server_with_test_input.py` | Two-stage validation: sample check + optional test_input check in a single `/generate` call |
| Final | `api_server_final.py` | Production-ready merge of all versions. Clean code extraction, HTML unescaping, `/solutions` listing, `/cleanup` endpoint. |

#### Multimodal Evolution (`Final/`)
- `wo_img_full.py` — Final server code for problems **without** images (text-only pipeline)
- `w_img_try2.py` — Final server code with **full multimodal support**: PIL image fetching with retries, `{{PHOTO_ID:X}}` placeholder matching, image URL fallback in prompt, configurable `MAX_IMAGES_IN_PROMPT`

### Phase 3: Production (`my-codegen-api2`)

The final production version deployed at [my-codegen-api2](https://github.com/spearb0lt/my-codegen-api2) incorporates all learnings:
- Multimodal image pipeline with multi-layered fallback
- `google-genai` SDK (≥2.5.0) with `thinking_level="high"`
- `/generate`, `/test`, `/test2`, `/download`, `/solutions`, `/cleanup` endpoints
- Configurable via environment variables
- Deployed on Render

---

## API Endpoints (Production)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check, model config status |
| `/generate` | POST | Upload ZIP → get validated solution |
| `/test` | POST | Re-run solution on new test input, regenerate on failure |
| `/test2` | POST | JSON-based: post local run results, trigger regeneration if failed |
| `/download/{id}/{file}` | GET | Download solution artifacts |
| `/solutions/{id}/files` | GET | List files for a solution |
| `/solutions` | GET | List all solution IDs |
| `/cleanup/{id}` | DELETE | Remove solution artifacts |

---

## ZIP File Format

```
MyQ.zip
├── statement.txt          # Problem statement (required)
├── sample_in.txt          # Sample input (required)
├── sample_out.txt         # Sample expected output (required)
├── image_1_uri.txt        # (optional) Contains URL to a problem diagram
├── photo_2_uri.txt        # (optional) Another image URL
└── diagram.png            # (optional) Direct image file
```

File matching uses **heuristics** (not exact names):
- Statement: filename contains `statement` or `problem`
- Sample input: contains `sample_in`, `sample.in`, `input`, or `sample-input`
- Sample output: contains `sample_out`, `sample.out`, `output`, or `sample-output`
- Images: image extensions OR files with `image`/`photo` in the name

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Web Framework | FastAPI |
| LLM | Google Gemini 2.5 Pro (via `google-genai` SDK) |
| Reasoning | Thinking mode enabled for maximum reasoning depth |
| Image Processing | Pillow (PIL) |
| HTTP Client | requests (for image fetching) |
| Code Execution | subprocess with configurable timeout |
| Deployment | Render (free tier) |

---

## Setup & Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### Using CP_GEN Client Tools

1. **Prepare your problem ZIP** with `statement.txt`, `sample_in.txt`, `sample_out.txt` (and optional image files)

2. **Generate a solution:**
```python
import requests, json
from pathlib import Path

url = "https://my-codegen-api2.onrender.com/generate"
files = {"file": ("problem.zip", open("MyQ.zip", "rb"), "application/zip")}
r = requests.post(url, files=files, timeout=600)
j = r.json()
print(j["status"], "attempt:", j.get("attempt"))
Path("coding_solution.py").write_text(j["solution"])
```

3. **Test with real competition input:**
```python
r = requests.post(
    "https://my-codegen-api2.onrender.com/test",
    data={"solution_id": j["solution_id"]},
    files={"test_file": open("test_input.txt", "rb")}
)
print(r.json()["status"])
```

4. **Local run + server-side fix (HyperTest pattern):**
```bash
python CP_GEN/For_Image/1runner2.py \
    --server https://my-codegen-api2.onrender.com \
    --solution-file my_solution.py \
    --test-file test_input.txt \
    --max-iters 4 \
    --time-limit 60
```

### Running Your Own Server

```bash
export GOOGLE_API_KEY="your-key-here"
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | (required) | Gemini API key |
| `MODEL_NAME` | `gemini-2.5-pro` | Model for code generation |
| `MAX_GENERATION_ATTEMPTS` | `4` | Max retries per request |
| `EXECUTION_TIMEOUT` | `300` | Seconds before killing a solution run |
| `SOLUTIONS_DIR` | `solutions` | Directory for persisted artifacts |
| `ENABLE_IMAGE_DOWNLOAD` | `1` | Enable fetching images from URLs |
| `MAX_IMAGES_IN_PROMPT` | `6` | Cap on images included in prompt |
| `PYTHON_PATH` | `python` | Python executable for running solutions |

---

## Key Design Decisions

1. **Iterative self-correction over single-shot generation** — The LLM gets up to 4 attempts, each time receiving the full error context (diff, stderr, or timeout message). This dramatically improves solve rates on harder problems.

2. **Local execution + server regeneration (HyperTest pattern)** — Run code locally (faster, no server timeout limits), post failures to `/test2` for LLM-based fixes. Best of both worlds.

3. **Image URL fallback** — If PIL can't decode an image, the URL is still embedded as text in the prompt. The model can reason about the problem using URL context or training knowledge.

4. **File heuristics over strict naming** — ZIP file matching is fuzzy (contains-based) rather than exact, accommodating Meta Hacker Cup's varying file naming across rounds.

5. **Unified diff for wrong answers** — When output doesn't match, the retry prompt includes a unified diff so the model can see exactly what went wrong line by line.

---

## Repository Structure

```
my-codegen-api/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── CP_GEN/                            # Client-side automation tools
│   ├── For_Image/                     # For problems with diagrams/images
│   │   ├── 1runner.py                 # Full test2 automation (always contacts server)
│   │   ├── 1runner2.py                # Post-on-error automation (contacts server only on failure)
│   │   ├── coding_solution_gen.py     # Example generated solution (graph BFS problem)
│   │   └── coding_solution_downloaded.py
│   ├── For_Non_Image/                 # For text-only problems
│   │   ├── coding_solution_gen.py     # Example generated solution (DP/minimax problem)
│   │   └── test_input.txt             # Example test input
│   └── Manual_Tester/                 # Ad-hoc manual testing notebooks
│
├── Initial-Prototype/                 # Phase 1: First iterations
│   ├── Agentic_way/                   # Standalone scripts (no server)
│   │   ├── gem1_60.py                 # First complete generate-test loop
│   │   └── gem1_61.py                 # Test+fix refinement loop
│   ├── 1st complete_api_v1/           # First FastAPI server
│   │   ├── api_server.py              # Server with /generate only
│   │   └── caller_v2_mix.py           # Client demonstrating all interaction patterns
│   └── final_code_where_image_uri_used.py  # First image/PHOTO_ID handling
│
├── Final-Prototype/                   # Phase 2: Iterative API refinement
│   ├── api_server_two_step.py         # v1: Added /test endpoint
│   ├── api_server_two_step_v2.py      # v2: Added /download, file upload
│   ├── api_server_two_step_v3.py      # v3: Direct code in /test, escape handling
│   ├── api_server_two_step_v4.py      # v4: Merged features, longer timeout
│   ├── api_server_with_test_input.py  # Two-stage validation variant
│   ├── api_server_final.py            # Production-ready merge
│   ├── server_caller_v2_mix.py        # Client script for testing all endpoints
│   ├── Final/                         # Final server variants
│   │   ├── wo_img_full.py             # Text-only pipeline (no images)
│   │   └── w_img_try2.py              # Full multimodal pipeline
│   └── HyperTest (Waterfall-Prototyping)/  # Local test+fix harness
│       ├── runner.py                  # test2 automation script
│       ├── runner2.py                 # Post-on-error variant
│       └── coding_solution_gen.py     # Example: generated solution for robots problem
```

---

## License

Private repository. Built for Meta Hacker Cup 2025 AI Track participation.

---

## Related

- **Production API**: [github.com/spearb0lt/my-codegen-api2](https://github.com/spearb0lt/my-codegen-api2)
- **Meta Hacker Cup 2025**: [facebook.com/codingcompetitions/hacker-cup](https://www.facebook.com/codingcompetitions/hacker-cup)
