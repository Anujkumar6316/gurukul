# Gurukul DeepEval Benchmark Suite

Standalone LLM-as-a-judge benchmarking package for evaluating the Gurukul offline curriculum assistant.

This suite is **completely isolated** from the core bot runtime. It uses **DeepEval** and custom **G-Eval** rubrics to score chatbot responses against Balbharati textbook cards, verify trilingual language consistency, enforce conversational offer etiquette, and evaluate calibrated abstentions.

---

## Architecture & Dependency Separation

To keep Gurukul's core offline deployment (FastEmbed, ONNX Runtime, FastAPI, Gradio) lightweight, benchmarking dependencies are placed in `[project.optional-dependencies]` in `pyproject.toml`:

```toml
[project.optional-dependencies]
benchmark = [
    "deepeval>=2.0.0"
]
```

- **Core Bot (`app/offline/`)**: 0 imports or dependencies on DeepEval.
- **Benchmark Suite (`benchmarks/`)**: Isolated evaluation runner.

To install benchmark dependencies:
```bash
uv sync --extra benchmark
# or
pip install -e ".[benchmark]"
```

---

## Artifact-Driven Workflow

The runner is built around **artifacts** for both input and output:

```
[Input Artifact]                              [Output Artifacts]
• batch_debug_results.json  ──► DeepEval ──►  • benchmark_report.md
• or queries.txt                 Judge        • benchmark_scorecard.json
```

### 1. Input Resolution
The runner automatically discovers input artifacts:
- Checks the path passed via `--input <file>`.
- If no path is given, auto-discovers the newest `.json` or `.txt` artifact in the conversation artifact directory (`$GURUKUL_ARTIFACT_DIR`) or `./artifacts/`.

#### Supported Input Formats:
1. **Batch Debugger JSON (`batch_debug_results.json`)**:
   - Exported from Gradio Studio Tab 4 (`🔬 Batch Debugger`) or `POST /api/debug/batch`.
   - Contains pre-computed answers, card IDs, citations, follow-up offers, and latencies.
   - **Evaluates immediately with zero re-inference overhead**.
2. **Raw Query Text (`queries.txt`)**:
   - One query per line.
   - Automatically executes queries through `OfflineFactBot` first, then sends the outputs to the judge.

### 2. Output Artifacts
Results are written back to the artifact directory:
- `benchmark_report.html`: Interactive, styled HTML report with KPI scorecards, progress bars, live search, and status filters.
- `benchmark_report.md`: Formatted scorecard table and summary markdown.
- `benchmark_scorecard.json`: Machine-readable raw metrics and evaluation metadata.

---

## Configuration via `.env` (`benchmarks/settings.py`)

Settings are loaded automatically from `benchmarks/.env` (or project root `.env`):

```ini
# benchmarks/.env

# 1. OpenAI-Compatible Model Configuration (vLLM, Ollama, LM Studio, Groq, OpenRouter)
OPENAI_API_KEY=your_openai_api_key_or_EMPTY
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL_ID=gpt-4o-mini

# Examples for local/remote providers:
#   - Local vLLM:       OPENAI_BASE_URL=http://localhost:8000/v1      OPENAI_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct
#   - Local Ollama:     OPENAI_BASE_URL=http://localhost:11434/v1     OPENAI_MODEL_ID=llama3.1:8b
#   - Local LM Studio:  OPENAI_BASE_URL=http://localhost:1234/v1      OPENAI_MODEL_ID=qwen2.5:7b
#   - Groq Cloud:       OPENAI_BASE_URL=https://api.groq.com/openai/v1 OPENAI_MODEL_ID=llama-3.1-8b-instant
#   - OpenRouter:       OPENAI_BASE_URL=https://openrouter.ai/api/v1  OPENAI_MODEL_ID=deepseek/deepseek-chat

# 2. Native Gemini (Alternative)
# GEMINI_API_KEY=your_gemini_api_key_here
# BENCHMARK_JUDGE=gemini-1.5-flash

# 3. Scoring Thresholds (0.0 to 1.0)
BENCHMARK_THRESHOLD_LANG=0.8
BENCHMARK_THRESHOLD_OFFER=0.8
BENCHMARK_THRESHOLD_GROUNDING=0.7
BENCHMARK_THRESHOLD_ABSTENTION=0.7
```


---

## The 4 Custom G-Eval Rubrics (`benchmarks/rubrics.py`)

| Rubric | Criteria | Pass Threshold |
| :--- | :--- | :---: |
| **Language Consistency** | Student query, factual answer, and follow-up offer must strictly share the same language (`mr` Marathi, `hi` Hindi, `en` English). Mixed languages fail. | $\ge 0.8$ |
| **Follow-up Offer Quality** | Follow-up must be formatted as an inviting conversational offer (*"Would you like to know about..."*). Heavily penalizes unsolicited quizzes or exam questions. | $\ge 0.8$ |
| **Curriculum Grounding** | Factual answer must be strictly supported by retrieved Balbharati textbook cards without hallucination. Proper polite explanation on abstention. | $\ge 0.7$ |
| **Abstention Accuracy** | Evaluates whether the bot correctly answered in-syllabus queries and politely abstained on out-of-syllabus queries. | $\ge 0.7$ |

---

## Quickstart & Commands

### 0. Banner & Command Guide (No Arguments)
```bash
python -m benchmarks.main
```
Prints the benchmark banner with the active judge configuration, input artifact discovery
status, thresholds, and the full command guide for both workflows below.

### 1. Validate Pipeline (Dry Run - No API Keys Needed)
```bash
uv run poe benchmark --dry-run
```
Discovers the input artifact, parses records, generates test cases, and exports a dry-run `benchmark_report.md`.

### 2. Run with Google Gemini Judge (Fast & Recommended)
```bash
export GEMINI_API_KEY="your-gemini-api-key"
uv run poe benchmark --input batch_debug_results.json --judge gemini-1.5-flash
```

### 3. Run with OpenAI Judge
```bash
export OPENAI_API_KEY="your-openai-api-key"
uv run poe benchmark --input batch_debug_results.json --judge gpt-4o-mini
```

### 4. Run 100% Offline with Local Ollama
```bash
uv run poe benchmark --input batch_debug_results.json --judge ollama/llama3.1:8b
```

---

## Two Ways to Feed the Benchmark

The runner auto-detects the artifact type from the file extension.

### A. Batch Debugger JSON (`batch_debug_results.json`) — Pre-computed Answers
Exported from Gradio Studio Tab 4 (`🔬 Batch Debugger`) or `POST /api/debug/batch`.
Contains answers, card IDs, citations, follow-up offers, and latencies, so the judge
evaluates immediately with **zero re-inference overhead**.

```bash
# Place the export in the artifact directory, then:
python -m benchmarks.main --input batch_debug_results.json --dry-run
python -m benchmarks.main --input batch_debug_results.json --judge gemini-1.5-flash
```

### B. Raw Query Text (`queries.txt`) — One Query Per Line
Each query is executed through `OfflineFactBot` first (this part runs fully offline),
and only the generated answers are sent to the judge.

```bash
printf 'who was shivaji maharaj\nप्रकाशसंश्लेषण म्हणजे काय\nhow to build a rocket to mars\n' > queries.txt
python -m benchmarks.main --input queries.txt --dry-run
python -m benchmarks.main --input queries.txt --judge gemini-1.5-flash
```

If `--input` is omitted, the runner auto-discovers the newest `.json` or `.txt`
artifact in `$GURUKUL_ARTIFACT_DIR` (or `./artifacts/`).

### C. Gurukul Studio UI (Tab 5: 🏆 Benchmark Suite)
Run benchmarks directly from the web interface without using the command line:
1. Launch Gradio Studio (`poe studio` or `python main.py --service studio`).
2. Switch to **Tab 5: 🏆 Benchmark Suite**.
3. Choose input artifact, upload files, or paste queries directly.
4. Select your judge model or toggle **Dry Run** for instant offline validation.
5. Click **🚀 Run Benchmark** to execute and view the interactive HTML dashboard, Markdown scorecard, and JSON metadata directly inside the browser.

---

## CLI Options

```bash
uv run python -m benchmarks.main [OPTIONS]

Options:
  --input TEXT        Path or filename of the input artifact (.json or .txt)
  --judge TEXT        Judge model identifier (default: gemini-1.5-flash)
  --output-dir PATH   Custom output directory for report artifacts
  --dry-run           Validate pipeline without making LLM API calls
  --help              Show this message and exit
```

---

## Running Automated Tests

```bash
uv run python -m unittest tests/test_benchmark_suite.py -v
```
All benchmark tests run with mocked/offline fixtures and do not require external LLM API credentials.
