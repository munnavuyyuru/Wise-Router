# RoutellM — Token-Efficient Routing Agent

**AMD Developer Hackathon: ACT II - Track 1: Hybrid Token-Efficient Routing Agent**

RoutellM is a containerized Track 1 agent that reads tasks from `/input/tasks.json`, writes one `{task_id, answer}` object per task to `/output/results.json`, and keeps Fireworks usage as low as possible without sacrificing accuracy. It does this by solving easy tasks locally first and only calling Fireworks when the task is outside the deterministic safety envelope.

## Runtime Contract

- Input: `/input/tasks.json`
- Output: `/output/results.json`
- Output format: JSON array of objects with exactly `task_id` and `answer`
- Fireworks config: `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, `ALLOWED_MODELS`
- Local dry-runs: safe even when Fireworks env vars are missing; the router falls back to deterministic answers or placeholders instead of crashing

## Architecture

```text
/input/tasks.json
        |
        v
+------------------+
| TaskClassifier   |  all-MiniLM-L6-v2 embeddings + LogisticRegression
+------------------+  8 task types: math, sentiment, code_debug, code_gen,
        |               summarization, ner, logic, factual
        |
        +--> deterministic solvers (zero-token)
        |     math_solver, sentiment (with intensifiers + negation + contrast),
        |     summarizer, ner, logic_solver, code_tools (8 templates + 6 debug patterns),
        |     factual (knowledge base lookup)
        |
        +--> Fireworks fallback if solver is not confident
                  |
                  v
        +--> model_selector chooses the cheapest sufficient allowed model
             (4 cost tiers filtered by ALLOWED_MODELS)
                  |
                  v
        +--> fireworks_client sends OpenAI-compatible calls with
             reasoning_effort="none" and cross-model fallback
                  |
                  v
        +--> batching groups up to 5 same-category tasks per API call
                  |
                  v
/output/results.json
```

## Deterministic Solvers

The deterministic layer is intentionally conservative. If a solver is not confident, it returns `None` and the task falls back to Fireworks instead of guessing.

| Solver | File | What it handles |
|--------|------|----------------|
| Math | `math_solver.py` | Arithmetic, %, average, equations (pure Python, no sympy) |
| Sentiment | `sentiment.py` | Positive/negative/neutral with negation, contrast, and intensifier scaling (very×1.5, extremely×2.0, etc.) |
| NER | `ner.py` | Emails, URLs, money, dates, phones, percents, numbers |
| Summarizer | `summarizer.py` | Extractive frequency-based, respects sentence count constraint |
| Logic | `logic_solver.py` | Simple ordering chains (tallest/slowest/greatest) |
| Code Tools | `code_tools.py` | 6 debug patterns (mutable defaults, missing return, off-by-one, recursion base, `==` vs `=` , missing colon) + 8 code gen templates (sort, reverse, factorial, fibonacci, palindrome, is_prime, fizzbuzz, gcd) |
| Factual | `factual.py` | Knowledge base lookup for capitals, inventors, constants |

## Fireworks Routing

- All Fireworks calls go through `app/llm/fireworks_client.py`.
- The client uses `FIREWORKS_BASE_URL` exactly as provided.
- The client refuses any model not present in `ALLOWED_MODELS`.
- `app/llm/model_selector.py` ranks allowed models by cost tier and prefers the cheapest sufficient option.
- Every payload includes `extra_body={"reasoning_effort": "none"}` to suppress hidden think tokens (saves 300-2000 tokens per call).
- On failure, `generate_with_fallback()` tries the next model in the tier before giving up.
- A time budget guard stops new Fireworks calls 20 seconds before the 10-minute container deadline.
- `app/llm/batcher.py` groups up to 5 same-category tasks into a single API call, amortizing system prompt overhead.

### Cost Tiers

| Task Type | Primary Model | Fallback |
|-----------|--------------|----------|
| sentiment, ner, summarization, factual | gemma-3-1b-it ($0.10/M) | gemma-3-4b-it ($0.20/M) |
| math, logic | gemma-3-4b-it ($0.20/M) | minimax-m3 ($0.50/M) |
| code_debug | gemma-3-4b-it ($0.20/M) | kimi-k2p7-code ($1.00/M) |
| code_gen | kimi-k2p7-code ($1.00/M) | — |

## Web Interface

RoutellM ships with a Streamlit dashboard that connects to a FastAPI backend:

- **Dashboard** — accuracy, token cost, routing distribution charts
- **Batch Evaluate** — upload JSON tasks, run evaluation with adjustable threshold
- **Live Demo** — single-prompt routing with step-by-step flow visualization
- **Cost Compare** — compare RoutellM cost vs always-using an expensive baseline model

```bash
# Start both backend and frontend locally
pip install -r requirements.txt
uvicorn app.api_server:app --host 0.0.0.0 --port 8000 &
streamlit run streamlit_app/Home.py --server.port 8501 --server.address 0.0.0.0
# Frontend: http://localhost:8501
# Backend:  http://localhost:8000/api/health
```

Or with docker-compose:

```bash
docker compose up -d
```

## Local Development

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Run the tests:

```bash
pytest tests/ -v
```

Run the representative 24-task local evaluator:

```bash
python scripts/eval_local.py --dry-run
```

Run the main entrypoint against the sample fixture:

```bash
python -m app.main --input tests/fixtures/tasks_track1_sample.json --output /tmp/results.json
cat /tmp/results.json
```

Dry-run mode (no API key needed):

```bash
python -m app.main --input tests/fixtures/tasks_track1_sample.json --output /tmp/results.json
```

## Docker

Build the image:

```bash
docker build -t routellm .
```

Run it with mounted input/output folders:

```bash
docker run --rm \
  -e FIREWORKS_API_KEY="$FIREWORKS_API_KEY" \
  -e FIREWORKS_BASE_URL="$FIREWORKS_BASE_URL" \
  -e ALLOWED_MODELS="$ALLOWED_MODELS" \
  -v "$(pwd)/input:/input" \
  -v "$(pwd)/output:/output" \
  routellm
```

Docker Compose (includes Web UI):

```bash
docker compose up -d
# Frontend: http://localhost:8501
```

PowerShell equivalent:

```powershell
docker run --rm `
  -e FIREWORKS_API_KEY=$env:FIREWORKS_API_KEY `
  -e FIREWORKS_BASE_URL=$env:FIREWORKS_BASE_URL `
  -e ALLOWED_MODELS=$env:ALLOWED_MODELS `
  -v "${PWD}/input:/input" `
  -v "${PWD}/output:/output" `
  routellm
```

## Allowed Models

`ALLOWED_MODELS` may contain either short names or full Fireworks paths; short names are normalized to `accounts/fireworks/models/<name>`.

```text
gemma-3-1b-it
gemma-3-4b-it
minimax-m3
kimi-k2p7-code
```

Recommended local export:

```bash
export FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1"
export ALLOWED_MODELS="accounts/fireworks/models/gemma-3-1b-it,accounts/fireworks/models/gemma-3-4b-it,accounts/fireworks/models/minimax-m3,accounts/fireworks/models/kimi-k2p7-code"
```

## Example I/O

Input:

```json
[
  {"task_id": "math_001", "prompt": "What is 18 + 24?"},
  {"task_id": "sent_001", "prompt": "What is the sentiment of this review: \"The room was small, but the view was absolutely stunning.\""}
]
```

Output:

```json
[
  {"task_id": "math_001", "answer": "42"},
  {"task_id": "sent_001", "answer": "positive"}
]
```

## Commands

- `pytest tests/ -v` runs the unit suite (62 tests).
- `python scripts/eval_local.py --dry-run` runs the representative 24-task Track 1 fixture.
- `python scripts/eval_local.py --threshold 85` runs eval with a custom accuracy threshold.
- `python scripts/generate_training_data.py` regenerates synthetic training data (560 samples, 8 types).
- `docker compose up -d` builds and runs the full stack (API + Web UI).

## Files of Interest

| File | Purpose |
|------|---------|
| `app/main.py` | CLI entrypoint, input/output handling |
| `app/router.py` | Classify → deterministic → Fireworks orchestration |
| `app/api_server.py` | FastAPI backend for the Streamlit frontend |
| `app/config.py` | Environment parsing and defaults |
| `app/classifier/task_classifier.py` | all-MiniLM-L6-v2 embeddings + LogisticRegression |
| `app/llm/fireworks_client.py` | OpenAI-compatible Fireworks client with reasoning suppression and cross-model fallback |
| `app/llm/model_selector.py` | Cost-tiered model selection filtered by ALLOWED_MODELS |
| `app/llm/batcher.py` | Groups same-category tasks into batched API calls |
| `app/deterministic/` | 7 zero-token solvers |
| `scripts/eval_local.py` | Track 1 local evaluator |
| `streamlit_app/` | Streamlit dashboard with 3 pages |
| `tests/fixtures/tasks_track1_sample.json` | Representative 24-task fixture |

## Track 1 Notes

- Deterministic wins first; Fireworks is only for tasks the local solver cannot handle confidently.
- The router never calls a model outside `ALLOWED_MODELS`.
- The router never hardcodes a Fireworks URL.
- The repository contains no committed secrets.
- Accuracy is prioritized over token savings. If a task is uncertain, it falls back instead of guessing.
