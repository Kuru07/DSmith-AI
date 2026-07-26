# DSmith AI — Autonomous Data Science Agent

> An end-to-end, self-repairing data science pipeline powered by **Gemini**, **LangGraph**, and **scikit-learn**.  
> Upload a raw CSV dataset, specify a target column — DSmith AI autonomously cleans it, selects an ML problem type, trains and evaluates baseline models, and returns a production-ready model artifact.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Modules & Components](#modules--components)
5. [Agent Workflow](#agent-workflow)
6. [Setup & Installation](#setup--installation)
7. [Running the Server](#running-the-server)
8. [Running the Agent Programmatically](#running-the-agent-programmatically)
9. [Running Tests](#running-tests)
10. [API Reference](#api-reference)
11. [Environment Variables](#environment-variables)
12. [Tech Stack](#tech-stack)
13. [Deployment](#deployment)

---

## Overview

DSmith AI is a **FastAPI-based** autonomous agent backend. Given a CSV file upload and a target column name, the agent:

1. **Inspects** the raw dataset and builds a structured profile.
2. **Generates** Python preprocessing code via Gemini with explicit target-leakage-prevention rules.
3. **Validates** the generated code for security and syntax issues.
4. **Executes** the code in an isolated, UUID-namespaced workspace.
5. **Verifies** the cleaned output quality (missing-value regression check).
6. **Self-repairs** any failed stage — up to a configurable retry limit.
7. **Analyzes** the ML problem type (classification or regression).
8. **Generates** a complete scikit-learn training script.
9. **Trains** 2–3 baseline models on a held-out test split.
10. **Exports** `metrics.json` and `best_model.joblib` to the job workspace.
11. **Returns** a structured JSON response with results, metrics, and download URLs.
12. **Serves** the cleaned dataset and trained model via dedicated download endpoints.

---

## Architecture

```
POST /analyze  (CSV file + target_column)
        │
        ▼
┌──────────────────────────────────────────────────────┐
│                      main.py                         │
│  0. Cleanup expired workspaces (tools/cleanup.py)    │
│  1. Validate file type (.csv only)                   │
│  2. Validate target column exists                    │
│  3. Save upload to uploads/<uuid>.csv                │
│  4. Call run_autonomous_cleaning(file, target)       │
│  5. Return JSON + /download/{job_id} URLs            │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────┐
        │    LangGraph Pipeline    │
        │                          │
        │  ┌─────────┐             │
        │  │ INSPECT │             │  ← inspect_dataset()
        │  └────┬────┘             │
        │       ▼                  │
        │  ┌──────────┐            │
        │  │ GENERATE │            │  ← LLM (Gemini) + target rules
        │  └────┬─────┘            │
        │       ▼                  │
        │  ┌──────────┐            │
        │  │ VALIDATE │            │  ← AST security check
        │  └────┬─────┘            │
        │       │ valid            │
        │  ┌────▼─────┐            │
        │  │ EXECUTE  │            │  ← subprocess isolation
        │  └────┬─────┘            │
        │       │ success          │
        │  ┌────▼─────┐            │
        │  │  VERIFY  │            │  ← missing-value regression check
        │  └────┬─────┘            │
        │       │ pass             │
        │  ┌────▼──────────┐       │
        │  │ ANALYZE ML    │       │  ← LLM (Gemini)
        │  └────┬──────────┘       │
        │       ▼                  │
        │  ┌─────────────────┐     │
        │  │ GENERATE TRAIN  │     │  ← LLM (Gemini)
        │  └────┬────────────┘     │
        │       ▼                  │
        │  ┌─────────────────┐     │
        │  │ VALIDATE TRAIN  │     │  ← AST security check
        │  └────┬────────────┘     │
        │       │ valid            │
        │  ┌────▼────────────┐     │
        │  │ EXECUTE TRAIN   │     │  ← subprocess (120s timeout)
        │  └────┬────────────┘     │
        │       │ success          │
        │  ┌────▼────────────┐     │
        │  │ VERIFY TRAINING │     │  ← checks metrics.json + best_model.joblib
        │  └─────────────────┘     │
        │                          │
        │  Any failure → REPAIR    │  ← LLM self-repair (up to max_retries)
        └──────────────────────────┘
```

---

## Project Structure

```text
DSmith AI/
├── .env                          # API keys (git-ignored)
├── .gitignore
├── main.py                       # FastAPI application — upload, validate, dispatch, download
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── models/                       # Pydantic schemas for LLM structured outputs and AgentState
│   ├── __init__.py
│   └── schemas.py                # CleaningResult, MLDecision, TrainingPlan,
│                                 # RepairResult, TrainingRepair, AgentState
│
├── agent/                        # Core agent logic
│   ├── __init__.py
│   ├── data_agent.py             # Cleaning code generation and repair (LLM)
│   ├── ml_agent.py               # ML problem analysis, training code generation and repair (LLM)
│   └── graph.py                  # LangGraph state machine — full end-to-end workflow
│
├── tools/                        # Stateless utility functions
│   ├── __init__.py
│   ├── dataset_tools.py          # CSV profiling (shape, dtypes, missing values, stats)
│   ├── code_validator.py         # AST-based security and syntax validator
│   ├── python_executor.py        # Subprocess-based isolated code runner
│   ├── workspace.py              # Per-job UUID workspace creation
│   ├── cleanup.py                # Expired workspace deletion (1-hour TTL)
│   └── model_test.py             # Quick utility to inspect a saved joblib model
│
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── data/
│   │   └── sample_employee.csv   # Shared sample dataset for all tests
│   ├── test_cleaning.py          # Unit tests for dataset inspection (no LLM)
│   ├── test_agent.py             # Unit tests for the data cleaning agent (LLM)
│   ├── test_ml_agent.py          # Unit tests for the ML agent (LLM)
│   ├── test_graph.py             # Integration test — full LangGraph pipeline (LLM)
│   └── test_end_to_end.py        # HTTP integration tests via FastAPI TestClient (LLM)
│
├── uploads/                      # Uploaded CSVs saved here temporarily (git-ignored)
└── workspace/                    # Auto-generated per-job workspaces (git-ignored)
    └── <uuid>/
        ├── input.csv             # Copy of the uploaded dataset
        ├── cleaned.csv           # Cleaned output (served by GET /download/{id}/cleaned)
        ├── generated_training.py # Saved training script
        ├── metrics.json          # Model evaluation metrics
        └── best_model.joblib     # Best trained pipeline (served by GET /download/{id}/model)
```

---

## Modules & Components

### `models/schemas.py`

All Pydantic models used as structured output schemas for LLM responses, plus the shared `AgentState` TypedDict.

| Class | Purpose |
|---|---|
| `CleaningResult` | LLM response for data cleaning (summary, plan, code) |
| `MLDecision` | LLM response for problem type and model selection |
| `TrainingPlan` | LLM response for a generated training script |
| `RepairResult` | LLM response for a repaired cleaning script |
| `TrainingRepair` | LLM response for a repaired training script |
| `AgentState` | Full shared state of the LangGraph workflow |

---

### `agent/data_agent.py`

LLM-powered functions for the **data cleaning** stage. All cleaning and repair prompts include explicit **target-leakage-prevention rules** — the target column is never imputed, never dropped, and never used as a feature-engineering source.

| Function | Description |
|---|---|
| `generate_cleaning_code_from_profile(profile, target_column)` | Generates preprocessing code from a profile dict |
| `generate_cleaning_code(file_path)` | Profiles a file and generates cleaning code |
| `analyze_dataset(file_path)` | Returns a plain-text cleaning analysis |
| `repair_cleaning_code(profile, target_column, previous_code, error)` | Repairs a failed cleaning script |

---

### `agent/ml_agent.py`

LLM-powered functions for the **ML training** stage.

| Function | Description |
|---|---|
| `analyze_ml_problem(profile, target_column)` | Determines problem type and selects 2–3 baseline models |
| `generate_training_code(profile, target, type, models)` | Generates a complete scikit-learn training script |
| `repair_training_code(profile, target, type, models, code, error)` | Repairs a failed training script |

---

### `agent/graph.py`

The central LangGraph state machine. Defines all nodes, routing functions, and the compiled graph.

**Cleaning nodes:** `inspect → generate → validate → execute → verify → repair`  
**ML nodes:** `analyze_ml → generate_training → validate_training → execute_training → verify_training → repair_training`

| Function | Description |
|---|---|
| `build_graph()` | Assembles and compiles the full LangGraph workflow |
| `run_autonomous_cleaning(file_path, target_column)` | Public entry point — runs the complete two-stage pipeline |
| `validate_target(profile, target_column)` | Checks that a target column exists in the dataset profile |

---

### `tools/dataset_tools.py`

`inspect_dataset(file_path) -> dict`  
Profiles a CSV and returns: shape, column dtypes, missing/unique counts, numeric statistics (mean, median, min, max), low-cardinality column details, and top-5 sample rows.

---

### `tools/code_validator.py`

`validate_generated_code(code) -> dict`  
Parses generated Python via AST and blocks dangerous imports (`subprocess`, `socket`, `requests`) and builtins (`eval`, `exec`, `compile`, `__import__`).

---

### `tools/python_executor.py`

`execute_python_code(code, working_directory, timeout_seconds=60) -> dict`  
Writes code to a temp file, executes it in a subprocess inside the workspace, and returns `success`, `exit_code`, `stdout`, and `stderr`. Auto-cleans temp files.

---

### `tools/workspace.py`

`create_workspace(source_file) -> Path`  
Creates a unique UUID-named directory under `workspace/` and copies the source dataset as `input.csv`.

---

### `tools/cleanup.py`

`cleanup_expired_workspaces() -> None`  
Scans `workspace/` and deletes any subdirectory whose last modification time exceeds `WORKSPACE_EXPIRY_SECONDS` (default: 1 hour). Called at the start of every `/analyze` request — no background scheduler needed. Errors on individual directories are caught and logged so a single locked workspace cannot abort the sweep.

---

### `tools/model_test.py`

Quick utility script to inspect a saved `best_model.joblib` artifact using `joblib.load`. Update the path inside the file and run directly to verify a saved model.

---

## Agent Workflow

### Stage 1 — Data Cleaning

```
inspect
   └─► generate (LLM + target rules)
           └─► validate (AST)
                   ├─✓─► execute (subprocess)
                   │          ├─✓─► verify → [Stage 2]
                   │          └─✗─► repair (LLM) ─► validate ...
                   └─✗─► repair (LLM) ─► validate ...
                                          └── (max retries → fail)
```

### Stage 2 — ML Training

```
analyze_ml (LLM)
   └─► generate_training (LLM)
           └─► validate_training (AST)
                   ├─✓─► execute_training (subprocess, 120s)
                   │          ├─✓─► verify_training → END ✓
                   │          └─✗─► repair_training (LLM) ─► validate_training ...
                   └─✗─► repair_training (LLM) ─► validate_training ...
                                                    └── (max retries → fail)
```

---

## Setup & Installation

### Prerequisites
- Python 3.11+
- A valid [Google Gemini API Key](https://aistudio.google.com/app/apikey)

### 1. Clone / navigate to the project

```bash
cd "d:\DSmith AI"
```

### 2. Create and activate a virtual environment

```powershell
# Create
python -m venv .venv

# Activate — PowerShell
.venv\Scripts\Activate.ps1

# Activate — CMD
.venv\Scripts\activate.bat

# Activate — Linux / macOS
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```ini
GEMINI_API_KEY=your_gemini_api_key_here
```

---

## Running the Server

```bash
uvicorn main:app --reload
```

Server starts at [http://127.0.0.1:8000](http://127.0.0.1:8000).  
Interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Running the Agent Programmatically

```python
from agent.graph import run_autonomous_cleaning

result = run_autonomous_cleaning(
    file_path="path/to/your/dataset.csv",
    target_column="YourTargetColumn"
)

print("Success:        ", result["success"])
print("Problem Type:   ", result["problem_type"])
print("Best Model:     ", result["best_model"])
print("Metrics:        ", result["metrics"])
print("Workspace:      ", result["workspace"])
```

---

## Running Tests

Run the full test suite with pytest from the project root:

```bash
pytest tests/ -v
```

Individual test modules and what they cover:

| File | LLM Calls | Description |
|---|---|---|
| `tests/test_cleaning.py` | ❌ None | Fast unit tests for `inspect_dataset` — validates CSV profiling |
| `tests/test_agent.py` | ✅ Yes | Tests data cleaning code generation via Gemini |
| `tests/test_ml_agent.py` | ✅ Yes | Tests ML problem detection and training code generation |
| `tests/test_graph.py` | ✅ Yes | Full LangGraph pipeline integration test (cleaning + ML) |
| `tests/test_end_to_end.py` | ✅ Yes | HTTP-level integration tests via FastAPI `TestClient` |

> **Note:** Tests marked with ✅ make real Gemini LLM calls and execute generated code in subprocesses. They can take several minutes to complete.

---

## API Reference

Base URL: `http://127.0.0.1:8000`  
Interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

### `GET /`

Returns service identity and status.

```json
{
  "name": "DSmith AI",
  "status": "running",
  "description": "Autonomous Data Science Agent"
}
```

---

### `GET /health`

Lightweight health check — returns `200` when the server is up.

```json
{ "status": "healthy" }
```

---

### `POST /analyze`

Upload a CSV file and specify a target column. The agent runs the full autonomous cleaning and ML pipeline.

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | `File` | ✅ | CSV dataset to analyze |
| `target_column` | `string` (Form) | ✅ | Name of the supervised-learning target column |

**Example — cURL:**

```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -F "file=@dataset.csv" \
  -F "target_column=Salary"
```

**Example — Python (`requests`):**

```python
import requests

with open("dataset.csv", "rb") as f:
    response = requests.post(
        "http://127.0.0.1:8000/analyze",
        files={"file": ("dataset.csv", f, "text/csv")},
        data={"target_column": "Salary"},
    )

print(response.json())
```

**Success Response `200`:**

```json
{
  "success": true,
  "original_filename": "dataset.csv",
  "target_column": "Salary",
  "problem_type": "regression",
  "problem_reasoning": "...",
  "selected_models": ["RandomForestRegressor", "LinearRegression"],
  "best_model": "RandomForestRegressor",
  "metrics": {
    "problem_type": "regression",
    "target": "Salary",
    "models": {
      "RandomForestRegressor": { "mae": 4200.0, "rmse": 6100.0, "r2": 0.87 },
      "LinearRegression":      { "mae": 5500.0, "rmse": 7800.0, "r2": 0.81 }
    },
    "best_model": "RandomForestRegressor"
  },
  "cleaning": {
    "summary": "...",
    "plan": ["...", "..."],
    "retries": 0
  },
  "training": {
    "retries": 0
  },
  "downloads": {
    "cleaned_dataset": "/download/<job_id>/cleaned",
    "trained_model":   "/download/<job_id>/model"
  }
}
```

**Error Responses:**

| Status | Condition |
|---|---|
| `400` | No file provided, non-CSV file, empty target, or target column not found in dataset |
| `413` | Uploaded file exceeds the 20 MB size limit |
| `500` | Agent failed to complete analysis, or unexpected internal error |

**Error `400` — target not found:**

```json
{
  "detail": {
    "message": "Target column does not exist.",
    "target_column": "BadColumn",
    "available_columns": ["Age", "Salary", "Department"]
  }
}
```

---

### `GET /download/{job_id}/cleaned`

Download the cleaned CSV dataset produced by a completed analysis job.

The workspace is retained on disk for **1 hour** after the job completes, giving the client time to fetch artifacts before they are deleted by the next cleanup sweep.

| Parameter | Type | Description |
|---|---|---|
| `job_id` | `string` (path) | UUID returned in the `/analyze` response `downloads.cleaned_dataset` field |

**Example — cURL:**

```bash
curl -O "http://127.0.0.1:8000/download/<job_id>/cleaned"
```

**Example — Python:**

```python
response = requests.get(f"http://127.0.0.1:8000/download/{job_id}/cleaned")
with open("cleaned_dataset.csv", "wb") as f:
    f.write(response.content)
```

| Status | Condition |
|---|---|
| `200` | Returns `cleaned_dataset.csv` (`text/csv`) |
| `400` | `job_id` is not a valid UUID |
| `404` | File not found — job expired or cleaning stage failed |

---

### `GET /download/{job_id}/model`

Download the best trained model artifact produced by a completed analysis job.

The file is a joblib-serialised scikit-learn `Pipeline`. Load it with `joblib.load('best_model.joblib')`.

| Parameter | Type | Description |
|---|---|---|
| `job_id` | `string` (path) | UUID returned in the `/analyze` response `downloads.trained_model` field |

**Example — cURL:**

```bash
curl -O "http://127.0.0.1:8000/download/<job_id>/model"
```

**Example — Python:**

```python
import joblib, requests

response = requests.get(f"http://127.0.0.1:8000/download/{job_id}/model")
with open("best_model.joblib", "wb") as f:
    f.write(response.content)

model = joblib.load("best_model.joblib")
```

| Status | Condition |
|---|---|
| `200` | Returns `best_model.joblib` (`application/octet-stream`) |
| `400` | `job_id` is not a valid UUID |
| `404` | File not found — job expired or training stage failed |

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | ✅ Yes | Google Gemini API key used for all LLM inference |

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn |
| LLM Provider | Google Gemini (`gemini-3.5-flash-lite`) |
| LLM Orchestration | LangChain + LangGraph |
| Data Processing | pandas |
| Machine Learning | scikit-learn + joblib |
| Schema Validation | Pydantic v2 |
| Code Safety | Python `ast` module |
| Environment | python-dotenv |
| Testing | pytest + FastAPI TestClient |

---

## Deployment



Set the `GEMINI_API_KEY` environment variable in your hosting platform's settings panel.

> **Ephemeral filesystems:** Platforms like Heroku use ephemeral disks — workspace files are lost on dyno restart. For persistent artifact storage, replace the `workspace/` directory writes with an object storage backend (e.g. AWS S3, Google Cloud Storage).

---

> **Workspace Lifecycle:** Each job runs in an isolated `workspace/<uuid>/` directory. Artifacts are kept for **1 hour** so clients can download them via the `/download` endpoints. On the next `/analyze` request, `cleanup_expired_workspaces()` removes any directories older than the TTL. Concurrent runs never interfere with each other.
