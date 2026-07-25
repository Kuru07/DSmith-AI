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
11. **Returns** a structured JSON response with results, metrics, and diagnostics.

---

## Architecture

```
POST /analyze  (CSV file + target_column)
        │
        ▼
┌──────────────────────────────────────────────────────┐
│                      main.py                         │
│  1. Validate file type (.csv only)                   │
│  2. Validate target column exists                    │
│  3. Save upload to uploads/<uuid>.csv                │
│  4. Call run_autonomous_cleaning(file, target)       │
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
├── main.py                       # FastAPI application — upload, validate, dispatch
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
│   └── model_test.py             # Quick utility to inspect a saved joblib model
│
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── sample_data.csv           # Sample raw dataset
│   ├── cleaned.csv               # Pre-cleaned dataset (for ML-only tests)
│   ├── test_tools.py             # Unit tests for dataset tools
│   ├── test_agent.py             # Tests for data agent functions
│   ├── test_graph.py             # End-to-end cleaning graph test
│   ├── test_ml_agent.py          # Full end-to-end pipeline test (cleaning + ML)
│   ├── test_ml_agent_only.py     # ML pipeline test only (skips cleaning)
│   └── test_saved_training.py    # Re-runs a saved training script without LLM calls
│
├── uploads/                      # Uploaded CSVs saved here (git-ignored)
└── workspace/                    # Auto-generated per-job workspaces (git-ignored)
    └── <uuid>/
        ├── input.csv             # Copy of the uploaded dataset
        ├── cleaned.csv           # Cleaned output
        ├── generated_training.py # Saved training script
        ├── metrics.json          # Model evaluation metrics
        └── best_model.joblib     # Best trained pipeline
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

LLM-powered functions for the **data cleaning** stage. All cleaning and repair prompts now include explicit **target-leakage-prevention rules** — the target column is never imputed, never dropped, and never used as a feature-engineering source.

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

| Command | Description |
|---|---|
| `python -m tests.test_ml_agent` | Full end-to-end pipeline (cleaning + ML, uses LLM) |
| `python -m tests.test_ml_agent_only` | ML pipeline only — uses a pre-cleaned CSV, skips cleaning LLM calls |
| `python -m tests.test_saved_training` | Re-executes a saved training script — zero LLM calls |
| `python -m tests.test_graph` | Cleaning graph only |
| `python -m tests.test_tools` | Dataset inspection and code execution unit tests |

---

## API Reference

Base URL: `http://127.0.0.1:8000`  
Interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

### `GET /`

```json
{
  "name": "DSmith AI",
  "status": "running",
  "description": "Autonomous Data Science Agent"
}
```

---

### `GET /health`

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
  }
}
```

**Error Responses:**

| Status | Condition |
|---|---|
| `400` | No file provided, non-CSV file, empty target, or target column not found in dataset |
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

---

> **Workspaces:** Each job runs in an isolated `workspace/<uuid>/` directory. Concurrent runs never interfere with each other. The workspace contains `input.csv`, `cleaned.csv`, `generated_training.py`, `metrics.json`, and `best_model.joblib`.
