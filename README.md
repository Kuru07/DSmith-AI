# DSmith AI — Autonomous Data Science Agent

> An end-to-end, self-repairing data science pipeline powered by **Gemini**, **LangGraph**, and **scikit-learn**.  
> DSmith AI ingests a raw CSV dataset, autonomously cleans it, selects an appropriate ML problem type, trains and evaluates baseline models, and exports a production-ready model artifact — all without human intervention.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Modules & Components](#modules--components)
5. [Agent Workflow](#agent-workflow)
6. [Setup & Installation](#setup--installation)
7. [Running the Agent](#running-the-agent)
8. [Running Tests](#running-tests)
9. [API Reference](#api-reference)
10. [Environment Variables](#environment-variables)
11. [Tech Stack](#tech-stack)

---

## Overview

DSmith AI is a **FastAPI-based** backend that exposes an autonomous data science agent. Given a CSV file and a target column, the agent:

1. **Inspects** the raw dataset and builds a structured profile.
2. **Generates** Python preprocessing code via an LLM (Gemini).
3. **Validates** the generated code for security and syntax issues.
4. **Executes** the code in an isolated workspace.
5. **Verifies** the cleaned output for quality (missing-value regression check).
6. **Self-repairs** if any stage fails — up to a configurable number of retries.
7. **Analyzes** the ML problem type (classification or regression).
8. **Generates** a complete scikit-learn training script.
9. **Trains** selected baseline models and evaluates them on a held-out test set.
10. **Exports** `metrics.json` and `best_model.joblib` to the job workspace.

---

## Architecture

```
Raw CSV
   │
   ▼
┌─────────────┐     LLM (Gemini)      ┌─────────────────────┐
│   INSPECT   │ ──────────────────►  │      GENERATE       │
│  (profile)  │                       │  (cleaning code)    │
└─────────────┘                       └──────────┬──────────┘
                                                  │
                                         ┌────────▼────────┐
                                         │    VALIDATE     │
                                         │  (AST + rules)  │
                                         └────────┬────────┘
                                    valid │        │ invalid
                                          │   ┌────▼──────┐
                                          │   │   REPAIR  │◄──┐
                                          │   └────┬──────┘   │
                                          │        │           │
                                 ┌────────▼────────▼──┐       │
                                 │      EXECUTE       │       │
                                 │  (subprocess run)  │       │
                                 └────────┬───────────┘       │
                                 success  │   failure ────────┘
                                          │
                                 ┌────────▼───────────┐
                                 │      VERIFY        │
                                 │  (quality checks)  │
                                 └────────┬───────────┘
                                          │  pass
                                          ▼
                               ┌─────────────────────┐
                               │   ANALYZE ML PROBLEM│  (LLM)
                               └──────────┬──────────┘
                                          │
                               ┌──────────▼──────────┐
                               │  GENERATE TRAINING  │  (LLM)
                               └──────────┬──────────┘
                                          │
                               ┌──────────▼──────────┐
                               │  VALIDATE TRAINING  │
                               └──────────┬──────────┘
                                          │
                               ┌──────────▼──────────┐
                               │  EXECUTE TRAINING   │
                               └──────────┬──────────┘
                                          │
                               ┌──────────▼──────────┐
                               │  VERIFY TRAINING    │
                               │  metrics.json ✓     │
                               │  best_model.joblib ✓│
                               └─────────────────────┘
```

---

## Project Structure

```text
DSmith AI/
├── .env                        # API keys (git-ignored)
├── .gitignore
├── main.py                     # FastAPI application entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── models/                     # Pydantic schemas for LLM structured outputs
│   ├── __init__.py
│   └── schemas.py              # CleaningResult, MLDecision, TrainingPlan,
│                               # RepairResult, TrainingRepair, AgentState
│
├── agent/                      # Core agent logic
│   ├── __init__.py
│   ├── data_agent.py           # Cleaning code generation, analysis, and repair (LLM)
│   ├── ml_agent.py             # ML problem analysis, training code generation, and repair (LLM)
│   └── graph.py                # LangGraph state machine — full end-to-end workflow
│
├── tools/                      # Stateless utility functions
│   ├── __init__.py
│   ├── dataset_tools.py        # CSV profiling (shape, dtypes, missing values, stats)
│   ├── code_validator.py       # AST-based security/syntax validator
│   ├── python_executor.py      # Subprocess-based isolated code runner
│   └── workspace.py            # Per-job UUID workspace creation
│
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── sample_data.csv         # Sample raw dataset for testing
│   ├── cleaned.csv             # Pre-cleaned dataset (for ML-only tests)
│   ├── test_tools.py           # Unit tests for dataset tools
│   ├── test_agent.py           # Tests for data agent functions
│   ├── test_graph.py           # End-to-end cleaning graph test
│   ├── test_ml_agent.py        # Full end-to-end pipeline test (cleaning + ML)
│   ├── test_ml_agent_only.py   # ML pipeline test only (skips cleaning)
│   └── test_saved_training.py  # Re-runs a saved training script without LLM calls
│
├── workspace/                  # Auto-generated per-job workspaces (git-ignored)
└── uploads/                    # Uploaded datasets via API (git-ignored)
```

---

## Modules & Components

### `models/schemas.py`
All Pydantic models used as structured output schemas for LLM responses, plus the shared `AgentState` TypedDict used across the LangGraph workflow.

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
LLM-powered functions for the **data cleaning** stage.

| Function | Description |
|---|---|
| `generate_cleaning_code_from_profile(profile)` | Generates preprocessing code from a dataset profile dict |
| `generate_cleaning_code(file_path)` | Profiles a file then generates cleaning code |
| `analyze_dataset(file_path)` | Returns a plain-text analysis of cleaning requirements |
| `repair_cleaning_code(profile, previous_code, error)` | Repairs a failed cleaning script |

---

### `agent/ml_agent.py`
LLM-powered functions for the **ML training** stage.

| Function | Description |
|---|---|
| `analyze_ml_problem(profile, target_column)` | Determines problem type and selects baseline models |
| `generate_training_code(profile, target, type, models)` | Generates a complete scikit-learn training script |
| `repair_training_code(profile, target, type, models, code, error)` | Repairs a failed training script |

---

### `agent/graph.py`
The central LangGraph state machine. Defines all nodes, routing functions, and the compiled graph.

**Cleaning nodes:** `inspect → generate → validate → execute → verify → repair`  
**ML nodes:** `analyze_ml → generate_training → validate_training → execute_training → verify_training → repair_training`

**Key functions:**

| Function | Description |
|---|---|
| `build_graph()` | Assembles and compiles the full LangGraph workflow |
| `run_autonomous_cleaning(file_path, target_column)` | Public entry point — runs the complete pipeline |

---

### `tools/dataset_tools.py`
`inspect_dataset(file_path) -> dict`  
Profiles a CSV and returns: shape, column dtypes, missing/unique counts, numeric statistics (mean, median, min, max), low-cardinality column details, and top-5 sample rows.

---

### `tools/code_validator.py`
`validate_generated_code(code) -> dict`  
Parses generated Python code via AST and blocks dangerous imports (`subprocess`, `socket`, `requests`) and dangerous builtins (`eval`, `exec`, `compile`, `__import__`).

---

### `tools/python_executor.py`
`execute_python_code(code, working_directory, timeout_seconds=60) -> dict`  
Writes code to a temp file, executes it in a subprocess within the workspace, and returns `success`, `exit_code`, `stdout`, and `stderr`. Cleans up temp files automatically.

---

### `tools/workspace.py`
`create_workspace(source_file) -> Path`  
Creates a unique UUID-named job directory under `workspace/` and copies the source dataset as `input.csv`.

---

## Agent Workflow

### Stage 1 — Data Cleaning

```
inspect → generate → validate ──✓──► execute ──✓──► verify ──✓──► [Stage 2]
                        │                  │              │
                        └──✗──► repair ◄──┘              └──✗──► repair
                                   │
                              (max retries → fail)
```

### Stage 2 — ML Training

```
analyze_ml → generate_training → validate_training ──✓──► execute_training ──✓──► verify_training → END
                                         │                        │
                                         └──✗──► repair_training ◄┘
                                                      │
                                               (max retries → fail)
```

---

## Setup & Installation

### Prerequisites
- Python 3.11+
- A valid [Gemini API Key](https://aistudio.google.com/app/apikey)

### 1. Clone / Navigate to the project

```bash
cd "d:\DSmith AI"
```

### 2. Create and activate a virtual environment

```powershell
# Create
python -m venv .venv

# Activate (PowerShell)
.venv\Scripts\Activate.ps1

# Activate (CMD)
.venv\Scripts\activate.bat
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

## Running the Agent

### Programmatic usage

```python
from agent.graph import run_autonomous_cleaning

result = run_autonomous_cleaning(
    file_path="path/to/your/dataset.csv",
    target_column="YourTargetColumn"
)

print("Success:", result["success"])
print("Best Model:", result["best_model"])
print("Metrics:", result["metrics"])
print("Workspace:", result["workspace"])
```

### Via the FastAPI server

Start the development server:

```bash
uvicorn main:app --reload
```

The API will be available at [http://127.0.0.1:8000](http://127.0.0.1:8000).

---

## Running Tests

### Full end-to-end pipeline (cleaning + ML)
```bash
python -m tests.test_ml_agent
```

### ML pipeline only (uses a pre-cleaned dataset, skips LLM cleaning calls)
```bash
python -m tests.test_ml_agent_only
```

### Re-run a previously saved training script (zero LLM calls)
```bash
python -m tests.test_saved_training
```

### Cleaning graph only
```bash
python -m tests.test_graph
```

### Dataset and code execution tools
```bash
python -m tests.test_tools
```

---

## API Reference

Base URL: `http://127.0.0.1:8000`

Interactive documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### `GET /`
Health ping.

**Response:**
```json
{ "message": "Data Science Agent is Running" }
```

### `GET /health`
Environment validation.

**Response:**
```json
{ "status": "Healthy", "gemini_configured": true }
```

### `POST /analyze`
Submit a dataset filename for analysis.

**Request body:**
```json
{ "filename": "dataset.csv" }
```

**Response:**
```json
{ "message": "Received Dataset, dataset.csv!" }
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | ✅ Yes | Google Gemini API key for LLM inference |

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn |
| LLM Provider | Google Gemini (`gemini-3.5-flash-lite`) |
| LLM Orchestration | LangChain + LangGraph |
| Data Processing | pandas |
| Machine Learning | scikit-learn |
| Schema Validation | Pydantic v2 |
| Code Safety | Python `ast` module |
| Environment | python-dotenv |

---

> **Note:** All job workspaces are created under the `workspace/` directory and are self-contained. Each run uses a unique UUID-named folder so concurrent runs do not interfere with each other.
