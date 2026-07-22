# DSmith AI

Autonomous Data Science Agent API (v1.0.0).

This is a FastAPI-based backend service that serves as the entry point for the autonomous Data Science Agent.

## Features

- **FastAPI Framework**: High-performance, easy-to-use API framework.
- **Pydantic Validation**: Robust input parsing and validation using Pydantic models.
- **Health Check**: Automatic validation of required keys (e.g., `GEMINI_API_KEY`).
- **Automatic Docs**: Interactive documentation generated via Swagger UI.
- **Dataset Analysis**: Core tool suite to inspect, clean, and profile CSV datasets.
- **Python Execution Engine**: Utility to safely execute custom Python scripts locally with configurable timeouts.

---

## Project Structure

```text
DSmith AI/
├── .venv/               # Local virtual environment
├── .env                 # Environment variables (git-ignored)
├── .gitignore           # Git ignore list
├── main.py              # Main FastAPI application
├── requirements.txt     # Project dependencies
├── README.md            # Project documentation (this file)
├── tests/               # Testing suite
│   ├── __init__.py
│   └── test_tools.py    # Scripts to test dataset and code execution utilities
└── tools/               # Core utility tools for the agent
    ├── __init__.py
    ├── dataset_tools.py # Tools for dataset inspection, profiling, and metadata extraction
    └── python_executor.py # Subprocess-based safe Python execution engine
```

---

## Utilities & Tools

The project includes core helper modules under `tools/` that the Data Science agent can utilize:

### 1. Dataset Tools (`tools/dataset_tools.py`)
Provides the `inspect_dataset(file_path: str)` function, which:
- Validates the dataset format (currently supports `.csv`).
- Returns shape metadata (number of rows and columns).
- Profiles all columns (data types, missing value count, and unique value counts).
- Performs numeric column profiling (generates mean, median, min, max values).
- Detects low-cardinality categorical columns (unique value count <= 20) with sample values.
- Retrieves the top 5 sample rows for preview.

### 2. Python Executor (`tools/python_executor.py`)
Provides the `execute_python_code(code: str, working_directory: str, timeout_seconds: int = 60)` function, which:
- Safely writes arbitrary Python code to a temporary file in the workspace directory.
- Runs it in a separate subprocess using the active Python executable.
- Captures `stdout`, `stderr`, and exit status codes.
- Implements a configurable execution timeout.
- Automatically cleans up the temporary files after execution.

---

## Setup & Installation

Follow these steps to set up the project locally:

### 1. Clone/Navigate to the directory
Ensure you are in the project root directory:
```bash
cd "d:\DSmith AI"
```

### 2. Configure Environment Variables
Create a file named `.env` in the root directory (already configured) and add your Gemini API credentials:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Create and Activate Virtual Environment
If you haven't already, create a Python virtual environment:
```powershell
python -m venv .venv
```

Activate the virtual environment:
- On Windows (PowerShell):
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
- On Windows (CMD):
  ```cmd
  .venv\Scripts\activate.bat
  ```
- On Linux/macOS:
  ```bash
  source .venv/bin/activate
  ```

### 4. Install Dependencies
Install all required libraries from `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## Running the Application

Start the local Uvicorn development server:
```bash
uvicorn main:app --reload
```

The application will run on [http://127.0.0.1:8000](http://127.0.0.1:8000).

---

## Running Tests

To verify that the dataset inspection and Python execution tools function correctly, execute the following command from the root directory:

```bash
python -m tests.test_tools
```

---

## API Documentation & Testing

FastAPI automatically generates interactive API documentation. Once the server is running, you can access:

- **Swagger UI (Interactive Docs)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc (Alternative Docs)**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### API Endpoints

#### 1. Root Endpoint
- **URL**: `/`
- **Method**: `GET`
- **Response**:
  ```json
  {
    "message": "Data Science Agent is Running"
  }
  ```

#### 2. Health Status
- **URL**: `/health`
- **Method**: `GET`
- **Response**:
  ```json
  {
    "status": "Healthy",
    "gemini_configured": true
  }
  ```

#### 3. Analyze Dataset
- **URL**: `/analyze`
- **Method**: `POST`
- **Request Body** (`application/json`):
  ```json
  {
    "filename": "dataset.csv"
  }
  ```
- **Response**:
  ```json
  {
    "message": "Received Dataset, dataset.csv!"
  }
  ```
