# DSmith AI

Autonomous Data Science Agent API (v1.0.0).

This is a FastAPI-based backend service that serves as the entry point for the autonomous Data Science Agent.

## Features

- **FastAPI Framework**: High performance, easy-to-use API framework.
- **Pydantic Validation**: Robust input parsing and validation using Pydantic models.
- **Health Check**: Automatic validation of required keys (e.g., `GEMINI_API_KEY`).
- **Automatic Docs**: Interactive documentation generated via Swagger UI.

---

## Project Structure

```text
DSmith AI/
├── .venv/               # Local virtual environment
├── .env                 # Environment variables (git-ignored)
├── .gitignore           # Git ignore list
├── main.py              # Main FastAPI application
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation (this file)
```

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
    "message": "Recieved Dataset, dataset.csv!"
  }
  ```
