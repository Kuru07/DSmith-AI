"""
main.py — DSmith AI FastAPI Application Entry Point

Defines the HTTP API for the DSmith AI autonomous data science agent.
Handles file upload validation, dataset inspection, and dispatches the
full cleaning + ML training pipeline via the LangGraph agent graph.

Endpoints:
    GET  /                              → Service identity and status
    GET  /health                        → Health check
    POST /analyze                       → Full autonomous cleaning + ML pipeline
    GET  /download/{job_id}/cleaned     → Download the cleaned dataset CSV
    GET  /download/{job_id}/model       → Download the trained model (.joblib)
"""

from uuid import UUID
from typing import final
import shutil
from pathlib import Path
from uuid import uuid4
from fastapi.responses import FileResponse
from tools.cleanup import cleanup_expired_workspaces

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from agent.graph import run_autonomous_cleaning
from tools.dataset_tools import inspect_dataset

# Maximum allowed upload size: 20 MB
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB

app = FastAPI(
    title="DSmith AI",
    description="Autonomous Data Science Agent",
    version="1.0.0",
)


# ---------------------------------------------------------
# ROOT
# ---------------------------------------------------------

@app.get("/")
def root():
    """Return service identity and running status."""
    return {
        "name": "DSmith AI",
        "status": "running",
        "description": "Autonomous Data Science Agent",
    }


# ---------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------

@app.get("/health")
def health():
    """Lightweight health check — returns 200 when the server is up."""
    return {
        "status": "healthy"
    }


# ---------------------------------------------------------
# ANALYZE DATASET
# ---------------------------------------------------------


@app.post("/analyze")
def analyze_dataset(
    file: UploadFile = File(...),
    target_column: str = Form(...),
):
    """
    Upload a CSV dataset and specify the target column.

    DSmith AI will autonomously:

    1. Inspect the dataset
    2. Generate preprocessing code
    3. Validate and execute the code
    4. Repair failures if necessary
    5. Verify the cleaned dataset
    6. Determine classification/regression
    7. Select ML models
    8. Generate and execute training code
    9. Evaluate models
    10. Return the best model and metrics
    """

    # Opportunistic cleanup: delete stale workspaces from previous jobs
    # before starting a new one. Avoids needing a background scheduler.
    cleanup_expired_workspaces()

    # -----------------------------------------------------
    # 1. Validate filename — reject requests with no file
    # -----------------------------------------------------

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file provided.",
        )

    # -----------------------------------------------------
    # 2. Validate file type — only CSV is accepted
    # -----------------------------------------------------

    extension = Path(file.filename).suffix.lower()

    if extension != ".csv":
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported.",
        )

    # -----------------------------------------------------
    # 3. Validate target — must not be blank
    # -----------------------------------------------------

    target_column = target_column.strip()

    if not target_column:
        raise HTTPException(
            status_code=400,
            detail="Target column cannot be empty.",
        )

    # -----------------------------------------------------
    # 4. Create uploads directory if it does not exist
    # -----------------------------------------------------

    upload_dir = Path("uploads")

    upload_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # 5. Generate a unique filename to prevent collisions
    #    between concurrent uploads
    # -----------------------------------------------------

    upload_id = str(uuid4())

    upload_path = upload_dir / f"{upload_id}.csv"
    

    try:

        # -------------------------------------------------
        # 6. Save the uploaded CSV to disk
        # -------------------------------------------------

        with upload_path.open("wb") as buffer:
            shutil.copyfileobj(
                file.file,
                buffer,
            )

        file_size = upload_path.stat().st_size

        # Reject files that exceed the size limit
        if file_size > MAX_FILE_SIZE:
            upload_path.unlink(missing_ok=True)

            raise HTTPException(
                status_code=413,
                detail="File too large. Maximum allowed size is 20 MB.",
            )

        # -------------------------------------------------
        # 7. Inspect dataset before running the agent
        #    — confirms the file is a valid, parseable CSV
        # -------------------------------------------------

        try:

            profile = inspect_dataset(str(upload_path))

        except Exception as exc:

            print(
                f"Dataset inspection failed: {exc}"
            )

            raise HTTPException(
                status_code=400,
                detail=(
                    "The uploaded file could not "
                    "be processed as a valid CSV."
                ),
            )

        # -------------------------------------------------
        # 8. Extract column names from the dataset profile
        # -------------------------------------------------

        columns = profile.get(
            "columns",
            []
        )

        column_names = [
            column["name"]
            for column in columns
        ]

        # -------------------------------------------------
        # 9. Validate that the requested target column
        #    actually exists in the uploaded dataset
        # -------------------------------------------------

        if target_column not in column_names:

            raise HTTPException(
                status_code=400,
                detail={
                    "message":
                        "Target column does not exist.",

                    "target_column":
                        target_column,

                    "available_columns":
                        column_names,
                },
            )

        # -------------------------------------------------
        # 10. Dispatch the autonomous LangGraph agent
        #     — runs the full cleaning + ML pipeline
        # -------------------------------------------------

        print("\n================================")
        print("DSmith AI Analysis Started")
        print("================================")

        print(
            f"File: {file.filename}"
        )

        print(
            f"Target: {target_column}"
        )
        print(str(upload_path))

        result = run_autonomous_cleaning(
            file_path=str(upload_path),
            target_column=target_column,
        )

        # Extract the UUID-named job directory from the result so we can
        # build download URLs that the client can use to fetch artifacts.
        workspace_path = Path(
            result["workspace"]
        )

        # The workspace directory name IS the job ID (a UUID string)
        job_id = workspace_path.name


        # -------------------------------------------------
        # 11. Handle agent-level failure
        #     — the graph ran but could not complete
        # -------------------------------------------------

        if not result.get("success"):

            print("\nAgent analysis failed.")

            print(
                result.get("error_message")
            )

            raise HTTPException(
                status_code=500,
                detail={
                    "message":
                        "DSmith AI could not complete "
                        "the analysis.",

                    "error":
                        result.get(
                            "error_message"
                        ),
                },
            )

        # -------------------------------------------------
        # 12. Return the structured success response
        # -------------------------------------------------

        print("\n================================")
        print("DSmith AI Analysis Completed")
        print("================================")

        return {
            "success": True,

            # Original upload metadata
            "original_filename":
                file.filename,

            "target_column":
                target_column,

            # ML problem classification
            "problem_type":
                result.get(
                    "problem_type"
                ),

            "problem_reasoning":
                result.get(
                    "problem_reasoning"
                ),

            # Model selection and evaluation
            "selected_models":
                result.get(
                    "selected_models"
                ),

            "best_model":
                result.get(
                    "best_model"
                ),

            "metrics":
                result.get(
                    "metrics"
                ),

            # Cleaning stage diagnostics
            "cleaning": {

                "summary":
                    result.get(
                        "summary"
                    ),

                "plan":
                    result.get(
                        "cleaning_plan"
                    ),

                # Number of LLM repair attempts needed
                "retries":
                    result.get(
                        "cleaning_retry_count",
                        0,
                    ),
            },

            # Training stage diagnostics
            "training": {

                # Number of LLM repair attempts needed
                "retries":
                    result.get(
                        "training_retry_count",
                        0,
                    ),
            },
            # Pre-built download URLs for the artifacts produced by this job.
            # The workspace is retained on disk for WORKSPACE_EXPIRY_SECONDS
            # (1 hour) so the client has time to fetch them.
            "downloads": {

                "cleaned_dataset":
                    f"/download/{job_id}/cleaned",

                "trained_model":
                    f"/download/{job_id}/model",
            },
        }

    # -----------------------------------------------------
    # Re-raise our own HTTP errors unchanged so FastAPI
    # returns the correct status code to the caller
    # -----------------------------------------------------

    except HTTPException:
        raise

    # -----------------------------------------------------
    # Catch unexpected server errors and return a clean
    # 500 without leaking internal tracebacks
    # -----------------------------------------------------

    except Exception as exc:

        print(
            f"Unexpected analysis error: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail="Internal analysis error.",
        )

    # -----------------------------------------------------
    # Always close the uploaded file handle and remove
    # temporary files regardless of success or failure
    # -----------------------------------------------------
    finally:

        file.file.close()

        if upload_path.exists():
            upload_path.unlink()

            print(
                f"[CLEANUP] Deleted upload: {upload_path}"
            )

# ---------------------------------------------------------
# JOB ID VALIDATION HELPER
# ---------------------------------------------------------

def validate_job_id(job_id: str):
    """
    Validate that `job_id` is a well-formed UUID string.

    The job ID doubles as the workspace directory name, so rejecting
    malformed values prevents path-traversal attempts and gives the
    client a clear error before any filesystem access occurs.

    Raises:
        HTTPException 400 — if the value is not a valid UUID.
    """
    try:
        UUID(job_id)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid job ID."
        )

# ---------------------------------------------------------
# DOWNLOAD ENDPOINTS
# ---------------------------------------------------------

@app.get("/download/{job_id}/cleaned")
def download_cleaned_dataset(
    job_id: str
):
    """
    Download the cleaned CSV dataset produced by a completed analysis job.

    The file is served from workspace/<job_id>/cleaned.csv — the output
    written by the execute node of the LangGraph cleaning pipeline.

    Returns:
        200 + cleaned_dataset.csv  — file download
        400                        — job_id is not a valid UUID
        404                        — cleaned.csv not found (job may have
                                     expired or the cleaning stage failed)
    """
    validate_job_id(job_id)

    file_path = (
        Path("workspace")
        / job_id
        / "cleaned.csv"
    )

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Cleaned dataset not found."
        )

    return FileResponse(
        path=file_path,
        filename="cleaned_dataset.csv",
        media_type="text/csv"
    )

@app.get("/download/{job_id}/model")
def download_trained_model(
    job_id: str
):
    """
    Download the best trained model artifact produced by a completed job.

    The file is served from workspace/<job_id>/best_model.joblib — a
    joblib-serialised scikit-learn Pipeline written by the training script
    during the ML stage. Load it with `joblib.load('best_model.joblib')`.

    Returns:
        200 + best_model.joblib    — file download (application/octet-stream)
        400                        — job_id is not a valid UUID
        404                        — model file not found (job may have
                                     expired or the training stage failed)
    """
    validate_job_id(job_id)

    file_path = (
        Path("workspace")
        / job_id
        / "best_model.joblib"
    )

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Trained model not found."
        )

    return FileResponse(
        path=file_path,
        filename="best_model.joblib",
        media_type="application/octet-stream"
    )
