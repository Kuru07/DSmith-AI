"""
main.py — DSmith AI FastAPI Application Entry Point

Defines the HTTP API for the DSmith AI autonomous data science agent.
Handles file upload validation, dataset inspection, and dispatches the
full cleaning + ML training pipeline via the LangGraph agent graph.

Endpoints:
    GET  /         → Service identity and status
    GET  /health   → Health check
    POST /analyze  → Full autonomous cleaning + ML pipeline
"""

from typing import final
import shutil
from pathlib import Path
from uuid import uuid4

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
