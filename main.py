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
    # 1. Validate filename
    # -----------------------------------------------------

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file provided.",
        )

    # -----------------------------------------------------
    # 2. Validate file type
    # -----------------------------------------------------

    extension = Path(file.filename).suffix.lower()

    if extension != ".csv":
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported.",
        )

    # -----------------------------------------------------
    # 3. Validate target
    # -----------------------------------------------------

    target_column = target_column.strip()

    if not target_column:
        raise HTTPException(
            status_code=400,
            detail="Target column cannot be empty.",
        )

    # -----------------------------------------------------
    # 4. Create uploads directory
    # -----------------------------------------------------

    upload_dir = Path("uploads")

    upload_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # 5. Generate unique filename
    # -----------------------------------------------------

    upload_id = str(uuid4())

    upload_path = (
        upload_dir
        / f"{upload_id}.csv"
    )

    try:

        # -------------------------------------------------
        # 6. Save uploaded CSV
        # -------------------------------------------------

        with upload_path.open("wb") as buffer:
            shutil.copyfileobj(
                file.file,
                buffer,
            )

        # -------------------------------------------------
        # 7. Inspect dataset before running agent
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
        # 8. Get available columns
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
        # 9. Validate target exists
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
        # 10. Run autonomous agent
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

        result = run_autonomous_cleaning(
            file_path=str(upload_path),
            target_column=target_column,
        )

        # -------------------------------------------------
        # 11. Handle agent failure
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
        # 12. Successful API response
        # -------------------------------------------------

        print("\n================================")
        print("DSmith AI Analysis Completed")
        print("================================")

        return {
            "success": True,

            "original_filename":
                file.filename,

            "target_column":
                target_column,

            "problem_type":
                result.get(
                    "problem_type"
                ),

            "problem_reasoning":
                result.get(
                    "problem_reasoning"
                ),

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

            "cleaning": {

                "summary":
                    result.get(
                        "summary"
                    ),

                "plan":
                    result.get(
                        "cleaning_plan"
                    ),

                "retries":
                    result.get(
                        "cleaning_retry_count",
                        0,
                    ),
            },

            "training": {

                "retries":
                    result.get(
                        "training_retry_count",
                        0,
                    ),
            },
        }

    # -----------------------------------------------------
    # Preserve our HTTP errors
    # -----------------------------------------------------

    except HTTPException:
        raise

    # -----------------------------------------------------
    # Unexpected server errors
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
    # Always close uploaded file
    # -----------------------------------------------------

    finally:

        file.file.close()