"""
tests/test_end_to_end.py — FastAPI HTTP Integration Tests

Tests the full HTTP API layer via FastAPI's TestClient.
Covers:
  - Health check endpoint
  - File-type validation (non-CSV rejected)
  - Target column validation (missing column rejected)
  - Full end-to-end pipeline via POST /analyze (cleaning + ML)

These tests exercise the entire stack from HTTP request down to the
LangGraph agent and back, making them slower integration tests rather
than unit tests.
"""

from pathlib import Path

from fastapi.testclient import TestClient

from main import app


# Shared TestClient instance for all tests in this module
client = TestClient(app)

# Path to the sample CSV used for valid-upload tests
SAMPLE_FILE = Path(
    "tests/data/sample_employee.csv"
)


def test_health_endpoint():
    """
    GET /health should return HTTP 200 and {"status": "healthy"}.
    Confirms the server is up and reachable.
    """

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    assert response.json() == {
        "status": "healthy"
    }


def test_invalid_file_rejected():
    """
    POST /analyze with a non-CSV file (e.g. .txt) should return HTTP 400.
    The API must reject any upload whose file extension is not .csv.
    """

    response = client.post(
        "/analyze",
        files={
            "file": (
                "invalid.txt",
                b"hello world",
                "text/plain",
            )
        },
        data={
            "target_column":
                "Remote_Work"
        },
    )

    assert response.status_code == 400


def test_invalid_target_rejected():
    """
    POST /analyze with a valid CSV but a non-existent target column
    should return HTTP 400 and report the bad column name.

    This verifies the column-existence guard that runs after dataset
    inspection and before the agent is invoked.
    """

    with SAMPLE_FILE.open("rb") as file:

        response = client.post(
            "/analyze",

            files={
                "file": (
                    SAMPLE_FILE.name,
                    file,
                    "text/csv",
                )
            },

            data={
                "target_column":
                    "COLUMN_DOES_NOT_EXIST"
            },
        )

    assert response.status_code == 400


def test_full_analysis():
    """
    POST /analyze with the sample employee CSV and a valid target column
    should run the complete autonomous pipeline and return HTTP 200.

    Asserts the response contains:
      - success flag set to True
      - correct target column echoed back
      - problem_type correctly detected as classification
      - non-empty selected_models, best_model, and metrics fields
    """

    with SAMPLE_FILE.open("rb") as file:

        response = client.post(
            "/analyze",

            files={
                "file": (
                    SAMPLE_FILE.name,
                    file,
                    "text/csv",
                )
            },

            data={
                "target_column":
                    "Remote_Work"
            },
        )

    assert response.status_code == 200

    result = response.json()

    # Pipeline must report success
    assert result["success"] is True

    # Target column should be echoed back unchanged
    assert (
        result["target_column"]
        == "Remote_Work"
    )

    # Remote_Work is a binary column — should be classified as classification
    assert (
        result["problem_type"]
        == "classification"
    )

    # At least one model should have been selected and evaluated
    assert result[
        "selected_models"
    ]

    assert result[
        "best_model"
    ]

    assert result[
        "metrics"
    ]