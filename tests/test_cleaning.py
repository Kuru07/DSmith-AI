"""
tests/test_cleaning.py — Dataset Inspection Unit Tests

Tests the `inspect_dataset` tool from `tools/dataset_tools.py` against
the sample employee CSV.  These are fast, deterministic unit tests that
do not make any LLM calls.

Covers:
  - Sample data file existence check
  - Dataset profile structure (shape + columns)
  - Presence of known expected column names
"""

from pathlib import Path

from tools.dataset_tools import inspect_dataset


# Path to the shared sample dataset used across the test suite
SAMPLE_FILE = Path("tests/data/sample_employee.csv")


def test_dataset_exists():
    """Sample dataset required by the test suite should exist."""

    assert SAMPLE_FILE.exists()


def test_dataset_inspection():
    """
    inspect_dataset should return a valid profile dict containing at
    minimum a 'shape' key and a 'columns' key, with non-zero dimensions.

    This confirms the CSV was parsed correctly and basic metadata was
    extracted without errors.
    """

    profile = inspect_dataset(str(SAMPLE_FILE))

    assert profile is not None

    # Profile must expose dataset dimensions
    assert "shape" in profile
    assert "columns" in profile

    # The sample dataset must have at least one row and column
    assert profile["shape"]["rows"] > 0
    assert profile["shape"]["columns"] > 0


def test_expected_columns_exist():
    """
    The sample employee dataset should contain the known columns that
    the rest of the test suite depends on: Age, Salary, and Remote_Work.

    Age and Salary are numeric features; Remote_Work is the binary
    classification target used in most integration tests.
    """

    profile = inspect_dataset(str(SAMPLE_FILE))

    column_names = [
        column["name"]
        for column in profile["columns"]
    ]

    expected_columns = [
        "Age",
        "Salary",
        "Remote_Work",
    ]

    for column in expected_columns:
        assert column in column_names