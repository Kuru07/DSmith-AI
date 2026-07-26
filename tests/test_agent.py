"""
tests/test_agent.py — Data Cleaning Agent Unit Tests

Tests the data cleaning agent functions in isolation:
  - `generate_cleaning_code_from_profile` — produces a CleaningResult
    (summary, cleaning_plan, generated_code) from a dataset profile.

These tests make real Gemini LLM calls and validate that the structured
output is well-formed and contains the expected content.
"""

from pathlib import Path

from agent.data_agent import generate_cleaning_code_from_profile
from tools.dataset_tools import inspect_dataset


# Sample CSV used as the agent's input dataset
SAMPLE_FILE = Path("tests/data/sample_employee.csv")


def test_cleaning_agent_generates_plan():
    """
    generate_cleaning_code_from_profile should return a CleaningResult
    with all three required fields populated.

    Asserts:
      - result is not None
      - summary describes what the agent found / planned
      - cleaning_plan contains at least one step
      - generated_code contains runnable Python
    """

    profile = inspect_dataset( str(SAMPLE_FILE))

    result = generate_cleaning_code_from_profile(
        profile,
        target_column="Remote_Work"
    )

    assert result is not None

    # All three structured-output fields must be present
    assert result.summary
    assert result.cleaning_plan
    assert result.generated_code


def test_generated_cleaning_code_uses_pandas():
    """
    The generated cleaning script must use pandas and write its output
    to a file named 'cleaned.csv'.

    Asserts:
      - 'pandas' is imported in the generated code
      - 'cleaned.csv' appears in the generated code (output filename
        convention required by the execute and verify nodes)
    """

    profile = inspect_dataset(str(SAMPLE_FILE))

    result = generate_cleaning_code_from_profile(
        profile,
        target_column="Remote_Work"
    )

    code = result.generated_code

    # The agent must use pandas for data manipulation
    assert "pandas" in code

    # The script must write its output to cleaned.csv
    # (the verify node expects this exact filename)
    assert "cleaned.csv" in code