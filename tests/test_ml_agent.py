"""
tests/test_ml_agent.py — ML Agent Unit Tests

Tests the ML agent functions from `agent/ml_agent.py` in isolation:
  - `analyze_ml_problem`   — classifies problem type and selects models
  - `generate_training_code` — produces a complete scikit-learn script

These tests make real Gemini LLM calls, so they are integration-level
and may take a minute or more to complete.

Covers:
  - Classification detection (binary target: Remote_Work)
  - Regression detection (continuous target: Salary)
  - Training code generation (uses cleaned.csv + train_test_split)
"""

from pathlib import Path

from agent.ml_agent import (
    analyze_ml_problem,
    generate_training_code,
)

from tools.dataset_tools import inspect_dataset


# Sample CSV used as the agent's input dataset
SAMPLE_FILE = Path("tests/data/sample_employee.csv")


def get_profile():
    """
    Helper: inspect the sample CSV and return its profile dict.
    Called at the start of each test to avoid module-level side effects.
    """

    return inspect_dataset( str(SAMPLE_FILE))


def test_classification_detection():
    """
    analyze_ml_problem should detect 'classification' when the target
    column (Remote_Work) is a low-cardinality binary feature.

    Also verifies that at least one scikit-learn classifier is selected.
    """

    profile = get_profile()

    decision = analyze_ml_problem(
        dataset_profile=profile,
        target_column="Remote_Work",
    )

    # Remote_Work is binary → should be classified as classification
    assert decision.problem_type == "classification"

    # At least one classifier must be recommended
    assert len(
        decision.selected_models
    ) >= 1


def test_regression_detection():
    """
    analyze_ml_problem should detect 'regression' when the target column
    (Salary) is a continuous numeric feature.

    Also verifies that at least one scikit-learn regressor is selected.
    """

    profile = get_profile()

    decision = analyze_ml_problem(
        dataset_profile=profile,
        target_column="Salary",
    )

    # Salary is continuous → should be classified as regression
    assert decision.problem_type == "regression"

    # At least one regressor must be recommended
    assert len(
        decision.selected_models
    ) >= 1


def test_training_code_generation():
    """
    generate_training_code should produce a non-empty Python training
    script that:
      - Reads data from 'cleaned.csv' (the execute node's output)
      - Uses 'train_test_split' for an 80/20 train-test split

    The Remote_Work target and the classification problem type are passed
    in, matching the output of test_classification_detection.
    """

    profile = get_profile()

    # First, determine the problem type and model selection
    decision = analyze_ml_problem(
        dataset_profile=profile,
        target_column="Remote_Work",
    )

    # Generate the training script based on the ML decision
    training = generate_training_code(
        dataset_profile=profile,
        target_column="Remote_Work",
        problem_type=decision.problem_type,
        selected_models=decision.selected_models,
    )

    # The script must be non-empty
    assert training.generated_code

    code = training.generated_code

    # The script must load data from cleaned.csv (pipeline convention)
    assert "cleaned.csv" in code

    # The script must split data into train/test sets
    assert "train_test_split" in code