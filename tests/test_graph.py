"""
tests/test_graph.py — LangGraph Pipeline Integration Test

Directly invokes `run_autonomous_cleaning` (the LangGraph entry point)
without going through the HTTP layer.  Tests that the complete two-stage
pipeline (data cleaning → ML training) executes successfully end-to-end
and returns a well-formed result dict.

This test makes real LLM calls and runs generated code in a subprocess,
so it is an integration test and may take a minute or more to complete.
"""

from pathlib import Path

from agent.graph import run_autonomous_cleaning


# Sample CSV used as the pipeline's input dataset
SAMPLE_FILE = Path(
    "tests/data/sample_employee.csv"
)


def test_complete_graph_execution():
    """
    Run the full LangGraph pipeline (cleaning + ML) against the sample
    employee dataset and assert the result is successful and complete.

    Checks:
      - result is not None
      - success flag is True
      - problem_type is correctly detected as "classification"
      - selected_models list is non-empty
      - metrics dict is present (produced by execute_training node)
      - best_model key is populated (selected by verify_training node)
    """

    result = run_autonomous_cleaning(
        file_path=str(SAMPLE_FILE),
        target_column="Remote_Work",
    )

    # The graph must return a result
    assert result is not None

    # The pipeline must report overall success
    assert result.get("success") is True

    # Remote_Work is binary — the agent should detect classification
    assert (
        result.get("problem_type")
        == "classification"
    )

    # At least one scikit-learn model must have been selected
    assert result.get(
        "selected_models"
    )

    # metrics.json must have been produced and parsed
    assert result.get(
        "metrics"
    )

    # verify_training node must identify the winning model
    assert result.get(
        "best_model"
    )