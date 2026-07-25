import json
import shutil
from pathlib import Path

from agent.ml_agent import (
    analyze_ml_problem,
    generate_training_code,
)

from tools.dataset_tools import inspect_dataset
from tools.code_validator import validate_generated_code
from tools.python_executor import execute_python_code


# ============================================================
# TEST CONFIGURATION
# ============================================================

# Use an already-cleaned CSV here.
CLEANED_CSV = "tests/cleaned.csv"

# Change this for classification/regression testing.
TARGET_COLUMN = "Salary"

TEST_WORKSPACE = Path("workspace/ml_test")


# ============================================================
# HELPER
# ============================================================

def prepare_workspace():

    if TEST_WORKSPACE.exists():
        shutil.rmtree(TEST_WORKSPACE)

    TEST_WORKSPACE.mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.copy(
        CLEANED_CSV,
        TEST_WORKSPACE / "cleaned.csv"
    )

    return TEST_WORKSPACE


# ============================================================
# ML-ONLY TEST
# ============================================================

def test_ml_pipeline():

    print("\n========================================")
    print("       DSmith AI - ML ONLY TEST")
    print("========================================")

    # --------------------------------------------------------
    # STEP 1: Prepare workspace
    # --------------------------------------------------------

    print("\n[1] PREPARING WORKSPACE")

    workspace = prepare_workspace()

    cleaned_path = workspace / "cleaned.csv"

    assert cleaned_path.exists(), (
        "cleaned.csv was not copied to workspace"
    )

    print("✓ Workspace created")
    print("✓ cleaned.csv available")


    # --------------------------------------------------------
    # STEP 2: Inspect pre-cleaned dataset
    # NO GEMINI CALL
    # --------------------------------------------------------

    print("\n[2] INSPECTING CLEANED DATASET")

    profile = inspect_dataset( str(cleaned_path))

    print(
        f"Rows: {profile['shape']['rows']}"
    )

    print(
        f"Columns: {profile['shape']['columns']}"
    )

    column_names = [
        column["name"]
        for column in profile["columns"]
    ]

    print("\nAvailable columns:")

    for name in column_names:
        print("-", name)


    # --------------------------------------------------------
    # STEP 3: Validate target locally
    # --------------------------------------------------------

    print("\n[3] VALIDATING TARGET")

    assert TARGET_COLUMN in column_names, (
        f"Target '{TARGET_COLUMN}' does not exist."
    )

    print(
        f"✓ Target exists: {TARGET_COLUMN}"
    )


    # --------------------------------------------------------
    # STEP 4: Analyze ML problem
    # GEMINI CALL #1
    # --------------------------------------------------------

    print("\n[4] ANALYZING ML PROBLEM")

    decision = analyze_ml_problem(
        dataset_profile=profile,
        target_column=TARGET_COLUMN
    )

    print(
        "Problem Type:",
        decision.problem_type
    )

    print(
        "Reasoning:",
        decision.reasoning
    )

    print("\nSelected Models:")

    for model_name in decision.selected_models:
        print("-", model_name)


    assert decision.problem_type in {
        "classification",
        "regression"
    }

    assert len(decision.selected_models) > 0


    # --------------------------------------------------------
    # STEP 5: Generate training code
    # GEMINI CALL #2
    # --------------------------------------------------------

    print("\n[5] GENERATING TRAINING CODE")

    training_plan = generate_training_code(
        dataset_profile=profile,
        target_column=TARGET_COLUMN,
        problem_type=decision.problem_type,
        selected_models=decision.selected_models
    )

    training_code = training_plan.generated_code

    generated_script_path = workspace / "generated_training.py"

    generated_script_path.write_text(
        training_code,
        encoding="utf-8"
    )

    print(
        f"✓ Generated code saved: {generated_script_path}"
    )

    print("\nExplanation:")
    print(training_plan.explanation)

    print("\n========== GENERATED CODE ==========")

    print(training_code)

    print("====================================")


    assert training_code
    assert len(training_code) > 0


    # --------------------------------------------------------
    # STEP 6: Validate generated code
    # NO GEMINI CALL
    # --------------------------------------------------------

    print("\n[6] VALIDATING TRAINING CODE")

    validation = validate_generated_code(
        training_code
    )

    if not validation["valid"]:

        print("✗ Training code rejected")
        print(
            "Reason:",
            validation["reason"]
        )

        raise AssertionError(
            validation["reason"]
        )

    print("✓ Training code passed validation")


    # --------------------------------------------------------
    # STEP 7: Execute training
    # NO GEMINI CALL
    # --------------------------------------------------------

    print("\n[7] EXECUTING TRAINING")

    execution = execute_python_code(
        code=training_code,
        working_directory=str(workspace),
        timeout_seconds=120
    )

    print(
        "Exit Code:",
        execution.get("exit_code")
    )

    print("\nSTDOUT:")
    print(
        execution.get("stdout", "")
    )

    if execution.get("stderr"):

        print("\nSTDERR:")
        print(
            execution["stderr"]
        )

    assert execution["success"], (
        "Training execution failed:\n"
        + execution.get("stderr", "")
    )

    print("✓ Training executed successfully")


    # --------------------------------------------------------
    # STEP 8: Verify metrics.json
    # --------------------------------------------------------

    print("\n[8] VERIFYING METRICS")

    metrics_path = (
        workspace / "metrics.json"
    )

    assert metrics_path.exists(), (
        "metrics.json was not created"
    )

    with open(
        metrics_path,
        "r",
        encoding="utf-8"
    ) as file:

        metrics = json.load(file)

    print(
        json.dumps(
            metrics,
            indent=2
        )
    )

    assert isinstance(metrics, dict)

    assert "best_model" in metrics, (
        "best_model missing from metrics.json"
    )

    print(
        "\n✓ Best Model:",
        metrics["best_model"]
    )


    # --------------------------------------------------------
    # STEP 9: Verify saved model
    # --------------------------------------------------------

    print("\n[9] VERIFYING MODEL ARTIFACT")

    model_path = (
        workspace / "best_model.joblib"
    )

    assert model_path.exists(), (
        "best_model.joblib was not created"
    )

    assert model_path.stat().st_size > 0, (
        "best_model.joblib is empty"
    )

    print("✓ best_model.joblib exists")

    print(
        "Model size:",
        model_path.stat().st_size,
        "bytes"
    )


    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print("\n========================================")
    print("          ML TEST PASSED ✓")
    print("========================================")

    print(
        "Problem Type:",
        decision.problem_type
    )

    print(
        "Target:",
        TARGET_COLUMN
    )

    print(
        "Models Tested:",
        decision.selected_models
    )

    print(
        "Best Model:",
        metrics["best_model"]
    )


if __name__ == "__main__":
    test_ml_pipeline()