import json
from pathlib import Path

from tools.code_validator import validate_generated_code
from tools.python_executor import execute_python_code


# ============================================================
# CONFIGURATION
# ============================================================

WORKSPACE = Path("workspace/ml_test")

CLEANED_FILE = WORKSPACE / "cleaned.csv"
GENERATED_SCRIPT = WORKSPACE / "generated_training.py"
METRICS_FILE = WORKSPACE / "metrics.json"
MODEL_FILE = WORKSPACE / "best_model.joblib"


# ============================================================
# TEST
# ============================================================

def test_saved_training():

    print("\n========================================")
    print("   DSmith AI - SAVED TRAINING TEST")
    print("========================================")


    # --------------------------------------------------------
    # STEP 1: Verify required input files
    # --------------------------------------------------------

    print("\n[1] CHECKING SAVED FILES")

    assert WORKSPACE.exists(), (
        f"Workspace does not exist: {WORKSPACE}"
    )

    assert CLEANED_FILE.exists(), (
        "cleaned.csv does not exist"
    )

    assert GENERATED_SCRIPT.exists(), (
        "generated_training.py does not exist"
    )

    print("✓ Workspace exists")
    print("✓ cleaned.csv exists")
    print("✓ generated_training.py exists")


    # --------------------------------------------------------
    # STEP 2: Read saved generated code
    # --------------------------------------------------------

    print("\n[2] LOADING SAVED TRAINING CODE")

    training_code = GENERATED_SCRIPT.read_text(
        encoding="utf-8"
    )

    assert training_code.strip(), (
        "generated_training.py is empty"
    )

    print(
        f"✓ Loaded {len(training_code)} characters"
    )


    # --------------------------------------------------------
    # STEP 3: Validate saved code
    # --------------------------------------------------------

    print("\n[3] VALIDATING SAVED CODE")

    validation = validate_generated_code(
        training_code
    )

    if not validation["valid"]:

        print("✗ Validation failed")
        print(
            "Reason:",
            validation.get("reason")
        )

        raise AssertionError(
            validation.get(
                "reason",
                "Code validation failed"
            )
        )

    print("✓ Saved training code passed validation")


    # --------------------------------------------------------
    # STEP 4: Delete previous output artifacts
    #
    # Important:
    # We KEEP:
    #   cleaned.csv
    #   generated_training.py
    #
    # We DELETE:
    #   metrics.json
    #   best_model.joblib
    #
    # This proves the saved script can recreate them.
    # --------------------------------------------------------

    print("\n[4] REMOVING OLD OUTPUT ARTIFACTS")

    if METRICS_FILE.exists():

        METRICS_FILE.unlink()

        print("✓ Removed old metrics.json")

    else:

        print("- No old metrics.json")


    if MODEL_FILE.exists():

        MODEL_FILE.unlink()

        print("✓ Removed old best_model.joblib")

    else:

        print("- No old best_model.joblib")


    assert not METRICS_FILE.exists()
    assert not MODEL_FILE.exists()


    # --------------------------------------------------------
    # STEP 5: Execute saved generated code
    #
    # ZERO Gemini calls.
    # --------------------------------------------------------

    print("\n[5] EXECUTING SAVED TRAINING CODE")

    execution = execute_python_code(
        code=training_code,
        working_directory=str(WORKSPACE),
        timeout_seconds=120
    )

    print(
        "Exit Code:",
        execution.get("exit_code")
    )

    print("\nSTDOUT:")

    print(
        execution.get(
            "stdout",
            ""
        )
    )

    if execution.get("stderr"):

        print("\nSTDERR:")

        print(
            execution["stderr"]
        )


    assert execution["success"], (
        "Saved training code execution failed:\n"
        + execution.get(
            "stderr",
            "Unknown execution error"
        )
    )

    print("✓ Saved training code executed successfully")


    # --------------------------------------------------------
    # STEP 6: Verify metrics.json was recreated
    # --------------------------------------------------------

    print("\n[6] VERIFYING RECREATED METRICS")

    assert METRICS_FILE.exists(), (
        "Training succeeded but metrics.json "
        "was not recreated"
    )

    metrics = json.loads(
        METRICS_FILE.read_text(
            encoding="utf-8"
        )
    )

    print(
        json.dumps(
            metrics,
            indent=2
        )
    )


    # --------------------------------------------------------
    # STEP 7: Validate metrics schema
    # --------------------------------------------------------

    print("\n[7] VALIDATING METRICS SCHEMA")

    required_keys = {
        "problem_type",
        "target",
        "models",
        "best_model"
    }

    missing_keys = (
        required_keys - metrics.keys()
    )

    assert not missing_keys, (
        f"Missing metrics keys: {missing_keys}"
    )

    assert metrics["problem_type"] in {
        "classification",
        "regression"
    }

    assert isinstance(
        metrics["target"],
        str
    )

    assert isinstance(
        metrics["models"],
        dict
    )

    assert len(metrics["models"]) > 0

    assert isinstance(
        metrics["best_model"],
        str
    )

    print("✓ problem_type exists")
    print("✓ target exists")
    print("✓ models exists")
    print("✓ best_model exists")


    # --------------------------------------------------------
    # STEP 8: Verify classification metrics
    # --------------------------------------------------------

    if metrics["problem_type"] == "classification":

        print(
            "\n[8] VALIDATING CLASSIFICATION METRICS"
        )

        required_metrics = {
            "accuracy",
            "precision",
            "recall",
            "f1"
        }

        for model_name, model_metrics in (
            metrics["models"].items()
        ):

            print(
                f"\nChecking: {model_name}"
            )

            missing = (
                required_metrics
                - model_metrics.keys()
            )

            assert not missing, (
                f"{model_name} missing metrics: "
                f"{missing}"
            )

            for metric_name in required_metrics:

                value = model_metrics[
                    metric_name
                ]

                assert isinstance(
                    value,
                    (int, float)
                )

                assert 0 <= value <= 1, (
                    f"{model_name} "
                    f"{metric_name} outside "
                    f"expected range: {value}"
                )

            print("✓ accuracy valid")
            print("✓ precision valid")
            print("✓ recall valid")
            print("✓ f1 valid")


    # --------------------------------------------------------
    # STEP 9: Verify best model really has highest F1
    # --------------------------------------------------------

    if metrics["problem_type"] == "classification":

        print(
            "\n[9] VERIFYING BEST MODEL SELECTION"
        )

        calculated_best_model = max(
            metrics["models"],
            key=lambda model_name:
                metrics["models"][model_name]["f1"]
        )

        print(
            "Reported Best Model:",
            metrics["best_model"]
        )

        print(
            "Calculated Best Model:",
            calculated_best_model
        )

        assert (
            metrics["best_model"]
            == calculated_best_model
        ), (
            "best_model does not have "
            "the highest F1 score"
        )

        print(
            "✓ Best model correctly selected "
            "using F1"
        )


    # --------------------------------------------------------
    # STEP 10: Verify model artifact
    # --------------------------------------------------------

    print("\n[10] VERIFYING SAVED MODEL")

    assert MODEL_FILE.exists(), (
        "best_model.joblib was not recreated"
    )

    model_size = (
        MODEL_FILE.stat().st_size
    )

    assert model_size > 0, (
        "best_model.joblib is empty"
    )

    print("✓ best_model.joblib exists")

    print(
        f"✓ Model size: {model_size} bytes"
    )


    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print("\n========================================")
    print("      SAVED TRAINING TEST PASSED ✓")
    print("========================================")

    print(
        "Problem Type:",
        metrics["problem_type"]
    )

    print(
        "Target:",
        metrics["target"]
    )

    print(
        "Models:",
        list(metrics["models"].keys())
    )

    print(
        "Best Model:",
        metrics["best_model"]
    )


if __name__ == "__main__":
    test_saved_training()