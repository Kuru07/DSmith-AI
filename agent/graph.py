"""LangGraph state machine definition for the autonomous data cleaning workflow."""
import json
from typing import TypedDict, Optional
from pathlib import Path

# pyrefly: ignore [missing-import]
from langgraph.graph import (
    StateGraph,
    START,
    END
)

from tools.dataset_tools import inspect_dataset
from agent.data_agent import generate_cleaning_code_from_profile,repair_cleaning_code
from tools.code_validator import validate_generated_code
from tools.python_executor import execute_python_code
from tools.workspace import create_workspace
from agent.ml_agent import analyze_ml_problem,generate_training_code, repair_training_code
from models.schemas import AgentState



def analyze_ml_node(
    state: AgentState
) -> dict:
    """Determines problem type and selects baseline ML models using the LLM."""
    print("\n[NODE] ANALYZE ML PROBLEM")

    decision = analyze_ml_problem(
        dataset_profile=state["cleaned_profile"],
        target_column=state["target_column"]
    )

    return {
        "problem_type": decision.problem_type,
        "problem_reasoning": decision.reasoning,
        "selected_models": decision.selected_models
    }

def validate_target(
    profile: dict,
    target_column: str
) -> bool:
    """Returns True if the target column exists in the dataset profile."""

    columns = [
        column["name"]
        for column in profile["columns"]
    ]

    return target_column in columns

def inspect_node(state: AgentState) -> dict:
    """Inspects the input dataset to generate a profile."""

    print("\n[NODE] INSPECT")

    profile = inspect_dataset(state["input_path"])

    return {
        "dataset_profile": profile
    }

def generate_node(state: AgentState) -> dict:
    """Generates preprocessing code based on the dataset profile."""

    print("\n[NODE] GENERATE")

    decision = generate_cleaning_code_from_profile(
        state["dataset_profile"]
    )

    return {
        "summary": decision.summary,
        "cleaning_plan": decision.cleaning_plan,
        "generated_code": decision.generated_code,
        "validation_error": None,
        "error_message": None,
    }

def validate_node(state: AgentState) -> dict:
    """Validates the generated python code for security and syntax."""

    print("\n[NODE] VALIDATE")

    result = validate_generated_code(
        state["generated_code"]
    )

    if result["valid"]:
        return {
            "validation_error": None
        }

    return {
        "validation_error": result["reason"],
        "error_message": result["reason"]
    }

def route_after_validation(state: AgentState) -> str:

    if state.get("validation_error"):
        return "repair"

    return "execute"

def execute_node(state: AgentState) -> dict:
    """Executes the validated python code in an isolated workspace."""

    print("\n[NODE] EXECUTE")

    result = execute_python_code(
        code=state["generated_code"],
        working_directory=state["workspace"]
    )

    if result["success"]:
        return {
            "execution_result": result,
            "error_message": None
        }

    error = result.get(
        "stderr",
        result.get("error", "Unknown execution error")
    )

    return {
        "execution_result": result,
        "error_message": error
    }

def route_after_execution(state: AgentState) -> str:

    result = state["execution_result"]

    if result["success"]:
        return "verify"

    return "repair"

def total_missing(profile: dict) -> int:
    return sum(
        column["missing"]
        for column in profile["columns"]
    )

def verify_node(state: AgentState) -> dict:
    """Verifies the output dataset after execution."""

    print("\n[NODE] VERIFY")

    cleaned_path = (
        Path(state["workspace"])
        / "cleaned.csv"
    )

    if not cleaned_path.exists():
        return {
            "success": False,
            "error_message":
                "cleaned.csv was not created."
        }

    cleaned_profile = inspect_dataset(
       str(cleaned_path)
    )

    before = total_missing(
        state["dataset_profile"]
    )

    after = total_missing(
        cleaned_profile
    )

    if after > before:
        return {
            "cleaned_profile": cleaned_profile,
            "success": False,
            "error_message": (
                "Preprocessing increased missing "
                f"values from {before} to {after}."
            )
        }

    return {
        "cleaned_profile": cleaned_profile,
        "success": True,
        "error_message": None
    }

def route_verification(state: AgentState) -> str:

    if state.get("success"):
        return "done"

    if state.get("retry_count", 0) >= state.get(
        "max_retries", 3
    ):
        return "fail"

    return "repair"

def repair_node(state: AgentState) -> dict:
    """Attempts to repair failed generated code using LLM."""

    print("\n[NODE] REPAIR")

    retry_count = state.get(
        "retry_count",
        0
    ) + 1

    repaired = repair_cleaning_code(
        profile=state["dataset_profile"],
        previous_code=state["generated_code"],
        error_message=state["error_message"]
    )

    print(
        f"Repair attempt {retry_count}"
    )

    return {
        "generated_code": repaired.generated_code,
        "retry_count": retry_count,
        "validation_error": None,
        "error_message": None
    }

def route_after_failure(state: AgentState) -> str:

    retry_count = state.get(
        "retry_count",
        0
    )

    max_retries = state.get(
        "max_retries",
        3
    )

    if retry_count >= max_retries:
        return "fail"

    return "repair"

def fail_node(state: AgentState) -> dict:
    """Handles the terminal failure state when retries are exhausted."""

    print("\n[NODE] FAILED")

    return {
        "success": False
    }

def generate_training_node(
    state: AgentState
) -> dict:
    """Generates a Python training script via LLM based on the cleaned dataset profile."""
    print("\n[NODE] GENERATE TRAINING CODE")

    result = generate_training_code(
        dataset_profile=state["cleaned_profile"],
        target_column=state["target_column"],
        problem_type=state["problem_type"],
        selected_models=state["selected_models"]
    )


    return {
        "training_code": result.generated_code,
        "error_message": None
    }

def execute_training_node(
    state: AgentState
) -> dict:
    """Executes the generated training script in the isolated workspace."""
    print("\n[NODE] EXECUTE TRAINING")

    result = execute_python_code(
        code=state["training_code"],
        working_directory=state["workspace"],
        timeout_seconds=120
    )

    if result["success"]:
        return {
            "training_result": result,
            "error_message": None
        }

    return {
        "training_result": result,
        "error_message": (
            result.get("stderr")
            or result.get("error")
            or "Unknown training error"
        )
    }

def verify_training_node(
    state: AgentState
) -> dict:
    """Verifies that metrics.json and best_model.joblib were created after training."""
    print("\n[NODE] VERIFY TRAINING")

    workspace = Path(
        state["workspace"]
    )

    metrics_path = (
        workspace / "metrics.json"
    )

    model_path = (
        workspace / "best_model.joblib"
    )

    if not metrics_path.exists():
        return {
            "success": False,
            "error_message":
                "metrics.json was not created."
        }

    if not model_path.exists():
        return {
            "success": False,
            "error_message":
                "best_model.joblib was not created."
        }

    with open(
        metrics_path,
        "r",
        encoding="utf-8"
    ) as file:

        metrics = json.load(file)

    return {
        "metrics": metrics,
        "best_model": metrics.get(
            "best_model"
        ),
        "success": True,
        "error_message": None
    }

def repair_training_node(
    state: AgentState
) -> dict:
    """Asks the LLM to fix a failed training script and increments the retry counter."""
    print("\n[NODE] REPAIR TRAINING")

    retry_count = (
        state.get("training_retry_count", 0) + 1
    )

    repaired = repair_training_code(
        dataset_profile=state["cleaned_profile"],
        target_column=state["target_column"],
        problem_type=state["problem_type"],
        selected_models=state["selected_models"],
        previous_code=state["training_code"],
        error_message=state["error_message"]
    )

    print(
        f"Training repair attempt {retry_count}"
    )

    return {
        "training_code": repaired.generated_code,
        "training_retry_count": retry_count,
        "error_message": None
    }

def validate_training_node(state: AgentState) -> dict:
    """Validates generated training code for security and syntax before execution."""
    print("\n[NODE] VALIDATE TRAINING")

    result = validate_generated_code(
        state["training_code"]
    )

    if result["valid"]:
        return {
            "validation_error": None,
            "error_message": None
        }

    return {
        "validation_error": result["reason"],
        "error_message": result["reason"]
    }

def route_validation(state: AgentState) -> str:

    if not state.get("validation_error"):
        return "execute"

    if state.get("retry_count", 0) >= state.get(
        "max_retries", 3
    ):
        return "fail"

    return "repair"

def route_execution(state: AgentState) -> str:

    if state["execution_result"]["success"]:
        return "verify"

    if state.get("retry_count", 0) >= state.get(
        "max_retries", 3
    ):
        return "fail"

    return "repair"

def route_training_validation(state: AgentState) -> str:
    """Routes after training validation: execute if valid, repair or fail otherwise."""

    if not state.get("validation_error"):
        return "execute_training"

    retries = state.get(
        "training_retry_count",
        0
    )

    max_retries = state.get(
        "max_training_retries",
        3
    )

    if retries >= max_retries:
        return "fail"

    return "repair_training"

def route_training_execution(
    state: AgentState
) -> str:
    """Routes after training execution: verify if succeeded, repair or fail otherwise."""

    if state["training_result"]["success"]:
        return "verify_training"

    retries = state.get(
        "training_retry_count",
        0
    )

    max_retries = state.get(
        "max_training_retries",
        3
    )

    if retries >= max_retries:
        return "fail"

    return "repair_training"

def build_graph():
    """Builds and compiles the LangGraph state machine."""

    builder = StateGraph(AgentState)

    builder.add_node(
        "inspect",
        inspect_node
    )

    builder.add_node(
        "generate",
        generate_node
    )

    builder.add_node(
        "validate",
        validate_node
    )

    builder.add_node(
        "execute",
        execute_node
    )

    builder.add_node(
        "repair",
        repair_node
    )

    builder.add_node(
        "verify",
        verify_node
    )

    builder.add_node(
        "fail",
        fail_node
    )

    builder.add_node(
    "analyze_ml",
    analyze_ml_node
)

    builder.add_node(
        "generate_training",
        generate_training_node
    )

    builder.add_node(
        "validate_training",
        validate_training_node
    )

    builder.add_node(
        "execute_training",
        execute_training_node
    )

    builder.add_node(
        "repair_training",
        repair_training_node
    )

    builder.add_node(
        "verify_training",
        verify_training_node
    )

    builder.add_edge(
        START,
        "inspect"
    )

    builder.add_edge(
        "inspect",
        "generate"
    )

    builder.add_edge(
        "generate",
        "validate"
    )

    builder.add_edge(
    "analyze_ml",
    "generate_training"
    )

    builder.add_edge(
        "generate_training",
        "validate_training"
    )

    builder.add_edge(
        "repair_training",
        "validate_training"
    )
    

    builder.add_conditional_edges(
        "validate",
        route_validation,
        {
            "execute": "execute",
            "repair": "repair",
            "fail": "fail"
        }
    )

    builder.add_conditional_edges(
        "execute",
        route_execution,
        {
            "verify": "verify",
            "repair": "repair",
            "fail": "fail"
        }
    )

    builder.add_conditional_edges(
    "execute_training",
    route_training_execution,
    {
        "verify_training": "verify_training",
        "repair_training": "repair_training",
        "fail": "fail"
    }
    )

    builder.add_edge(
        "repair",
        "validate"
    )

    builder.add_conditional_edges(
        "verify",
        route_verification,
        {
            "done": "analyze_ml",
            "repair": "repair",
            "fail": "fail"
        }
    )

    builder.add_conditional_edges(
        "validate_training",
        route_training_validation,
        {
            "execute_training": "execute_training",
            "repair_training": "repair_training",
            "fail": "fail"
        }
    )

    builder.add_edge(
        "fail",
        END
    )

    return builder.compile()



graph = build_graph()


def run_autonomous_cleaning(
    file_path: str,
    target_column: str
):
    """Entry point to execute the autonomous data cleaning workflow on a given file."""

    workspace = create_workspace(
        file_path
    )

    input_path = (
        workspace / "input.csv"
    )

    initial_state: AgentState = {
        "input_path": str(input_path),
        "workspace": str(workspace),
        "cleaning_retry_count": 0,
        "max_cleaning_retries": 3,
        "target_column": target_column,
        "training_retry_count": 0,
        "max_training_retries": 3,
        
        "success": False
    }

    result = graph.invoke(
        initial_state
    )

    return result

