"""Pydantic models and schemas for structured LLM outputs."""

# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field
from typing import TypedDict, Optional

# Define schema structure for LLM structured output response
class CleaningResult(BaseModel):
    # A short textual summary of what the code will do
    summary:str

    # List of sequential cleaning steps to be executed
    cleaning_plan: list[str]

    # The actual standalone Python script that performs the preprocessing
    generated_code: str = Field(
        description = "Executable Python preprocessing code"
    )

class MLDecision(BaseModel):
    """LLM response schema for ML problem type analysis and model selection."""

    problem_type: str = Field(
        description=(
            "Either 'classification' or 'regression'"
        )
    )

    reasoning: str

    selected_models: list[str]

class TrainingPlan(BaseModel):
    """LLM response schema for a generated ML training script."""

    explanation: str

    generated_code: str

class RepairResult(BaseModel):
    """LLM response schema for a repaired data-cleaning script."""
    explanation: str
    generated_code: str

class TrainingRepair(BaseModel):
    """LLM response schema for a repaired ML training script."""
    explanation: str
    generated_code: str

class AgentState(TypedDict, total=False):
    """State definition for the autonomous data cleaning agent workflow."""
    input_path: str
    workspace: str

    dataset_profile: dict

    summary: str
    cleaning_plan: list[str]
    generated_code: str

    validation_error: Optional[str]
    execution_result: dict

    cleaned_profile: dict

    error_message: Optional[str]

    cleaning_retry_count: int
    training_retry_count: int
    max_cleaning_retries: int
    max_training_retries: int

    success: bool

    target_column: str

    problem_type: str
    problem_reasoning: str

    selected_models: list[str]

    training_code: str
    training_result: dict

    metrics: dict
    best_model: str