"""ML planning, training code generation, and repair functions for the autonomous ML agent."""

import json

# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field

from agent.data_agent import model
from models.schemas import MLDecision, TrainingPlan, TrainingRepair


# Structured output models bound to their response schemas
ml_decision_model = model.with_structured_output(MLDecision)
training_model = model.with_structured_output(TrainingPlan)
training_repair_model = model.with_structured_output(TrainingRepair)

def analyze_ml_problem(
    dataset_profile: dict,
    target_column: str
) -> MLDecision:
    """Ask the LLM to determine the problem type (classification/regression) and select baseline models."""

    prompt = f"""
You are the machine learning planning component
of an autonomous data science system.

Determine the machine learning problem type and
select suitable baseline models.

DATASET PROFILE:

{json.dumps(dataset_profile, indent=2, default=str)}

TARGET COLUMN:

{target_column}

Rules:

1. problem_type must be exactly:
   - classification
   OR
   - regression

2. Use the target's characteristics to determine
   the problem type.

3. Select 2-3 sensible baseline models.

4. Prefer reliable scikit-learn models.

5. Do not generate training code yet.

6. Do not invent columns or dataset properties.

7. Avoid unnecessarily complex deep learning models
   for small tabular datasets.
"""

    return ml_decision_model.invoke(prompt)

def generate_training_code(
    dataset_profile: dict,
    target_column: str,
    problem_type: str,
    selected_models: list[str]
) -> TrainingPlan:
    """Generate a complete standalone training script for the given problem type and models."""

    prompt = f"""
You are the ML implementation component of an
autonomous data science agent.

Generate a complete standalone Python training script.

DATASET PROFILE:

{json.dumps(dataset_profile, indent=2, default=str)}

TARGET:
{target_column}

PROBLEM TYPE:
{problem_type}

MODELS:
{selected_models}

STRICT RULES:

- Read "cleaned.csv".
- Use pandas and scikit-learn.
- Do not access the internet.
- Do not install packages.
- Do not use subprocess.
- Separate features X and target y.
- Prevent target leakage.
- Handle categorical and numerical features properly.
- Use a train/test split.
- Use random_state=42 where supported.
- Use appropriate preprocessing pipelines.
- Train every selected model.
- Evaluate every model on the SAME held-out test set.
- Save the best trained pipeline to "best_model.joblib".
- The script must run standalone.

STRICT METRICS OUTPUT FORMAT:

The generated script MUST save a file named "metrics.json".

The JSON MUST use exactly this top-level structure:

{{
    "problem_type": "{problem_type}",
    "target": "{target_column}",
    "models": {{
        "<model_name>": {{
            "<metric_name>": <numeric_value>
        }}
    }},
    "best_model": "<best_model_name>"
}}

Rules for metrics.json:

- "problem_type" is mandatory.
- "target" is mandatory.
- "models" is mandatory.
- "best_model" is mandatory.
- Store every trained model and its metrics inside "models".
- "best_model" must contain the exact name of the selected best model.
- Metric values must come from actual model evaluation.
- Do not invent or estimate metric values.
"""

    if problem_type == "classification":
        prompt += """
CLASSIFICATION REQUIREMENTS:

For every model report:

- accuracy
- precision
- recall
- f1

Use appropriate averaging for multiclass targets.

Choose the best model primarily using F1 score.

The model with the highest F1 score must be stored
in the "best_model" field of metrics.json.
"""

    elif problem_type == "regression":
        prompt += """
REGRESSION REQUIREMENTS:

For every model report:

- mae
- rmse
- r2

Choose the best model primarily using RMSE.

The model with the lowest RMSE must be stored
in the "best_model" field of metrics.json.
"""

    return training_model.invoke(prompt)
def repair_training_code(
    dataset_profile: dict,
    target_column: str,
    problem_type: str,
    selected_models: list[str],
    previous_code: str,
    error_message: str
) -> TrainingRepair:
    """Repair a failed training script by feeding the error back to the LLM."""

    prompt = f"""
You are repairing failed machine-learning training code
for an autonomous data science agent.

The previous training script failed during actual execution.

TARGET COLUMN:
{target_column}

PROBLEM TYPE:
{problem_type}

SELECTED MODELS:
{selected_models}

DATASET PROFILE:
{json.dumps(dataset_profile, indent=2, default=str)}

FAILED TRAINING CODE:
{previous_code}

ACTUAL EXECUTION ERROR:
{error_message}

Fix the actual cause of the error and return a complete
standalone Python training script.

STRICT RULES:

- Read the dataset from "cleaned.csv".
- Preserve the target column.
- Prevent target leakage.
- Use a train/test split.
- Use pandas and scikit-learn.
- Use only the selected models.
- Handle numeric and categorical features appropriately.
- Do not access the internet.
- Do not install packages.
- Do not use subprocess.
- Do not invent columns.
- Save the best trained pipeline to "best_model.joblib".

STRICT METRICS OUTPUT FORMAT:

The repaired script MUST save "metrics.json"
using exactly this top-level structure:

{{
    "problem_type": "{problem_type}",
    "target": "{target_column}",
    "models": {{
        "<model_name>": {{
            "<metric_name>": <numeric_value>
        }}
    }},
    "best_model": "<best_model_name>"
}}

- Preserve this metrics.json structure even while repairing.
- All metric values must come from actual model evaluation.
- Store every trained model's metrics inside "models".
- "best_model" must contain the exact selected model name.
"""

    if problem_type == "classification":
        prompt += """
For classification:

- Report accuracy, precision, recall and f1.
- Use appropriate averaging for multiclass targets.
- Select best_model using the highest F1 score.
"""

    elif problem_type == "regression":
        prompt += """
For regression:

- Report mae, rmse and r2.
- Select best_model using the lowest RMSE.
"""

    return training_repair_model.invoke(prompt)