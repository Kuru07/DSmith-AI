"""LLM-powered data cleaning, analysis, and repair functions for the autonomous data science agent."""
import os
import json

from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from langchain_google_genai import ChatGoogleGenerativeAI
from tools.dataset_tools import inspect_dataset
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field
from models.schemas import CleaningResult 
from tools.code_validator import validate_generated_code
from tools.python_executor import execute_python_code
from tools.workspace import create_workspace
from models.schemas import RepairResult

# Load environmental variables from .env file
load_dotenv()

# Initialize the Gemini model for structured and text output
model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0
)

# Configure model to return a structured CleaningResult schema
structured_model = model.with_structured_output(
    CleaningResult
)

def generate_cleaning_code_from_profile(
    profile: dict,
    target_column: str
) -> CleaningResult:

    profile_json = json.dumps(
        profile,
        indent=2,
        default=str
    )

    prompt = f"""
You are the DATA CLEANING component of an autonomous
data science agent.

Your responsibility is ONLY to improve data quality and
produce a human-readable cleaned version of the original
dataset.

You must generate:
1. A conservative cleaning plan.
2. Complete executable Python code implementing that plan.

The output cleaned.csv is intended both for:
- human download and inspection
- later machine-learning processing

Therefore, cleaned.csv MUST remain structurally close to
the original dataset.

TARGET COLUMN:
{target_column}

=========================================================
STRICT SEPARATION OF RESPONSIBILITIES
=========================================================

This stage performs DATA CLEANING ONLY.

DO NOT perform machine-learning feature preprocessing.

Specifically, DO NOT:

- One-hot encode categorical columns.
- Label encode categorical columns.
- Ordinal encode columns for modelling.
- Scale or normalize numerical columns.
- Standardize numerical columns.
- Create dummy variables.
- Perform feature selection.
- Perform dimensionality reduction.
- Create polynomial features.
- Convert categorical columns into model-ready numbers.
- Drop identifier columns merely because they are poor
  ML features.
- Drop PII columns merely because they should not be used
  for modelling.
- Transform the dataset into a feature matrix.
- Train or evaluate machine-learning models.

Those operations belong exclusively to the ML training
stage.

=========================================================
TARGET HANDLING RULES
=========================================================

The target column is the supervised-learning label.

- Never impute missing target values.
- Never generate synthetic target values.
- Never derive features from the target.
- Never encode the target for feature preprocessing.
- Never drop the target column itself.
- If the target contains missing values, drop ONLY rows
  where the target value is missing.
- Never use information from the target to clean feature
  columns.

The target should remain human-readable in cleaned.csv.

=========================================================
ALLOWED DATA CLEANING
=========================================================

Perform cleaning only when justified by the dataset
profile.

Examples of allowed operations include:

- Filling missing NUMERIC FEATURE values using an
  appropriate conservative statistic such as median.
- Filling missing CATEGORICAL FEATURE values using an
  appropriate conservative strategy such as mode or an
  explicit "Unknown" category.
- Removing exact duplicate rows when clearly detected.
- Standardizing obvious whitespace inconsistencies.
- Converting values to appropriate data types when safe.
- Parsing date columns into datetime when appropriate.
- Fixing clearly invalid or inconsistent representations
  only when supported by the dataset profile.

For date columns:

- You may parse them into a consistent date representation.
- DO NOT automatically create year/month/day feature
  columns.
- Preserve the original semantic information in a
  human-readable form.

=========================================================
COLUMN PRESERVATION
=========================================================

Preserve original columns whenever possible.

Identifier, name, email, phone, and other PII-like columns
must NOT be removed simply because they are unsuitable for
machine learning.

The ML training stage will decide which columns should be
excluded from model features.

Only remove a column if there is a clear DATA QUALITY
reason supported by the dataset profile, not merely a
machine-learning reason.

=========================================================
CODE REQUIREMENTS
=========================================================

- Use only columns that actually exist.
- Never invent dataset statistics.
- Read only "input.csv".
- Save the cleaned result as "cleaned.csv".
- Use pandas.
- Keep preprocessing conservative.
- Preserve human readability.
- Preserve the target column.
- Do not access the network.
- Do not use subprocesses.
- Do not train ML models.
- The generated code must run as a standalone script.
- Print a short cleaning summary after execution.

DATASET PROFILE:

{profile_json}
"""

    return structured_model.invoke(prompt)

def generate_cleaning_code(
    file_path: str
) -> CleaningResult:

    # Profile the dataset to analyze its columns and properties
    profile = inspect_dataset.func(file_path)

    profile_json = json.dumps(
        profile,
        indent=2,
        default=str
    )

    # Prompt containing specific instructions and constraints for data cleaning code generation
    prompt = f"""
You are the preprocessing component of an autonomous
data science agent.

Analyze the dataset profile below and create an
appropriate preprocessing plan.

You must also generate executable Python code.

RULES:

- Use only columns that actually exist.
- Never invent dataset statistics.
- Read the dataset from "input.csv".
- Save the processed dataset as "cleaned.csv".
- Use pandas for preprocessing.
- Preserve the target column.
- Do not train any ML models yet.
- Do not delete rows/columns unless there is a clear
  reason from the supplied profile.
- Keep preprocessing conservative.
- The generated code must run as a standalone script.
- Print a short preprocessing summary after execution.

Dataset profile:

{profile_json}
"""

    # Invoke LLM to generate the cleaning plan and python code
    return structured_model.invoke(prompt)

def analyze_dataset(file_path:str):
    # Profile dataset for high-level text analysis
    profile = inspect_dataset.func(file_path)
    profile_json = json.dumps(
        profile,
        indent=2,
        default=str
    )

    prompt = f"""
You are the DATA CLEANING analysis component of an
autonomous data science agent.

Analyze the dataset profile below.

Your task is ONLY to identify DATA QUALITY problems and
recommend conservative cleaning operations.

This stage produces a human-readable cleaned.csv that
should remain structurally close to the original dataset.

DO NOT recommend machine-learning feature preprocessing.

Specifically, do not recommend:

- One-hot encoding
- Label encoding
- Feature scaling
- Normalization
- Standardization
- Feature selection
- Dimensionality reduction
- Creating dummy variables
- Dropping identifiers because they are poor predictors
- Dropping PII because it should not be used for modelling
- Date feature engineering such as creating year/month/day
- Model training

Those responsibilities belong to the later ML stage.

You MAY recommend genuine data-quality operations such as:

- Handling missing feature values
- Removing exact duplicate rows
- Fixing safe data-type issues
- Standardizing obvious whitespace inconsistencies
- Parsing dates consistently
- Fixing clearly supported representation problems

Do not invent columns, statistics, outliers, duplicates,
or inconsistencies that are not supported by the profile.

Dataset profile:

{profile_json}

Explain:

1. Important DATA QUALITY issues detected.
2. Cleaning actions required.
3. Why each cleaning action is appropriate.
4. Which issues should be left for the ML preprocessing
   stage instead of modifying cleaned.csv.

Do not generate or train a machine-learning model.
"""

    # Generate a readable analysis explanation using the text model
    response = model.invoke(prompt)

    return response.content


def run_cleaning_agent(
    file_path: str
) -> dict:

    # 1. Create a unique job workspace and copy the dataset
    workspace = create_workspace.func(file_path)

    input_path = workspace / "input.csv"

    # 2. Ask model to generate a cleaning plan and code
    decision = generate_cleaning_code(
        str(input_path)
    )

    # 3. Check for security/safety issues and syntax validity in generated code
    validation = validate_generated_code.func(
        decision.generated_code
    )

    if not validation["valid"]:
        return {
            "success": False,
            "stage": "validation",
            "reason": validation["reason"]
        }

    # 4. Safely execute the validated code inside the workspace
    execution = execute_python_code.func(
        code=decision.generated_code,
        working_directory=str(workspace)
    )

    if not execution["success"]:
        return {
            "success": False,
            "stage": "execution",
            "workspace": str(workspace),
            "summary": decision.summary,
            "cleaning_plan": decision.cleaning_plan,
            "generated_code": decision.generated_code,
            "execution": execution
        }

    cleaned_path = workspace / "cleaned.csv"

    if not cleaned_path.exists():
        return {
            "success": False,
            "stage": "verification",
            "reason": "Generated code executed successfully but cleaned.csv was not created.",
            "workspace": str(workspace),
            "summary": decision.summary,
            "cleaning_plan": decision.cleaning_plan,
            "generated_code": decision.generated_code,
            "execution": execution
        }

    # 5. Profile the final cleaned dataset for validation
    cleaned_profile = inspect_dataset.func(str(cleaned_path))

    return {
        "success": True,
        "stage": "completed",
        "workspace": str(workspace),
        "summary": decision.summary,
        "cleaning_plan": decision.cleaning_plan,
        "generated_code": decision.generated_code,
        "execution": execution,
        "cleaned_profile": cleaned_profile
    }




# Configure model to return a structured RepairResult schema
repair_model = model.with_structured_output(
    RepairResult
)

def repair_cleaning_code(
    profile: dict,
    target_column: str,
    previous_code: str,
    error_message: str
) -> RepairResult:
    """Ask the LLM to fix a failed data-cleaning script given the error and original profile."""

    prompt = f"""
You are repairing DATA CLEANING Python code generated by
an autonomous data science agent.

The previous cleaning script failed during execution.

Your job is ONLY to fix the actual execution failure while
preserving the original cleaning objective.

Do NOT expand the scope of the cleaning operation.

TARGET COLUMN:
{target_column}

DATASET PROFILE:
{json.dumps(profile, indent=2, default=str)}

FAILED CODE:
{previous_code}

ACTUAL EXECUTION ERROR:
{error_message}

=========================================================
STRICT CLEANING / ML BOUNDARY
=========================================================

The repaired script must perform DATA CLEANING ONLY.

DO NOT:

- One-hot encode columns.
- Label encode columns.
- Create dummy variables.
- Scale or normalize features.
- Standardize features.
- Perform feature selection.
- Perform dimensionality reduction.
- Create model-specific features.
- Split data into train/test sets.
- Train ML models.
- Drop identifiers merely because they are unsuitable
  model features.
- Drop PII merely because it should not be used for ML.
- Transform the dataset into a model-ready feature matrix.
- Automatically expand dates into year/month/day features.

cleaned.csv must remain human-readable and structurally
close to input.csv.

=========================================================
TARGET RULES
=========================================================

- The target column is "{target_column}".
- Never impute missing target values.
- Never generate synthetic target values.
- Never derive features from the target.
- Never drop the target column itself.
- If the target contains missing values, drop only rows
  where the target value is missing.
- Do not use target-derived information to clean features.
- Keep the target human-readable.

=========================================================
REPAIR RULES
=========================================================

- Fix the ACTUAL cause of the supplied error.
- Make the smallest reasonable correction.
- Do not introduce unrelated transformations.
- Do not invent columns.
- Do not invent dataset statistics.
- Read only "input.csv".
- Save the result as "cleaned.csv".
- Use pandas.
- Preserve original columns whenever possible.
- Date columns may be parsed consistently but should not
  automatically be expanded into ML features.
- Do not access the network.
- Do not use subprocesses.
- Return a complete standalone Python script.
- Print a short cleaning summary after execution.
"""
    return repair_model.invoke(prompt)