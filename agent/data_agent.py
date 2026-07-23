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

# Load environmental variables from .env file
load_dotenv()

# Initialize the Gemini model for structured and text output
model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0
)

# Configure model to return a structured CleaningResult schema
structured_model = model.with_structured_output(
    CleaningResult
)


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
You are an autonomous data science agent.

Analyze the following dataset profile.

Your current task is ONLY to determine what data
cleaning and preprocessing should be performed.

Do not invent columns or statistics.
Use only information present in the profile.

Dataset profile:

{profile_json}

Explain:
1. Important issues detected
2. Cleaning actions required
3. Why each action is appropriate

Do not train a machine learning model yet.
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