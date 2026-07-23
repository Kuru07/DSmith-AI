# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field

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