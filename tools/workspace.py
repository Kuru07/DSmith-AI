import shutil
from pathlib import Path
from uuid import uuid4

def create_workspace(
    source_file: str
) -> Path:
    """Create a unique temporary workspace directory for a job and copy the source dataset file into it as 'input.csv'."""

    # Generate a unique job ID using uuid4
    job_id = str(uuid4())

    # Create the workspace directory path
    workspace = Path("workspace") / job_id

    # Create directory and parent directories if they don't exist
    workspace.mkdir(
        parents=True,
        exist_ok=True
    )

    # Copy the input dataset file to the workspace as input.csv
    shutil.copy(
        source_file,
        workspace / "input.csv"
    )

    return workspace