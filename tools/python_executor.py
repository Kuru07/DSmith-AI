import subprocess
import sys
import tempfile
from pathlib import Path

def execute_python_code(
    code:str,
    working_directory: str,
    timeout_seconds: int = 60
) -> dict :
    """Safely execute Python code locally in a separate subprocess within the specified working directory, enforcing a configurable timeout, and return the execution results (success, exit code, stdout, stderr)."""
    # Create path object for working directory
    workspace=Path(working_directory)
    # Ensure working directory exists
    workspace.mkdir(parents=True,exist_ok=True)

    script_path = None

    try:
        # Write the python code block to a temporary file inside the workspace
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            dir=workspace,
            delete=False,
            encoding="utf-8"
        ) as temp_file:

            temp_file.write(code)
            script_path=Path(temp_file.name)

        # Run the temporary Python file in a separate subprocess with timeout
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout_seconds
        )

        # Return stdout, stderr, and exit status
        return {
            "success": result.returncode == 0,
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
        
    except subprocess.TimeoutExpired:
        # Return error if subprocess execution timed out
        return {
            "success": False,
            "error": "Execution timed out."
        }

    except Exception as exc:
        # Return standard error details if any exceptions occurred
        return {
            "success": False,
            "error": str(exc)
        }

    finally:
        # Always clean up and delete the temporary script file
        if script_path and script_path.exists():
            script_path.unlink()