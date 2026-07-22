from asyncio import subprocess
import tempfile
import subprocess
import sys
import tempfile
from pathlib import Path

def execute_python_code(
    code:str,
    working_directory: str,
    timeout_seconds: int = 60
) -> dict :
    workspace=Path(working_directory)
    workspace.mkdir(parents=True,exist_ok=True)

    script_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            dir=workspace,
            delete=False,
            encoding="utf-8"
        ) as temp_file:

            temp_file.write(code)
            script_path=Path(temp_file.name)

        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout_seconds
        )

        return {
            "success": result.returncode == 0,
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
        
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Execution timed out."
        }

    except Exception as exc:
        return {
            "success": False,
            "error": str(exc)
        }

    finally:
        if script_path and script_path.exists():
            script_path.unlink()