import json

from tools.dataset_tools import inspect_dataset
from tools.code_validator import validate_generated_code

# Unsafe code string to test validator security checks
code = """
import subprocess

subprocess.run(["something"])
"""

# Run the validator on the unsafe code snippet
valid_output=validate_generated_code.func(
    code
)

# Output validation result
print(valid_output)