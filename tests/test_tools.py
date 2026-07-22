import json

from tools.dataset_tools import inspect_dataset


print("\n==============================")
print("ORIGINAL DATASET")
print("==============================")

original = inspect_dataset(
    "workspace/test_run/input.csv"
)

print(
    json.dumps(
        original,
        indent=2,
        default=str
    )
)


print("\n==============================")
print("CLEANED DATASET")
print("==============================")

cleaned = inspect_dataset(
    "workspace/test_run/cleaned.csv"
)

print(
    json.dumps(
        cleaned,
        indent=2,
        default=str
    )
)