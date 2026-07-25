from agent.graph import run_autonomous_cleaning


result = run_autonomous_cleaning(
    file_path="tests/sample_data.csv",
    target_column="Remote_Work"
)


print("\n========== END-TO-END RESULT ==========")

print("Success:", result.get("success"))
print("Target:", result.get("target_column"))
print("Problem Type:", result.get("problem_type"))
print("Selected Models:", result.get("selected_models"))
print("Cleaning Retries:", result.get("cleaning_retry_count"))
print("Training Retries:", result.get("training_retry_count"))
print("Best Model:", result.get("best_model"))
print("Metrics:", result.get("metrics"))
print("Error:", result.get("error_message"))
print("Workspace:", result.get("workspace"))