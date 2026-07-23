from agent.data_agent import run_cleaning_agent

# Run the cleaning agent on the sample CSV dataset
result = run_cleaning_agent(
    "tests/sample_data.csv"
)

# Print agent outcome details
print("\n=== AGENT SUMMARY ===")
print(result["summary"])

# Print step-by-step cleaning plan
print("\n=== CLEANING PLAN ===")
for step in result["cleaning_plan"]:
    print("-", step)

# Print the generated Python preprocessing code
print("\n=== GENERATED CODE ===")
print(result["generated_code"])

# Print execution results (success/exit code/stdout/stderr)
print("\n=== EXECUTION ===")
print(result["execution"])

# Print verification status of the cleaned dataset
print("\n=== VERIFICATION ===")

if result.get("cleaned_profile"):
    cleaned_profile = result["cleaned_profile"]

    print("Cleaned dataset successfully verified.")

    # Output details of the cleaned dataset
    print("\nShape:")
    print(cleaned_profile["shape"])

    print("\nColumns:")
    for column in cleaned_profile["columns"]:
        print(
            f"{column['name']} -> "
            f"type={column['dtype']}, "
            f"missing={column['missing']}"
        )

else:
    print("Verification failed.")
    print(result.get("reason"))
