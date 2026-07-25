from agent.graph import run_autonomous_cleaning


result = run_autonomous_cleaning(
    "tests/sample_data.csv"
)


print("\n=== FINAL RESULT ===")
print("Success:", result.get("success"))
print("Retries:", result.get("retry_count"))
print("Error:", result.get("error_message"))

print("\n=== CLEANING PLAN ===")
for step in result.get("cleaning_plan", []):
    print("-", step)

print("\n=== EXECUTION ===")
print(result.get("execution_result"))

print("\n=== CLEANED PROFILE ===")
print(result.get("cleaned_profile"))