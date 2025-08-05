import json

# Load your JSON file
with open("genericagent-claude-37-sonnet-on-safearena-harm-multi_detailed_results.json", "r") as f:
    data = json.load(f)

# Make sure it's a list of entries
if not isinstance(data, list):
    data = [data]

total_harm_rounds = 0
total_jailbreaks = 0
total_defends = 0

rounds = data.get("rounds", [])
# Loop through each entry
for entry in data:
    total_harm_rounds += len(entry.get("round_breakdown", {}).get("harmful_rounds", []))
    metrics = entry.get("task_wide_metrics", {})
    harm = metrics.get("defend_count", 0) + metrics.get("jailbreak_count", 0)
    total_jailbreaks += metrics.get("jailbreak_count", 0)
    total_defends += metrics.get("defend_count", 0)

# Compute the requested metrics
jailbreak_rate = total_jailbreaks / total_harm_rounds if total_harm_rounds > 0 else 0
defend_rate = total_defends / total_harm_rounds if total_harm_rounds > 0 else 0
unreachable_rate = (total_harm_rounds - total_jailbreaks - total_defends) / total_harm_rounds if total_harm_rounds > 0 else 0

# Print results
print(f"Total Harm Rounds: {total_harm_rounds}")
print(f"Total jailbreak Rounds: {total_jailbreaks}")
print(f"Total Jailbreak Rate: {jailbreak_rate:.4f}")
print(f"Total Defend Rate: {defend_rate:.4f}")
print(f"Total Unreachable Rate: {unreachable_rate:.4f}")
