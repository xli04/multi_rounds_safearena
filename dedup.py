import json

# Read the file
with open('data/safe_multi_round.json', 'r') as f:
    data = json.load(f)

# Extract metadata and original_tasks
metadata = data.get('metadata', {})
original_tasks = data.get('original_tasks', [])

# Get unique rounds by task_id + step combination
seen = set()
unique_rounds = []

for item in data['rounds']:
    task_id = item.get('original_task_id', '')
    description = item.get('description', '')
    identifier = f"{task_id}_{description}"
    
    if identifier not in seen:
        seen.add(identifier)
        unique_rounds.append(item)
    else:
        print(f"Removing duplicate: {identifier}")

# Create new data structure
clean_data = {
    'metadata': metadata,
    'original_tasks': original_tasks,
    'rounds': unique_rounds
}

# Save the cleaned data
with open('data/safe_multi_round_clean.json', 'w') as f:
    json.dump(clean_data, f, indent=2)

print(f"Original rounds count: {len(data['rounds'])}")
print(f"Cleaned rounds count: {len(unique_rounds)}")
print(f"Removed {len(data['rounds']) - len(unique_rounds)} duplicates")

# Verify tasks coverage
task_ids = set()
for item in unique_rounds:
    task_id = item.get('original_task_id', '')
    if task_id:
        task_ids.add(task_id)

task_ids = sorted(list(task_ids))
print(f"Tasks covered: {len(task_ids)}")
if task_ids:
    print(f"First task: {task_ids[0]}, Last task: {task_ids[-1]}")
