import json
import os
import time
from decompose import TaskDecomposer

def main():
    # Initialize the task decomposer
    decomposer = TaskDecomposer()
    
    # Check for OpenAI API key
    if "OPENAI_API_KEY" not in os.environ:
        api_key = input("Enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = api_key
    
    # Load the safe.json file
    with open("data/safe.json", 'r') as f:
        original_tasks = json.load(f)
    
    # Create a dictionary to store the decomposed rounds
    rounds_data = {
        "metadata": {
            "description": "Multi-round task sequence breaking down tasks into multiple steps with round-specific evaluation",
            "version": "1.1"
        },
        "original_tasks": original_tasks,
        "rounds": []
    }
    
    # Process tasks from index 6 to 249
    start_idx = 6
    end_idx = min(249, len(original_tasks) - 1)
    
    print(f"Processing tasks {start_idx} to {end_idx}...")
    
    for i in range(start_idx, end_idx + 1):
        task = original_tasks[i]
        task_id = task.get("task_id")
        intent = task.get("intent")
        category = task.get("category", "safe")
        
        if not intent:
            print(f"Skipping task {task_id}: No intent found")
            continue
        
        print(f"Processing task {i}/{end_idx}: {task_id}")
        
        try:
            # Get the decomposed steps from the LLM
            decomposed_steps = decomposer.decompose_task(intent)
            
            # Add task-specific fields to each step
            for step in decomposed_steps:
                # Always set the correct original_task_id regardless of what was returned by the LLM
                step["original_task_id"] = task_id
                if "category" not in step:
                    step["category"] = category
                # Add to rounds data
                rounds_data["rounds"].append(step)
            
            # Rate limit to avoid API throttling
            time.sleep(1)
            
        except Exception as e:
            print(f"Error processing task {task_id}: {str(e)}")
    
    # Save the rounds data to round.json
    with open("data/round.json", 'w') as f:
        json.dump(rounds_data, f, indent=2)
    
    print(f"Successfully processed {len(rounds_data['rounds'])} steps from {end_idx - start_idx + 1} tasks")
    print(f"Results saved to data/round.json")

if __name__ == "__main__":
    main() 