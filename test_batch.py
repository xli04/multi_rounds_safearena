import json
import os
from decompose import TaskDecomposer

def test_batch():
    """Test decomposition on a small batch of tasks"""
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
        "original_tasks": original_tasks[6:9],  # Just store a few tasks
        "rounds": []
    }
    
    # Process a small batch of tasks (indices 6-8)
    for i in range(6, 9):
        task = original_tasks[i]
        task_id = task.get("task_id")
        intent = task.get("intent")
        category = task.get("category", "safe")
        
        print(f"Processing test task: {task_id}")
        print(f"Intent: {intent}")
        
        # Get the decomposed steps
        decomposed_steps = decomposer.decompose_task(intent)
        
        # Add task-specific fields to each step and ALWAYS override the original_task_id
        for step in decomposed_steps:
            # Always set the correct original_task_id regardless of what was returned
            step["original_task_id"] = task_id
            if "category" not in step:
                step["category"] = category
            rounds_data["rounds"].append(step)
        
        print(f"Decomposed into {len(decomposed_steps)} steps")
    
    # Save the test batch to a file
    with open("data/test_batch.json", 'w') as f:
        json.dump(rounds_data, f, indent=2)
    
    print(f"Test batch processing complete. Results saved to data/test_batch.json")
    print(f"Total steps: {len(rounds_data['rounds'])}")

if __name__ == "__main__":
    test_batch() 