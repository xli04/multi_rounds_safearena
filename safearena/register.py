import logging
import os

from browsergym.core.registration import register_task

from . import config, task
from .multi_round_task import MultiRoundSafeArenaTask

logger = logging.getLogger(__name__)
ALL_SAFEARENA_TASK_IDS = []
ALL_MULTI_ROUND_TASK_IDS = []

# Check if multi-round data exists
# First check if a custom path is specified in environment variables
custom_path = os.getenv("SAFEARENA_MULTI_ROUND_DATA_PATH")
if custom_path:
    multi_round_data_path = custom_path
    print(f"Using custom multi-round data path from environment: {multi_round_data_path}")
else:
    multi_round_data_path = os.path.join(config.safearena_data_dir, "sample_multi_round.json")
    print(f"Using default multi-round data path: {multi_round_data_path}")

multi_round_mode = os.getenv("SAFEARENA_MULTI_ROUND", "").lower() in ['true', '1']

# Register all SafeArena tasks
for task_id in config.TASK_IDS:
    gym_id = f"safearena.{task_id}"
    
    # Determine which task class to use based on mode
    if multi_round_mode and os.path.exists(multi_round_data_path):
        task_class = MultiRoundSafeArenaTask
        task_kwargs = {
            "task_id": task_id,
            "multi_round_data_path": multi_round_data_path
        }
        ALL_MULTI_ROUND_TASK_IDS.append(gym_id)
        logger.info(f"Registered multi-round task {gym_id} with data from {multi_round_data_path}")
        
        # Verify multi-round data actually contains this task
        import json
        try:
            with open(multi_round_data_path, 'r') as f:
                multi_data = json.load(f)
                rounds = [r for r in multi_data.get("rounds", []) if r.get("original_task_id") == task_id]
                logger.info(f"Found {len(rounds)} rounds for task {task_id} in multi-round data")
        except Exception as e:
            logger.error(f"Error verifying multi-round data for {task_id}: {e}")
    else:
        task_class = task.GenericSafeArenaTask
        task_kwargs = {"task_id": task_id}
        ALL_SAFEARENA_TASK_IDS.append(gym_id)
        logger.debug(f"Registered single-round task {gym_id}")
    
    register_task(
        gym_id,
        task_class,
        task_kwargs=task_kwargs,
    )

# Log registration summary
if multi_round_mode and os.path.exists(multi_round_data_path):
    logger.info(f"Registered a total of {len(ALL_MULTI_ROUND_TASK_IDS)} multi-round SafeArena tasks.")
else:
    logger.info(f"Registered a total of {len(ALL_SAFEARENA_TASK_IDS)} SafeArena tasks. The first 5 tasks are: {ALL_SAFEARENA_TASK_IDS[:5] if len(ALL_SAFEARENA_TASK_IDS) >= 5 else ALL_SAFEARENA_TASK_IDS}")