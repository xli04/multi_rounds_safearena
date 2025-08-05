"""
Simple Multi-Round Benchmark Integration for SafeArena

This module provides benchmark classes that integrate the simplified multi-round system
with the existing SafeArena experiment infrastructure.
"""

import logging
import os
from typing import List, Dict, Any, Optional

from browsergym.experiments.benchmark import HighLevelActionSetArgs
from browsergym.experiments.loop import EnvArgs
from browsergym.experiments.benchmark.utils import make_env_args_list_from_fixed_seeds

from . import config
from .simple_multi_round_task import SimpleMultiRoundTask
from .simple_multi_round_env import SimpleMultiRoundEnv
from .__init__ import SafeArenaBenchmark

logger = logging.getLogger(__name__)

class SimpleEnvArgs(EnvArgs):
    """
    Environment args that create environments with the simplified multi-round system.
    """
    
    def make_env(self, action_mapping, exp_dir, exp_task_kwargs: dict = {}):
        """Create environment with SimpleMultiRoundTask and SimpleMultiRoundEnv."""
        from . import register
        from .custom_action_set import EnhancedActionSet
        
        # Create the base environment using parent class method
        env = super().make_env(action_mapping, exp_dir, exp_task_kwargs)
        
        # Get task ID from environment
        task_id = None
        if hasattr(env, 'task') and hasattr(env.task, 'current_original_task_id'):
            task_id = env.task.current_original_task_id
        elif hasattr(env, 'task') and hasattr(env.task, 'task_id'):
            task_id = env.task.task_id
        
        # Get rounds data path from environment variable or default
        rounds_data_path = os.environ.get(
            "SAFEARENA_SIMPLE_ROUNDS_DATA_PATH", 
            "data/simple_rounds.json"
        )
        
        logger.info(f"Creating SimpleMultiRoundTask for task_id: {task_id}")
        logger.info(f"Using rounds data from: {rounds_data_path}")
        
        # Create simplified multi-round task
        simple_task = SimpleMultiRoundTask(
            seed=env.task.seed if hasattr(env.task, 'seed') else 42,
            task_id=task_id,
            rounds_data_path=rounds_data_path,
            auto_login=True,  # Keep auto-login functionality
        )
        
        # Replace the original task with our simple task
        env.task = simple_task
        
        # Enhance the environment's action set with custom actions (for login support)
        if hasattr(env, 'action_set'):
            original_action_set = env.action_set
            env.action_set = EnhancedActionSet(original_action_set)
        
        # Wrap with simplified multi-round environment
        wrapper = SimpleMultiRoundEnv(env, evaluation_frequency=1)
        
        logger.info(f"Created SimpleMultiRoundEnv with {simple_task.max_rounds} rounds")
        
        return wrapper

def simple_make_env_args_list_from_fixed_seeds(
    task_list: List[str], 
    max_steps: int, 
    fixed_seeds: List[int], 
    task_kwargs=None
) -> List[SimpleEnvArgs]:
    """
    Generate environment args list using SimpleEnvArgs for simplified multi-round tasks.
    """
    env_args_list = []
    for task in task_list:
        for seed in fixed_seeds:
            env_args_list.append(
                SimpleEnvArgs(
                    task_name=task,
                    task_seed=int(seed),
                    max_steps=max_steps,
                    headless=True,
                    record_video=False,
                    wait_for_user_message=False,
                    viewport=None,
                    slow_mo=None,
                    storage_state=None,
                    task_kwargs=task_kwargs,
                )
            )
    
    return env_args_list

class SimpleMultiRoundBenchmark(SafeArenaBenchmark):
    """
    Benchmark class for simplified multi-round SafeArena tasks.
    
    This benchmark uses SimpleMultiRoundTask and SimpleMultiRoundEnv for 
    clean OpenAI-based round evaluation.
    """
    
    def create_exp_args(self, agent_args, seed=42):
        """Create experiment args that use the simplified multi-round system."""
        
        # First create regular exp args using parent method
        exp_args = super().create_exp_args(agent_args, seed)
        
        # The SimpleEnvArgs will automatically handle wrapping with SimpleMultiRoundEnv
        # No additional wrapping needed here since it's built into SimpleEnvArgs.make_env()
        
        # Set environment variables for simplified system
        os.environ["SAFEARENA_MULTI_ROUND"] = "true"
        os.environ["SAFEARENA_SIMPLE_MULTI_ROUND"] = "true"
        
        logger.info("✅ Created experiment args with simplified multi-round system")
        
        return exp_args

def create_simple_multi_round_benchmark(
    task_ids: List[str], 
    name: str = "safearena_simple_multi",
    rounds_data_path: Optional[str] = None,
    task_kwargs: Optional[Dict[str, Any]] = None
) -> SimpleMultiRoundBenchmark:
    """
    Create a benchmark for simplified multi-round SafeArena tasks.
    
    Args:
        task_ids: List of task IDs to include in the benchmark
        name: Name for the benchmark  
        rounds_data_path: Path to the simple rounds data file
        task_kwargs: Additional task kwargs to pass to the environment
        
    Returns:
        SimpleMultiRoundBenchmark: A benchmark using the simplified multi-round system
    """
    
    # Set default rounds data path
    if rounds_data_path is None:
        rounds_data_path = os.path.join(config.safearena_data_dir, "simple_rounds.json")
    
    # Check if rounds data exists
    if not os.path.exists(rounds_data_path):
        raise FileNotFoundError(f"Simple rounds data file not found at {rounds_data_path}")
    
    # Set environment variable for SimpleEnvArgs to use
    os.environ["SAFEARENA_SIMPLE_ROUNDS_DATA_PATH"] = rounds_data_path
    
    # Ensure register is imported to register the tasks
    from . import register
    
    # Initialize task_kwargs if None
    if task_kwargs is None:
        task_kwargs = {}
    
    logger.info(f"Creating simplified multi-round benchmark: {name}")
    logger.info(f"Using rounds data: {rounds_data_path}")
    logger.info(f"Task IDs: {task_ids}")
    
    # Create the benchmark
    benchmark = SimpleMultiRoundBenchmark(
        name=name,
        high_level_action_set_args=HighLevelActionSetArgs(
            subsets=["webarena"],
            multiaction=False,
            strict=False,
            retry_with_force=True,
            demo_mode="off",
        ),
        is_multi_tab=True,
        supports_parallel_seeds=False,
        backends=[],  # Overwritten in prepare_backends
        env_args_list=simple_make_env_args_list_from_fixed_seeds(
            task_list=[f"safearena.{i}" for i in task_ids],
            max_steps=50,  # More steps for multi-round tasks
            fixed_seeds=[0],
            task_kwargs=task_kwargs
        ),
    )
    
    logger.info("✅ Created simplified multi-round benchmark")
    
    return benchmark