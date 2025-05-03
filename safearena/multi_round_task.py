import json
import logging
import os
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple

import playwright.sync_api
from browsergym.webarena.task import GenericWebArenaTask

from .task import GenericSafeArenaTask
from .config import all_configs_raw_dict

logger = logging.getLogger(__name__)

class MultiRoundSafeArenaTask(GenericSafeArenaTask):
    """
    A task class that supports multi-round execution for SafeArena.
    
    Each original task is broken down into multiple rounds, where each round
    has its own intent and description. The state from one round is preserved
    for the next round.
    """
    
    def __init__(
        self,
        seed: int,
        task_id: Optional[str] = None,
        intent_template_id: Optional[int] = None,
        with_na_hint: bool = False,
        with_homepage_hint: bool = False,
        multi_round_data_path: str = "data/sample_multi_round.json",
    ) -> None:
        # Initialize the parent class with the original task_id
        super().__init__(
            seed=seed,
            task_id=task_id,
            intent_template_id=intent_template_id,
            with_na_hint=with_na_hint,
            with_homepage_hint=with_homepage_hint,
        )
        
        # Load multi-round data
        self.multi_round_data = self._load_multi_round_data(multi_round_data_path)
        
        # Initialize round state
        self.current_round_index = 0
        self.max_rounds = 0
        self.current_original_task_id = None
        self.rounds_for_current_task = []
        
        # Set current task rounds if a task_id is provided
        if task_id is not None:
            self._setup_rounds_for_task(task_id)
    
    def _load_multi_round_data(self, data_path: str) -> Dict[str, Any]:
        """Load the multi-round data from the specified path."""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Multi-round data file {data_path} not found.")
        
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        return data
    
    def _setup_rounds_for_task(self, task_id: str) -> None:
        """Set up the rounds for the specified task."""
        import os
        import json
        
        # First try to reload the data directly from the source to ensure we have the latest
        try:
            if os.path.exists(self.multi_round_data_path):
                with open(self.multi_round_data_path, 'r') as f:
                    self.multi_round_data = json.load(f)
                    logger.info(f"Reloaded multi-round data from {self.multi_round_data_path}")
        except Exception as e:
            logger.error(f"Error reloading multi-round data: {e}")
        
        # Find rounds for the current task
        self.rounds_for_current_task = [
            round_data for round_data in self.multi_round_data.get("rounds", [])
            if round_data.get("original_task_id") == task_id
        ]
        
        if not self.rounds_for_current_task:
            # Try alternate task ID formats (e.g., with/without prefix)
            alternate_task_ids = []
            
            # Try without "safearena." prefix
            if task_id.startswith("safearena."):
                alternate_task_ids.append(task_id[len("safearena."):])
            # Try with "safearena." prefix
            else:
                alternate_task_ids.append(f"safearena.{task_id}")
                
            # Try with just the category and ID number
            parts = task_id.split(".")
            if len(parts) >= 2:
                # Extract just "category.id" if there are multiple parts
                alternate_task_ids.append(f"{parts[-2]}.{parts[-1]}")
            
            logger.info(f"Trying alternate task IDs: {alternate_task_ids}")
            
            # Check each alternate ID
            for alt_id in alternate_task_ids:
                alt_rounds = [
                    round_data for round_data in self.multi_round_data.get("rounds", [])
                    if round_data.get("original_task_id") == alt_id
                ]
                if alt_rounds:
                    logger.info(f"Found rounds using alternate task ID: {alt_id}")
                    self.rounds_for_current_task = alt_rounds
                    break
        
        # If still no rounds found, create a single-round fallback
        if not self.rounds_for_current_task:
            logger.info(f"No multi-round data found for task {task_id} or any alternate IDs. Creating fallback.")
            
            # Create a single round using the original task's intent
            original_intent = self.intent
            self.rounds_for_current_task = [{
                "original_task_id": task_id,
                "round_index": 0,
                "intent": original_intent,
                "description": "Complete the task",
                # No round-specific evaluation criteria
            }]
        
        self.max_rounds = len(self.rounds_for_current_task)
        self.current_original_task_id = task_id
        
        # Detailed logging of round information
        logger.info(f"🔢🔢🔢 Task {task_id} set up with {self.max_rounds} rounds")
        
        # Store round data in environment variables for resilience
        for i, round_data in enumerate(self.rounds_for_current_task):
            round_intent = round_data.get("intent", "No intent specified")
            round_desc = round_data.get("description", "No description")
            logger.info(f"  Round {i+1}/{self.max_rounds}: {round_desc} - {round_intent[:50]}...")
            
            # Store in environment for cross-component access
            os.environ[f"SAFEARENA_ROUND_{i}_INTENT"] = round_intent
            os.environ[f"SAFEARENA_ROUND_{i}_DESCRIPTION"] = round_desc
            
        # Explicitly set environment variables so other components can detect multi-round setup
        os.environ["SAFEARENA_MULTI_ROUND_CONFIGURED"] = "true"
        os.environ["SAFEARENA_MULTI_ROUND_COUNT"] = str(self.max_rounds)
        os.environ["SAFEARENA_CURRENT_TASK_ID"] = task_id
    
    def setup(self, page: playwright.sync_api.Page) -> Tuple[str, Dict]:
        """
        Set up the task for the current round.
        
        Instead of using the original task intent, use the intent from the current round.
        """
        from .helper.evaluators import safearena_eval_router
        import os
        
        # Use parent class setup but replace goal with the current round intent
        goal, d = super().setup(page)
        
        # Override the goal with the current round intent
        if self.current_round_index < self.max_rounds:
            current_round = self.rounds_for_current_task[self.current_round_index]
            
            # Try different possible keys for intent
            if "intent" in current_round:
                goal = current_round.get("intent", goal)
                logger.info(f"Using intent key from round data: {goal}")
            elif "round_intent" in current_round:
                goal = current_round.get("round_intent", goal)
                logger.info(f"Using round_intent key from round data: {goal}")
            
            # Add round description to the goal for context
            description = ""
            if "description" in current_round:
                description = current_round.get("description", "")
            elif "round_description" in current_round:
                description = current_round.get("round_description", "")
                
            if description:
                goal = f"{description}: {goal}"
            
            # Store the goal for this round in the task (so it can be accessed reliably)
            self.intent = goal
            
            # Also store in environment variable for backup access
            os.environ["SAFEARENA_CURRENT_INTENT"] = goal
            os.environ[f"SAFEARENA_ROUND_{self.current_round_index}_INTENT"] = goal
                
            logger.info(f"Setting up round {self.current_round_index + 1}/{self.max_rounds} with goal: {goal}")
        
        # Set up evaluator for the current round
        self.evaluator = safearena_eval_router(self.config_file)
        
        return goal, d
    
    def next_round(self) -> None:
        """
        Advance to the next round
        
        This method is kept for backward compatibility but delegates to advance_to_next_round
        """
        logger.info("next_round() called - using advance_to_next_round() instead")
        self.advance_to_next_round()
    
    def reset(self) -> None:
        """
        Reset the task to the first round.
        """
        self.current_round_index = 0
        current_round = self.get_current_round_info()
        self.intent = current_round.get("round_intent", self.original_intent)
        logger.info(f"Task {self.current_original_task_id} reset to round 1/{self.max_rounds}")
        
        # Set up the first round
        logger.info(f"Setting up round 1/{self.max_rounds} with goal: {self.intent}")
    
    def _capture_round_state(self) -> dict:
        """
        Capture any state that needs to be preserved between rounds.
        
        This could include form data, session information, or other
        task-specific state needed for continuity.
        
        Returns:
            dict: State information to preserve
        """
        # By default, preserve the config_file which contains task-specific information
        state = {
            'config_file': self.config_file,
            'current_original_task_id': self.current_original_task_id,
        }
        
        # Add any task-specific state here
        return state
    
    def _apply_round_state(self, state: dict) -> None:
        """
        Apply preserved state from previous round.
        
        Args:
            state: The state captured from the previous round
        """
        # Ensure we maintain the same config_file across rounds
        if 'config_file' in state and state['config_file'] is not None:
            # Only update if we don't already have a config_file
            if self.config_file is None:
                self.config_file = state['config_file']
                logger.info(f"Applied previous round's config_file")
        
        # Add any additional state application logic here
    
    def is_final_round(self) -> bool:
        """Check if this is the final round for the current task."""
        return self.current_round_index == self.max_rounds - 1
    
    def get_current_round_info(self) -> Dict[str, Any]:
        """Get information about the current round."""
        if self.current_round_index < self.max_rounds:
            # Get the original round data
            round_data = self.rounds_for_current_task[self.current_round_index]
            
            # Create a more consistent format with both "intent" and "round_intent" keys
            result = dict(round_data)
            
            # Always include intent under both keys to ensure compatibility
            if "intent" in round_data and "round_intent" not in round_data:
                result["round_intent"] = round_data["intent"]
            elif "round_intent" in round_data and "intent" not in round_data:
                result["intent"] = round_data["round_intent"]
                
            # Do the same for description
            if "description" in round_data and "round_description" not in round_data:
                result["round_description"] = round_data["description"]
            elif "round_description" in round_data and "description" not in round_data:
                result["description"] = round_data["round_description"]
                
            logger.info(f"🔍🔍🔍 Current round {self.current_round_index + 1} info: {result}")
            return result
            
        return {}
    
    def evaluate(self, trajectory) -> Dict[str, Any]:
        """
        Custom evaluation for the current round.
        
        This overrides the default evaluator to use round-specific criteria.
        """
        # First try using the round-specific evaluation if available
        current_round = self.get_current_round_info()
        
        # Log the trajectory length for debugging
        logger.info(f"Evaluating round {self.current_round_index + 1}/{self.max_rounds} with trajectory of length {len(trajectory)}")
        
        # Get the last state for URL checking
        last_state = trajectory[-1] if trajectory else {}
        current_url = last_state.get('url', '')
        
        # Basic result structure
        result = {
            "success": False,
            "reason": "Not completed yet",
            "current_url": current_url,
            "steps_taken": len(trajectory),
            "round_index": self.current_round_index,
            "round_of_total": f"{self.current_round_index + 1}/{self.max_rounds}"
        }
        
        # If this is a multi-round task and has round-specific evaluation criteria
        if current_round and "round_eval" in current_round:
            # Use the round-specific evaluation criteria
            eval_criteria = current_round["round_eval"]
            round_intent = current_round.get("round_intent", "")
            
            # Add round-specific information
            result["round_intent"] = round_intent
            
            # Check URL-based criteria if specified
            if "target_urls" in eval_criteria:
                target_urls = eval_criteria["target_urls"]
                for target in target_urls:
                    if target.lower() in current_url.lower():
                        result["success"] = True
                        result["reason"] = f"Found target URL pattern: {target}"
                        logger.info(f"Round {self.current_round_index + 1} evaluation: Success - {result['reason']}")
                        return result
            
            # Check for subtasks if specified
            if "subtasks" in eval_criteria:
                subtasks = []
                any_completed = False
                
                for i, subtask in enumerate(eval_criteria["subtasks"]):
                    subtask_desc = subtask.get("description", f"Subtask {i+1}")
                    completed = False
                    
                    # Check for completion criteria based on URL
                    if "url_pattern" in subtask and subtask["url_pattern"] in current_url:
                        completed = True
                        any_completed = True
                    
                    # Check for completion criteria based on element presence
                    if "element_present" in subtask:
                        # This would require checking the DOM for a specific element
                        # We'll simplify here, but real implementation would check DOM
                        pass
                    
                    subtasks.append({
                        "description": subtask_desc,
                        "completed": completed
                    })
                
                result["subtasks"] = subtasks
                
                # If any subtask is complete, consider partial progress
                if any_completed:
                    result["partial_progress"] = True
        
        # Fallback to standard evaluation if no round-specific criteria
        standard_eval = self.evaluator(self.intent, trajectory)
        
        # Combine round-specific and standard evaluation
        if standard_eval.get("success", False):
            result["success"] = True
            result["reason"] = standard_eval.get("reason", "Standard evaluation succeeded")
        
        # Add any additional information from standard evaluation
        for key, value in standard_eval.items():
            if key not in result:
                result[key] = value
        
        # Log the evaluation result
        logger.info(f"Round {self.current_round_index + 1} evaluation result: {result.get('success')}")
        
        return result
    
    def _evaluate_round_content(self, trajectory, round_eval: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the trajectory based on round-specific criteria.
        
        Args:
            trajectory: The trajectory to evaluate
            round_eval: Round-specific evaluation criteria
            
        Returns:
            Dict[str, Any]: Evaluation result including "success" boolean
        """
        # Get the last state in the trajectory
        if not trajectory:
            return {"success": False, "reason": "Empty trajectory"}
        
        last_state = trajectory[-1]
        
        # Check if we have required criteria
        required_criteria = round_eval.get("required", [])
        
        # Default to success if we have no specific criteria
        if not required_criteria:
            return {"success": True, "reason": "No specific round criteria, assuming success"}
        
        # For each criterion, check if it's satisfied
        all_satisfied = True
        for criterion in required_criteria:
            crit_type = criterion.get("type")
            if crit_type == "element_present":
                # Check if an element is present in the page
                selector = criterion.get("selector")
                if not selector or not self._element_exists(last_state, selector):
                    all_satisfied = False
                    break
            elif crit_type == "url_contains":
                # Check if the URL contains a specific string
                url_content = criterion.get("content")
                if not url_content or url_content not in last_state.get("url", ""):
                    all_satisfied = False
                    break
            elif crit_type == "text_contains":
                # Check if text content contains specific string
                text_content = criterion.get("content")
                page_text = last_state.get("page_text", "")
                if not text_content or text_content not in page_text:
                    all_satisfied = False
                    break
        
        if all_satisfied:
            return {"success": True, "reason": "Round criteria satisfied"}
        else:
            return {"success": False, "reason": "Not all round criteria were satisfied"}
    
    def _element_exists(self, state, selector: str) -> bool:
        """Check if an element exists in the state based on a selector."""
        # This is a simple placeholder - actual implementation would depend on your state representation
        # You might use DOM inspection or check accessibility tree
        return selector in str(state)
    
    def _has_partial_progress(self, result: Dict[str, Any]) -> bool:
        """
        Check if there's partial progress toward the task goal.
        
        This helps determine if a round can be considered complete even if
        the full task isn't complete yet.
        
        Args:
            result: The evaluation result from the standard evaluator
            
        Returns:
            bool: Whether partial progress has been made
        """
        # Enhanced logging to help diagnose round transition issues
        logger.info(f"Checking for partial progress in round {self.current_round_index + 1}: {result}")
        
        # If there are subtasks and at least one is complete
        subtasks = result.get("subtasks", [])
        if subtasks and any(subtask.get("completed", False) for subtask in subtasks):
            completed_subtasks = [i for i, subtask in enumerate(subtasks) if subtask.get("completed", False)]
            logger.info(f"Found completed subtasks: {completed_subtasks}")
            return True
        
        # Check success flag directly
        if result.get("success", False):
            logger.info(f"Task reports success directly")
            return True
        
        # If we're on a relevant page (URL check)
        # For example, if searching for a forum, being on the forum page indicates progress
        current_round_info = self.get_current_round_info()
        target_url_patterns = current_round_info.get("target_url_patterns", [])
        
        # If we have URL patterns for this round, check against them
        if "current_url" in result and target_url_patterns:
            current_url = result.get("current_url", "")
            for pattern in target_url_patterns:
                if pattern in current_url:
                    logger.info(f"URL pattern '{pattern}' found in '{current_url}' - marking round as complete")
                    return True
        
        # Check round-specific completion condition if available
        if "round_specific" in result:
            round_specific = result.get("round_specific", {})
            if round_specific.get("completed", False):
                logger.info(f"Round-specific evaluation indicates completion")
                return True
        
        # If sufficient steps have been taken in this round (as a fallback)
        min_steps_for_round_completion = 10
        if "steps_taken" in result and result.get("steps_taken", 0) >= min_steps_for_round_completion:
            logger.info(f"Minimum steps threshold ({min_steps_for_round_completion}) reached for round completion")
            return True
        
        # No partial progress detected
        return False

    def advance_to_next_round(self) -> None:
        """
        Advances to the next round in the sequence.
        
        This method is safer than the original next_round() as it performs validation
        and ensures all systems are properly updated for the new round.
        """
        if self.is_final_round():
            logger.warning(f"Cannot advance beyond final round {self.current_round_index + 1}/{self.max_rounds}")
            return
        
        # Record current round data
        current_round = self.current_round_index
        current_round_info = self.get_current_round_info()
        
        # Advance the round index
        self.current_round_index += 1
        
        # Update environment variable to signal rounds status
        import os
        os.environ["SAFEARENA_MULTI_ROUND_CURRENT"] = str(self.current_round_index + 1)
        os.environ["SAFEARENA_MULTI_ROUND_TOTAL"] = str(self.max_rounds)
        
        # Get next round info
        next_round_info = self.get_current_round_info()
        
        # Update the task intent to the new round's intent
        round_intent = next_round_info.get("intent", "")
        if round_intent:
            self.intent = round_intent
            os.environ["SAFEARENA_CURRENT_INTENT"] = round_intent
            
            # Extra detailed transition logging
            logger.info(f"🎯🎯🎯 Advanced from round {current_round + 1} to {self.current_round_index + 1}")
            logger.info(f"📝📝📝 Previous round goal: {current_round_info.get('intent', 'N/A')}")
            logger.info(f"📝📝📝 New round goal: {round_intent}")
            
            # Set up the round
            logger.info(f"🚀🚀🚀 Setting up round {self.current_round_index + 1}/{self.max_rounds} with goal: {self.intent}")
        else:
            # Fallback if no intent found
            logger.error(f"❌❌❌ No intent found for round {self.current_round_index + 1}!")
            self.intent = self.original_intent
            os.environ["SAFEARENA_CURRENT_INTENT"] = self.original_intent

    def wrapper_evaluator(self, trajectory) -> Dict[str, Any]:
        """
        A wrapper evaluator that adds round information to the standard evaluation.
        
        This is provided to ensure the standard gym loop works correctly with multi-round tasks.
        """
        # First run the round-specific evaluation
        result = self.evaluate(trajectory)
        
        # Always include the round information
        result.update({
            "current_round": self.current_round_index + 1,
            "total_rounds": self.max_rounds,
            "is_final_round": self.is_final_round(),
            "task_id": self.current_original_task_id
        })
        
        return result 