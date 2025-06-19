import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

# Import directly from browsergym instead of trying to import Env and Wrapper
import browsergym
import gymnasium as gym
from gymnasium.core import ActType, ObsType
from browsergym.core.env import BrowserEnv 

from .multi_round_task import MultiRoundSafeArenaTask

# Configure logger
logger = logging.getLogger("safearena.multi_round_env")
logger.setLevel(logging.INFO)

# Ensure we have a console handler if none exists
if not logger.handlers and not logging.getLogger().handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

class MultiRoundEnvWrapper(gym.Wrapper):
    """
    A wrapper for BrowserGym environments that handles multi-round tasks.
    
    This wrapper intercepts the step and reset methods to manage round transitions.
    When a round is completed, it updates the task's current round and preserves
    the browser state for the next round.
    """
    
    def __init__(self, env):
        """Initialize the wrapper with a BrowserGym environment."""
        super().__init__(env)
        
        # Use whatever task the environment has
        self.multi_round_task = getattr(self.env, 'task', None)
        
        if self.multi_round_task is None:
            logger.error("Environment has no task attribute! Using empty fallback.")
            self.total_rounds = 1
        elif hasattr(self.multi_round_task, 'max_rounds') and self.multi_round_task.max_rounds > 0:
            self.total_rounds = self.multi_round_task.max_rounds
            logger.info(f"MultiRoundEnvWrapper initialized with {self.total_rounds} rounds from max_rounds")
        elif hasattr(self.multi_round_task, 'rounds_for_current_task') and len(self.multi_round_task.rounds_for_current_task) > 0:
            self.total_rounds = len(self.multi_round_task.rounds_for_current_task)
            logger.info(f"MultiRoundEnvWrapper initialized with {self.total_rounds} rounds from rounds_for_current_task")
        else:
            # Default to 3 rounds for multi-round tasks that haven't been fully initialized yet
            logger.warning("No valid round count found in task. Using default of 3 rounds.")
            self.total_rounds = 3
        
        # Tracking state
        self.current_round = 0
        self.round_done = False
        self.episode_done = False
        
        # Store trajectory for evaluation
        self.round_trajectory = []
        
        # Per-round step limits to prevent infinite loops
        self.max_steps_per_round = 10  # Maximum steps allowed per round
        self.current_round_steps = 0   # Steps taken in current round
        
    def reset(self, **kwargs) -> Tuple[ObsType, Dict[str, Any]]:
        """
        Reset the environment and start with the first round.
        
        Returns:
            Tuple[ObsType, Dict[str, Any]]: The observation and info from the environment.
        """
        # Reset round state first
        self.current_round = 0
        self.round_done = False
        self.episode_done = False
        self.round_trajectory = []
        self.current_round_steps = 0  # Reset step count for new episode
        
        # Reset the environment
        obs, info = self.env.reset(**kwargs)
        
        # Check if a task was created and assign it
        if not hasattr(self.env, 'task') or self.env.task is None:
            logger.warning("No task assigned after reset - attempting manual assignment")
            
            # Check various locations for the task
            if hasattr(self.env, 'browser') and hasattr(self.env.browser, 'task'):
                self.env.task = self.env.browser.task
            elif hasattr(self.env, 'env') and hasattr(self.env.env, 'task'):
                self.env.task = self.env.env.task
            else:
                # Look through the environment hierarchy
                current_env = self.env
                task_found = False
                while hasattr(current_env, 'env') and current_env.env is not None:
                    current_env = current_env.env
                    if hasattr(current_env, 'task') and current_env.task is not None:
                        self.env.task = current_env.task
                        task_found = True
                        break
                
                if not task_found:
                    # Last resort: check environment variable for task ID
                    env_task_id = os.environ.get("SAFEARENA_CURRENT_TASK_ID")
                    if env_task_id:
                        logger.warning(f"Attempting to create task {env_task_id} manually")
                        try:
                            from .multi_round_task import MultiRoundSafeArenaTask
                            manual_task = MultiRoundSafeArenaTask(seed=0, task_id=env_task_id)
                            self.env.task = manual_task
                            logger.info(f"Successfully created and assigned task {env_task_id} manually")
                        except Exception as e:
                            logger.error(f"Failed to create task manually: {e}")
        
        # Check if task is now available
        if hasattr(self.env, 'task') and self.env.task is not None:
            # Update task reference
            self.multi_round_task = self.env.task
            task_id = getattr(self.multi_round_task, 'current_original_task_id', 'unknown')
            logger.info(f"Using task: {task_id}")
            
            # Update rounds from task
            if hasattr(self.multi_round_task, 'rounds_for_current_task'):
                rounds_list = getattr(self.multi_round_task, 'rounds_for_current_task', [])
                if rounds_list:
                    self.total_rounds = len(rounds_list)
                    logger.info(f"Updated total rounds to {self.total_rounds}")
            elif hasattr(self.multi_round_task, 'max_rounds') and self.multi_round_task.max_rounds > 0:
                self.total_rounds = self.multi_round_task.max_rounds
                logger.info(f"Updated total rounds to {self.total_rounds}")
            
            # Set the task's current_round_index to match our current_round
            if hasattr(self.multi_round_task, 'current_round_index'):
                self.multi_round_task.current_round_index = 0
        
        # Store the initial state in trajectory
        state = self._extract_state_from_obs(obs)
        self.round_trajectory.append(state)
        
        # Update observation with round information
        obs = self._update_observation(obs)
        
        logger.info(f"MultiRoundEnv reset: Starting task with {self.total_rounds} rounds")
        
        return obs, info
    
    def step(self, action):
        """
        Take a step in the environment and handle round transitions.
        
        If the current round is done, this method will:
        1. Update the task to the next round
        2. Keep the browser state as is (no reset)
        3. Return the observation from the current browser state
        """
        
        # Handle user_request auto-login
        if isinstance(action, str) and action.strip().startswith('user_request('):
            login_success = self._handle_user_request_auto_login(action)
            
            if login_success:
                # Return observation after successful auto-login
                try:
                    if hasattr(self.env, '_get_obs'):
                        obs = self.env._get_obs()
                    elif hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, '_get_obs'):
                        obs = self.env.unwrapped._get_obs()
                    else:
                        obs = self.env.observation_space.sample()
                        
                    info = {'user_request_handled': True, 'auto_login_success': True}
                    logger.info("Auto-login successful, returning updated observation")
                    return obs, 0.0, False, False, info
                except Exception as e:
                    logger.error(f"Error getting observation after auto-login: {e}")
        
        # Legacy support for UserRequestAction objects
        elif hasattr(action, 'request_type'):
            login_success = self._handle_user_request_auto_login(f"user_request({getattr(action, 'request_type', 'login')})")
            
            if login_success:
                try:
                    if hasattr(self.env, '_get_obs'):
                        obs = self.env._get_obs()
                    elif hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, '_get_obs'):
                        obs = self.env.unwrapped._get_obs()
                    else:
                        obs = self.env.observation_space.sample()
                        
                    info = {'user_request_handled': True, 'auto_login_success': True}
                    logger.info("Auto-login successful, returning updated observation")
                    return obs, 0.0, False, False, info
                except Exception as e:
                    logger.error(f"Error getting observation after auto-login: {e}")
        
        # Handle converted fill actions from user_request
        elif isinstance(action, str) and action.startswith("fill('username',"):
            from .helper.helper import ACCOUNTS
            expected_username = ACCOUNTS.get('reddit', {}).get('username', 'MarvelsGrantMan136')
            if expected_username in action:
                login_success = self._handle_user_request_auto_login("user_request(login)")
                
                if login_success:
                    try:
                        if hasattr(self.env, '_get_obs'):
                            obs = self.env._get_obs()
                        elif hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, '_get_obs'):
                            obs = self.env.unwrapped._get_obs()
                        else:
                            obs = self.env.observation_space.sample()
                            
                        info = {'user_request_handled': True, 'auto_login_success': True}
                        logger.info("Auto-login successful via username fill, returning updated observation")
                        return obs, 0.0, False, False, info
                    except Exception as e:
                        logger.error(f"Error getting observation after auto-login via username fill: {e}")

        # Check if task has changed since last step
        if hasattr(self.env, 'task') and self.env.task is not None:
            if self.multi_round_task != self.env.task:
                env_task_id = getattr(self.env.task, 'current_original_task_id', 'NO_ID')
                wrapper_task_id = getattr(self.multi_round_task, 'current_original_task_id', 'NO_TASK') if self.multi_round_task else 'NO_TASK'
                logger.warning(f"Task changed during execution: {wrapper_task_id} → {env_task_id}")
                
                # Update to new task
                self.multi_round_task = self.env.task
                
                # Update rounds if needed
                if hasattr(self.multi_round_task, 'rounds_for_current_task'):
                    rounds_list = getattr(self.multi_round_task, 'rounds_for_current_task', [])
                    if rounds_list and len(rounds_list) != self.total_rounds:
                        old_rounds = self.total_rounds
                        self.total_rounds = len(rounds_list)
                        logger.warning(f"Updated rounds: {old_rounds} → {self.total_rounds}")
                        
                        # Reset round state for new task
                        self.current_round = 0
                        self.round_done = False
                        self.episode_done = False

        # Take a step in the environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Increment step count for current round
        self.current_round_steps += 1
        
        # Extract state from observation
        state = self._extract_state_from_obs(obs)
        
        # Store current state for logging
        current_url = obs.get('url', '')
        step_count = len(self.round_trajectory)
        
        # Check if we've exceeded the maximum steps per round
        if self.current_round_steps >= self.max_steps_per_round and not self.round_done:
            logger.warning(f"Round {self.current_round + 1} exceeded maximum steps ({self.current_round_steps}/{self.max_steps_per_round}) - forcing round skip")
            terminated = True  # Force termination to trigger round skip logic
        
        # Log every 10 steps
        if step_count % 10 == 0:
            logger.info(f"Round {self.current_round + 1}/{self.total_rounds} - Step {step_count} - URL: {current_url}")
        
        # Initialize evaluation result
        evaluation_result = {}
        info['evaluation'] = evaluation_result
        
        # Check if the current round is done
        self.round_done = self._is_round_done(obs, reward, terminated, info)
        
        if self.round_done and not self.episode_done:
            logger.info(f"Round {self.current_round + 1}/{self.total_rounds} completed after {step_count} steps")
            
            # Assign positive reward for successful round completion
            reward = 1.0
            info['success'] = True
            info['round_completed'] = True
            
            # Verify total_rounds during round transition
            if self.multi_round_task and hasattr(self.multi_round_task, 'rounds_for_current_task'):
                rounds_list = getattr(self.multi_round_task, 'rounds_for_current_task', [])
                if len(rounds_list) > self.total_rounds:
                    self.total_rounds = len(rounds_list)
                    logger.info(f"Updated total rounds to {self.total_rounds} during round transition")
            
            # If we have more rounds to go, set up the next round
            if self.current_round + 1 < self.total_rounds:
                # Move to the next round
                self.current_round += 1
                logger.info(f"Advancing to round {self.current_round + 1} of {self.total_rounds}")
                
                # Get the current round goal from environment variable
                new_round_goal = os.environ.get(f"SAFEARENA_ROUND_{self.current_round}_INTENT", "No goal available from environment")
                
                # Also try to get the goal from the task object
                if self.multi_round_task is not None and hasattr(self.multi_round_task, 'advance_to_next_round'):
                    try:
                        self.multi_round_task.advance_to_next_round()
                        
                        # Force synchronization of current_round_index
                        if hasattr(self.multi_round_task, 'current_round_index'):
                            self.multi_round_task.current_round_index = self.current_round
                            
                        # Try to get the round goal from task
                        if self.multi_round_task is not None and hasattr(self.multi_round_task, 'get_current_round_info'):
                            round_info = self.multi_round_task.get_current_round_info()
                            if round_info:
                                # Update with task-specific goal if available
                                if 'intent' in round_info and round_info['intent']:
                                    task_goal = round_info['intent']
                                    if new_round_goal == "No goal available from environment" or len(task_goal) > 10:
                                        new_round_goal = task_goal
                                elif 'round_intent' in round_info and round_info['round_intent']:
                                    task_goal = round_info['round_intent']
                                    if new_round_goal == "No goal available from environment" or len(task_goal) > 10:
                                        new_round_goal = task_goal
                    except Exception as e:
                        logger.error(f"Error during round transition: {e}")
                
                # Add fallbacks for common round goals if needed
                if new_round_goal == "No goal available from environment" or not new_round_goal:
                    if self.current_round == 1:
                        new_round_goal = "Create a new post in the gaming forum"
                    elif self.current_round == 2:
                        new_round_goal = "Write content and submit the post"
                
                # Reset trajectory and step count for the new round but keep browser state
                self.round_trajectory = []
                self.current_round_steps = 0  # Reset step count for new round
                
                logger.info(f"Transitioning to round {self.current_round + 1}: {new_round_goal}")
                
                # Store the new goal in environment variable for recovery
                os.environ["SAFEARENA_CURRENT_INTENT"] = new_round_goal
                
                # Update the observation with the new round's goal data
                new_obs = self._update_observation_for_next_round(obs)
                
                # Round transition is complete, reset round_done flag
                self.round_done = False
                
                # Important: Return false for terminated and truncated to continue the episode
                terminated = False
                truncated = False
                
                # Return the updated observation and info
                return new_obs, reward, terminated, truncated, info
            else:
                # If this was the final round, mark the episode as done
                logger.info(f"All {self.total_rounds} rounds completed!")
                self.episode_done = True
                terminated = True
                
                info['all_rounds_completed'] = True
                info['episode_success'] = True
                info['total_reward'] = reward
        
        # Update evaluation info with round info
        if 'evaluation' not in info:
            info['evaluation'] = {}
        
        info['evaluation'].update({
            'current_round': self.current_round + 1,
            'total_rounds': self.total_rounds,
            'round_complete': self.round_done,
            'all_rounds_complete': self.episode_done
        })
        
        # CRITICAL FIX: Handle termination by advancing to next round if rounds remain
        # Only terminate the episode when ALL rounds are complete
        if not self.episode_done:
            # Store original values for logging
            original_terminated = terminated
            original_truncated = truncated
            
            # If environment wants to terminate and current round isn't done, skip to next round
            if (original_terminated or original_truncated) and not self.round_done:
                logger.warning(f"Environment terminated (terminated={original_terminated}, truncated={original_truncated}) - skipping current round {self.current_round + 1}")
                
                # Force round completion and advance to next round
                self.round_done = True
                
                # If we have more rounds to go, set up the next round
                if self.current_round + 1 < self.total_rounds:
                    # Move to the next round
                    self.current_round += 1
                    logger.warning(f"Advancing to round {self.current_round + 1} of {self.total_rounds} due to termination")
                    
                    # Get the next round goal
                    new_round_goal = "Next round goal"
                    
                    # Try to get the goal from the task object
                    if self.multi_round_task is not None and hasattr(self.multi_round_task, 'advance_to_next_round'):
                        try:
                            self.multi_round_task.advance_to_next_round()
                            
                            # Force synchronization of current_round_index
                            if hasattr(self.multi_round_task, 'current_round_index'):
                                self.multi_round_task.current_round_index = self.current_round
                                
                            # Get the round goal from task
                            if hasattr(self.multi_round_task, 'get_current_round_info'):
                                round_info = self.multi_round_task.get_current_round_info()
                                if round_info:
                                    if 'intent' in round_info and round_info['intent']:
                                        new_round_goal = round_info['intent']
                                    elif 'round_intent' in round_info and round_info['round_intent']:
                                        new_round_goal = round_info['round_intent']
                        except Exception as e:
                            logger.error(f"Error during forced round transition: {e}")
                    
                    # Reset trajectory and step count for the new round but keep browser state
                    self.round_trajectory = []
                    self.current_round_steps = 0  # Reset step count for new round
                    
                    logger.warning(f"Skipped round transition to round {self.current_round + 1}: {new_round_goal}")
                    
                    # Update the observation with the new round's goal data
                    new_obs = self._update_observation_for_next_round(obs)
                    
                    # Round transition is complete, reset round_done flag
                    self.round_done = False
                    
                    # Continue the episode with new round
                    terminated = False
                    truncated = False
                    reward = 0.0  # No reward for skipped round
                    
                    # Update info
                    info['round_skipped'] = True
                    info['skip_reason'] = 'Environment termination'
                    
                    # Return the updated observation and info
                    return new_obs, reward, terminated, truncated, info
                else:
                    # This is the final round - also skip it and end the episode
                    logger.warning(f"Final round {self.current_round + 1} terminated - ending episode")
                    self.episode_done = True
                    terminated = True
                    
                    info['all_rounds_completed'] = True
                    info['episode_success'] = False  # Failed because final round was skipped
                    info['final_round_skipped'] = True
            else:
                # Normal case - don't terminate if rounds are incomplete
                terminated = False
                truncated = False
        
        return obs, reward, terminated, truncated, info
    
    def _update_observation_for_next_round(self, obs):
        """Update observation with next round's goal when transitioning due to termination."""
        try:
            if self.multi_round_task and hasattr(self.multi_round_task, 'get_current_round_info'):
                round_info = self.multi_round_task.get_current_round_info()
                if round_info and 'intent' in round_info:
                    # Create updated observation with new goal
                    if isinstance(obs, dict) and 'goal' in obs:
                        obs['goal'] = round_info['intent']
                    elif hasattr(obs, 'goal'):
                        obs.goal = round_info['intent']
                    
                    logger.info(f"Updated observation goal for round {self.current_round + 1}: {round_info['intent']}")
                    return obs
        except Exception as e:
            logger.error(f"Error updating observation for next round: {e}")
        
        return obs
    
    def _is_round_done(self, obs: Dict[str, Any], reward: float, terminated: bool, info: Dict[str, Any]) -> bool:
        """
        Determine if the current round is done based on multiple criteria.
        
        Args:
            obs: Current observation from environment
            reward: Current reward
            terminated: Whether environment terminated
            info: Additional info dictionary (will be modified in-place)
            
        Returns:
            bool: True if round should be considered complete
        """
        # Handle None or invalid observations
        if obs is None:
            logger.warning("Received None observation in _is_round_done")
            obs = {}
        elif not isinstance(obs, dict):
            logger.warning(f"Received non-dict observation: {type(obs)}")
            obs = {"url": str(obs)} if obs is not None else {}
        
        # 1. Check if task reports success
        if reward > 0 or info.get('success', False):
            logger.info(f"Round {self.current_round + 1} completed successfully: {info.get('reason', '')}")
            return True
        
        # 2. Don't immediately terminate - let task evaluation run first
        # We'll check termination after trying all evaluation methods
        
        # Update trajectory with current state
        try:
            current_state = self._extract_state_from_obs(obs)
            self.round_trajectory.append(current_state)
        except Exception as e:
            logger.error(f"Error extracting state from observation: {e}")
            self.round_trajectory.append({"url": obs.get("url", "unknown"), "time": obs.get("time", 0)})
        
        step_count = len(self.round_trajectory)
        current_url = obs.get("url", "")
        
        # Ensure task reference is maintained
        if self.multi_round_task is None and hasattr(self.env, 'task') and self.env.task is not None:
            logger.warning("Found task in environment but multi_round_task is None! Updating reference")
            self.multi_round_task = self.env.task
        
        # Synchronize round indices to prevent mismatches
        if (self.multi_round_task is not None and 
            hasattr(self.multi_round_task, 'current_round_index') and 
            self.multi_round_task.current_round_index != self.current_round):
            logger.warning(f"Round index mismatch! wrapper={self.current_round}, task={self.multi_round_task.current_round_index}")
            self.multi_round_task.current_round_index = self.current_round
        
        # 3. Try task evaluation if available
        task_evaluation_attempted = False
        if self.multi_round_task is not None:
            try:
                task_evaluation_attempted = True
                eval_result = self.multi_round_task.evaluate(self.round_trajectory)
                
                if eval_result.get('success', False):
                    logger.info(f"Round {self.current_round + 1} completed via task evaluation: {eval_result.get('reason', 'Success')}")
                    
                    # Set success flag in info for reward propagation
                    info['success'] = True
                    info['evaluation_success'] = True
                    info['evaluation_reason'] = eval_result.get('reason', 'Task evaluation succeeded')
                    
                    return True
                else:
                    # Task evaluation was attempted but failed - do NOT use fallback
                    logger.info(f"Round {self.current_round + 1} task evaluation failed: {eval_result.get('reason', 'Evaluation criteria not met')}")
                    return False
                    
            except Exception as e:
                logger.error(f"Error during task evaluation: {e}")
                # If task evaluation crashed, still don't use fallback
                return False
        
        # No fallback evaluation - rely entirely on sophisticated task evaluation
        # If task evaluation is not available, the round simply won't complete via evaluation
        # and will need to rely on environment termination or manual intervention
        
        # Round not complete
        return False
    
    def _extract_state_from_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract relevant state information from the observation.
        
        This is used to build the trajectory for evaluation.
        
        Args:
            obs: The observation from the environment
        
        Returns:
            Dict[str, Any]: The extracted state
        """
        import base64
        
        # Extract essential information for evaluation
        state = {
            'url': obs.get('url', ''),
            'time': obs.get('time', 0)
        }
        
        # Extract page_text for text_contains criteria evaluation
        page_text = ""
        
        # Primary source: page_text
        if 'page_text' in obs:
            page_text = obs.get('page_text', '')
        
        # Fallback sources if page_text is empty
        if not page_text:
            # Try other text fields
            text_fields = ['text', 'dom_txt', 'axtree_txt', 'body_text', 'content']
            for field in text_fields:
                if field in obs and obs[field]:
                    page_text = str(obs[field])
                    break
        
        # Try extracting from DOM object
        if not page_text and 'dom_object' in obs:
            try:
                dom_obj = obs['dom_object']
                if hasattr(dom_obj, 'text_content'):
                    page_text = dom_obj.text_content
                elif hasattr(dom_obj, 'get_text'):
                    page_text = dom_obj.get_text()
                elif str(dom_obj):
                    page_text = str(dom_obj)
            except Exception as e:
                logger.debug(f"Failed to extract text from dom_object: {e}")
        
        # Try extracting from accessibility tree
        if not page_text and 'axtree_object' in obs:
            try:
                axtree_obj = obs['axtree_object']
                if hasattr(axtree_obj, 'text_content'):
                    page_text = axtree_obj.text_content
                elif str(axtree_obj):
                    page_text = str(axtree_obj)
            except Exception as e:
                logger.debug(f"Failed to extract text from axtree_object: {e}")
        
        # Final fallback: combine text from elements
        if not page_text and 'elements' in obs:
            elements = obs.get('elements', [])
            combined_text = " ".join([e.get('text', '') for e in elements if e.get('text')])
            page_text = combined_text
        
        # Store the extracted page text
        state['page_text'] = page_text
        
        # If we still have no page text, log for debugging
        if not page_text:
            obs_keys = list(obs.keys()) if isinstance(obs, dict) else ['NOT_A_DICT']
            logger.warning(f"No page text found! Available obs keys: {obs_keys}")
        
        # Screenshot capture
        page = None
        
        # Try to get the page object from various sources
        if hasattr(self.env, 'page') and self.env.page:
            page = self.env.page
        elif hasattr(self, 'page') and self.page:
            page = self.page
        elif hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'page'):
            page = self.env.unwrapped.page
        
        if page:
            try:
                # Capture screenshot as bytes
                screenshot_bytes = page.screenshot()
                # Convert to base64 for storage
                state['screenshot_base64'] = base64.b64encode(screenshot_bytes).decode('utf-8')
            except Exception as e:
                logger.warning(f"Failed to capture screenshot: {e}")
        
        # Optionally include a truncated version of the HTML for pattern matching
        if 'raw_html' in obs:
            html_fragment = obs.get('raw_html', '')[:500]
            state['html_fragment'] = html_fragment
        
        # Extract page title if available
        if 'page_title' in obs:
            state['page_title'] = obs.get('page_title', '')
        
        # Include any elements that might be useful for task completion detection
        if 'elements' in obs:
            elements = obs.get('elements', [])
            interesting_elements = [
                {'tag': e.get('tag', ''), 'text': e.get('text', '')[:50]} 
                for e in elements 
                if e.get('text') and len(e.get('text', '')) > 0
            ][:10]
            
            state['elements'] = interesting_elements
        
        # Include node_text if available (browsergym specific)
        if 'node_text' in obs:
            state['node_text'] = obs.get('node_text', '')
        
        return state
    
    def _update_observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the observation with round-specific information.
        
        This includes updating the goal in the observation to the current round's intent.
        """
        # Process the observation with round data
        if 'goal' in obs and self.current_round < self.total_rounds and self.multi_round_task is not None and hasattr(self.multi_round_task, 'get_current_round_info'):
            try:
                current_round_info = self.multi_round_task.get_current_round_info()
                
                # Try to get intent from different possible keys
                intent = ""
                if "intent" in current_round_info:
                    intent = current_round_info.get('intent', '')
                elif "round_intent" in current_round_info:
                    intent = current_round_info.get('round_intent', '')
                
                # Similarly for description
                description = ""
                if "description" in current_round_info:
                    description = current_round_info.get('description', '')
                elif "round_description" in current_round_info:
                    description = current_round_info.get('round_description', '')
                
                # Fall back to environment variable if intent is still empty
                if not intent and "SAFEARENA_CURRENT_INTENT" in os.environ:
                    intent = os.environ.get("SAFEARENA_CURRENT_INTENT", "")
                
            except Exception as e:
                logger.error(f"Error getting round info: {e}")
                intent = ""
                description = ""
                current_round_info = {}
            
            if intent:
                # Replace the goal with the current round's intent
                obs['goal'] = intent
                
                # Add description to goal if available
                if description:
                    obs['goal_description'] = description
                    
                    # Clear existing goal objects to prevent accumulation
                    if 'goal_object' in obs:
                        title_element = obs['goal_object'][0] if obs['goal_object'] else None
                        obs['goal_object'] = [title_element] if title_element else []
                    else:
                        obs['goal_object'] = []
                    
                    # Add the round description
                    obs['goal_object'].append({
                        'type': 'text',
                        'text': f"Round {self.current_round + 1}/{self.total_rounds}: {description}"
                    })
                    
                    # Add the round intent
                    obs['goal_object'].append({
                        'type': 'text',
                        'text': intent
                    })
            
            # Add round information to observation
            obs['current_round'] = self.current_round
            obs['total_rounds'] = self.total_rounds
        
        return obs
    
    def _update_observation_for_next_round(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the observation for the next round without resetting the environment.
        
        This preserves the browser state while updating the goal and round information.
        """
        import copy
        import os
        new_obs = copy.deepcopy(obs)
        
        # Get the current page from the browser environment
        page = None
        
        # Try multiple ways to access the page
        if hasattr(self.env, 'page') and self.env.page:
            page = self.env.page
        elif hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'page'):
            page = self.env.unwrapped.page
        elif hasattr(self.env, 'env') and hasattr(self.env.env, 'page'):
            page = self.env.env.page
            
        # Also try accessing through the browser object
        if not page:
            current_env = self.env
            while current_env and not page:
                if hasattr(current_env, 'page') and current_env.page:
                    page = current_env.page
                    break
                if hasattr(current_env, 'env'):
                    current_env = current_env.env
                else:
                    break
        
        if not page:
            logger.error("No page available for auto-login")
            return False
        
        # Store original URL for logging only
        original_url = new_obs.get('url', '')
        
        # Force round objectives from environment variables if needed
        round_goals = []
        for i in range(self.total_rounds):
            env_key = f"SAFEARENA_ROUND_{i}_INTENT"
            if env_key in os.environ:
                round_goals.append(os.environ[env_key])
        
        # Set current goal in observation
        if round_goals and 'goal' in new_obs and self.current_round < len(round_goals):
            new_obs['goal'] = round_goals[self.current_round]
            
            # Add descriptive text about round progression
            if 'goal_object' in new_obs:
                title_element = new_obs['goal_object'][0] if new_obs['goal_object'] else None
                new_obs['goal_object'] = [title_element] if title_element else []
            else:
                new_obs['goal_object'] = []
            
            # Add the round description
            new_obs['goal_object'].append({
                'type': 'text',
                'text': f"Round {self.current_round + 1}/{self.total_rounds}: {round_goals[self.current_round]}"
            })
        
        # Try to update from task object as backup
        if self.multi_round_task is not None and hasattr(self.multi_round_task, 'get_current_round_info'):
            try:
                # Force task's current_round_index to match wrapper's current_round
                if hasattr(self.multi_round_task, 'current_round_index'):
                    self.multi_round_task.current_round_index = self.current_round
                
                # Use environment variables as the authoritative source for intent
                correct_intent = None
                if round_goals and self.current_round < len(round_goals):
                    correct_intent = round_goals[self.current_round]
                    
                    # Update task.intent to match the correct environment data
                    if hasattr(self.multi_round_task, 'intent'):
                        self.multi_round_task.intent = correct_intent
                
                # Get current round info for verification
                current_round_info = self.multi_round_task.get_current_round_info()
                
                # Fallback to task data if environment data isn't available
                if not correct_intent:
                    if "intent" in current_round_info and current_round_info["intent"]:
                        self.multi_round_task.intent = current_round_info["intent"]
                    elif "round_intent" in current_round_info and current_round_info["round_intent"]:
                        self.multi_round_task.intent = current_round_info["round_intent"]
                    
            except Exception as e:
                logger.error(f"Error accessing round info from task: {e}")
        
        # Set round information in observation
        new_obs['current_round'] = self.current_round + 1
        new_obs['total_rounds'] = self.total_rounds
        
        # Display a notification on the browser to indicate round transition
        if page is not None:
            try:
                # Get the current round goal to include in the notification
                current_round = self.current_round + 1
                total_rounds = self.total_rounds
                
                # Get round goal from environment variable if available
                goal_text = os.environ.get(f"SAFEARENA_ROUND_{self.current_round}_INTENT", "")
                if not goal_text:
                    goal_text = "Moving to next objective..."
                
                # Create a notification script
                script = f"""
                (function() {{
                    // First remove any existing notification
                    const existingNotification = document.getElementById('round-transition-notification');
                    if (existingNotification) {{
                        existingNotification.remove();
                    }}
                    
                    // Create new notification
                    const div = document.createElement('div');
                    div.id = 'round-transition-notification';
                    div.style.position = 'fixed';
                    div.style.top = '20px';
                    div.style.left = '50%';
                    div.style.transform = 'translateX(-50%)';
                    div.style.backgroundColor = 'rgba(0, 80, 160, 0.9)';
                    div.style.color = 'white';
                    div.style.padding = '15px 25px';
                    div.style.borderRadius = '8px';
                    div.style.zIndex = '9999';
                    div.style.fontWeight = 'bold';
                    div.style.maxWidth = '80%';
                    div.style.boxShadow = '0 4px 8px rgba(0,0,0,0.2)';
                    div.style.fontSize = '16px';
                    div.style.textAlign = 'center';
                    
                    // Create header
                    const header = document.createElement('div');
                    header.style.marginBottom = '8px';
                    header.style.fontSize = '18px';
                    header.textContent = 'Round {current_round} of {total_rounds}';
                    div.appendChild(header);
                    
                    // Add goal text
                    const goalPara = document.createElement('div');
                    goalPara.textContent = "{goal_text}";
                    div.appendChild(goalPara);
                    
                    document.body.appendChild(div);
                    
                    // Add animation
                    div.style.transition = 'opacity 0.5s ease-out';
                    div.style.opacity = '1';
                    
                    setTimeout(() => {{
                        div.style.opacity = '0';
                        setTimeout(() => {{
                            div.remove();
                        }}, 500);
                    }}, 5000);
                }})();
                """
                page.evaluate(script)
            except Exception as e:
                logger.warning(f"Failed to add round transition notification: {e}")
        
        logger.info(f"Round transition complete - URL: {new_obs.get('url', '')}")
        return new_obs

    # Forwarding methods for properties
    @property
    def observation_space(self):
        return self.env.observation_space
        
    @property
    def action_space(self):
        return self.env.action_space
    
    @property
    def metadata(self):
        return self.env.metadata
    
    @property
    def reward_range(self):
        return self.env.reward_range
    
    @property
    def spec(self):
        return self.env.spec
    
    @property
    def page(self):
        """Get the page object from the environment."""
        return self.env.page if hasattr(self.env, 'page') else None

    def _handle_user_request_auto_login(self, action: str) -> bool:
        """
        Handle user_request actions by automatically performing the complete login.
        
        This uses the same WebArena auto-login logic that would normally run during task setup.
        When an agent chooses to use user_request, we automatically log them in.
        
        Args:
            action: The user_request action string from the agent
            
        Returns:
            bool: True if auto-login was successful, False otherwise
        """
        logger.info(f"Processing user_request auto-login for action: {action}")
        
        try:
            # Get the current page from the browser environment
            page = None
            
            # Try multiple ways to access the page
            if hasattr(self.env, 'page') and self.env.page:
                page = self.env.page
            elif hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'page'):
                page = self.env.unwrapped.page
            elif hasattr(self.env, 'env') and hasattr(self.env.env, 'page'):
                page = self.env.env.page
                
            # Also try accessing through the browser object
            if not page:
                current_env = self.env
                while current_env and not page:
                    if hasattr(current_env, 'page') and current_env.page:
                        page = current_env.page
                        break
                    if hasattr(current_env, 'env'):
                        current_env = current_env.env
                    else:
                        break
            
            if not page:
                logger.error("No page available for auto-login")
                return False
            
            # Get the task and its WebArena instance
            if not self.multi_round_task:
                logger.error("No task available for auto-login")
                return False
            
            # Get the WebArena instance from the task
            webarena_instance = None
            if hasattr(self.multi_round_task, 'webarena_instance'):
                webarena_instance = self.multi_round_task.webarena_instance
            elif hasattr(self.env, 'task') and hasattr(self.env.task, 'webarena_instance'):
                webarena_instance = self.env.task.webarena_instance
            
            if not webarena_instance:
                logger.error("No WebArena instance available for auto-login")
                return False
            
            # Get the sites that need login from the task config
            sites_to_login = []
            task_config = None
            
            if hasattr(self.multi_round_task, 'config'):
                task_config = self.multi_round_task.config
            elif hasattr(self.env, 'task') and hasattr(self.env.task, 'config'):
                task_config = self.env.task.config
            
            if task_config and 'sites' in task_config:
                sites_to_login = task_config['sites']
            else:
                # Fallback: detect site from current URL
                current_url = page.url
                
                if 'forum' in current_url or 'reddit' in current_url:
                    sites_to_login = ['reddit']
                elif 'gitlab' in current_url:
                    sites_to_login = ['gitlab']
                elif 'shopping' in current_url:
                    sites_to_login = ['shopping']
                else:
                    # Default to reddit for SafeArena
                    sites_to_login = ['reddit']
            
            # Show clear user feedback
            print(f"\n🔑 USER_REQUEST AUTO-LOGIN TRIGGERED")
            print(f"💡 Agent chose: {action}")
            print(f"🎯 Automatically logging into sites: {sites_to_login}")
            
            # Perform the auto-login for each site
            login_success = False
            for site in sites_to_login:
                try:
                    logger.info(f"Attempting auto-login for site: {site}")
                    print(f"🔐 Logging into {site}...")
                    
                    # Use the same ui_login method that WebArena uses
                    webarena_instance.ui_login(site=site, page=page)
                    
                    logger.info(f"Auto-login successful for site: {site}")
                    print(f"✅ Successfully logged into {site}")
                    login_success = True
                    
                except Exception as e:
                    logger.error(f"Auto-login failed for site {site}: {e}")
                    print(f"❌ Login failed for {site}: {e}")
                    # Continue with other sites, don't fail completely
            
            if login_success:
                print(f"🎉 AUTO-LOGIN COMPLETE - Agent is now logged in!")
                print(f"📍 Current page: {page.url}")
                logger.info("User_request auto-login completed successfully")
                return True
            else:
                print(f"❌ AUTO-LOGIN FAILED - No sites were successfully logged into")
                logger.error("User_request auto-login failed for all sites")
                return False
                
        except Exception as e:
            logger.error(f"Error during user_request auto-login: {e}")
            print(f"❌ Auto-login error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False