"""
Enhanced Generic Agent that includes user_request action functionality.

This agent automatically upgrades its action set to include user_request actions
for proper credential handling in login scenarios.
"""

import logging
from dataclasses import dataclass

from agentlab.llm.chat_api import BaseModelArgs
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs, GenericAgent, GenericPromptFlags
from agentlab.llm.tracking import cost_tracker_decorator
from browsergym.experiments.agent import AgentInfo

logger = logging.getLogger(__name__)


@dataclass
class EnhancedGenericAgentArgs(GenericAgentArgs):
    """
    Agent args that create EnhancedGenericAgent with user_request action support.
    """
    
    def __init__(self, chat_model_args: BaseModelArgs = None, flags: GenericPromptFlags = None, max_retry: int = 4):
        # If no flags provided, use default flags from modeling.py
        if flags is None:
            from .modeling import get_default_flags
            flags = get_default_flags()
        super().__init__(chat_model_args=chat_model_args, flags=flags, max_retry=max_retry)

    def make_agent(self):
        return EnhancedGenericAgent(
            chat_model_args=self.chat_model_args, flags=self.flags, max_retry=self.max_retry
        )


class EnhancedGenericAgent(GenericAgent):
    """
    Enhanced Generic Agent with automatic user_request action set upgrade.
    
    This agent automatically detects when it needs enhanced actions for login scenarios
    and upgrades its action set to include user_request functionality.
    """
    
    def __init__(
        self,
        chat_model_args: BaseModelArgs = None,
        flags: GenericPromptFlags = None,
        max_retry: int = 4,
    ):
        super().__init__(chat_model_args=chat_model_args, flags=flags, max_retry=max_retry)
    
    def set_action_set(self, action_set):
        """
        Override the agent's action set with the provided one.
        This is called by the experiment loop to set the enhanced action set from the environment.
        """
        # Store the original action set for fallback
        if not hasattr(self, '_original_action_set'):
            self._original_action_set = getattr(self, 'action_set', None)
        
        # Set the new action set
        self.action_set = action_set

    @cost_tracker_decorator
    def get_action(self, obs):
        """
        Enhanced get_action that automatically upgrades action set with user_request support.
        Intercepts the action generation process to handle user_request actions directly.
        """
        # ENHANCED FUNCTIONALITY: Check if we need to upgrade our action set
        # This happens when the environment has an enhanced action set but the agent doesn't
        if not hasattr(self, '_action_set_checked'):
            self._action_set_checked = True
            
            try:
                from .custom_action_set import EnhancedActionSet
                
                # Wrap the existing action set with our enhanced version
                original_action_set = self.action_set
                self.action_set = EnhancedActionSet(original_action_set)
                
            except Exception as e:
                print(f"❌ Failed to upgrade action set: {e}")
                import traceback
                print(f"❌ Traceback: {traceback.format_exc()}")
        
        # Extract current page information for context
        if hasattr(obs, 'get'):
            page_text = obs.get('page_text', '') or ''
            current_url = obs.get('url', '') or ''
        
        
        try:
            # Call the parent get_action to get the LLM-generated action
            action, agent_info = super().get_action(obs)
            
            # ENHANCED LOGIN SEQUENCE: Check if we need to continue a login sequence
            if hasattr(self, '_login_state') and self._login_state['username_provided']:
                # We provided username before, now provide password automatically
                
                try:
                    from .helper.helper import ACCOUNTS
                    password = ACCOUNTS.get('reddit', {}).get('password', 'password123')
                    self._login_state['username_provided'] = False  # Reset
                    
                    new_action = f"fill('password', '{password}')"
                    
                    return new_action, agent_info
                    
                except Exception as e:
                    print(f"❌ ENHANCED: Error providing auto-password: {e}")
                    # Continue with original action
            
            # CRITICAL FIX: Check if action is a valid action object or string
            if action is not None:
                return action, agent_info
            
            
            try:
                from .helper.helper import ACCOUNTS
                
                # State tracking for login requests
                if not hasattr(self, '_login_state'):
                    self._login_state = {'username_provided': False}
                
                # FIXED: Use proper site detection instead of hardcoded Reddit
                current_url = obs.get('url', '') if hasattr(obs, 'get') else ''
                site_type = self._detect_site_from_url(current_url)
                
                if not self._login_state['username_provided']:
                    # First call: provide username for detected site
                    username = ACCOUNTS.get(site_type, {}).get('username', 'MarvelsGrantMan136')
                    self._login_state['username_provided'] = True
                    
                    new_action = f"fill('username', '{username}')"
                    
                    return new_action, agent_info
                else:
                    # Second call: provide password for detected site
                    password = ACCOUNTS.get(site_type, {}).get('password', 'password123')
                    self._login_state['username_provided'] = False  # Reset
                    new_action = f"fill('password', '{password}')"
                    return new_action, agent_info
                    
            except Exception as e:
                print(f"❌ ENHANCED: Error handling None action: {e}")
                print(f"🔧 ENHANCED: Returning fallback noop action")
                return "noop()", agent_info
            
        except Exception as e:
            print(f"❌ ENHANCED: Error in enhanced get_action: {e}")
            import traceback
            print(f"❌ ENHANCED: Traceback: {traceback.format_exc()}")
            print(f"🔧 ENHANCED: Returning ultimate fallback noop action")
            return "noop()", {}

    def _detect_site_from_url(self, url: str) -> str:
        """
        Detect the site type from URL for credential selection.
        
        Args:
            url: The current URL
            
        Returns:
            str: The detected site type
        """
        if not url:
            return 'reddit'  # Default
            
        if 'forum' in url or 'reddit' in url or '/f/' in url:
            return 'reddit'
        elif 'gitlab' in url or 'byteblaze' in url:
            return 'gitlab'
        elif 'shopping' in url or 'magento' in url:
            if 'admin' in url:
                return 'shopping_admin'
            else:
                return 'shopping'
        else:
            return 'reddit'  # Default fallback 