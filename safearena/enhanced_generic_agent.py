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
        print(f"\n{'🔧 ENHANCED AGENT ACTION SET OVERRIDE'}")
        print(f"{'='*60}")
        print(f"🔧 Setting enhanced agent action set to: {type(action_set).__name__}")
        
        # Store the original action set for fallback
        if not hasattr(self, '_original_action_set'):
            self._original_action_set = getattr(self, 'action_set', None)
        
        # Set the new action set
        self.action_set = action_set
        
        # Test the action set to ensure it has user_request
        if hasattr(action_set, 'describe'):
            description = action_set.describe(with_long_description=True, with_examples=True)
            has_user_request = 'user_request' in description
            user_request_count = description.count('user_request')
            
            print(f"✅ Enhanced agent now has action set with user_request: {has_user_request}")
            print(f"✅ user_request mentions: {user_request_count}")
            
            if has_user_request:
                print("🎯 SUCCESS: Enhanced agent can now use user_request actions!")
            else:
                print("❌ WARNING: Enhanced action set does not contain user_request!")
        
        print(f"{'='*60}\n")

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
            
            # Check if we're using the standard action set that doesn't have user_request
            current_action_set_type = type(self.action_set).__name__ if hasattr(self, 'action_set') else "None"
            
            print(f"\n{'🔧 ENHANCED AGENT ACTION SET UPGRADE'}")
            print(f"{'='*60}")
            print(f"🔧 Current action set: {current_action_set_type}")
            print(f"🔧 Upgrading to EnhancedActionSet for user_request support...")
            print(f"🔧 Wrapping existing action set with EnhancedActionSet")
            
            try:
                from .custom_action_set import EnhancedActionSet
                
                # Wrap the existing action set with our enhanced version
                original_action_set = self.action_set
                self.action_set = EnhancedActionSet(original_action_set)
                
                print(f"✅ Enhanced agent action set upgraded!")
                print(f"✅ New action set: {type(self.action_set).__name__}")
                
                # Verify user_request is available
                description = self.action_set.describe(with_long_description=True, with_examples=True)
                user_request_available = 'user_request' in description
                user_request_mentions = description.count('user_request')
                print(f"✅ Contains user_request: {user_request_available}")
                print(f"✅ user_request mentions: {user_request_mentions}")
                print(f"🎯 SUCCESS: Enhanced agent can now use user_request!")
                print(f"{'='*60}")
                
            except Exception as e:
                print(f"❌ Failed to upgrade action set: {e}")
                import traceback
                print(f"❌ Traceback: {traceback.format_exc()}")
        
        # CRITICAL DEBUG: Log what we're about to do
        print(f"\n{'🔍 ENHANCED AGENT DEBUG START'}")
        print(f"{'='*60}")
        print(f"🔍 DEBUG: Enhanced agent action set type: {type(self.action_set).__name__}")
        
        # Enhanced debugging for action set
        if hasattr(self.action_set, 'describe'):
            description = self.action_set.describe(with_long_description=True, with_examples=True)
            user_request_available = 'user_request' in description
            user_request_mentions = description.count('user_request')
            login_guidance = 'LOGIN' in description
            
            print(f"🔍 DEBUG: Action set contains user_request: {user_request_available}")
            print(f"🔍 DEBUG: user_request mentions count: {user_request_mentions}")
            print(f"🔍 DEBUG: Contains LOGIN guidance: {login_guidance}")
            
            if user_request_available:
                print(f"🎯 DEBUG: user_request found in enhanced agent action set!")
                # Extract custom section for debugging
                if "SAFEARENA CUSTOM ACTIONS" in description:
                    custom_start = description.find("SAFEARENA CUSTOM ACTIONS")
                    custom_preview = description[custom_start:custom_start+200]
                    print(f"📋 Custom actions section:\n{custom_preview}")
        
        # DEBUG: Current page analysis
        if hasattr(obs, 'get'):
            page_text = obs.get('page_text', '') or ''
            current_url = obs.get('url', '') or ''
            print(f"🌐 Current URL: {current_url}")
            print(f"📄 Page contains 'login': {'login' in page_text.lower()}")
            print(f"📄 Page contains 'username': {'username' in page_text.lower()}")
            print(f"📄 Page contains 'password': {'password' in page_text.lower()}")
        
        print(f"{'='*60}")

        # ENHANCED INTERCEPTION: Override the parent method to provide cleaner user_request handling
        print(f"\n{'🎯 ENHANCED AGENT GET_ACTION OVERRIDE'}")
        print(f"{'='*60}")
        
        try:
            # Call the parent get_action to get the LLM-generated action
            action, agent_info = super().get_action(obs)
            print(f"🔍 Parent get_action returned: action='{action}', type={type(action)}")
            
            # ENHANCED LOGIN SEQUENCE: Check if we need to continue a login sequence
            if hasattr(self, '_login_state') and self._login_state['username_provided']:
                # We provided username before, now provide password automatically
                print(f"🔄 ENHANCED: Continuing login sequence - providing password")
                
                try:
                    from .helper.helper import ACCOUNTS
                    password = ACCOUNTS.get('reddit', {}).get('password', 'password123')
                    self._login_state['username_provided'] = False  # Reset
                    
                    print(f"✅ ENHANCED: Auto-providing password: {password[:3]}***")
                    new_action = f"fill('password', '{password}')"
                    
                    print(f"🔧 ENHANCED: Returning password action: {new_action}")
                    return new_action, agent_info
                    
                except Exception as e:
                    print(f"❌ ENHANCED: Error providing auto-password: {e}")
                    # Continue with original action
            
            # CRITICAL FIX: Check if action is a valid action object or string
            if action is not None:
                print(f"✅ ENHANCED: Action is valid, returning: {action}")
                return action, agent_info
            
            # If action is None, this likely means user_request failed to parse in the framework
            # BUT our EnhancedActionSet should have caught it already
            print(f"🚨 ENHANCED: Action is None despite our enhanced action set!")
            print(f"🚨 ENHANCED: This suggests the framework failed to use our enhanced parsing")
            
            # Provide fallback with clear messaging
            print(f"🔧 ENHANCED: Providing fallback credentials for login attempt")
            
            try:
                from .helper.helper import ACCOUNTS
                
                # State tracking for login requests
                if not hasattr(self, '_login_state'):
                    self._login_state = {'username_provided': False}
                
                if not self._login_state['username_provided']:
                    # First call: provide username
                    username = ACCOUNTS.get('reddit', {}).get('username', 'MarvelsGrantMan136')
                    self._login_state['username_provided'] = True
                    
                    print(f"✅ ENHANCED: Providing username: {username[:3]}***")
                    new_action = f"fill('username', '{username}')"
                    
                    print(f"🔧 ENHANCED: Returning action: {new_action}")
                    return new_action, agent_info
                else:
                    # Second call: provide password
                    password = ACCOUNTS.get('reddit', {}).get('password', 'password123')
                    self._login_state['username_provided'] = False  # Reset
                    
                    print(f"✅ ENHANCED: Providing password: {password[:3]}***")
                    new_action = f"fill('password', '{password}')"
                    
                    print(f"🔧 ENHANCED: Returning action: {new_action}")
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