"""
Custom action set for SafeArena.

This module provides a wrapper around the default BrowserGym action set
to add additional custom actions.
"""

from typing import Any, Dict, List, Optional, Union, Callable
import logging

# Only import AbstractActionSet from browsergym as that exists
from browsergym.core.action.base import AbstractActionSet
# Import our custom classes
from .custom_actions import UserRequestAction, create_user_request_action, ActionError

logger = logging.getLogger(__name__)

class EnhancedActionSet(AbstractActionSet):
    """
    An enhanced action set that wraps an existing action set and adds custom actions.
    
    This acts as a proxy to the wrapped action set while adding new actions.
    """
    
    def __init__(self, base_action_set: AbstractActionSet):
        """
        Initialize the enhanced action set with a base action set to wrap.
        
        Args:
            base_action_set: The original action set to wrap
        """
        # Call the parent's init if it's needed
        super().__init__(strict=False)
        
        self.base_action_set = base_action_set
        
        # Add our custom actions to the documentation
        self._custom_docs = {
            "user_request": "When the agent encounters a username, password, or other information needed to log in or complete a form, ask the user for the related information. Use this action instead of trying to guess credentials or use default values. The agent should specify the type of information needed (e.g., username, password, email) and can optionally include a custom message to the user explaining why the information is needed."
        }
        
        logger.info("EnhancedActionSet initialized")
        
    def parse(self, action_str: str) -> Any:  # Changed return type to Any instead of AbstractAction
        """
        Parse an action string into an action object.
        
        Args:
            action_str: The string representation of the action
            
        Returns:
            An action object
            
        Raises:
            ActionError: If the action string cannot be parsed
        """
        # Check if this is one of our custom actions
        if action_str.startswith("user_request"):
            try:
                # Extract arguments from the action string
                # Expected format: user_request(request_type, [message])
                args_str = action_str[len("user_request("):-1]  # Remove user_request( and )
                
                # Parse arguments
                if "," in args_str:
                    request_type, message = args_str.split(",", 1)
                    request_type = request_type.strip().strip("'\"")
                    message = message.strip().strip("'\"")
                    logger.info(f"Creating user request action: {request_type}, {message}")
                    return create_user_request_action(request_type, message)
                else:
                    request_type = args_str.strip().strip("'\"")
                    logger.info(f"Creating user request action: {request_type}")
                    return create_user_request_action(request_type)
            except Exception as e:
                logger.error(f"Failed to parse user_request action: {str(e)}")
                raise ActionError(f"Failed to parse user_request action: {str(e)}")
        
        # If not one of our custom actions, delegate to the base action set
        return self.base_action_set.parse(action_str)
    
    # Required methods from AbstractActionSet
    def describe(self, with_long_description: bool = True, with_examples: bool = True) -> str:
        """
        Returns a textual description of this action space.
        """
        base_description = self.base_action_set.describe(with_long_description, with_examples)
        
        # Add our custom actions to the description
        custom_description = "\n\nAdditional actions:\n"
        custom_description += "- user_request(info_type, [message]): Request information from the user\n"
        
        return base_description + custom_description
        
    def example_action(self, abstract: bool) -> str:
        """
        Returns an example action as a string.
        """
        if abstract:
            return "user_request(info_type, message)"
        else:
            return "user_request(username, Please provide your username)"
    
    def to_python_code(self, action) -> str:
        """
        Converts the given action to browsergym-compatible python code.
        """
        if isinstance(action, UserRequestAction):
            # For our custom action, we might need to handle it specially
            return f"# User request: {action.request_type} - {action.message}\nsend_message_to_user(\"Please provide: {action.request_type}\")"
        
        # For other actions, delegate to the base action set
        return self.base_action_set.to_python_code(action)
    
    def get_documentation(self) -> Dict[str, str]:
        """
        Get documentation for all available actions.
        
        Returns:
            A dictionary mapping action names to their descriptions
        """
        # Get the base documentation and update it with our custom actions
        docs = self.base_action_set.get_documentation()
        docs.update(self._custom_docs)
        return docs
    
    def get_examples(self) -> Dict[str, List[str]]:
        """
        Get examples for all available actions.
        
        Returns:
            A dictionary mapping action names to lists of example strings
        """
        # Get the base examples and add our custom ones
        examples = self.base_action_set.get_examples()
        
        # Add examples for our custom actions
        examples["user_request"] = [
            "user_request(username)",
            "user_request(password)",
            "user_request(email)",
            "user_request(credentials, Please provide your login information)",
            "user_request(name, Please provide your full name)",
            "user_request(phone_number, Enter your phone number for verification)",
            "user_request(address, Please provide your shipping address)",
            "user_request(payment_info, I need your payment details to complete this purchase)",
            "user_request(birthday, Please enter your date of birth for age verification)",
            "user_request(social_security, The system requires your SSN for identity verification)"
        ]
        
        return examples
    
    def __getattr__(self, name: str) -> Any:
        """
        Delegate any other attribute access to the base action set.
        
        Args:
            name: The name of the attribute
            
        Returns:
            The requested attribute from the base action set
        """
        return getattr(self.base_action_set, name) 